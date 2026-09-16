import os
import time
import argparse
import tempfile
import warnings
import sys
import torch
import pyvista as pv
import numpy as np
import pandas as pd
import vtk
from collections import defaultdict, deque
from tqdm import tqdm

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    AsDiscrete, Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Spacingd, NormalizeIntensityd, CropForegroundd, Invertd, SaveImaged,
    Activationsd, AsDiscreted, Orientationd, KeepLargestConnectedComponentd
)
from monai.networks.nets import SegResNet
from vmtk import vmtkscripts

try:
    import resource
except ImportError:
    resource = None

warnings.filterwarnings("ignore")

def get_peak_ram_gb():
    """ Returns the Peak RAM used by the entire OS process up to this point in GB. """
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    return 0.0

def get_peak_vram_gb(device=None):
    """ Returns the Peak VRAM reserved by PyTorch up to this point in GB. """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    return 0.0

def reset_vram_stats(device=None):
    """ Resets the PyTorch VRAM high-water mark for step-by-step profiling. """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

class VascularSegmentation:
    """ Handles the SegResNet inference to generate a segmentation mask from a medical image. """
    def __init__(self, model_path, input_image, input_label=None, output_dir="output"):
        self.model_path = model_path
        self.input_image = input_image
        self.input_label = input_label
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        """ Executes the deep learning sliding window inference and saves the output. """
        print(f"  -> Initializing Model on {self.device}...")

        keys = ["image", "label"] if self.input_label else ["image"]
        spacing_mode = ("bilinear", "nearest") if self.input_label else ("bilinear",)

        test_transforms = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            Orientationd(keys=keys, axcodes="LAS", labels=None),
            CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            Spacingd(keys=keys, pixdim=(1, 1, 1), mode=spacing_mode),
            EnsureTyped(keys=keys, track_meta=True),
        ])

        data_dict = {"image": self.input_image}
        if self.input_label:
            data_dict["label"] = self.input_label

        test_ds = Dataset(data=[data_dict], transform=test_transforms)
        test_loader = DataLoader(test_ds, num_workers=1, batch_size=1, pin_memory=True, shuffle=False)

        post_pred_transforms = Compose([
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=True,
                to_tensor=True,
            ),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=self.output_dir,
                output_postfix="",
                output_ext=".seg.nii.gz",
                resample=False,
                separate_folder=False
            ),
        ])

        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            init_filters=48,
            blocks_down=(1, 2, 2, 4),
            dropout_prob=0.2
        ).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith("_model."):
                new_state_dict[k.replace("_model.", "", 1)] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.eval()

        print("  -> Running Sliding Window Inference...")
        with torch.no_grad():
            for test_data in tqdm(test_loader, desc="  -> Segmenting Volume"):
                test_inputs = test_data["image"].to(self.device)
                test_data["pred"] = sliding_window_inference(
                    inputs=test_inputs,
                    roi_size=(96, 96, 128),
                    sw_batch_size=8,
                    predictor=model.forward,
                    overlap=0.5
                )

                decollated_data = decollate_batch(test_data)
                for data in decollated_data:
                    post_pred_transforms(data)

class SpacebarPuncher:
    """ Handles the Step 1 interactive 3D rendering and hole punching process for root selection. """
    def __init__(self, vtk_surface, hole_radius=4.0):
        self.vtk_surface = vtk_surface
        self.hole_radius = hole_radius
        self.original_mesh = pv.wrap(self.vtk_surface)
        
        self.plotter = pv.Plotter(title="Step 1: Interactive Root Selection", window_size=[1600, 1200])
        self.plotter.set_background("white")
        
        self.mesh_actor = self.plotter.add_mesh(
            self.original_mesh, 
            color='#d94c4c', 
            specular=0.2, 
            ambient=0.2, 
            diffuse=0.8, 
            smooth_shading=True
        )
        
        self.selected_point = None
        self.marker_actor = None

        instructions = (
            "Step 1: Select Aortic Root\n"
            "--------------------------\n"
            "1. Hover your mouse over the anatomical Aortic Root.\n"
            "2. Press 'SPACE' to place the origin marker.\n"
            "3. Press 'Q' to confirm, punch the hole, and extract network."
        )
        self.plotter.add_text(instructions, position='upper_left', font_size=12, color='black')
        self.plotter.add_key_event('space', self._space_pressed)

    def _space_pressed(self):
        """ Triggers a spatial ray-cast from the camera to the mesh surface. """
        pos = self.plotter.iren.interactor.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.AddPickList(self.mesh_actor)
        picker.PickFromListOn()
        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)

        if picker.GetCellId() != -1:
            pt = picker.GetPickPosition()
            self._select_point(pt)

    def _select_point(self, point):
        """ Places a green marker sphere at the ray-casted coordinate. """
        self.selected_point = point
        if self.marker_actor is not None:
            self.plotter.remove_actor(self.marker_actor)
        
        marker = pv.Sphere(radius=self.hole_radius * 1.5, center=point)
        self.marker_actor = self.plotter.add_mesh(marker, color='#3cb44b', pickable=False)

    def run(self):
        """ Displays the interactive window, cuts the hole, and returns interaction metrics. """
        interaction_start_time = time.time()
        self.plotter.camera.azimuth -= 100
        saved_cpos = self.plotter.show(return_cpos=True)
        interaction_time = time.time() - interaction_start_time

        open_mesh = self.original_mesh.copy()
        if self.selected_point is not None:
            distances = np.linalg.norm(open_mesh.points - self.selected_point, axis=1)
            open_mesh["dist_to_click"] = distances
            clipped = open_mesh.clip_scalar(scalars="dist_to_click", value=self.hole_radius, invert=False)
            open_mesh = clipped.extract_surface()

        return open_mesh, self.selected_point, saved_cpos, interaction_time

class TopologyViewer:
    """ Displays the Step 2 verified topological endpoints with deletion capability. """
    def __init__(self, vtk_surface, source_point, target_points, saved_cpos=None, hole_radius=4.0):
        self.vtk_surface = vtk_surface
        self.source_point = source_point
        self.target_points = target_points
        self.hole_radius = hole_radius
        self.saved_cpos = saved_cpos
        self.original_mesh = pv.wrap(self.vtk_surface)
        
        self.plotter = pv.Plotter(title="Step 2: Network Topology Review", window_size=[1600, 1200])
        self.plotter.set_background("white")
        
        self.plotter.add_mesh(
            self.original_mesh, 
            color='#d94c4c', 
            specular=0.2, 
            ambient=0.2, 
            diffuse=0.8, 
            smooth_shading=True
        )
        
        if self.source_point is not None:
            root_marker = pv.Sphere(radius=self.hole_radius * 1.5, center=self.source_point)
            self.plotter.add_mesh(root_marker, color='#3cb44b', pickable=False)
            
        self.target_actors = {}
        if self.target_points is not None and len(self.target_points) > 0:
            for pt in self.target_points:
                marker = pv.Sphere(radius=self.hole_radius, center=pt)
                actor = self.plotter.add_mesh(marker, color='#ffe119', pickable=True)
                self.target_actors[actor] = pt

        instructions = (
            "Step 2: Topology Verification\n"
            "-----------------------------\n"
            "Green Sphere: Selected Aortic Root (Source Point)\n"
            "Yellow Spheres: Detected Distal Endpoints (Target Points)\n"
            "Actions:\n"
            "- Hover over any invalid yellow endpoint and press 'R' to delete.\n"
            "- Press 'Q' to confirm and extract centerlines."
        )
        self.plotter.add_text(instructions, position='upper_left', font_size=12, color='black')
        self.plotter.add_key_event('r', self._r_pressed)

    def _r_pressed(self):
        """ Removes hovered endpoint actors from both the viewport and target array. """
        pos = self.plotter.iren.interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
        actor = picker.GetActor()
        
        if actor in self.target_actors:
            self.plotter.remove_actor(actor)
            del self.target_actors[actor]
            self.target_points = np.array(list(self.target_actors.values()))

    def run(self):
        """ Shows the topology validation window and returns pruned endpoints and time. """
        interaction_start_time = time.time()
        if self.saved_cpos is not None:
            self.plotter.camera_position = self.saved_cpos
        self.plotter.show()
        interaction_time = time.time() - interaction_start_time
        
        return self.target_points, interaction_time

class CenterlineProcessing:
    """ Computes the topological centerlines, geometry, and cross-sections via VMTK. """
    def __init__(self, input_seg_file, surface_out_file, geom_out_file, radius=4.0, distance_back=3.0):
        self.input_seg_file = input_seg_file
        self.surface_out_file = surface_out_file
        self.geom_out_file = geom_out_file
        self.radius = radius
        self.distance_back = distance_back

    def read_and_process_image(self):
        """ Reconstructs and smooths a 3D surface from the binary volumetric mask. """
        print("  -> Reading segmentation image into VMTK...")
        reader = vmtkscripts.vmtkImageReader()
        reader.InputFileName = self.input_seg_file
        reader.Orientation = 'sagittal'
        reader.Flip = [0, 0, 0]
        reader.Execute()

        print("  -> Running Marching Cubes to generate 3D Surface...")
        marching_cubes = vmtkscripts.vmtkMarchingCubes()
        marching_cubes.Image = reader.Image
        marching_cubes.Level = 0.5
        marching_cubes.Execute()

        print("  -> Smoothing Surface (30 Iterations)...")
        smoothing = vmtkscripts.vmtkSurfaceSmoothing()
        smoothing.Surface = marching_cubes.Surface
        smoothing.PassBand = 0.01
        smoothing.NumberOfIterations = 30
        smoothing.Execute()

        print(f"  -> Saving Smoothed Surface to {self.surface_out_file}...")
        writer = vmtkscripts.vmtkSurfaceWriter()
        writer.Surface = smoothing.Surface
        writer.OutputFileName = self.surface_out_file
        writer.Execute()

        return smoothing.Surface

    def get_network_endpoints(self, network_polydata):
        """ Extracts the terminal leaf nodes from a VMTK network polydata. """
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(network_polydata)
        cleaner.PointMergingOn()
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(1e-5)
        cleaner.Update()
        clean_network = cleaner.GetOutput()

        adj = defaultdict(set)
        for i in range(clean_network.GetNumberOfCells()):
            cell = clean_network.GetCell(i)
            if cell.GetCellType() in (vtk.VTK_LINE, vtk.VTK_POLY_LINE):
                for j in range(cell.GetNumberOfPoints() - 1):
                    u = cell.GetPointId(j)
                    v = cell.GetPointId(j + 1)
                    adj[u].add(v)
                    adj[v].add(u)

        leaves = [node for node, neighbors in adj.items() if len(neighbors) == 1]
        points = []

        for leaf in leaves:
            current_node = leaf
            previous_node = None
            accumulated_dist = 0.0

            while accumulated_dist < self.distance_back:
                neighbors = adj[current_node]
                if len(neighbors) > 2 and previous_node is not None:
                    break
                valid_neighbors = [n for n in neighbors if n != previous_node]
                if not valid_neighbors:
                    break
                next_node = valid_neighbors[0]

                pt1 = np.array(clean_network.GetPoint(current_node))
                pt2 = np.array(clean_network.GetPoint(next_node))
                accumulated_dist += np.linalg.norm(pt1 - pt2)

                previous_node = current_node
                current_node = next_node

            pt = clean_network.GetPoint(current_node)
            points.append(pt)

        return np.array(points)

    def run_up_to_geometry(self):
        """ Coordinates surface generation, the two-stage topological interaction, and centerline extraction. """
        vtk_surface = self.read_and_process_image()

        print("\n  -> Awaiting user interaction (Step 1: Root Selection)...")
        puncher = SpacebarPuncher(vtk_surface, hole_radius=self.radius)
        open_mesh, selected_point, saved_cpos, interaction_time_1 = puncher.run()

        if selected_point is None:
            print("  -> ERROR: No point was selected. Exiting Centerline Processing.")
            return None, None, None, interaction_time_1

        print(f"\n  -> Extracting network topology on open mesh...")
        network_extractor = vmtkscripts.vmtkNetworkExtraction()
        network_extractor.Surface = open_mesh
        network_extractor.Execute()

        endpoints = self.get_network_endpoints(network_extractor.Network)

        if len(endpoints) < 2:
            print(f"  -> ERROR: Found {len(endpoints)} points. Minimum 2 required.")
            return None, None, None, interaction_time_1

        distances = np.linalg.norm(endpoints - selected_point, axis=1)
        source_idx = np.argmin(distances)
        source_point = endpoints[source_idx]
        target_points = np.delete(endpoints, source_idx, axis=0)
        
        print("\n  -> Awaiting user interaction (Step 2: Topology Verification)...")
        viewer = TopologyViewer(
            vtk_surface=vtk_surface,
            source_point=source_point,
            target_points=target_points,
            saved_cpos=saved_cpos,
            hole_radius=self.radius
        )
        final_target_points, interaction_time_2 = viewer.run()
        total_interaction_time = interaction_time_1 + interaction_time_2
        
        if len(final_target_points) == 0:
            print("  -> ERROR: All target points were removed. Exiting Centerline Processing.")
            return None, None, None, total_interaction_time

        print("  -> Computing Centerlines (This may take a moment)...")
        centerlines_extractor = vmtkscripts.vmtkCenterlines()
        centerlines_extractor.Surface = vtk_surface
        centerlines_extractor.SeedSelectorName = 'pointlist'
        centerlines_extractor.SourcePoints = source_point.tolist()
        centerlines_extractor.TargetPoints = final_target_points.flatten().tolist()
        centerlines_extractor.AppendEndPoints = 1
        centerlines_extractor.Execute()

        if centerlines_extractor.Centerlines.GetNumberOfPoints() == 0:
            print("\n  -> ERROR: VMTK failed to generate centerlines. Target points could not be reached.")
            return None, None, None, total_interaction_time

        print("  -> Smoothing Centerlines...")
        smoothing = vmtkscripts.vmtkCenterlineSmoothing()
        smoothing.Centerlines = centerlines_extractor.Centerlines
        smoothing.SmoothingFactor = 0.2
        smoothing.Execute()

        print("  -> Computing Centerline Geometry (Curvature, Torsion, etc.)...")
        geometry = vmtkscripts.vmtkCenterlineGeometry()
        geometry.Centerlines = smoothing.Centerlines
        geometry.Execute()

        print(f"  -> Saving Centerline Geometry to {self.geom_out_file}...")
        writer = vmtkscripts.vmtkSurfaceWriter()
        writer.Surface = geometry.Centerlines
        writer.OutputFileName = self.geom_out_file
        writer.Execute()

        geom_pv = pv.wrap(geometry.Centerlines)
        return geom_pv, geometry.Centerlines, vtk_surface, total_interaction_time

    def compute_cross_sections(self, centerlines, closed_surface):
        """ Evaluates localized cross-sectional metrics perpendicular to the extracted centerlines. """
        print("  -> Computing Cross Sections...")
        sections = vmtkscripts.vmtkCenterlineSections()
        sections.Centerlines = centerlines
        sections.Surface = closed_surface
        sections.Execute()
        return pv.wrap(sections.CenterlineSections)

class CenterlineNode:
    """ Datastructure for constructing a continuous topological tree of the vascular centerlines. """
    def __init__(self, coord):
        self.coord = np.array(coord, dtype=float)
        self.children = []
        self.parent = None
        self.cumulative_length = 0.0
        self.root_coord = self.coord
        self.incremental_tortuosity = 1.0

    def add_child(self, child_node):
        """ Links sequential centerline nodes to compute integrated length and tortuosity. """
        if child_node not in self.children:
            child_node.parent = self
            self.children.append(child_node)

class FeatureExporter:
    """ Maps VMTK geometric attributes and outputs a unified CSV formatted file. """
    def __init__(self, patient_id, geom_mesh, cross_mesh, output_dir="output"):
        self.patient_id = patient_id
        self.geom_mesh = geom_mesh
        self.cross_mesh = cross_mesh
        self.output_dir = output_dir

    def export(self):
        """ Aggregates centerline features, cross-sectional profiles, and topological data to CSV. """
        expected_columns = [
            'Centerline_X', 'Centerline_Y', 'Centerline_Z',
            'CrossCenter_X', 'CrossCenter_Y', 'CrossCenter_Z',
            'Length', 'Tortuosity', 'MaximumInscribedSphereRadius', 'Curvature', 'Torsion',
            'EdgePCoordArray', 'EdgeArray_0', 'EdgeArray_1',
            'FrenetTangent_X', 'FrenetTangent_Y', 'FrenetTangent_Z',
            'FrenetNormal_X', 'FrenetNormal_Y', 'FrenetNormal_Z',
            'FrenetBinormal_X', 'FrenetBinormal_Y', 'FrenetBinormal_Z',
            'CenterlineSectionArea', 'CenterlineSectionMinSize',
            'CenterlineSectionMaxSize', 'CenterlineSectionShape',
            'CenterlineSectionClosed'
        ]

        if self.geom_mesh is None:
            df = pd.DataFrame(columns=expected_columns)
            csv_filename = os.path.join(self.output_dir, f"{self.patient_id}.csv")
            df.to_csv(csv_filename, index=False)
            return

        cleaned_branches = []
        cleaned_cross_centers = []
        cleaned_point_data = defaultdict(list)
        cleaned_cell_data = defaultdict(list)

        cross_centers = self.cross_mesh.cell_centers().points if self.cross_mesh is not None else None
        raw_cross_data = {}
        if self.cross_mesh is not None:
            for key in ['CenterlineSectionArea', 'CenterlineSectionMinSize', 'CenterlineSectionMaxSize',
                        'CenterlineSectionShape', 'CenterlineSectionClosed']:
                if key in self.cross_mesh.cell_data:
                    raw_cross_data[key] = self.cross_mesh.cell_data[key]

        pts_count = 0
        for i in range(self.geom_mesh.n_cells):
            cell = self.geom_mesh.extract_cells(i)
            n_pts = cell.n_points
            pts = cell.points[::-1]

            valid_indices = [0]
            last_valid_pt = pts[0]
            for j in range(1, len(pts)):
                if np.linalg.norm(pts[j] - last_valid_pt) < 10.0:
                    valid_indices.append(j)
                    last_valid_pt = pts[j]

            cleaned_branches.append(pts[valid_indices])

            for key in ['MaximumInscribedSphereRadius', 'Curvature', 'Torsion', 'EdgePCoordArray', 'EdgeArray',
                        'FrenetTangent', 'FrenetNormal', 'FrenetBinormal']:
                if key in cell.point_data:
                    arr = cell.point_data[key][::-1]
                    cleaned_point_data[key].append(arr[valid_indices])

            if cross_centers is not None:
                branch_cc = cross_centers[pts_count: pts_count + n_pts]
                cleaned_cross_centers.append(branch_cc[valid_indices])

            for key, raw_arr in raw_cross_data.items():
                branch_cdata = raw_arr[pts_count: pts_count + n_pts]
                cleaned_cell_data[key].append(branch_cdata[valid_indices])

            pts_count += n_pts

        geom_points = np.concatenate(cleaned_branches)
        total_pts = len(geom_points)
        cross_points = np.concatenate(cleaned_cross_centers) if cleaned_cross_centers else np.full((total_pts, 3),
                                                                                                   np.nan)

        coord_to_node = {}
        for pts in cleaned_branches:
            prev_node = None
            for pt in pts:
                key = tuple(np.round(pt, 5))
                node = coord_to_node.get(key)
                if node is None:
                    node = CenterlineNode(pt)
                    coord_to_node[key] = node

                if prev_node is not None and node not in prev_node.children:
                    prev_node.add_child(node)
                prev_node = node

        roots = [n for n in coord_to_node.values() if n.parent is None]
        q = deque(roots)
        seen = set([id(r) for r in roots])

        while q:
            parent_node = q.popleft()
            for child_node in parent_node.children:
                if id(child_node) not in seen:
                    dist = np.linalg.norm(child_node.coord - parent_node.coord)
                    child_node.cumulative_length = parent_node.cumulative_length + dist
                    child_node.root_coord = parent_node.root_coord
                    straight_dist = np.linalg.norm(child_node.coord - child_node.root_coord)
                    if straight_dist > 0:
                        child_node.incremental_tortuosity = child_node.cumulative_length / straight_dist
                    q.append(child_node)
                    seen.add(id(child_node))

        calculated_lengths = []
        calculated_tortuosity = []
        for pts in cleaned_branches:
            for pt in pts:
                key = tuple(np.round(pt, 5))
                node = coord_to_node[key]
                calculated_lengths.append(node.cumulative_length)
                calculated_tortuosity.append(node.incremental_tortuosity)

        def get_concatenated_pdata(key, cols=1):
            if key in cleaned_point_data and len(cleaned_point_data[key]) > 0:
                return np.concatenate(cleaned_point_data[key])
            else:
                shape = (total_pts,) if cols == 1 else (total_pts, cols)
                return np.full(shape, np.nan)

        edge_array = get_concatenated_pdata('EdgeArray', cols=2)
        frenet_t = get_concatenated_pdata('FrenetTangent', cols=3)
        frenet_n = get_concatenated_pdata('FrenetNormal', cols=3)
        frenet_b = get_concatenated_pdata('FrenetBinormal', cols=3)

        def get_concatenated_cdata(key):
            if key in cleaned_cell_data and len(cleaned_cell_data[key]) > 0:
                return np.concatenate(cleaned_cell_data[key])
            return np.full(total_pts, np.nan)

        data_dict = {
            'Centerline_X': geom_points[:, 0], 'Centerline_Y': geom_points[:, 1], 'Centerline_Z': geom_points[:, 2],
            'CrossCenter_X': cross_points[:, 0], 'CrossCenter_Y': cross_points[:, 1],
            'CrossCenter_Z': cross_points[:, 2],
            'Length': np.array(calculated_lengths),
            'Tortuosity': np.array(calculated_tortuosity),
            'MaximumInscribedSphereRadius': get_concatenated_pdata('MaximumInscribedSphereRadius', cols=1),
            'Curvature': get_concatenated_pdata('Curvature', cols=1),
            'Torsion': get_concatenated_pdata('Torsion', cols=1),
            'EdgePCoordArray': get_concatenated_pdata('EdgePCoordArray', cols=1),
            'EdgeArray_0': edge_array[:, 0], 'EdgeArray_1': edge_array[:, 1],
            'FrenetTangent_X': frenet_t[:, 0], 'FrenetTangent_Y': frenet_t[:, 1], 'FrenetTangent_Z': frenet_t[:, 2],
            'FrenetNormal_X': frenet_n[:, 0], 'FrenetNormal_Y': frenet_n[:, 1], 'FrenetNormal_Z': frenet_n[:, 2],
            'FrenetBinormal_X': frenet_b[:, 0], 'FrenetBinormal_Y': frenet_b[:, 1], 'FrenetBinormal_Z': frenet_b[:, 2],
            'CenterlineSectionArea': get_concatenated_cdata('CenterlineSectionArea'),
            'CenterlineSectionMinSize': get_concatenated_cdata('CenterlineSectionMinSize'),
            'CenterlineSectionMaxSize': get_concatenated_cdata('CenterlineSectionMaxSize'),
            'CenterlineSectionShape': get_concatenated_cdata('CenterlineSectionShape'),
            'CenterlineSectionClosed': get_concatenated_cdata('CenterlineSectionClosed')
        }

        df = pd.DataFrame(data_dict)
        csv_filename = os.path.join(self.output_dir, f"{self.patient_id}.csv")
        df.to_csv(csv_filename, index=False)

def main():
    """ Directs the automated pipeline handling arguments, execution flow, and final resource reporting. """
    parser = argparse.ArgumentParser(description="Full Automated Vascular Processing Pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained SegResNet checkpoint.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image (.nii.gz).")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for the final CSV.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID (e.g., '0'). Set to '-1' for CPU.")
    parser.add_argument("--radius", type=float, default=4.0, help="Hole punch radius for interactive selection.")
    parser.add_argument("--distance_back", type=float, default=3.0, help="Distance to step back from leaf nodes.")
    args = parser.parse_args()

    if args.gpu != "-1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.basename(args.input_image).replace(".nii.gz", "")
    patient_prefix = basename.split("_")[0]

    overall_peak_vram = 0.0

    print("  -> Creating failsafe empty CSV template...")
    failsafe_exporter = FeatureExporter(patient_prefix, None, None, args.output_dir)
    failsafe_exporter.export()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            seg_file = os.path.join(temp_dir, f"{basename}.seg.nii.gz")

            print("\n" + "=" * 60)
            print("  STEP 1: VASCULAR SEGMENTATION")
            print("=" * 60)

            reset_vram_stats()
            t0 = time.time()

            segmentation_task = VascularSegmentation(
                model_path=args.model_path,
                input_image=args.input_image,
                output_dir=temp_dir
            )
            segmentation_task.run()

            time_seg = time.time() - t0
            vram_seg = get_peak_vram_gb()
            ram_seg = get_peak_ram_gb()
            overall_peak_vram = max(overall_peak_vram, vram_seg)

            print(f"\n  [Step 1 Metrics] Time: {time_seg:.2f}s | Peak VRAM: {vram_seg:.2f} GB | Peak RAM: {ram_seg:.2f} GB")

            print("\n" + "=" * 60)
            print("  STEP 2: CENTERLINE & GEOMETRY EXTRACTION")
            print("=" * 60)

            reset_vram_stats()
            t0 = time.time()

            surface_file = os.path.join(args.output_dir, f"{basename}_smooth_surface.vtp")
            geom_file = os.path.join(args.output_dir, f"{basename}_centerline_geometry.vtp")

            centerline_task = CenterlineProcessing(
                input_seg_file=seg_file,
                surface_out_file=surface_file,
                geom_out_file=geom_file,
                radius=args.radius,
                distance_back=args.distance_back
            )

            geom_pv, centerlines_vtk, vtk_surface, total_interaction_time = centerline_task.run_up_to_geometry()

            time_geom = (time.time() - t0) - total_interaction_time
            vram_geom = get_peak_vram_gb()
            ram_geom = get_peak_ram_gb()
            overall_peak_vram = max(overall_peak_vram, vram_geom)

            print(f"\n  [Step 2 Metrics] Time: {time_geom:.2f}s | Peak VRAM: {vram_geom:.2f} GB | Peak RAM: {ram_geom:.2f} GB")

            if geom_pv is not None:
                print("\n" + "=" * 60)
                print("  STEP 3: CROSS SECTIONS & EXPORT")
                print("=" * 60)

                reset_vram_stats()
                t0 = time.time()

                cross_pv = centerline_task.compute_cross_sections(centerlines_vtk, vtk_surface)

                full_exporter = FeatureExporter(
                    patient_id=patient_prefix,
                    geom_mesh=geom_pv,
                    cross_mesh=cross_pv,
                    output_dir=args.output_dir
                )
                full_exporter.export()

                time_cross = time.time() - t0
                vram_cross = get_peak_vram_gb()
                ram_cross = get_peak_ram_gb()
                overall_peak_vram = max(overall_peak_vram, vram_cross)

                print(f"\n  [Step 3 Metrics] Time: {time_cross:.2f}s | Peak VRAM: {vram_cross:.2f} GB | Peak RAM: {ram_cross:.2f} GB")

    except Exception as e:
        print(f"\n  -> Python ERROR during processing: {e}")

    total_compute_time = time_seg + time_geom + (time_cross if 'time_cross' in locals() else 0.0)
    total_ram = get_peak_ram_gb()

    print("\n" + "=" * 60)
    print("  COMPUTATIONAL EFFICIENCY SUMMARY")
    print("=" * 60)
    print(f"  Total Computational Time : {total_compute_time / 60:.0f} min {total_compute_time % 60:.2f} sec")
    print(f"  Overall Peak VRAM        : {overall_peak_vram:.2f} GB")
    print(f"  Overall Peak OS RAM      : {total_ram:.2f} GB")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()