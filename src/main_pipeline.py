import os
import time
import argparse
import tempfile
import warnings
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

warnings.filterwarnings("ignore")


class VascularSegmentation:
    """
    Handles the SegResNet inference to generate a segmentation mask from a medical image.
    Saves the temporary segmentation output for VMTK consumption.
    """
    def __init__(self, model_path, input_image, input_label=None, output_dir="output"):
        self.model_path = model_path
        self.input_image = input_image
        self.input_label = input_label
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        """
        Executes the data loading, transforms, model inference, and saves the temporary segmentation output.
        """
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
    """
    Handles the interactive 3D rendering and selection process to punch a spherical
    hole in a PyVista/VTK surface mesh using the spacebar.
    """
    def __init__(self, vtk_surface, hole_radius=2.0):
        self.vtk_surface = vtk_surface
        self.hole_radius = hole_radius
        self.original_mesh = pv.wrap(self.vtk_surface)
        self.plotter = pv.Plotter(title="Spacebar Hole Puncher")
        self.mesh_actor = self.plotter.add_mesh(self.original_mesh, color='white')
        self.selected_point = None
        self.marker_actor = None

        instructions = (
            "1. Hover your mouse and press SPACE to place the marker.\n"
            "2. Press 'Q' to PUNCH the hole and proceed to centerline extraction."
        )
        self.plotter.add_text(instructions, position='upper_left', font_size=11)
        self.plotter.add_key_event('space', self._space_pressed)

    def _space_pressed(self):
        """
        Grabs hardware-scaled mouse coordinates directly from the VTK interactor
        and uses a cell picker to select the exact surface point.
        """
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
        """
        Saves the selected point coordinates and visually updates the marker.
        """
        self.selected_point = point
        if self.marker_actor is not None:
            self.plotter.remove_actor(self.marker_actor)
        marker = pv.Sphere(radius=self.hole_radius, center=point)
        self.marker_actor = self.plotter.add_mesh(marker, color='red', pickable=False)

    def run(self):
        """
        Displays the plotter window, clips the mesh based on the selected point,
        and returns the open mesh along with the point coordinates.
        """
        self.plotter.show()
        open_mesh = self.original_mesh.copy()

        if self.selected_point is not None:
            distances = np.linalg.norm(open_mesh.points - self.selected_point, axis=1)
            open_mesh["dist_to_click"] = distances
            clipped = open_mesh.clip_scalar(scalars="dist_to_click", value=self.hole_radius, invert=False)
            open_mesh = clipped.extract_surface(algorithm='dataset_surface')

        return open_mesh, self.selected_point


class CenterlineProcessing:
    """
    Manages the full VMTK pipeline. Separates geometry calculation and cross-section
    calculation to allow for incremental data saving before potential segfaults.
    """
    def __init__(self, input_seg_file, surface_out_file, geom_out_file, radius=4.0, distance_back=3.0):
        self.input_seg_file = input_seg_file
        self.surface_out_file = surface_out_file
        self.geom_out_file = geom_out_file
        self.radius = radius
        self.distance_back = distance_back

    def read_and_process_image(self):
        """
        Reads a segmentation image, generates a smooth surface using marching cubes,
        and saves the smooth surface VTP.
        """
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
        """
        Extracts 3D coordinates by traversing backwards from the network leaves.
        """
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

        return np.array(points), leaves

    def run_up_to_geometry(self):
        """
        Executes image reading, network extraction, centerline computation,
        and geometry calculation. Returns intermediate objects safely.
        """
        vtk_surface = self.read_and_process_image()

        print("\n  -> Awaiting user interaction (Spacebar to punch hole)...")
        puncher = SpacebarPuncher(vtk_surface, hole_radius=self.radius)
        open_mesh, selected_point = puncher.run()

        if selected_point is None:
            print("  -> ERROR: No point was selected. Exiting Centerline Processing.")
            return None, None, None

        print(f"\n  -> Extracting network topology on open mesh...")
        network_extractor = vmtkscripts.vmtkNetworkExtraction()
        network_extractor.Surface = open_mesh
        network_extractor.Execute()

        endpoints, leaves = self.get_network_endpoints(network_extractor.Network)
        
        if len(endpoints) < 2:
            print(f"  -> ERROR: Found {len(endpoints)} points. Minimum 2 required.")
            return None, None, None

        print(f"  -> Found {len(leaves)} branch leaves. Pulled back {self.distance_back} units from each leaf.")

        distances = np.linalg.norm(endpoints - selected_point, axis=1)
        source_idx = np.argmin(distances)
        source_point = endpoints[source_idx]
        target_points = np.delete(endpoints, source_idx, axis=0)

        print(f"  -> Assigned 1 Source Point. Assigned {len(target_points)} Target Points.")
        print("  -> Computing Centerlines (This may take a moment)...")
        
        centerlines_extractor = vmtkscripts.vmtkCenterlines()
        centerlines_extractor.Surface = vtk_surface
        centerlines_extractor.SeedSelectorName = 'pointlist'
        centerlines_extractor.SourcePoints = source_point.tolist()
        centerlines_extractor.TargetPoints = target_points.flatten().tolist()
        centerlines_extractor.AppendEndPoints = 1
        centerlines_extractor.Execute()

        if centerlines_extractor.Centerlines.GetNumberOfPoints() == 0:
             print("\n  -> ERROR: VMTK failed to generate centerlines. Target points could not be reached.")
             return None, None, None

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
        
        return geom_pv, geometry.Centerlines, vtk_surface

    def compute_cross_sections(self, centerlines, closed_surface):
        """
        Computes cross-sections. This is the isolated operation that frequently triggers segmentation faults.
        """
        print("  -> Computing Cross Sections...")
        sections = vmtkscripts.vmtkCenterlineSections()
        sections.Centerlines = centerlines
        sections.Surface = closed_surface
        sections.Execute()
        return pv.wrap(sections.CenterlineSections)


class CenterlineNode:
    """
    Helper class for building the centerline tree to calculate cumulative Euclidean length and incremental tortuosity.
    """
    def __init__(self, coord):
        self.coord = np.array(coord, dtype=float)
        self.children = []
        self.parent = None
        self.cumulative_length = 0.0
        self.root_coord = self.coord
        self.incremental_tortuosity = 1.0

    def add_child(self, child_node):
        """
        Adds a child node to the current node, enforcing the tree hierarchy.
        """
        if child_node not in self.children:
            child_node.parent = self
            self.children.append(child_node)


class FeatureExporter:
    """
    Extracts centerline and cross-section data using synchronized cell-by-cell indexing,
    filters out VMTK ghost points, computes length and tortuosity, and saves to CSV.
    """
    def __init__(self, patient_id, geom_mesh, cross_mesh, output_dir="output"):
        self.patient_id = patient_id
        self.geom_mesh = geom_mesh
        self.cross_mesh = cross_mesh
        self.output_dir = output_dir

    def export(self):
        """
        Aligns arrays synchronously, handles NaNs, computes cumulative length and tortuosity, and writes to CSV.
        """
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
            for key in ['CenterlineSectionArea', 'CenterlineSectionMinSize', 'CenterlineSectionMaxSize', 'CenterlineSectionShape', 'CenterlineSectionClosed']:
                if key in self.cross_mesh.cell_data:
                    raw_cross_data[key] = self.cross_mesh.cell_data[key]

        pts_count = 0
        for i in range(self.geom_mesh.n_cells):
            cell = self.geom_mesh.extract_cells(i)
            n_pts = cell.n_points
            
            # Geometry points are reversed to flow Root -> Leaf
            pts = cell.points[::-1]

            valid_indices = [0]
            last_valid_pt = pts[0]
            for j in range(1, len(pts)):
                if np.linalg.norm(pts[j] - last_valid_pt) < 10.0:
                    valid_indices.append(j)
                    last_valid_pt = pts[j]

            cleaned_branches.append(pts[valid_indices])

            for key in ['MaximumInscribedSphereRadius', 'Curvature', 'Torsion', 'EdgePCoordArray', 'EdgeArray', 'FrenetTangent', 'FrenetNormal', 'FrenetBinormal']:
                if key in cell.point_data:
                    # Point data is attached to the geometry, so it must be reversed as well
                    arr = cell.point_data[key][::-1]
                    cleaned_point_data[key].append(arr[valid_indices])

            if cross_centers is not None:
                # Cross-section data natively flows Root -> Leaf, NO REVERSAL
                branch_cc = cross_centers[pts_count : pts_count + n_pts]
                cleaned_cross_centers.append(branch_cc[valid_indices])

            for key, raw_arr in raw_cross_data.items():
                # Cross-section cell data natively flows Root -> Leaf, NO REVERSAL
                branch_cdata = raw_arr[pts_count : pts_count + n_pts]
                cleaned_cell_data[key].append(branch_cdata[valid_indices])

            pts_count += n_pts

        geom_points = np.concatenate(cleaned_branches)
        total_pts = len(geom_points)

        cross_points = np.concatenate(cleaned_cross_centers) if cleaned_cross_centers else np.full((total_pts, 3), np.nan)

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

        calculated_lengths = np.array(calculated_lengths)
        calculated_tortuosity = np.array(calculated_tortuosity)

        def get_concatenated_pdata(key, cols=1):
            """
            Safely extracts and concatenates cleaned point data arrays.
            """
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
            """
            Safely extracts and concatenates cleaned cross-section cell data arrays.
            """
            if key in cleaned_cell_data and len(cleaned_cell_data[key]) > 0:
                return np.concatenate(cleaned_cell_data[key])
            else:
                return np.full(total_pts, np.nan)

        data_dict = {
            'Centerline_X': geom_points[:, 0], 'Centerline_Y': geom_points[:, 1], 'Centerline_Z': geom_points[:, 2],
            'CrossCenter_X': cross_points[:, 0], 'CrossCenter_Y': cross_points[:, 1], 'CrossCenter_Z': cross_points[:, 2],
            'Length': calculated_lengths,
            'Tortuosity': calculated_tortuosity,
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
        print(f"  -> Successfully filtered VMTK artifacts and saved synchronized data to {csv_filename}")


def main():
    """
    Parses arguments and orchestrates the pipeline, saving data incrementally.
    """
    overall_start = time.time()
    
    parser = argparse.ArgumentParser(description="Full Automated Vascular Processing Pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained SegResNet checkpoint.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image (.nii.gz).")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for the final CSV.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID (e.g., '0'). Set to '-1' for CPU.")
    parser.add_argument("--radius", type=float, default=4.0, help="Hole punch radius for interactive selection.")
    parser.add_argument("--distance_back", type=float, default=3.0, help="Distance to step back from network leaf nodes.")
    args = parser.parse_args()

    if args.gpu != "-1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.basename(args.input_image).replace(".nii.gz", "")
    patient_prefix = basename.split("_")[0]
    
    print("  -> Creating failsafe empty CSV template...")
    failsafe_exporter = FeatureExporter(
        patient_id=patient_prefix,
        geom_mesh=None,
        cross_mesh=None,
        output_dir=args.output_dir
    )
    failsafe_exporter.export()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            seg_file = os.path.join(temp_dir, f"{basename}.seg.nii.gz")
            
            print("\n" + "="*50)
            print("  STEP 1: VASCULAR SEGMENTATION")
            print("="*50)
            t0 = time.time()
            
            segmentation_task = VascularSegmentation(
                model_path=args.model_path,
                input_image=args.input_image,
                output_dir=temp_dir 
            )
            segmentation_task.run()
            print(f"  [Step 1 Completed in {time.time() - t0:.2f} seconds]")

            print("\n" + "="*50)
            print("  STEP 2: CENTERLINE & GEOMETRY EXTRACTION")
            print("="*50)
            
            surface_file = os.path.join(args.output_dir, f"{basename}_smooth_surface.vtp")
            geom_file = os.path.join(args.output_dir, f"{basename}_centerline_geometry.vtp")

            centerline_task = CenterlineProcessing(
                input_seg_file=seg_file,
                surface_out_file=surface_file,
                geom_out_file=geom_file,
                radius=args.radius,
                distance_back=args.distance_back
            )
            
            geom_pv, centerlines_vtk, vtk_surface = centerline_task.run_up_to_geometry()

            if geom_pv is not None:
                print("\n  -> Centerline Geometry succeeded! Updating CSV with geometry data before risking cross-sections...")
                partial_exporter = FeatureExporter(
                    patient_id=patient_prefix,
                    geom_mesh=geom_pv,
                    cross_mesh=None,
                    output_dir=args.output_dir
                )
                partial_exporter.export()

                cross_pv = centerline_task.compute_cross_sections(centerlines_vtk, vtk_surface)
                
                print("\n  -> Cross Sections succeeded! Updating CSV with full structural data...")
                full_exporter = FeatureExporter(
                    patient_id=patient_prefix,
                    geom_mesh=geom_pv,
                    cross_mesh=cross_pv,
                    output_dir=args.output_dir
                )
                full_exporter.export()

    except Exception as e:
        print(f"\n  -> Python ERROR during processing: {e}")

    total_seconds = time.time() - overall_start
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    print(f"\n PIPELINE COMPLETED in {minutes} minutes and {seconds:.2f} seconds.")


if __name__ == '__main__':
    main()