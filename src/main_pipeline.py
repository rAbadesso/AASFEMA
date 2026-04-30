import os
import argparse
import torch
import pyvista as pv
import numpy as np
import pandas as pd
import vtk
from collections import defaultdict
from tqdm import tqdm

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    AsDiscrete, Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Spacingd, NormalizeIntensityd, CropForegroundd, Invertd, SaveImaged,
    Activationsd, AsDiscreted, Orientationd, KeepLargestConnectedComponentd
)
from monai.networks.nets import SwinUNETR
from vmtk import vmtkscripts


class VascularSegmentation:
    """
    Handles the SwinUNETR inference to generate a segmentation mask from a medical image.
    Saves the segmentation output and calculates the Dice score if an input label is provided.
    """
    def __init__(self, model_path, input_image, input_label=None, output_dir="output"):
        self.model_path = model_path
        self.input_image = input_image
        self.input_label = input_label
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        """
        Executes the data loading, transforms, model inference, and saves the .seg.nii.gz file.
        """
        print(f"Using device: {self.device} for Segmentation Inference")

        keys = ["image", "label"] if self.input_label else ["image"]
        spacing_mode = ("bilinear", "nearest") if self.input_label else ("bilinear",)

        test_transforms = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            Orientationd(keys=keys, axcodes="LAS", labels=None),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
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

        if self.input_label:
            post_label = AsDiscrete(to_onehot=2)
            post_pred_metric = AsDiscrete(argmax=True, to_onehot=2)
            dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        model = SwinUNETR(
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
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

        with torch.no_grad():
            for test_data in tqdm(test_loader, desc="Testing"):
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

                    if self.input_label:
                        label_for_metric = post_label(data["label"].to(self.device))
                        pred_for_metric = post_pred_metric(data["pred"])
                        dice_metric(y_pred=[pred_for_metric], y=[label_for_metric])

        if self.input_label:
            mean_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"\nDice Score on input: {mean_dice:.4f}")


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
            open_mesh = clipped.extract_surface()

        return open_mesh, self.selected_point


class CenterlineProcessing:
    """
    Manages the full VMTK pipeline, combining image reading, network extraction,
    interactive node selection, centerline generation, and VTP file saving.
    """
    def __init__(self, input_seg_file, geom_file, cross_file, radius=4.0, distance_back=3.0):
        self.input_seg_file = input_seg_file
        self.geom_file = geom_file
        self.cross_file = cross_file
        self.radius = radius
        self.distance_back = distance_back

    def read_and_process_image(self):
        """
        Reads a segmentation image, applies orientation and flipping, generates a
        surface using marching cubes, and smooths the resulting surface.
        """
        reader = vmtkscripts.vmtkImageReader()
        reader.InputFileName = self.input_seg_file
        reader.Orientation = 'sagittal'
        reader.Flip = [0, 0, 0]
        reader.Execute()

        marching_cubes = vmtkscripts.vmtkMarchingCubes()
        marching_cubes.Image = reader.Image
        marching_cubes.Level = 0.5
        marching_cubes.Execute()

        smoothing = vmtkscripts.vmtkSurfaceSmoothing()
        smoothing.Surface = marching_cubes.Surface
        smoothing.PassBand = 0.01
        smoothing.NumberOfIterations = 30
        smoothing.Execute()

        return smoothing.Surface

    def get_network_endpoints(self, network_polydata):
        """
        Extracts 3D coordinates by traversing backwards from the network leaves (endpoints)
        until a physical distance is accumulated, halting early if a bifurcation or dead end is reached.
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

        print(f"\n--- Network Extraction Complete ---")
        print(f"Found {len(leaves)} branches. Pulling back {self.distance_back} units from each leaf.")

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

    def extract_network_and_endpoints(self, open_surface_mesh):
        """
        Executes vmtknetworkextraction on the given open surface to extract its vascular
        network topology, and passes the resulting network to the endpoint finder.
        """
        network_extractor = vmtkscripts.vmtkNetworkExtraction()
        network_extractor.Surface = open_surface_mesh
        network_extractor.Execute()

        return self.get_network_endpoints(network_extractor.Network)

    def compute_centerlines(self, closed_surface, endpoints, selected_point):
        """
        Calculates the distances between the user's selected hole point and all extracted
        target nodes to identify the source point, then computes vmtkCenterlines.
        """
        distances = np.linalg.norm(endpoints - selected_point, axis=1)
        source_idx = np.argmin(distances)

        source_point = endpoints[source_idx]
        target_points = np.delete(endpoints, source_idx, axis=0)

        print("\n--- Centerline Extraction ---")
        print(f"Source point assigned: {source_point}")
        print(f"Target points assigned: {len(target_points)}")

        centerlines_extractor = vmtkscripts.vmtkCenterlines()
        centerlines_extractor.Surface = closed_surface
        centerlines_extractor.SeedSelectorName = 'pointlist'
        centerlines_extractor.SourcePoints = source_point.tolist()
        centerlines_extractor.TargetPoints = target_points.flatten().tolist()
        centerlines_extractor.AppendEndPoints = 1
        centerlines_extractor.Execute()

        return centerlines_extractor.Centerlines, source_point, target_points

    def post_process_and_save(self, centerlines, closed_surface):
        """
        Applies smoothing to the extracted centerlines, computes their geometry and
        cross-sections, and saves the resulting VTP files to the specified paths.
        """
        smoothing = vmtkscripts.vmtkCenterlineSmoothing()
        smoothing.Centerlines = centerlines
        smoothing.SmoothingFactor = 0.2
        smoothing.Execute()

        geometry = vmtkscripts.vmtkCenterlineGeometry()
        geometry.Centerlines = smoothing.Centerlines
        geometry.Execute()

        writer_geom = vmtkscripts.vmtkSurfaceWriter()
        writer_geom.Surface = geometry.Centerlines
        writer_geom.OutputFileName = self.geom_file
        writer_geom.Execute()

        sections = vmtkscripts.vmtkCenterlineSections()
        sections.Centerlines = smoothing.Centerlines
        sections.Surface = closed_surface
        sections.Execute()

        writer_cross = vmtkscripts.vmtkSurfaceWriter()
        writer_cross.Surface = sections.CenterlineSections
        writer_cross.OutputFileName = self.cross_file
        writer_cross.Execute()

    def visualize_results(self, closed_surface, centerlines_polydata, source_point, target_points):
        """
        Visualizes the original closed surface mesh alongside the computed centerlines
        and extracted points using PyVista.
        """
        plotter = pv.Plotter(title="Final Verification: Centerlines & Target Points")

        plotter.add_mesh(closed_surface, color='lightblue', opacity=0.4)

        centerlines_mesh = pv.wrap(centerlines_polydata)
        plotter.add_mesh(centerlines_mesh, color='black', line_width=1, render_lines_as_tubes=True)

        plotter.add_points(np.array([source_point]), color='green', point_size=18, render_points_as_spheres=True)

        if len(target_points) > 0:
            plotter.add_points(target_points, color='red', point_size=15, render_points_as_spheres=True)

        instructions = (
            "Review Centerlines.\n"
            "Green = Source Point (Closest to hole)\n"
            "Red = Target Points\n"
            "Press 'Q' to exit."
        )
        plotter.add_text(instructions, position='upper_left', font_size=11)

        plotter.show()

    def run(self):
        """
        Executes the entire Centerline Processing workflow sequentially.
        """
        vtk_surface = self.read_and_process_image()

        puncher = SpacebarPuncher(vtk_surface, hole_radius=self.radius)
        open_mesh, selected_point = puncher.run()

        if selected_point is None:
            print("No point was selected. Exiting Centerline Processing.")
            return False

        endpoints = self.extract_network_and_endpoints(open_mesh)

        if len(endpoints) < 2:
            print("Not enough valid points found to compute centerlines (minimum 2 required). Exiting.")
            return False

        centerlines, source, targets = self.compute_centerlines(vtk_surface, endpoints, selected_point)

        self.post_process_and_save(centerlines, vtk_surface)
        self.visualize_results(pv.wrap(vtk_surface), centerlines, source, targets)
        return True


class FeatureExporter:
    """
    Extracts centerline and cross-section data from VTP files, aligns them,
    and exports a flattened multidimensional CSV file.
    """
    def __init__(self, patient_id, geom_file, cross_file, output_dir="output"):
        self.patient_id = patient_id
        self.geom_file = geom_file
        self.cross_file = cross_file
        self.output_dir = output_dir

    def export(self):
        """
        Extracts centerline points, cross-section centers, and their respective
        data arrays, reverses geometry cells to match order, and saves to CSV.
        """
        geom_mesh = pv.read(self.geom_file)
        cross_mesh = pv.read(self.cross_file)

        cells = [geom_mesh.extract_cells(i) for i in range(geom_mesh.n_cells)]

        geom_points = np.concatenate([c.points[::-1] for c in cells])

        keys = ['MaximumInscribedSphereRadius', 'Curvature', 'Torsion', 'EdgeArray',
                'EdgePCoordArray', 'FrenetTangent', 'FrenetNormal', 'FrenetBinormal']

        p_data = {k: np.concatenate([c.point_data[k][::-1] for c in cells]) for k in keys}

        cross_points = cross_mesh.cell_centers().points

        data_dict = {
            'Centerline_X': geom_points[:, 0],
            'Centerline_Y': geom_points[:, 1],
            'Centerline_Z': geom_points[:, 2],

            'CrossCenter_X': cross_points[:, 0],
            'CrossCenter_Y': cross_points[:, 1],
            'CrossCenter_Z': cross_points[:, 2],

            'MaximumInscribedSphereRadius': p_data['MaximumInscribedSphereRadius'],
            'Curvature': p_data['Curvature'],
            'Torsion': p_data['Torsion'],

            'EdgeArray_0': p_data['EdgeArray'][:, 0],
            'EdgeArray_1': p_data['EdgeArray'][:, 1],
            'EdgePCoordArray': p_data['EdgePCoordArray'],

            'FrenetTangent_X': p_data['FrenetTangent'][:, 0],
            'FrenetTangent_Y': p_data['FrenetTangent'][:, 1],
            'FrenetTangent_Z': p_data['FrenetTangent'][:, 2],

            'FrenetNormal_X': p_data['FrenetNormal'][:, 0],
            'FrenetNormal_Y': p_data['FrenetNormal'][:, 1],
            'FrenetNormal_Z': p_data['FrenetNormal'][:, 2],

            'FrenetBinormal_X': p_data['FrenetBinormal'][:, 0],
            'FrenetBinormal_Y': p_data['FrenetBinormal'][:, 1],
            'FrenetBinormal_Z': p_data['FrenetBinormal'][:, 2],

            'CenterlineSectionArea': cross_mesh.cell_data['CenterlineSectionArea'],
            'CenterlineSectionMinSize': cross_mesh.cell_data['CenterlineSectionMinSize'],
            'CenterlineSectionMaxSize': cross_mesh.cell_data['CenterlineSectionMaxSize'],
            'CenterlineSectionShape': cross_mesh.cell_data['CenterlineSectionShape'],
            'CenterlineSectionClosed': cross_mesh.cell_data['CenterlineSectionClosed']
        }

        df = pd.DataFrame(data_dict)
        csv_filename = os.path.join(self.output_dir, f"{self.patient_id}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Successfully saved cleanly aligned data to {csv_filename}")


def main():
    """
    Parses arguments and sequentially orchestrates the complete segmentation,
    centerline calculation, and feature export pipeline.
    """
    parser = argparse.ArgumentParser(description="Full Automated Vascular Processing Pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained SwinUNETR checkpoint (.ckpt).")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image (.nii.gz).")
    parser.add_argument("--input_label", type=str, default=None, help="Optional ground truth label (.seg.nii.gz).")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for all generated outputs.")
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

    filename = os.path.basename(args.input_image)
    basename = filename.replace(".nii.gz", "")
    patient_prefix = basename.split("_")[0]

    seg_file = os.path.join(args.output_dir, f"{basename}.seg.nii.gz")
    geom_file = os.path.join(args.output_dir, f"{basename}_centerline_geometry.vtp")
    cross_file = os.path.join(args.output_dir, f"{basename}_cross_sections.vtp")

    print("\n========================================")
    print("STEP 1: VASCULAR SEGMENTATION INFERENCE")
    print("========================================")
    segmentation_task = VascularSegmentation(
        model_path=args.model_path,
        input_image=args.input_image,
        input_label=args.input_label,
        output_dir=args.output_dir
    )
    segmentation_task.run()

    print("\n========================================")
    print("STEP 2: CENTERLINE EXTRACTION")
    print("========================================")
    centerline_task = CenterlineProcessing(
        input_seg_file=seg_file,
        geom_file=geom_file,
        cross_file=cross_file,
        radius=args.radius,
        distance_back=args.distance_back
    )
    success = centerline_task.run()

    if success:
        print("\n========================================")
        print("STEP 3: FEATURE EXPORT")
        print("========================================")
        export_task = FeatureExporter(
            patient_id=patient_prefix,
            geom_file=geom_file,
            cross_file=cross_file,
            output_dir=args.output_dir
        )
        export_task.export()
    else:
        print("\nPipeline interrupted during Centerline Processing.")


if __name__ == '__main__':
    main()