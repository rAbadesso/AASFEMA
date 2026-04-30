# Automated Aorta Segmentation and Feature Extraction for Medical Analysis

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3118/)
[![MONAI](https://img.shields.io/badge/MONAI-Deep%20Learning-green)](https://project-monai.github.io/)
[![VMTK](https://img.shields.io/badge/VMTK-Vascular%20Modeling-orange)](http://www.vmtk.org/)

> **Master's Dissertation Project**\
> *Author:* Ruben Filipe Nascimento Abadesso\
> *Institution:* Universidade da Beira Interior (UBI)

## Overview

This repository contains an automated computational framework designed to segment the aortic vessel tree and extract quantitative 3D geometric features from Computed Tomography Angiography (CTA) scans. 

Cardiovascular diseases, particularly aortic aneurysms and dissections, are often evaluated clinically using simple 1D maximum diameter measurements. This pipeline aims to bridge the gap between raw radiological data and actionable clinical insights by providing a fully automated, reproducible workflow for extracting complex morphologic descriptors (such as 3D tortuosity, local curvature, and torsion) that better reflect pathological remodeling.

## Architecture & Methodology

The pipeline integrates state-of-the-art deep learning with robust geometric modeling, split into two main computational phases:

1. **Volumetric Segmentation (MONAI + SwinUNETR):** The pipeline preprocesses heterogeneous CTA scans (orientation, Hounsfield Unit normalization, voxel resampling to 1x1x1 mm) and utilizes a SwinUNETR hierarchical vision transformer to accurately delineate the aorta and its principal branches.
2. **Geometric Analysis (VMTK):** The generated 3D masks are converted into high-fidelity surface meshes. Centerlines are extracted using Voronoi diagram-based algorithms, allowing for the precise calculation of orthogonal cross-sections and advanced shape descriptors along the vessel's longitudinal axis.

## Extracted Biomarkers

The final output is a flattened `.csv` file containing point-by-point data along the entire aortic network, including:
* Maximum Inscribed Sphere Radius (Diameter)
* Local Curvature & Torsion
* Frenet Frames (Tangent, Normal, Binormal vectors)
* Cross-Sectional Area
* Equivalent Ellipticity & Eccentricity metrics

## Project Structure

```text
.
├── environment.yml                                                  # Conda environment dependencies
├── script.sh                                                        # Main executable bash wrapper
├── README.md                                                        # Project documentation
├── Relatorio_do_Projeto_de_Dissertacao_Mestrado_Ruben_Abadesso.pdf  # Full Dissertation Document
├── src/
│   ├── main_pipeline.py                                             # Unified Python script handling all phases
│   ├── best-model-epoch=1339-val_dice=0.9170.ckpt                   # Trained SwinUNETR weights
│   └── D1.nii.gz                                                    # Example 3D input scan
└── output/                                                          # Generated output files
    ├── D1.seg.nii.gz                                                # Binary segmentation mask
    ├── D1_centerline_geometry.vtp                                   # 3D Centerlines and embedded metrics
    ├── D1_cross_sections.vtp                                        # 3D Orthogonal cross-sections
    └── D1.csv                                                       # Aligned geometric features dataset
```

## Installation

The pipeline requires specific versions of PyTorch, MONAI, VMTK, and PyVista. It is highly recommended to use the provided `environment.yml` to recreate the exact Conda environment.

```bash
# Clone the repository
git clone https://github.com/rAbadesso/AASFEMA.git
cd AASFEMA

# Create and activate the environment
conda env create -f environment.yml
conda activate VmtkMonai
```
*(it's not necessary to activate the environment, since when running script.sh, this will activate it automatically)*

## Usage

The primary entry point is the `script.sh` bash wrapper, which automates directory setup and triggers the Python pipeline.

```bash
# Make the script executable (only needed once)
chmod +x script.sh

# 1. Run the pipeline (Defaults to GPU 0)
./script.sh src/D1.nii.gz

# 2. Run on a specific GPU (e.g., GPU 1)
./script.sh src/D1.nii.gz 1

# 3. Run on CPU only
./script.sh src/D1.nii.gz -1
```

### Interactive Centerline Extraction
During execution, a PyVista 3D interactive window will appear displaying the segmented surface mesh:
1. Hover your cursor over the root/origin of the vascular tree (e.g., the aortic root).
2. Press **SPACE** to place a red marker (source point).
3. Press **'Q'** to confirm. 

The system will automatically detect the distal endpoints, compute the Voronoi centerlines, and display a final verification window showing the source (Green) and target (Red) points. Press **'Q'** again to finalize feature extraction and CSV export.

## Dataset & Training Performance
The segmentation model was trained and validated on the **MICCAI SEG.A. 2023 Challenge (Aortic Vessel Tree)** dataset, a multicenter dataset comprising 56 CTA scans with diverse pathologies and acquisition protocols. The SwinUNETR model achieved an average **Dice Similarity Coefficient (DSC) of 0.93** on the test set.
