# Organoid Segmentation

Instance segmentation of nuclei and membranes in 3D organoid microscopy images using [CellposeSAM](https://github.com/mario-koddenbrock/cellpose-adapt) and watershed-based membrane segmentation.

## Overview

This project provides tools to:

- **Segment nuclei** in 3D organoid images using a pretrained or finetuned CellposeSAM model
- **Segment membranes** via watershed transform using nuclei detections as markers
- **Finetune CellposeSAM** on custom organoid data with train/test evaluation and learning curve visualization
- **Evaluate segmentation** quality with F1 score, Jaccard index, and Average Precision at multiple IoU thresholds
- **Batch process** all organoids and export metrics to CSV

## Project Structure

```
organoid_segmentation/
├── run.py                      # Single organoid segmentation pipeline
├── process_all_organoids.py    # Batch processing with metrics export
├── finetune_nuclei.py          # Finetune CellposeSAM on organoid nuclei
├── sam3_prediction.py          # SAM3 text-prompted segmentation evaluation
├── convert_to_video.py         # Export 3D image stacks to MP4
├── find_intesity_range.py      # Hyperparameter tuning for membrane intensity
├── utils/
│   ├── base_config.py          # Configuration management (JSON/YAML)
│   ├── base_model.py           # Abstract base class for segmentation models
│   ├── cli.py                  # CLI argument parsing and setup
│   ├── preprocessing.py        # Intensity rescaling and contrast enhancement
│   ├── plotting.py             # Segmentation visualization (3-view plots)
│   ├── sam3model.py            # SAM3 text-prompted video segmentation wrapper
│   └── video_io.py             # MP4 video export from image volumes
├── configs/                    # Model hyperparameter configurations (JSON)
├── cluster/                    # SLURM job scripts for HPC
│   └── finetune_nuclei.sbatch
├── data/Organoids/             # Input data (not tracked in git)
├── models/                     # Finetuned model weights
├── results/                    # Metric CSV files
└── plots/                      # Segmentation visualization outputs
```

## Installation

```bash
git clone https://github.com/mario-koddenbrock/organoid_segmentation.git
cd organoid_segmentation
pip install -r requirements.txt
```

### Requirements

- Python 3.11+
- [cellpose-adapt](https://github.com/mario-koddenbrock/cellpose-adapt) (CellposeSAM)
- napari
- scikit-image
- opencv-python
- PyTorch (with CUDA or MPS support recommended)

## Data Layout

The expected data structure under `data/Organoids/`:

```
data/Organoids/
└── <organoid_folder>/
    ├── images_cropped_isotropic/
    │   └── *.tif                       # 4D volumes (Z, C, H, W)
    └── labelmaps/
        ├── Nuclei/
        │   └── *_nuclei-labels.tif     # 3D instance masks (Z, H, W)
        └── Membranes/
            └── *_membranes-labels.tif  # 3D instance masks (Z, H, W)
```

Images are multichannel 3D stacks where channel 0 is the nuclei channel (Hoechst) and channel 1 is the membrane channel (SiR-Actin).

## Usage

### Single Organoid Segmentation

```bash
python run.py --image <path_to_image.tif> --mask <path_to_mask.tif> --config configs/best_organoid_3d_nuclei_study_config.json
```

### Batch Processing

```bash
python process_all_organoids.py
```

Processes all organoids and saves per-image metrics (F1, Jaccard) to `results/segmentation_metrics.csv`.

### Finetuning CellposeSAM

Finetune the pretrained CellposeSAM model on 2D slices extracted from 3D organoid data:

```bash
python finetune_nuclei.py \
    --data_dir data/Organoids \
    --n_epochs 100 \
    --learning_rate 5e-5 \
    --batch_size 8
```

This will:
1. Load 3D volumes and slice them into 2D along the Z-axis
2. Split into train/test sets (80/20 by default)
3. Evaluate the pretrained model on the test set
4. Finetune and save the model
5. Evaluate the finetuned model on the test set
6. Save learning curves and AP comparison plots to `models/`

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data/Organoids` | Path to organoid data |
| `--folders` | 4 x 40x folders | Which organoid folders to include |
| `--n_epochs` | 100 | Training epochs |
| `--learning_rate` | 5e-5 | Learning rate |
| `--batch_size` | 8 | Patches per GPU step |
| `--test_fraction` | 0.2 | Fraction held out for testing |
| `--min_train_masks` | 5 | Minimum masks per image for training |
| `--min_pixels` | 64 | Minimum pixels per mask instance |

### Cluster (SLURM)

Submit the finetuning job on an HPC cluster:

```bash
cd /path/to/organoid_segmentation
sbatch cluster/finetune_nuclei.sbatch
```

The sbatch script is configured for a single GPU node with 16 CPUs and 64GB RAM. All hyperparameters are defined as variables at the top of the script for easy editing. Logs are written to `cluster/logs/`.

## Configs

JSON configuration files in `configs/` store CellposeSAM hyperparameters (flow threshold, cell probability threshold, min size, intensity percentiles). Two sets are provided:

- `best_*` configs: optimized via hyperparameter search
- `manual_*` configs: hand-tuned parameters

## License

MIT License. See [LICENSE](LICENSE) for details.
