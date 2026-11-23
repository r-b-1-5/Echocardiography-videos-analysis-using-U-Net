# EchoNet-Dynamic LV Segmentation and EF Estimation with U-Net

## Project overview

The implementation uses EchoNet-Dynamic videos and tracing annotations to create paired end-diastolic (ED) and end-systolic (ES) inputs. It trains a U-Net to segment the left-ventricular (LV) cavity in both phases and estimates ejection fraction (EF) using a single-view 20-disc Simpson approximation.

The model receives two grayscale channels:

- Channel 0: ED frame
- Channel 1: ES frame

The model produces four logits channels: ED background/LV and ES background/LV. The training objective remains the notebook's combination of Cross-Entropy, soft Dice loss, and differentiable EF MAE (EF MAE is divided by 100 before it is added to the objective).

The final hard-mask EF is an approximation based on predicted binary masks. The estimated EDV/ESV are relative pixel³ units because this dataset setup does not provide physical pixel spacing. The `EF < 50%` binary classification output is the notebook's user-defined task threshold; it should not be interpreted as an independent clinical diagnosis.

## Repository structure

```text
EchoNet-UNet/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── config.py       # defaults, validation, command-line arguments
    ├── dataset.py      # CSVs, official splits, videos, tracing masks, PyTorch Dataset
    ├── models.py       # Milesial U-Net import and local fallback architecture
    ├── losses.py       # differentiable Simpson volume and combined loss
    ├── training.py     # training/validation loop and best-checkpoint selection
    ├── evaluate.py     # hard-mask EF, test metrics, CSVs, figures, run summary
    ├── utils.py        # random seeds, device/environment, shared visualizations
    └── train.py        # Python command-line entry point
```

## Dataset setup

Download/access the EchoNet-Dynamic dataset under its applicable terms. The project expects a dataset directory containing:

```text
<DATA_ROOT>/
├── FileList.csv
├── VolumeTracings.csv
└── Videos/
    ├── <study-id>.mp4
    ├── ...
```

The data loader searches recursively for a directory containing both CSV files. It searches `/kaggle/input`, `/content`, and the current working directory when `--data-root` is omitted. If more than one dataset copy is attached, pass the intended dataset directory explicitly.

The official split column in `FileList.csv` is used. The code does not create a new random train/validation/test split. Optional sample limits select studies from each respective official split using the configured random seed.

## Installation

Use Python 3.10 or newer. Create and activate a virtual environment if you are running locally:

```bash
python -m venv .venv
```

Activate it, then install the dependencies:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a local NVIDIA GPU, install a PyTorch build compatible with your CUDA driver using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). The notebook's default AMP setting is enabled only when CUDA is available.

The model loader first tries to import [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), cloning it into the working directory if needed. If that repository cannot be obtained/imported (for example, if internet access is unavailable), it uses the U-Net-compatible fallback architecture included in `src/models.py`.

## Run with Python

Run commands from the project root (the directory containing `README.md`):

```bash
python -m src.train
```

This uses the notebook defaults:

- `IMAGE_SIZE=112`
- `BATCH_SIZE=8`
- `EPOCHS=50`
- `LEARNING_RATE=1e-4`
- `WEIGHT_DECAY=1e-5`
- `NUM_WORKERS=0`
- `N_SIMPSON_DISCS=20`
- all studies in the official train/validation/test splits

To specify the dataset and output folders:

```bash
python -m src.train \
  --data-root /path/to/echonet-dynamic \
  --output-dir ./echonet_results
```

### Small smoke run

Test the data and training pipeline with a small subset before using the full dataset:

```bash
python -m src.train \
  --epochs 2 \
  --train-limit 120 \
  --val-limit 30 \
  --test-limit 15
```

This samples studies independently from the original official splits. It is a pipeline check, not a substitute for the full experiment.

### Common options

```text
--data-root PATH          Dataset directory (optional if auto-discovery finds it)
--output-dir PATH         Output directory
--train-limit INTEGER     Max studies selected from official train split; default: all
--val-limit INTEGER       Max studies selected from official validation split; default: all
--test-limit INTEGER      Max studies selected from official test split; default: all
--epochs INTEGER          Number of epochs (default: 50)
--batch-size INTEGER      Batch size (default: 8)
--learning-rate FLOAT     Learning rate (default: 0.0001)
--weight-decay FLOAT      Weight decay (default: 0.00001)
--num-workers INTEGER     DataLoader workers (default: 0)
--image-size INTEGER      Image side length (default: 112)
--simpson-discs INTEGER   Number of Simpson discs (default: 20)
--ef-loss-weight FLOAT    EF loss weight (default: 1.0)
--task-ef-threshold FLOAT EF classification threshold (default: 50.0)
--seed INTEGER            Random seed (default: 42)
--disable-amp             Disable CUDA automatic mixed precision
```

See all current options with:

```bash
python -m src.train --help
```

## Outputs

The run writes the following files under the output directory:

| Artifact | Description |
|---|---|
| `best_unet_echonet.pth` | Best validation checkpoint (model, optimizer, epoch, history) |
| `best_unet_echonet_state_dict.pth` | Final saved state dictionary after restoring the best checkpoint |
| `test_predictions.csv` | Study-level clinical EF, soft/hard EF, EDV/ESV estimates, and threshold-task labels |
| `run_summary.txt` | Device, split counts, best validation objective, and test metrics |
| `training_curves.png` | Combined, CE, Dice, and EF training/validation curves |
| `learning_rate.png` | Learning-rate schedule |
| `ef_agreement_bland_altman.png` | EF agreement and Bland–Altman plots (when enough valid hard-mask values exist) |
| `hfre_task_confusion_matrix.png` | Threshold-task confusion matrix (when valid hard-mask EF values exist) |
| `hfre_task_roc.png` | Threshold-task ROC (when both target classes are present) |
| `volume_ef_distributions.png` | Predicted EDV, ESV, and hard-mask EF distributions |
| `five_test_segmentation_examples.png` | ED/ES reference and predicted masks for up to five test studies |
| `predicted_ed_es_masks.png` | ED/ES predicted contours for the first test study |
| `simpson_disc_widths.png` | ED/ES 20-disc widths for the first test study |
| `reference_simpson_geometry.png` | Simpson geometry check on one training reference mask |
| `example_training_pair.png` | Example ED/ES inputs and reconstructed reference masks |
| `example_raw_video_frame.png` | Raw first frame from the inspected example video |
| `pipeline_flowchart.png` | End-to-end workflow schematic |

Some evaluation plots are conditional: for example, ROC is saved only if the evaluated test subset contains both target classes.

## Other Method Details

- Official train/validation/test assignments from `FileList.csv`.
- Trace-based mask reconstruction with the CSV row order retained.
- ED frame selected by the largest traced LV mask area; ES by the smallest.
- Two-channel paired input and four-channel output layout.
- The original Milesial U-Net import attempt and local fallback.
- Horizontal flip augmentation with probability 0.5 on the training dataset only.
- AdamW optimizer, ReduceLROnPlateau scheduler, optional AMP, and gradient clipping at 1.0.
- Cross-Entropy + soft Dice + differentiable Simpson EF MAE objective.
- Best-checkpoint selection by validation combined loss.
- Hard-mask PCA/20-disc Simpson calculation for final EF reporting.
- Global and per-frame Dice/IoU, EF MAE/RMSE/R², threshold classification, and existing notebook visualizations.


**Note:** This repository is an experimental research implementation. It is not a clinically validated diagnostic tool.
