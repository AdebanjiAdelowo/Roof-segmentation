# Roof Segmentation

Binary semantic segmentation of rooftops in aerial/satellite imagery using a U-Net convolutional network, trained on a small labeled dataset (25 training images and masks).

## Overview

Given an RGBA aerial image, the model predicts a per-pixel mask indicating rooftop versus background. The task was carried out as a small applied exercise on a dataset of house imagery, with 25 labeled training images/masks and two held-out sets of 5 unlabeled test images.

## Methodology

The segmentation network is the U-Net architecture from [zhixuhao/unet](https://github.com/zhixuhao/unet) (`model.py`, `data.py`), adapted here for 4-channel (RGBA) rooftop imagery at 256x256 resolution.

Because only 25 training images were available, `ImageDataGenerator`-based augmentation (rotation, width/height shift, shear, zoom, horizontal flip) was used to generate additional training batches. The model was trained with the Adam optimizer and binary cross-entropy loss.

## Data

- Training: 25 RGBA images and corresponding binary masks (`data/train`)
- Test: two sets of 5 unlabeled images (`data/test`), used only for qualitative inspection

## Results

The model was evaluated qualitatively by visually comparing predicted masks to the input test images, since no ground-truth masks were available for the test sets. No quantitative metric (e.g. IoU, pixel accuracy) was computed against a labeled test set. Predicted masks for the test images are written to `data/test` by `saveResult`.

## Repository Structure

```
Roof-segmentation/
├── main.py                                  # Training/inference entry point
├── model.py                                 # U-Net architecture (from zhixuhao/unet)
├── data.py                                  # Data generators, augmentation, mask visualization
├── Adelowo_rooftop_segment.ipynb            # Exploration and first training run
├── Adelowo_rooftop_segment_update.ipynb     # Updated training/inference notebook
├── dataPrepare.ipynb                        # Dataset preparation
└── data/                                    # Training and test images/masks
```

## Installation

```bash
pip install tensorflow keras numpy scikit-image matplotlib
```

## Usage

```bash
python main.py
```

or open `Adelowo_rooftop_segment_update.ipynb` and run all cells. The notebook was originally run on Google Colab with the dataset mounted from Google Drive; paths in the notebook (`main_path`, `DATA_DIR`) should be adjusted for a local run.

## Limitations

- The training set is small (25 images), so generalization beyond the provided data distribution is untested.
- No quantitative segmentation metric is reported; evaluation was visual only.
- The U-Net implementation is reused directly from zhixuhao/unet rather than written from scratch.

## Possible Extensions

Alternative or pretrained backbone architectures, transfer learning from a model pretrained on a larger segmentation dataset, and quantitative evaluation (IoU, Dice, pixel accuracy) against a labeled test set.

## References

- zhixuhao, [unet](https://github.com/zhixuhao/unet): Keras U-Net implementation used as the base architecture.
- Ronneberger, Fischer, Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
