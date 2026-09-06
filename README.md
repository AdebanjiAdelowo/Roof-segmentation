# Roof Segmentation

Binary semantic segmentation of rooftops in aerial/satellite imagery using a U-Net convolutional network, trained on a small labeled dataset (25 training images and masks).

## Overview

Given an RGBA aerial image, the model predicts a per-pixel mask indicating rooftop versus background. The task was carried out as a small applied exercise on a dataset of house imagery, with 25 labeled training images/masks. The training notebook applies the model to two held-out sets of 5 unlabeled test images; only one of those sets (`data/test/images`) is included in this repository.

## Methodology

The segmentation network is the U-Net architecture from [zhixuhao/unet](https://github.com/zhixuhao/unet) (`model.py`, `data.py`), adapted here for 4-channel (RGBA) rooftop imagery at 256x256 resolution.

Because only 25 training images were available, `ImageDataGenerator`-based augmentation (rotation, width/height shift, shear, zoom, horizontal flip) was used to generate additional training batches. The model was trained with the Adam optimizer and binary cross-entropy loss.

## Data

- Training: 25 RGBA images and corresponding binary masks (`data/train`)
- Test: 5 unlabeled RGBA images (`data/test/images`), used only for qualitative inspection. `data/test/labels` holds the model's own predicted masks written there by `saveResult`, not ground truth. A second 5-image test set (`dida_images`, referenced in the notebook) is part of the original take-home dataset and is not included in this repository.

## Results

The model was evaluated qualitatively by visually comparing predicted masks to the input test images, since no ground-truth masks were available for the test sets. No quantitative metric (e.g. IoU, pixel accuracy) was computed against a labeled test set. Predicted masks (raw sigmoid output, not thresholded) for the test images are written to `data/test/labels` by `saveResult`.

## Repository Structure

```
Roof-segmentation/
├── main.py                                  # Original zhixuhao/unet template (unmodified, not used for this task)
├── model.py                                 # U-Net architecture (from zhixuhao/unet)
├── data.py                                  # Data generators, augmentation, mask visualization
├── Adelowo_rooftop_segment.ipynb            # Exploration and first training run
├── Adelowo_rooftop_segment_update.ipynb     # Updated training/inference notebook
├── dataPrepare.ipynb                        # Dataset preparation
└── data/                                    # Training and test images/masks
```

## Installation

```bash
pip install "tensorflow<2.16" numpy scikit-image matplotlib
```

`data.py` calls `from keras.preprocessing.image import ImageDataGenerator`. As of Keras 3 (the default when `tensorflow>=2.16` is installed unpinned), `ImageDataGenerator` is no longer exposed at that public path (verified directly against an installed Keras 3.13.2 tree: the `keras/preprocessing/` submodule no longer exists at all, and the class survives only internally under `keras.src.legacy.preprocessing`), so this import fails on a fresh, unpinned install. Pinning `tensorflow<2.16` installs the matching pre-Keras-3 `keras` dependency for which this code was written. `model.py` separately calls `tf.keras.optimizers.Adam(...)` without ever importing `tensorflow as tf` in that file; this is very likely a `NameError` on current Keras (which does not leak a bare `tf` name through the wildcard imports `model.py` uses) but was not re-run end to end in this pass to confirm dynamically, since this machine's own TensorFlow install is independently broken (unrelated protobuf conflict) and reproducing a clean legacy environment was out of scope here.

## Usage

Open `Adelowo_rooftop_segment_update.ipynb` and run all cells; this is the actual rooftop training/inference workflow, calling `unet(input_size=(256,256,4))` for RGBA input and `trainGenerator(..., image_color_mode='rgba')`. The notebook was originally run on Google Colab with the dataset mounted from Google Drive; paths in the notebook (`main_path`, `DATA_DIR`) should be adjusted for a local run.

`main.py` is the unmodified `zhixuhao/unet` template entry point (grayscale membrane-segmentation demo). It has not been adapted to this repository's data or task: it points at a `data/membrane` directory that does not exist here, and its `saveResult` call does not match the current `saveResult` signature in `data.py`. Running `python main.py` as-is will fail; use the notebook instead.

## Limitations

- The training set is small (25 images), so generalization beyond the provided data distribution is untested.
- No quantitative segmentation metric is reported; evaluation was visual only.
- The U-Net implementation is reused directly from zhixuhao/unet rather than written from scratch.

## Possible Extensions

Alternative or pretrained backbone architectures, transfer learning from a model pretrained on a larger segmentation dataset, and quantitative evaluation (IoU, Dice, pixel accuracy) against a labeled test set.

## References

- zhixuhao, [unet](https://github.com/zhixuhao/unet): Keras U-Net implementation used as the base architecture.
- Ronneberger, Fischer, Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
