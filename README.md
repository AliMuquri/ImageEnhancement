# ImageEnhancement

## Overview

The **ImageEnhancement** project focuses on training an Enhanced Deep Super-Resolution (EDSR) model using the DIV2K dataset at a 4x upscaling factor. One significant challenge this project addresses is enabling the upscaling of large images that might exceed the GPU's capacity by implementing a tiling procedure.

### Key Features

- Training of EDSR model on DIV2K datasets (4x upscaling).
- Development of a tiling procedure for upscaling large images.
- Automatic folder structure creation and data extraction from DIV2K datasets.
- Utilization of the TensorFlow `tf.data.Dataset` API for a data pipeline with data augmentation.
- PSNR (Peak Signal-to-Noise Ratio) used as a metric for model evaluation.
- Mean Squared Error (MSE) employed as the loss function during training.

## Usage

1. **Training and Inference with EDSR Model**

   The `main_program.py` file provides examples and scripts for both training the EDSR model and using it for inference. You can follow these steps to train your model and enhance images:

   - Run the training script, which will train the EDSR model using the DIV2K dataset (4x upscalingg).
   - Use the trained model for image enhancement and super-resolution tasks.

2. **Tiling for Large Images**

   The project introduces a `use_tile()` method that allows the EDSR model instance to process large images during inference by applying tiling techniques. This method takes large images, tiles them, processes each tile individually, and then reconstructs the enhanced image.

