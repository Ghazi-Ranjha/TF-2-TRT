# TensorFlow to TensorRT Conversion

This repository provides a script to convert a TensorFlow model to TensorRT in different precision modes (float32, float16, int8) for optimized deployment on NVIDIA GPUs. The script was initially developed in Google Colab.

## Overview

This script leverages TensorFlow-TensorRT integration to optimize the inference performance of a pre-trained InceptionV3 model. The conversion is performed in multiple precision modes, allowing you to choose between float32, float16, and int8 for deployment based on your requirements.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- TensorFlow-GPU 2.0
- TensorRT Runtime

## Usage

### Google Colab

If you prefer running the script in Google Colab, you can use the provided Colab notebook [here](link-to-colab-notebook). Follow the instructions in the notebook to perform the conversion.

### Local Environment

1. Clone the repository:

    ```bash
    git clone https://github.com/Ghazi-Ranjha/TF-2-TRT.git
    ```

2. Navigate to the repository:

    ```bash
    cd TF-2-TRT
    ```

3. Run the script:

    ```bash
    python convert_to_tensorrt.py
    ```

4. Follow the instructions in the script to choose the precision mode and perform the conversion.

## Model and Data

The script uses the InceptionV3 model pre-trained on ImageNet. Sample images are included in the 'data' folder for testing.

## Results

The script benchmarks the performance of the optimized models in terms of throughput and provides comparisons between different precision modes.

## Folder Structure

- `data`: Contains sample images for testing.
- `inceptionv3_saved_model`: Original TensorFlow SavedModel.
- `inceptionv3_saved_model_TFTRT_FP32`: TF-TRT optimized model in float32 precision.
- `inceptionv3_saved_model_TFTRT_FP16`: TF-TRT optimized model in float16 precision.
- `inceptionv3_saved_model_TFTRT_INT8`: TF-TRT optimized model in int8 precision.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- NVIDIA TensorRT: [https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)

Feel free to contribute and open issues if you encounter any problems or have suggestions for improvement.

