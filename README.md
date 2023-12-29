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

The script in Google Colab, you can use the provided .ipynb or .py files. [here](link-to-colab-notebook).

## Network Transformation

TF-TRT performs several important transformations and optimizations to the neural network graph. Layers with unused outputs are eliminated to avoid unnecessary computation. Additionally, convolution, bias, and ReLU layers are fused where possible to form a single layer, improving overall efficiency.

### Source:[Speed up TensorFlow Inference on GPUs with TensorRT](https://blog.tensorflow.org/2018/04/speed-up-tensorflow-inference-on-gpus-tensorRT.html)

<div align="center">
    <img width="700px" src='https://2.bp.blogspot.com/-nc-poLV8CNc/XhOI1wfgGjI/AAAAAAAACQI/3FlNTSKKrqMyTzR5XC5RCNnVuUY5EGmhQCLcBGAsYHQ/s1600/fig2.png' />
    <p style="text-align: center;color:gray">Figure (a): An example convolutional model with multiple convolutional and activation layers before optimization</p>
    <p style="text-align: center;color:gray">Figure (c): Horizontal layer fusion</p>
</div>

Please refer to the [TF-TRT User Guide](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#supported-ops) for a comprehensive list of supported operators.

## TF-TRT Workflow

Below, you can see a typical workflow of TF-TRT:

### Source:[High performance inference with TensorRT Integration](https://medium.com/tensorflow/high-performance-inference-with-tensorrt-integration-c4d78795fbfe)

<div align="center">
    <img width="600px" src='https://miro.medium.com/max/875/1*hD_4k9bTEXnjuLHcaoFQRQ.png' />
</div>

<div align="center">
    <img width="600px" src='https://miro.medium.com/max/875/1*DwxO-QF6Bz-H4aurRBIrjw.png' />
</div>

To perform graph conversion, we use `TrtGraphConverterV2`, passing it the directory of a saved model, and any updates we wish to make to its conversion parameters.

```python
from tensorflow.python.compiler.tensorrt import trt_convert as trt

trt.TrtGraphConverterV2(
    input_saved_model_dir=None,
    conversion_params=TrtConversionParams(precision_mode='FP32',
                                          max_batch_size=1
                                          minimum_segment_size=3,
                                          max_workspace_size_bytes=8000000000,
                                          use_calibration=True,
                                          maximum_cached_engines=1,
                                          is_dynamic_op=True,
                                          rewriter_config_template=None,
                                         )

### Conversion Parameters

Here is additional information about the most frequently adjusted conversion parameters.

* __precision_mode__: This parameter sets the precision mode; which can be one of FP32, FP16, or INT8. Precision lower than FP32, meaning FP16 and INT8, would improve the performance of inference. The FP16 mode uses Tensor Cores or half precision hardware instructions, if possible. The INT8 precision mode uses integer hardware instructions.

* __max_batch_size__: This parameter is the maximum batch size for which TF-TRT will optimize. At runtime, a smaller batch size may be chosen, but, not a larger one.

* __minimum_segment_size__: This parameter determines the minimum number of TensorFlow nodes in a TF-TRT engine, which means the TensorFlow subgraphs that have fewer nodes than this number will not be converted to TensorRT. Therefore, in general, smaller numbers such as 5 are preferred. This can also be used to change the minimum number of nodes in the optimized INT8 engines to change the final optimized graph to fine tune result accuracy.

* __max_workspace_size_bytes__: TF-TRT operators often require temporary workspace. This parameter limits the maximum size that any layer in the network can use. If insufficient scratch is provided, it is possible that TF-TRT may not be able to find an implementation for a given layer.

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

