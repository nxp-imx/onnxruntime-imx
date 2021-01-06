# NXP Release Notes

This document describes changes and limitations to ONNX Runtime for NXP releases on https://source.codeaurora.org/external/imx/onnxruntime-imx.

NXP supports ONNX Runtime as one of the [eIQ](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ) components for its [i.MX8](https://www.nxp.com/products/processors-and-microcontrollers/arm-processors/i-mx-applications-processors/i-mx-8-processors:IMX8-SERIES) and [Layerscape](https://www.nxp.com/products/processors-and-microcontrollers/arm-processors/layerscape-multicore-processors:QORIQ-ARM) processors. 
To run on CPU use:
* default CPU execution provider
* [ACL](./docs/execution_providers/ACL-ExecutionProvider.md)
* [Arm NN](./docs/execution_providers/ArmNN-ExecutionProvider.md)

To accelerate your NN using GPU or NPU (depending on availability), use:

* [VSI NPU](./docs/execution_providers/VsiNpu-ExecutionProvider.md)

To view all the changes, have a look at the official [GitHub changelog](https://github.com/microsoft/onnxruntime/releases):
* [1.1.2](https://github.com/microsoft/onnxruntime/releases/tag/v1.1.2)
* [1.1.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.1.1)
* [1.1.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.1.0)

### Known Limitations

##### 1.1.2 (rel_imx_5.4.47_2.2.0)
* Arm NN and ACL only support group=1 for normal conv and =channel for dw conv. Models such as [BVLC AlexNet](https://github.com/onnx/models/tree/master/vision/classification/alexnet) are not supported in ArmNN and ACL EPs. To run on CPU, CPU execution provider must be used.
* [BVLC GoogleNet](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/googlenet) has precision issues in Arm NN EP. ACL EP can be used instead.
* Conv2D in the VSI NPU EP might have precision issues (under very specific configurations), which are root caused in the OpenVX driver.
