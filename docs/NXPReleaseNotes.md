# Release Notes for NXP Yocto BSP

NXP releases prebuilt images for Yocto Linux BSP quaterly which are available on the [NXP website](https://www.nxp.com/design/software/embedded-software/i-mx-software/embedded-linux-for-i-mx-applications-processors:IMXLINUX) for download. Additionally there might be off-cycle releases which target new boards or features. Code is released on the [CodeAurora website](https://source.codeaurora.org/external/imx/onnxruntime-imx).

## LF 5.10.52-2.1.0

### Known Issues and Limitations
* The prebuilt image released on the NXP website is without the latest Arm NN and ACL execution provider fixes to the GEMM operation. If it is mandatory to use these execution providers we strongly recommend to build either the image or update the ONNX Runtime package for Yocto Linux manually from sources available in the latest commits on "lf-5.10.52_2.1.0" branch. Models from the ONNX Model Zoo which are affected: inception_v1, inception_v2, inception_v4_299, resnet50_v1, resnet50_v2, zfnet512, mobilenetv2-7
* VSI NPU execution provider has slightly different results from CPU implementations (caused by NPU HW) for the following models: mobilenet_v1_0.25_128, mobilenet_v1_1.0_224 (negligable), resnet50_v1 (minor)
* Python API is currently not supported
* Currently enabled Execution Providers are: CPU, VSI NPU, NNAPI, Arm NN and ACL

## Feedback
For questions, bug reports or other forms of discussion, please use [NXP Community](https://community.nxp.com/).
