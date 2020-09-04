## VsiNpu Execution Provider

NXP i.MX8 devices have the capability to accelerate neural networks on VeriSilicon's GPUs or NPUs. To enable it VeriSilicon's NN RT framework is used, which uses nnrt, ovxlib, and OpenVX libraries. In ONNX runtime, these libraries are called through the VSI NPU execution provider. OpenVX at the other end manages, which accelerator is used, thus it can be both the GPU, or the NPU if available.

### Build ArmNN execution provider
For build instructions, please see the [BUILD page](../../BUILD.md#VsiNpu).

### Using the ArmNN execution provider
#### C/C++
To use VsiNpu as execution provider for inferencing, there are 2 things which need to be done.

Include the VSI NPU header file located in `<onnx_include_dir>/onnxruntime/core/providers/vsi_npu` (make sure to add it to the include path in your makefile as well):
```
#include "vsi_npu_provider_factory.h"
```

Add the VSI NPU execution provider in your application:
```
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_VsiNpu(session_options, 0));
```
The C API details are [here](../C_API.md#c-api).

### Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../ONNX_Runtime_Perf_Tuning.md)

When/if using [onnxruntime_perf_test](../../onnxruntime/test/perftest), use the flag -e vsi_npu
