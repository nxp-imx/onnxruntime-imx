/****************************************************************************
 *
 *    Copyright (c) 2020 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include "core/framework/compute_capability.h"
#include "core/providers/vsi_npu/vsi_npu_provider_factory.h"
#include "vsi_npu_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {

struct VsiNpuProviderFactory : IExecutionProviderFactory {
    VsiNpuProviderFactory(int device_id) : device_id_(device_id) {}
    ~VsiNpuProviderFactory() override {}

    std::unique_ptr<IExecutionProvider> CreateProvider() override;

   private:
    int device_id_;
};

std::unique_ptr<IExecutionProvider> VsiNpuProviderFactory::CreateProvider() {
    VsiNpuExecutionProviderInfo info;
    info.device_id = device_id_;
    return onnxruntime::make_unique<VsiNpuExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_VsiNpu(int device_id) {
    return std::make_shared<onnxruntime::VsiNpuProviderFactory>(device_id);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_VsiNpu,
                    _In_ OrtSessionOptions* options,
                    int device_id) {
    options->provider_factories.push_back(
        onnxruntime::CreateExecutionProviderFactory_VsiNpu(device_id));
    return nullptr;
}
