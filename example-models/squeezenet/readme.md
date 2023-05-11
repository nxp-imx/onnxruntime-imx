# SqueezeNet 1.0 info

Source (ONNX Zoo): https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-9.tar.gz

By default, the model ONNX opset is 9. The one present in this repo is converted to opset 17 (latest for onnxruntime 1.13.1).

Opset update reproduction steps:
+ Environment with Python3 (3.8+) and ONNX (1.13+)
+ Donwload the original model with opset 9 from model zoo
+ Follow steps described in https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#converting-version-of-an-onnx-model-within-default-domain-aionnx
    +   ```
            ...
            converted_model = version_converter.convert_version(original_model, 17)
            ...
        ```
