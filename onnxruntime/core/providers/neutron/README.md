# Neutron Execution Provider

Minimal execution provider to prove LLM execution through ONNXRT

neutron: common ep related files
* platform: Neutron-software/platform specific related elements
* ops: implemented ops in neutron and/or cpu with neutron memory management
* tools: script to convert the model weights to Neutron format

Note:
In order to speed up the pre-packing time for 4bit models, the script
convert_ort_models_to_neutron.py can be used.
```sh
python3 -m onnxruntime.tools.convert_ort_models_to_neutron -i input.onnx
```
One model named 'input_neutron.onnx' will be created, then use the
'input_neutron.onnx' in application instead of 'input.onnx'.
