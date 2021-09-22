# E2E Example for Image Classification

1. create folder "test_images" and download the images, e.g.:
```shell
mkdir test_images
curl -o test_images/daisy.jpg https://raw.githubusercontent.com/microsoft/onnxruntime/v1.8.1/onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/test_images/daisy.jpg
curl -o test_images/rose.jpg https://raw.githubusercontent.com/microsoft/onnxruntime/v1.8.1/onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/test_images/rose.jpg
curl -o test_images/tulip.jpg https://raw.githubusercontent.com/microsoft/onnxruntime/v1.8.1/onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/test_images/tulip.jpg
```

2. call run.py to calibrate, quantize and run the quantized model, e.g.:
```shell
python run.py --input_model mobilenetv2-7.onnx --output_model mobilenetv2-7.quant.onnx --calibrate_dataset ./test_images/
```

## Licenses

Images are downloaded from the upstream onnxruntime repository resized. The original source and license is specified below:

| Image       | Source     | License     |
| ----------- | ---------- | ----------- |
| daisy.jpg | https://www.flickr.com/photos/17393884@N00/5547758 | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| rose.jpg | https://commons.wikimedia.org/wiki/File:Bouquet_-_Flickr_-_Muffet_(1).jpg | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| tulip.jpg | https://www.flickr.com/photos/42126397@N00/11746276 | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
