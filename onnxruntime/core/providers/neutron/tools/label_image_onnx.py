# Copyright 2025 NXP
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from PIL import Image
import numpy as np
import onnxruntime as ort

support_input_type_list = []
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def load_labels(filename: str):
    with open(filename) as f:
        return [line.strip() for line in f]

def preprocess_image_for_nchw(image_data):
    image_data = image_data.transpose([2, 0, 1])
    # Normalization of standard ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

def preprocess_image_for_nhwc(image_data):
    # Normalization of standard ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_data = image_data / np.float32(255.0)
    # Automatically broatcast to each channel
    image_data = (image_data - mean) / std
    # shape: [1, H, W, C] or [1, C, H, W]
    image_data = np.expand_dims(image_data, axis=0)
    return image_data

def preprocess(input_data, dtype, chw):
    if dtype == np.float32:
        input_data = np.array(input_data, dtype)
        return preprocess_image_for_nchw(input_data) if chw else preprocess_image_for_nhwc(input_data)
    elif dtype in [np.int8, np.uint8]:
        if chw:
            input_data = input_data.transpose([2, 0, 1])

        input_data = np.expand_dims(input_data, 0)

        if dtype == np.uint8:
            input_data = input_data.astype(np.uint8)
        else:
            input_data = input_data.astype(np.int32) - 128
            input_data = input_data.astype(np.int8) 
        return input_data
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default="grace_hopper.bmp")
    parser.add_argument("-e", "--env", required=True, help="cpu,vsinpu,neutron,gpu")
    parser.add_argument("-m", "--model_file", default="mobilenet_v2_1.0_224.onnx", help=".onnx model to be executed")
    parser.add_argument("-l", "--label_file", default="labels.txt", help="name of file containing labels")
    parser.add_argument("--num_threads", type=int)
    parser.add_argument("--no-softmax", action="store_true", help="Disable softmax layer in post-processing")
    args = parser.parse_args()

    sess_opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    if args.num_threads:
        sess_opts.intra_op_num_threads = args.num_threads
    if args.env == "neutron":
        providers=[('NeutronExecutionProvider',{'neutron_op_only': True}),'CPUExecutionProvider']
    if args.env == "vsinpu" or args.env == "gpu":
        providers=["VSINPUExecutionProvider"]

    session = ort.InferenceSession(
        args.model_file,
        sess_opts,
        providers=providers
    )

    input_info  =  session.get_inputs()[0]
    input_name  = input_info.name
    input_shape = input_info.shape
    output_info = session.get_outputs()[0]
    output_name = output_info.name

    # process layout
    if len(input_shape) == 4 and input_shape[1] == 3:
        _, C, H, W = input_shape
        chw = True
    else:
        _, H, W, C = input_shape
        chw = False

    #read image and resize
    img = Image.open(args.image).convert("RGB").resize((W, H),Image.Resampling.LANCZOS)
    required_dtype = input_info.type
    # Map onnx input type to numpy type
    onnx_to_numpy_dtype = {
        'tensor(float)': np.float32,
        'tensor(float32)': np.float32,
        'tensor(int8)': np.int8,
        'tensor(uint8)': np.uint8,
        'tensor(int32)': np.int32,
        'tensor(int64)': np.int64
    }

    dtype = onnx_to_numpy_dtype.get(required_dtype)     
    quntizedModel = False  
    if dtype in [ np.int8, np.uint8 ]:
        quntizedModel = True  
    elif dtype in [ np.int32, np.int64 ]:
        raise f"dtype:{dtype} is not supported!" 

    input_data = preprocess(img, dtype, chw)

    # Do inference
    _ = session.run([output_name], {input_name: input_data})
    outputs = session.run([output_name], {input_name: input_data})
    output_data = outputs[0]
 
    if quntizedModel:
        output_data = output_data.astype(np.int32) + 128
        output_data = output_data.astype(np.uint8)

    results = np.squeeze(output_data)

    if not args.no_softmax:
        results = softmax(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args.label_file)
    if len(results) == 1000:
        labels = labels[1:]
    for idx in top_k:
        if quntizedModel:
            print('{:08.6f}: {}'.format(float(results[idx] / 255.0), labels[idx]))
        else:
            print(f"{results[idx]:.6f}: {labels[idx]}")
