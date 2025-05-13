# Copyright (c) NXP. All rights reserved.
import onnx
import numpy as np
import math
import argparse
import multiprocessing
from ctypes import *
from onnx import numpy_helper, helper

def DecimalToFixedPoint(number, integer_bits=10, fraction_bits=6):
    """Converts an unsigned decimal number to a fixed-point equivalent."""
    sign = 0
    if number < 0:
        sign =1
        number *= -1

    elif number > 2**integer_bits-1:
      number = 2**integer_bits-1
    # Split integer and fractional parts
    integer_part = int(number)
    fractional_part = number - integer_part


    first_bit_obtained = (integer_part>0)
    bits_obtained = int(math.log2(integer_part)) + 1 if (integer_part>0) else 0
    bits_left = integer_bits-bits_obtained
    shift =0
    fixed_point_scale = integer_part

    if(first_bit_obtained):
      fraction_bits = bits_left
    else:
      while(fractional_part < 0.5 and shift < 2**fraction_bits-1):
        fractional_part*=2
        shift = shift +1

    # Convert fractional part to binary
    frac_bin = ""
    for _ in range(fraction_bits):
        if(shift >= 2**fraction_bits-1):
          break
        fractional_part *= 2
        bit = int(fractional_part)
        fixed_point_scale = fixed_point_scale*2 | bit
        shift+=1
        fractional_part -= bit  # Remove the integer part
        if (fractional_part == 0 ):
          break
    # Combine integer and fractional parts to get the fixed point decimal number
    scale_10bit = fixed_point_scale & 0b1111111111
    shift_6bit = shift & 0b111111
    return((scale_10bit * 2**(-shift_6bit)) * ((-1)**sign))

def DecimalToNeutron(number, integer_bits=10, fraction_bits=6):
    """Converts an unsigned decimal number to a fixed-point binary representation on Neutron."""
    if number < 0:
        raise ValueError("Number must be non-negative for unsigned representation.")
    elif number > 2**integer_bits-1:
      number = 2**integer_bits-1
    # Split integer and fractional parts
    integer_part = int(number)
    fractional_part = number - integer_part


    first_bit_obtained = (integer_part>0)
    bits_obtained = int(math.log2(integer_part)) + 1 if (integer_part>0) else 0
    bits_left = integer_bits-bits_obtained
    shift =0
    fixed_point_scale = integer_part

    if(first_bit_obtained):
      fraction_bits = bits_left
    else:
      while(fractional_part < 0.5 and shift < 2**fraction_bits-1):
        fractional_part*=2
        shift = shift +1

    # Convert fractional part to binary
    frac_bin = ""
    for _ in range(fraction_bits):
        if(shift >= 2**fraction_bits-1):
          break
        fractional_part *= 2
        bit = int(fractional_part)
        fixed_point_scale = fixed_point_scale*2 | bit
        shift+=1
        fractional_part -= bit  # Remove the integer part
        if (fractional_part == 0 ):
          break
    # Combine integer and fractional parts
    scale_10bit = fixed_point_scale & 0b1111111111
    shift_6bit = shift & 0b111111
    return(np.int16(((shift_6bit << 10) | scale_10bit)-65536*(shift_6bit >= 32 )))

def WeightPacker(B, rowsB, colsB, channelDensity, weightBits = 4, MACs = 16):
  length = math.ceil(channelDensity* colsB * weightBits / 8) * rowsB/channelDensity
  packedWeights = np.zeros(int(length), dtype = np.int8)
  cell_pointer = 0
  bit_pointer = 0
  for j in range(0,rowsB,channelDensity):
      cell_pointer = math.ceil(channelDensity* colsB * weightBits / 8) * (j//channelDensity)
      bit_pointer = 0
      for i in range(0,colsB,MACs):
        for m in range(0,channelDensity):
          for k in range(0,MACs):
            if(8-bit_pointer >= weightBits):
              extracted_bits = B[j+m, i+k] & (2**weightBits-1)
              packedWeights[cell_pointer] = np.int8( packedWeights[cell_pointer] | (extracted_bits << bit_pointer) )
              bit_pointer += weightBits
              cell_pointer += bit_pointer//8
              bit_pointer = bit_pointer%8
            else:
              fitting_bits = 8-bit_pointer
              remaining_bits = weightBits - fitting_bits
              extracted_bits =  B[j+m, i+k] & (2**fitting_bits-1)
              rem_extracted_bits =  (B[j+m, i+k] >> fitting_bits) & (2**remaining_bits-1)
              packedWeights[cell_pointer] = np.int8( packedWeights[cell_pointer] | (extracted_bits << bit_pointer) )
              cell_pointer += 1
              bit_pointer =0
              packedWeights[cell_pointer] = np.int8( packedWeights[cell_pointer] | (rem_extracted_bits << bit_pointer) )
              bit_pointer+= remaining_bits

  return packedWeights

def CalculateChannelDensity(embeddings_in, group_size, weight_bits=8,
                            decode_weights=False, use_decode_bias=False,
                            res_num_bytes=4, MACS=16, num_neutrons=4,
                            tcm_size=1024 * 1024, tcm_banks=16):
  scale = 1 if decode_weights else (weight_bits / 8.0)
  channel_density = 2 * MACS * num_neutrons
  tcm_per_bank = tcm_size / tcm_banks

  term1 = math.ceil(
      math.ceil(channel_density / num_neutrons * embeddings_in * scale)
      * num_neutrons / tcm_per_bank
  ) * tcm_per_bank

  if decode_weights:
      term2 = math.ceil(
          (channel_density * embeddings_in +
          (2 + 1 * use_decode_bias) * channel_density * embeddings_in / group_size +
           16 * 1024 * num_neutrons)
          / tcm_per_bank
      ) * tcm_per_bank
  else:
      term2 = 0

  offset_b = max(term1, term2)
  offset_a = math.ceil(embeddings_in / tcm_per_bank) * tcm_per_bank

  used_tcm = offset_a + 2 * offset_b + channel_density * res_num_bytes

  if tcm_size - used_tcm < 0:
    channel_density = MACS * num_neutrons

  return int(channel_density / num_neutrons)

def FactorToNeutronScaler(float_factor):
  casted_factor = cast(pointer(c_float(float_factor)), POINTER(c_int32)).contents.value
  scaler = (((casted_factor) & 0xffffffff)>>8) & 0x7fff
  exp_tmp = (((casted_factor) & 0xffffffff)>>23) & 0xff
  if (exp_tmp == 0):
    scaler = 0
  else:
    scaler = scaler | 0x8000
  exp_tmp = -(exp_tmp -142)
  if (exp_tmp > 63):
    exp = 63
  else:
    exp = exp_tmp
  scaler = (exp<<16) | scaler
  return scaler

def ComputeDecodeScales(N, blocksPerCol, scales):
  group_sacles = scales.reshape(N, blocksPerCol)
  channel_scales = np.max(np.abs(group_sacles), axis = 1) / 16
  channel_scales_expand = np.expand_dims(channel_scales, axis = 1)
  channel_scales_repeat = np.repeat(channel_scales_expand, blocksPerCol, axis = 1)
  decode_scales = group_sacles / channel_scales_repeat

  fixed_point_process = np.vectorize(DecimalToFixedPoint)
  fixedPointScales = fixed_point_process(decode_scales)

  scaler_process = np.vectorize(FactorToNeutronScaler)
  factors = scaler_process(channel_scales)

  return fixedPointScales, factors.astype(np.uint32)

def ComputeWeightAndBias(B, decodeScales, N, blocksPerCol, groupSize):
  arr = np.asarray(B, dtype=np.uint8).reshape(-1)
  high4 = (arr >> 4) & 0x0F
  low4 = arr & 0x0F

  temp = np.empty(arr.size * 2, dtype=np.int8)
  temp[0::2] = low4
  temp[1::2] = high4
  B_int8 = temp.reshape(N, -1) - 8 #zero point is 8

  bias = np.zeros([N])
  scales_repeated = np.repeat(decodeScales, groupSize, axis = 1)
  B_decode = B_int8 * scales_repeated
  B_decode = np.clip(np.floor(B_decode + 0.5), -128, 127)
  for i in range(N):
    bias[i] = np.sum(B_decode[i,:]) * -128

  decodeBiases = np.zeros([N, blocksPerCol], np.int8)
  mask_repeated = scales_repeated < 0
  mask = decodeScales<0
  B_int8[mask_repeated] = -B_int8[mask_repeated] -1
  decodeBiases[mask]= -decodeBiases[mask] + 1

  return B_int8, decodeBiases, bias.astype(np.int32)

def ScalesPacker(decodeScales, N, channelDensity, blocksPerCol):
  decodeScales = np.abs(decodeScales)

  neutronScales = np.zeros(N  * blocksPerCol, np.int16)
  for i in range(0, N, channelDensity):
    for k in range(channelDensity):
      for j in range(blocksPerCol):
        neutronScales[i * blocksPerCol + j * channelDensity + k] = DecimalToNeutron(decodeScales[i + k, j])
  return neutronScales

def ConvertWeightToNeutron(cvt_args):
  b_name, B, scales, K, N, blockSize = cvt_args
  print("Packing weight: ", b_name)
  blocksPerCol = (K + blockSize - 1) // blockSize
  channelDensity = CalculateChannelDensity(K, blockSize)

  decodeScales, factors = ComputeDecodeScales(N, blocksPerCol, scales)
  B_int8, decodeBiases, bias = ComputeWeightAndBias(B, decodeScales, N, blocksPerCol, blockSize)
  packedDecodeScales = ScalesPacker(decodeScales, N, channelDensity, blocksPerCol)
  packedWeight = WeightPacker(B_int8, N, K, channelDensity)

  raw = packedWeight.tobytes() + decodeBiases.tobytes() + packedDecodeScales.tobytes() + bias.tobytes() + factors.tobytes()
  return (b_name, np.frombuffer(raw, dtype=np.uint8))

class Index:
    IN_A = 0
    IN_B = 1
    SCALES = 2
    ZERO_POINTS = 3
    G_IDX = 4
    BIAS = 5

def Main(args):
    model = onnx.load(args.input)
    nodes = [x for x in model.graph.node if x.op_type == "MatMulNBits"]
    initializers = {init.name: init for init in model.graph.initializer}

    cvt_args = []
    for node in nodes:
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        K, N = attrs["K"], attrs["N"]
        block_size, bits = attrs["block_size"], attrs["bits"]
        if K % 16 or N % 128 or bits != 4:
            continue
        if len(node.input) > 3:
            continue

        b_name = node.input[Index.IN_B]
        b_tensor = initializers[b_name]
        b_data = numpy_helper.to_array(b_tensor)

        scales_name = node.input[Index.SCALES]
        scales_tensor = initializers[scales_name]
        scales_data = numpy_helper.to_array(scales_tensor)

        cvt_args.append((b_name, b_data, scales_data, K, N, block_size))

    with multiprocessing.Pool(processes=args.jobs) as pool:
        results = pool.map(ConvertWeightToNeutron, cvt_args)

    for b_name, packaged_data in results:
        b_tensor = initializers[b_name]
        b_tensor.CopyFrom(numpy_helper.from_array(packaged_data, name=b_name))

    onnx.save(model, args.output)

def CheckArgs(value):
    if not value.endswith(".onnx"):
        raise argparse.ArgumentTypeError("Input file must end with '.onnx'.")
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline pack weights to Neutron format.")
    parser.add_argument("-i", "--input", required=True, type=CheckArgs, help="Input model file name")
    parser.add_argument("-o", "--output", type=CheckArgs, help="Output model file name")
    parser.add_argument("-j", "--jobs", type=int, default=multiprocessing.cpu_count(), help="Number of jobs")
    args = parser.parse_args()

    if (args.output == None):
        args.output = args.input.replace(".onnx", "_neutron.onnx")
    Main(args)
