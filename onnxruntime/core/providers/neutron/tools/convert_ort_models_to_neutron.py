# Copyright 2026 NXP
import onnx
import numpy as np
import math
import argparse
import multiprocessing
import ctypes
import os
from ctypes import *
from onnx import numpy_helper, helper, shape_inference

# ---------------------------------------------------------------------------
# Load libNeutronDriver.so (provides the prepack API declared in
# NeutronDriverBmm.h). Lookup order:
#   1. $NEUTRON_DRIVER_LIB env var (explicit override)
#   2. system loader ("libNeutronDriver.so") -- target board install path
#   3. <script_dir>/libNeutronDriver.so (co-located fallback)
# ---------------------------------------------------------------------------
def _load_driver_lib():
    candidates = []
    env_path = os.environ.get("NEUTRON_DRIVER_LIB")
    if env_path:
        candidates.append(env_path)
    candidates.append("libNeutronDriver.so")
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "libNeutronDriver.so"))
    last_err = None
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to load libNeutronDriver.so. Tried: {candidates}. "
        f"Last error: {last_err}. Set NEUTRON_DRIVER_LIB to the .so path.")

_LIB = _load_driver_lib()
_c_free = ctypes.CDLL(None).free
_c_free.argtypes = [ctypes.c_void_p]
_c_free.restype = None

# ---------------------------------------------------------------------------
# C struct definitions (mirrors NeutronDriverBmm.h)
# ---------------------------------------------------------------------------
class _C_TilingResult(ctypes.Structure):
    _fields_ = [
        ("channelDensity", ctypes.c_int),
        ("lineDensity",    ctypes.c_int),
        ("numNeutrons",    ctypes.c_int),
        ("bPingPong",      ctypes.c_bool),
        ("divisions",      ctypes.c_int),
    ]

class _C_Dyn8(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int8)),
        ("size", ctypes.c_size_t),
        ("cap",  ctypes.c_size_t),
    ]

class _C_Dyn32(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int32)),
        ("size", ctypes.c_size_t),
        ("cap",  ctypes.c_size_t),
    ]

class _C_PrepackOut(ctypes.Structure):
    _fields_ = [
        ("Bpacked",   ctypes.POINTER(_C_Dyn8)),
        ("lengths",   ctypes.POINTER(_C_Dyn32)),
        ("tilingInfo", _C_TilingResult),
    ]

class _C_PrepackCfg(ctypes.Structure):
    _fields_ = [
        ("rearrange",     ctypes.c_bool),
        ("miniWeights",   ctypes.c_bool),
        ("weightBits",    ctypes.c_int),
        ("groupSize",     ctypes.c_int),
        ("useDecodeBias", ctypes.c_bool),
        ("compress",      ctypes.c_bool),
        ("numMacs",       ctypes.c_int),
        ("numNeutrons",   ctypes.c_int),
        ("tcmSize",       ctypes.c_int),
        ("numBanks",      ctypes.c_int),
    ]

# ---------------------------------------------------------------------------
# C function declarations
# ---------------------------------------------------------------------------

# prepack
_LIB.prepack.argtypes = [
    ctypes.POINTER(ctypes.c_int8),    # B
    ctypes.c_int,                      # rb
    ctypes.c_int,                      # cb
    ctypes.POINTER(_C_PrepackCfg),     # pckCfg
    ctypes.POINTER(_C_PrepackOut),     # pckOut
]
_LIB.prepack.restype = None

# dyn8_free / dyn32_free
_LIB.dyn8_free.argtypes = [ctypes.POINTER(_C_Dyn8)]
_LIB.dyn8_free.restype = None
_LIB.dyn32_free.argtypes = [ctypes.POINTER(_C_Dyn32)]
_LIB.dyn32_free.restype = None

# fetch_unp_organize_group_scales_double_int16  (for decode scales: double -> int16)
_LIB.fetch_unp_organize_group_scales_double_int16.argtypes = [
    ctypes.POINTER(ctypes.c_double),   # param_data
    ctypes.c_int,                       # rowsB
    ctypes.c_int,                       # colsB
    ctypes.c_int,                       # channelDensity
    ctypes.c_int,                       # MACs
    ctypes.c_int,                       # weightBits
    ctypes.c_int,                       # numNeutrons
    ctypes.c_int,                       # groupSize
    ctypes.c_int,                       # isBias
    ctypes.c_int,                       # divisions
    ctypes.POINTER(ctypes.c_int),       # out_size
]
_LIB.fetch_unp_organize_group_scales_double_int16.restype = ctypes.POINTER(ctypes.c_int16)

# fetch_unp_organize_group_scales_int32_int8  (for decode biases: int32 -> int8)
_LIB.fetch_unp_organize_group_scales_int32_int8.argtypes = [
    ctypes.POINTER(ctypes.c_int32),    # param_data
    ctypes.c_int,                       # rowsB
    ctypes.c_int,                       # colsB
    ctypes.c_int,                       # channelDensity
    ctypes.c_int,                       # MACs
    ctypes.c_int,                       # weightBits
    ctypes.c_int,                       # numNeutrons
    ctypes.c_int,                       # groupSize
    ctypes.c_int,                       # isBias
    ctypes.c_int,                       # divisions
    ctypes.POINTER(ctypes.c_int),       # out_size
]
_LIB.fetch_unp_organize_group_scales_int32_int8.restype = ctypes.POINTER(ctypes.c_int8)


# ---------------------------------------------------------------------------
# pack_weight_to_neutron  (uses C prepack + C scales packing from libNeutronDriver)
# ---------------------------------------------------------------------------
def pack_weight_to_neutron(B, rb, ca, weightBits=8, groupSize=-1,
                           decodeScales=None, decodeBiases=None, compress=True):
    miniWeights = (weightBits < 8)

    # Build C config -- mirrors the Python tiling_solver / packer defaults
    c_cfg = _C_PrepackCfg(
        rearrange     = miniWeights,   # 4-bit -> rearrange, 8-bit -> no rearrange
        miniWeights   = miniWeights,
        weightBits    = weightBits,
        groupSize     = groupSize if groupSize > 0 else -1,
        useDecodeBias = miniWeights,
        compress      = compress,
        numMacs       = 16,
        numNeutrons   = 4,
        tcmSize       = 1024 * 1024,
        numBanks      = 16,
    )

    # Pre-allocate Dyn8 / Dyn32 so the C prepack has valid pointers to write into
    c_dyn8  = _C_Dyn8()
    c_dyn32 = _C_Dyn32()
    c_out = _C_PrepackOut()
    c_out.Bpacked = ctypes.pointer(c_dyn8)
    c_out.lengths = ctypes.pointer(c_dyn32)

    B_contig = np.ascontiguousarray(B, dtype=np.int8)

    _LIB.prepack(
        B_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        ctypes.c_int(rb),
        ctypes.c_int(ca),
        ctypes.byref(c_cfg),
        ctypes.byref(c_out),
    )

    # -- extract packed weights --
    bp = c_out.Bpacked[0]
    output_compressed_stream = np.ctypeslib.as_array(bp.data, shape=(bp.size,)).copy()

    # -- extract compressed lengths --
    ln = c_out.lengths[0]
    compressed_lengths = np.ctypeslib.as_array(ln.data, shape=(ln.size,)).copy()

    # -- extract tiling info (needed for scales packing) --
    tiling = c_out.tilingInfo
    channelDensity = tiling.channelDensity
    _numNeutrons   = tiling.numNeutrons
    divisions      = tiling.divisions

    # free C-side weight/length buffers
    _LIB.dyn8_free(c_out.Bpacked)
    _LIB.dyn32_free(c_out.lengths)

    # -- pack scales & biases (4-bit only) --
    packed_scales = None
    packed_biases = None

    if miniWeights and decodeScales is not None:
        MACs = 16

        # scales: double -> int16
        scales_flat = np.ascontiguousarray(decodeScales, dtype=np.float64).ravel()
        out_size_s = ctypes.c_int(0)
        scales_ptr = _LIB.fetch_unp_organize_group_scales_double_int16(
            scales_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(rb), ctypes.c_int(ca),
            ctypes.c_int(channelDensity), ctypes.c_int(MACs),
            ctypes.c_int(weightBits), ctypes.c_int(_numNeutrons),
            ctypes.c_int(groupSize), ctypes.c_int(0),   # isBias=False
            ctypes.c_int(divisions), ctypes.byref(out_size_s),
        )
        packed_scales = np.ctypeslib.as_array(scales_ptr, shape=(out_size_s.value,)).copy()
        _c_free(scales_ptr)

        # biases: int32 -> int8
        biases_flat = np.ascontiguousarray(decodeBiases, dtype=np.int32).ravel()
        out_size_b = ctypes.c_int(0)
        biases_ptr = _LIB.fetch_unp_organize_group_scales_int32_int8(
            biases_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(rb), ctypes.c_int(ca),
            ctypes.c_int(channelDensity), ctypes.c_int(MACs),
            ctypes.c_int(weightBits), ctypes.c_int(_numNeutrons),
            ctypes.c_int(groupSize), ctypes.c_int(1),   # isBias=True
            ctypes.c_int(divisions), ctypes.byref(out_size_b),
        )
        packed_biases = np.ctypeslib.as_array(biases_ptr, shape=(out_size_b.value,)).copy()
        _c_free(biases_ptr)

    return (np.array(output_compressed_stream, dtype=np.int8),
            np.array(compressed_lengths, dtype=np.int32),
            packed_scales, packed_biases)


# ---------------------------------------------------------------------------
# Pure-Python helpers (no C equivalent)
# ---------------------------------------------------------------------------

def DecimalToFixedPoint(number, integer_bits=10, fraction_bits=6):
    """Converts an unsigned decimal number to a fixed-point equivalent."""
    sign = 0
    if number < 0:
        sign = 1
        number *= -1
    elif number > 2**integer_bits - 1:
        number = 2**integer_bits - 1

    integer_part = int(number)
    fractional_part = number - integer_part

    first_bit_obtained = (integer_part > 0)
    bits_obtained = int(math.log2(integer_part)) + 1 if (integer_part > 0) else 0
    bits_left = integer_bits - bits_obtained
    shift = 0
    fixed_point_scale = integer_part

    if first_bit_obtained:
        fraction_bits = bits_left
    else:
        while fractional_part < 0.5 and shift < 2**fraction_bits - 1:
            fractional_part *= 2
            shift = shift + 1

    frac_bin = ""
    for _ in range(fraction_bits):
        if shift >= 2**fraction_bits - 1:
            break
        fractional_part *= 2
        bit = int(fractional_part)
        fixed_point_scale = fixed_point_scale * 2 | bit
        shift += 1
        fractional_part -= bit
        if fractional_part == 0:
            break

    scale_10bit = fixed_point_scale & 0b1111111111
    shift_6bit = shift & 0b111111
    return (scale_10bit * 2**(-shift_6bit)) * ((-1)**sign)

def FactorToNeutronScaler(float_factor):
    casted_factor = cast(pointer(c_float(float_factor)), POINTER(c_int32)).contents.value
    scaler = (((casted_factor) & 0xffffffff) >> 8) & 0x7fff
    exp_tmp = (((casted_factor) & 0xffffffff) >> 23) & 0xff
    if exp_tmp == 0:
        scaler = 0
    else:
        scaler = scaler | 0x8000
    exp_tmp = -(exp_tmp - 142)
    if exp_tmp > 63:
        exp = 63
    else:
        exp = exp_tmp
    scaler = (exp << 16) | scaler
    return scaler


def ComputeDecodeScales(N, blocksPerCol, scales):
    group_sacles = scales.reshape(N, blocksPerCol)
    channel_scales = np.max(np.abs(group_sacles), axis=1) / 16
    channel_scales_expand = np.expand_dims(channel_scales, axis=1)
    channel_scales_repeat = np.repeat(channel_scales_expand, blocksPerCol, axis=1)
    decode_scales = group_sacles / channel_scales_repeat

    fixed_point_process = np.vectorize(DecimalToFixedPoint)
    fixedPointScales = fixed_point_process(decode_scales)

    scaler_process = np.vectorize(FactorToNeutronScaler)
    factors = scaler_process(channel_scales)

    return fixedPointScales, factors.astype(np.int32)


def ComputeWeightAndBias(B, decodeScales, N, blocksPerCol, groupSize):
    arr = np.asarray(B, dtype=np.uint8).reshape(-1)
    high4 = (arr >> 4) & 0x0F
    low4 = arr & 0x0F

    temp = np.empty(arr.size * 2, dtype=np.int8)
    temp[0::2] = low4
    temp[1::2] = high4
    B_int8 = temp.reshape(N, -1) - 8  # zero point is 8

    bias = np.zeros([N])
    scales_repeated = np.repeat(decodeScales, groupSize, axis=1)
    B_decode = B_int8 * scales_repeated
    B_decode = np.clip(np.floor(B_decode + 0.5), -128, 127)
    for i in range(N):
        bias[i] = np.sum(B_decode[i, :]) * -128

    decodeBiases = np.zeros([N, blocksPerCol], np.int8)
    mask_repeated = scales_repeated < 0
    mask = decodeScales < 0
    B_int8[mask_repeated] = -B_int8[mask_repeated] - 1
    decodeBiases[mask] = -decodeBiases[mask] + 1

    return B_int8, decodeBiases, bias.astype(np.int32)


# ---------------------------------------------------------------------------
# Weight conversion entry points
# ---------------------------------------------------------------------------

def ConvertWeightToNeutron4Bit(cvt_args):
    b_name, B, scales, K, N, blockSize = cvt_args
    weightBits = 4
    print("Packing weight: ", b_name)
    blocksPerCol = (K + blockSize - 1) // blockSize
    decodeScales, factors = ComputeDecodeScales(N, blocksPerCol, scales)
    B_int8, decodeBiases, bias = ComputeWeightAndBias(B, decodeScales, N, blocksPerCol, blockSize)

    packedWeight, compressedLengths, packedDecodeScales, packedDecodeBiases = pack_weight_to_neutron(
        B_int8, N, K, weightBits=weightBits, groupSize=blockSize,
        decodeScales=abs(decodeScales), decodeBiases=decodeBiases, compress=True)

    print(b_name, f"compression ratio is {len(packedWeight.tobytes()) / len(B.tobytes()):.2%}")

    magic_word = int(0x20250918).to_bytes(4, byteorder='little', signed=False)
    lengthsNum = compressedLengths.shape[0].to_bytes(4, byteorder='little', signed=False)
    packedWeightLenght = packedWeight.shape[0].to_bytes(4, byteorder='little', signed=False)

    raw = magic_word + packedWeightLenght + lengthsNum + packedDecodeBiases.tobytes() + \
          packedDecodeScales.tobytes() + bias.tobytes() + factors.tobytes() + \
          packedWeight.tobytes() + compressedLengths.tobytes()
    return (b_name, np.frombuffer(raw, dtype=np.uint8))


def ComputeBias8Bit(B, azp):
    rb, ca = B.shape[0], B.shape[1]
    bias = np.zeros([rb])

    multiplier = 1 if azp is None else -1 * azp.item()
    for i in range(rb):
        bias[i] = np.sum(B[i, :]) * multiplier

    def quant_format(float_factor):
        casted_factor = cast(pointer(c_float(float_factor)), POINTER(c_int32)).contents.value
        scaler = (((casted_factor) & 0xffffffff) >> 8) & 0x7fff
        exp_tmp = (((casted_factor) & 0xffffffff) >> 23) & 0xff
        if exp_tmp == 0:
            scaler = 0
        else:
            scaler = scaler | 0x8000
        exp_tmp = -(exp_tmp - 142)
        if exp_tmp > 63:
            exp = 63
        else:
            exp = exp_tmp
        scaler = (exp << 16) | scaler
        return scaler

    factor = quant_format(1.0)
    factors = np.full((rb, ), factor, dtype=np.float32)
    return bias.astype(np.int32), factors.astype(np.int32)


def ConvertWeightToNeutron8Bit(cvt_args):
    b_name, B, azp = cvt_args
    print("B type:", B.dtype, azp)
    B_trans = np.transpose(B.astype(np.int64))
    print("Packing weight: ", b_name)
    bias, factors = ComputeBias8Bit(B_trans, azp)
    packedWeight, compressedLengths, _, _ = pack_weight_to_neutron(
        B_trans, B.shape[1], B.shape[0], weightBits=8, groupSize=-1,
        decodeScales=None, decodeBiases=None, compress=True)
    magic_word = int(0x20250918).to_bytes(4, byteorder='little', signed=False)
    lengthsNum = compressedLengths.shape[0].to_bytes(4, byteorder='little', signed=False)
    packedWeightLenght = packedWeight.shape[0].to_bytes(4, byteorder='little', signed=False)

    print(b_name, f"compression ratio is {len(packedWeight.tobytes()) / len(B.tobytes()):.2%}")
    raw = magic_word + packedWeightLenght + lengthsNum + \
          bias.tobytes() + factors.tobytes() + \
          packedWeight.tobytes() + compressedLengths.tobytes()

    new = np.frombuffer(raw, dtype=np.int8)

    padded = np.pad(new, (0, B.shape[1] * B.shape[0] - len(raw)), mode='constant')
    reshaped = padded.reshape((B.shape[0], B.shape[1]))
    print("Finished weight: ", b_name)

    return (b_name, reshaped)


class NbitsIndex:
    IN_A = 0
    IN_B = 1
    SCALES = 2
    ZERO_POINTS = 3
    G_IDX = 4
    BIAS = 5
class Int2FloatIndex:
    IN_A = 0
    IN_B = 1
    A_SCALE = 2
    B_SCALE = 3
    A_ZERO_POINT = 4
    B_ZERO_POINT = 5
    BIAS = 6
class IntIndex:
    IN_A = 0
    IN_B = 1
    A_ZERO_POINT = 2
    B_ZERO_POINT = 3
    BIAS = 4

Op2Index = {"MatMulNBits": NbitsIndex, "MatMulIntegerToFloat": Int2FloatIndex,
            "MatMulInteger": IntIndex}


def Main(args):
    model = onnx.load(args.input)
    nodes_4bit = [x for x in model.graph.node if x.op_type == "MatMulNBits"]
    nodes_8bit = [x for x in model.graph.node if x.op_type in ["MatMulIntegerToFloat", "MatMulInteger"]]
    initializers = {init.name: init for init in model.graph.initializer}
    inferred_model = shape_inference.infer_shapes(model)

    def GetInitializerData(name):
        if name not in initializers:
            return None
        tensor = initializers[name]
        return numpy_helper.to_array(tensor)

    cvt_args = []
    for node in nodes_8bit:
        Index = Op2Index[node.op_type]
        b_name = node.input[Index.IN_B]
        b_data = GetInitializerData(b_name)

        azp_data = GetInitializerData(node.input[Index.A_ZERO_POINT])
        bzp_data = GetInitializerData(node.input[Index.B_ZERO_POINT])

        if b_data.shape[1] % 16 or b_data.shape[1] * 16 >= 1024*1024 or np.any(bzp_data != 0):
            continue

        cvt_args.append((b_name, b_data, azp_data))


    with multiprocessing.Pool(processes=args.jobs) as pool:
        results = pool.map(ConvertWeightToNeutron8Bit, cvt_args)

    for b_name, packaged_data in results:
        b_tensor = initializers[b_name]
        b_tensor.CopyFrom(numpy_helper.from_array(packaged_data, name=b_name))

    cvt_args = []
    for node in nodes_4bit:
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        K, N = attrs["K"], attrs["N"]
        block_size, bits = attrs["block_size"], attrs["bits"]
        if K % 16 or N % 128 or bits != 4:
            continue
        if len(node.input) > 3:
            continue

        b_name = node.input[NbitsIndex.IN_B]
        b_tensor = initializers[b_name]
        b_data = numpy_helper.to_array(b_tensor)

        scales_name = node.input[NbitsIndex.SCALES]
        scales_tensor = initializers[scales_name]
        scales_data = numpy_helper.to_array(scales_tensor)

        cvt_args.append((b_name, b_data, scales_data, K, N, block_size))

    with multiprocessing.Pool(processes=args.jobs) as pool:
        results = pool.map(ConvertWeightToNeutron4Bit, cvt_args)

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
