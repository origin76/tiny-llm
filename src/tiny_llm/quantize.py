import mlx.core as mx
from typing import Any

try:
    from extensions.tiny_llm_ext import quantized_matmul as _quantized_matmul
except ImportError as e:
    raise ImportError("Failed to load C++ extension: {}".format(e)) from e


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
    """
    Weight: (K,N/8) uint32 
    scales: (K,N/G) float16
    biases: (K,N/G) float16
    """
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    return _quantized_matmul(
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        a=a,
        b=b,
        transpose_b=transpose_b,
    )


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    output = quantized_matmul(
        scales=w.scales,
        biases=w.biases,
        group_size=w.group_size,
        bits=w.bits,
        a=x,
        b=w.weight,
        transpose_b=True,
    )
    if bias is not None:
        output = output + bias
    return output
