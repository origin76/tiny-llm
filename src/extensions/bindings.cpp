// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "tiny_llm_ext.h"
#include "axpby.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "tiny-llm extensions for MLX";

    m.def("load_library", &tiny_llm_ext::load_library, "device"_a, "path"_a);

    m.def("axpby", &tiny_llm_ext::axpby, "x"_a, "y"_a, "alpha"_a, "beta"_a, nb::kw_only(), "stream"_a = nb::none(),
          R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )");

	    m.def("quantized_matmul", &tiny_llm_ext::quantized_matmul, "scales"_a, "biases"_a, "group_size"_a, "bits"_a,
	          "a"_a, "b"_a, nb::kw_only(), "transpose_b"_a = false, "stream"_a = nb::none(),
	          R"(
	        Quantized matrix multiplication
	
	        Computes: a @ dequant(b, scales, biases)
	
	        Args:
	            scales (array): Quantization scales, shape [n, k/group_size].
	            biases (array): Quantization biases, shape [n, k/group_size].
	            group_size (int): Size of each quantization group.
	            bits (int): Number of bits per packed value (currently only 4).
	            a (array): Input activation, shape [..., m, k].
	            b (array): Quantized weight. If transpose_b=True: [n, k*bits/32];
	                       else: [k*bits/32, n].
	            transpose_b (bool): Whether to transpose b. Default False.

        Returns:
            array: Result of quantized matmul, shape [..., m, n]
      )");
}
