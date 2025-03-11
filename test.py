#!/usr/bin/env python3
"""
test.py - Unit tests for the CUDA & OpenMP Starter Kit for Python Developers.

This module verifies the correctness of reduction and matrix multiplication 
operations implemented in both the baseline Python/Numba code and the optimized 
C++/CUDA (and OpenMP) implementations.

Tests are executed using pytest. The module dynamically selects which backends 
to test based on whether CUDA is supported.

Usage:
    uv run pytest test.py -s -x
    or
    python -m pytest test.py -s -x
"""

import pytest
import numpy as np

from starter_kit_baseline import matmul as matmul_baseline, reduce as reduce_baseline
from starter_kit import supports_cuda, reduce_openmp, reduce_cuda, matmul_openmp, matmul_cuda

backends = ["openmp", "cuda"] if supports_cuda() else ["openmp"]

@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("backend", backends)
def test_reduce(dtype, backend):
    """
    Test the reduction operation for different data types and backends.

    This test generates a 1D array of random values, computes the expected sum
    using the baseline Python/Numba implementation, and compares it against 
    the result from the optimized C++/CUDA (or OpenMP) implementation.

    Parameters:
        dtype (np.dtype): The data type for the array elements (e.g., np.float32).
        backend (str): The backend to test ('openmp' or 'cuda').

    Raises:
        AssertionError: If the results differ by more than the acceptable tolerance.
    """
    # Generate random data
    data = (np.random.rand(1024) * 100).astype(dtype)

    # Get the expected result from the baseline implementation
    expected_result = reduce_baseline(data)

    # Get the result from the C++/CUDA implementation
    if backend == "openmp":
        result = reduce_openmp(data)
    elif backend == "cuda":
        result = reduce_cuda(data)

    # Compare the results
    np.testing.assert_allclose(result, expected_result, rtol=1e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("tile_size", [4, 8, 16, 32])
@pytest.mark.parametrize("backend", backends)
def test_matmul(dtype, tile_size, backend):
    """
    Test the matrix multiplication operation for different tile sizes, data types, 
    and backends.

    This test generates two random 2D matrices, computes the product using the 
    baseline implementation, and compares it to the product computed by the 
    optimized C++/CUDA (or OpenMP) implementation.

    Parameters:
        dtype (np.dtype): The data type for the matrix elements (e.g., np.float32).
        tile_size (int): The tile size to be used for the multiplication kernel.
        backend (str): The backend to test ('openmp' or 'cuda').

    Raises:
        AssertionError: If the output matrices differ by more than the acceptable tolerance.
    """
    # Generate random matrices
    a = (np.random.rand(256, 256) * 100).astype(dtype)
    b = (np.random.rand(256, 256) * 100).astype(dtype)

    # Get the expected result from the baseline implementation
    expected_result = matmul_baseline(a, b)

    # Get the result from the C++/CUDA implementation
    if backend == "openmp":
        result = matmul_openmp(a, b, tile_size=tile_size)
    elif backend == "cuda":
        result = matmul_cuda(a, b, tile_size=tile_size)

    # Compare the results
    np.testing.assert_allclose(result, expected_result, rtol=1e-2)


if __name__ == "__main__":
    pytest.main()
