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
from starter_kit import (
    supports_cuda,
    get_cuda_device_count,
    reduce_openmp,
    reduce_cuda,
    reduce_cuda_multigpu,
    matmul_openmp,
    matmul_cuda,
    matmul_cuda_multigpu,
)

backends = ["openmp", "cuda"] if supports_cuda() else ["openmp"]
multigpu_backends = ["cuda_multigpu"] if get_cuda_device_count() > 1 else []
all_backends = backends + multigpu_backends

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
@pytest.mark.parametrize("size", [1024, 8192, 65536])
@pytest.mark.parametrize("backend", multigpu_backends)
def test_reduce_multigpu(dtype, size, backend):
    """
    Test the multi-GPU reduction operation for different data types and sizes.

    This test verifies that multi-GPU reduction produces the same results
    as the baseline implementation for various data sizes.

    Parameters:
        dtype (np.dtype): The data type for the array elements.
        size (int): The size of the array to reduce.
        backend (str): The backend to test ('cuda_multigpu').

    Raises:
        AssertionError: If the results differ by more than the acceptable tolerance.
    """
    if not multigpu_backends:
        pytest.skip("Multi-GPU not available (requires 2+ GPUs)")

    # Generate random data
    data = (np.random.rand(size) * 100).astype(dtype)

    # Get the expected result from the baseline implementation
    expected_result = reduce_baseline(data)

    # Get the result from the multi-GPU implementation
    if backend == "cuda_multigpu":
        result = reduce_cuda_multigpu(data)

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


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("size", [128, 256, 512])
@pytest.mark.parametrize("tile_size", [8, 16, 32])
@pytest.mark.parametrize("backend", multigpu_backends)
def test_matmul_multigpu(dtype, size, tile_size, backend):
    """
    Test the multi-GPU matrix multiplication operation for different sizes and tile sizes.

    This test verifies that multi-GPU matrix multiplication produces the same results
    as the baseline implementation for various matrix sizes.

    Parameters:
        dtype (np.dtype): The data type for the matrix elements.
        size (int): The dimension of the square matrices.
        tile_size (int): The tile size to be used for the multiplication kernel.
        backend (str): The backend to test ('cuda_multigpu').

    Raises:
        AssertionError: If the output matrices differ by more than the acceptable tolerance.
    """
    if not multigpu_backends:
        pytest.skip("Multi-GPU not available (requires 2+ GPUs)")

    # Generate random matrices
    a = (np.random.rand(size, size) * 100).astype(dtype)
    b = (np.random.rand(size, size) * 100).astype(dtype)

    # Get the expected result from the baseline implementation
    expected_result = matmul_baseline(a, b)

    # Get the result from the multi-GPU implementation
    if backend == "cuda_multigpu":
        result = matmul_cuda_multigpu(a, b, tile_size=tile_size)

    # Compare the results
    np.testing.assert_allclose(result, expected_result, rtol=1e-2)


if __name__ == "__main__":
    pytest.main()
