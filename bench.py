#!/usr/bin/env python3
"""
Benchmark tests for the CUDA & OpenMP Starter Kit using "pytest-benchmark".

This module defines benchmarks for reduction and matrix multiplication operations
using parameterized kernels. The available kernels (baseline, OpenMP, and CUDA)
are collected in lists and then each benchmark test is parameterized over the kernel,
data type, and input size (and tile size for matrix multiplication).

Usage:
    uv run pytest --benchmark-enable bench.py
"""

import numpy as np
import pytest

from starter_kit_baseline import reduce as reduce_baseline, matmul as matmul_baseline
from starter_kit import reduce_openmp, reduce_cuda, matmul_openmp, matmul_cuda, supports_cuda

# Build lists of (name, kernel_function) for reduction and matrix multiplication.
REDUCTION_KERNELS = [
    ("baseline", reduce_baseline),
    ("openmp", reduce_openmp),
]
if supports_cuda():
    REDUCTION_KERNELS.append(("cuda", reduce_cuda))

MATMUL_KERNELS = [
    ("baseline", matmul_baseline),
    ("openmp", matmul_openmp),
]
if supports_cuda():
    MATMUL_KERNELS.append(("cuda", matmul_cuda))


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
@pytest.mark.parametrize("n", [2**i for i in range(10, 20)])
@pytest.mark.parametrize("kernel_name,kernel_func", REDUCTION_KERNELS)
def test_reduce(benchmark, dtype, n, kernel_name, kernel_func):
    """
    Benchmark a reduction kernel.

    Parameters:
        dtype (np.dtype): Data type for the input array.
        n (int): Size of the input array.
        kernel_name (str): Name of the kernel (baseline, openmp, cuda).
        kernel_func (function): The reduction function to benchmark.

    The test generates a random 1D array of size n, converts it to the given dtype,
    and then benchmarks the provided reduction kernel.
    """
    data = (np.random.rand(n) * 100).astype(dtype)

    # Wrap the kernel call in a lambda to delay execution until benchmarking.
    benchmark(lambda: kernel_func(data))


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
@pytest.mark.parametrize("n", [2**i for i in range(6, 11)])
@pytest.mark.parametrize("tile_size", [4, 8, 16, 32])
@pytest.mark.parametrize("kernel_name,kernel_func", MATMUL_KERNELS)
def test_matmul(benchmark, dtype, n, tile_size, kernel_name, kernel_func):
    """
    Benchmark a matrix multiplication kernel.

    Parameters:
        dtype (np.dtype): Data type for the input matrices.
        n (int): Dimension of the square matrices.
        tile_size (int): Tile size to use for the multiplication kernel.
        kernel_name (str): Name of the kernel (baseline, openmp, cuda).
        kernel_func (function): The matrix multiplication function to benchmark.

    The test generates a random n x n matrix (used as both input matrices) and benchmarks
    the provided kernel using the specified tile size.
    """
    a = (np.random.rand(n, n) * 100).astype(dtype)

    # Wrap the kernel call in a lambda.
    benchmark(lambda: kernel_func(a, a, tile_size=tile_size))
