import os
import platform
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc, get_python_lib

import pybind11
import numpy as np

is_macos = platform.system() == "Darwin"
is_linux = platform.system() == "Linux"
is_nvcc_available = os.system("which nvcc > /dev/null 2>&1") == 0
enable_openmp = not is_macos
enable_cuda = is_linux and is_nvcc_available


class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")

        for ext in self.extensions:
            if any(source.endswith(".cu") for source in ext.sources):
                if is_nvcc_available:
                    self.build_cuda_extension(ext)
                else:
                    self.build_gcc_extension(ext)
            else:
                super().build_extension(ext)

    def build_cuda_extension(self, ext):
        # Step 1: Compile CUDA kernels with NVCC (device code only)
        for source in ext.sources:
            if source.endswith(".cu"):
                self.compile_cuda(source)

        # Step 2: Compile host code (including PyBind11 bindings) with GCC
        # Treat .cu file as C++ for host compilation
        host_objects = []
        for source in ext.sources:
            if source.endswith(".cu"):
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_preargs=["-x", "c++"],
                    extra_postargs=[
                        "-fPIC",
                        "-std=c++17",
                        "-fdiagnostics-color=always",
                        "-D__CUDACC__",  # Tell the code that CUDA is available
                    ],
                )
                host_objects.extend(obj)
            else:
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_postargs=[
                        "-fPIC",
                        "-std=c++17",
                        "-fdiagnostics-color=always",
                    ],
                )
                host_objects.extend(obj)

        # Link all object files (host + device)
        all_objects = host_objects + [os.path.join(self.build_temp, "starter_kit.o")]
        self.compiler.link_shared_object(
            all_objects,
            self.get_ext_fullpath(ext.name),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=ext.language,
        )

    def build_gcc_extension(self, ext):
        # Check if compiling on macOS
        objects = []
        for source in ext.sources:
            if source.endswith(".cu"):
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_preargs=["-x", "c++"],
                    extra_postargs=[
                        "-fPIC",
                        "-std=c++17",
                        "-fdiagnostics-color=always",
                    ]
                    + (["-fopenmp"] if enable_openmp else []),
                    include_dirs=ext.include_dirs,
                )
            else:
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_postargs=[
                        "-fPIC",
                        "-std=c++17",
                        "-fdiagnostics-color=always",
                    ]
                    + (["-fopenmp"] if enable_openmp else []),
                    include_dirs=ext.include_dirs,
                )
            objects.extend(obj)

        # Link all object files
        self.compiler.link_shared_object(
            objects,
            self.get_ext_fullpath(ext.name),
            libraries=[lib for lib in ext.libraries if not lib.startswith("cu")],
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=ext.language,
        )

    def compile_cuda(self, source):
        # Compile CUDA device code only using NVCC
        import subprocess
        ext = self.extensions[0]
        output_dir = self.build_temp
        os.makedirs(output_dir, exist_ok=True)
        
        # Only include CUDA-related headers for device compilation
        cuda_include_dirs = [
            "/usr/local/cuda/include/",
            "/usr/include/cuda/",
            "cccl/cub/",
            "cccl/libcudacxx/include",
            "cccl/thrust/",
        ]
        # Filter to only existing directories
        cuda_include_dirs = [d for d in cuda_include_dirs if os.path.exists(d)]
        print(f"\n{'='*70}")
        print(f"CUDA Include Directories Found:")
        for d in cuda_include_dirs:
            print(f"  - {d}")
        if not cuda_include_dirs:
            print("  ⚠ WARNING: No CUDA include directories found!")
        print(f"{'='*70}\n")
        
        cuda_include_dirs_str = " ".join(f"-I{dir}" for dir in cuda_include_dirs)
        output_file = os.path.join(output_dir, "starter_kit.o")

        # Let's try inferring the compute capability from the GPU
        arch_code = "75"  # Default to Turing (T4 GPU)
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit

            device = cuda.Device(0)  # Get the default device
            major, minor = device.compute_capability()
            arch_code = f"{major}{minor}"
            print(f"Detected GPU Compute Capability: {arch_code}")
        except (ImportError, Exception) as e:
            print(f"Could not detect GPU, using default arch {arch_code}: {e}")

        # Compile device code only with nvcc - add define to skip host-only code
        cmd = (
            f"nvcc -dc {source} -o {output_file} -std=c++17 "
            f"-gencode=arch=compute_{arch_code},code=sm_{arch_code} "
            f"--expt-relaxed-constexpr --expt-extended-lambda "
            f"-D__CUDACC_RELAXED_CONSTEXPR__ -DNVCC_DEVICE_COMPILE "
            f"-Xcompiler -fPIC,-Wno-psabi {cuda_include_dirs_str} -O3 -g"
        )
        
        print(f"\n{'='*70}")
        print(f"NVCC Command:")
        print(f"{cmd}")
        print(f"{'='*70}\n")
        
        # Use subprocess to capture output
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\n{'='*70}")
            print(f"NVCC COMPILATION FAILED!")
            print(f"{'='*70}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"{'='*70}")
            print(f"STDERR:\n{result.stderr}")
            print(f"{'='*70}\n")
            raise RuntimeError(f"nvcc compilation of {source} failed with exit code {result.returncode}")
        else:
            print(f"- NVCC compilation successful")
            if result.stdout:
                print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")


__version__ = open("VERSION", "r").read().strip()

long_description = ""
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

# Get Python library path dynamically
python_lib_dir = get_python_lib(standard_lib=True)
python_lib_name = os.path.basename(python_lib_dir).replace(".so", "")

ext_modules = [
    Extension(
        "starter_kit",
        ["starter_kit.cu"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
            get_python_inc(),
            "cccl/cub/",
            "cccl/libcudacxx/include",
            "cccl/thrust/",
            "/usr/local/cuda/include/",
            "/usr/include/cuda/",
        ],
        library_dirs=[
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/wsl/lib",
            python_lib_dir,
        ],
        #
        libraries=[python_lib_name.replace(".a", "")]
        + (["cudart", "cuda", "cublas"] if enable_cuda else [])
        + (["gomp"] if enable_openmp else []),
        #
        extra_link_args=[f"-Wl,-rpath,{python_lib_dir}"]
        + (["-fopenmp"] if enable_openmp else []),
        language="c++",
    ),
]

setup(
    name="PyBindToGPUs",
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/PyBindToGPUs",
    description="Starter Kit project for CUDA- and OpenMP-accelerated Python projects.",
    long_description=long_description,
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.7",
)
