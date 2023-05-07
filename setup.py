from setuptools import setup, Extension

module = Extension(
    "synapsia",
    sources = ["pybind_synapsia.cpp"],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-arch arm64"]
)

setup(
    name="synapsia",
    ext_modules=[module]
)