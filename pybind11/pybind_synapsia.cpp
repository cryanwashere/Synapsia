#include </opt/anaconda3/lib/python3.8/site-packages/pybind11/include/pybind11/pybind11.h>
#include "pybind_example.c"

namespace py = pybind11;

PYBIND11_MODULE(pybind_example, m) 
{
    m.def("add", &add, "A function that adds two integers");
}