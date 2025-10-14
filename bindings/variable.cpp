// ag_bindings.cpp — single TU pybind11 bindings for ag::Variable and ops.
// Drop this in your bindings/ folder (or src/) and build with pybind11.
//
// It tries both "ag/..." and local includes so it works whether your headers
// live under include/ag/... or side-by-side with this file.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// ---- flexible includes ----
#include "ag/all.hpp"

// Optional: version string helper
#ifndef AG_BINDINGS_VERSION
#define AG_BINDINGS_VERSION "0.1.0"
#endif

namespace {
// A tiny context-manager wrapper for NoGradGuard so `with ag.nograd():` works
struct PyNoGradCtx {
  std::unique_ptr<ag::NoGradGuard> guard;
  PyNoGradCtx() = default;
  PyNoGradCtx& enter() { guard = std::make_unique<ag::NoGradGuard>(); return *this; }
  void exit(py::object, py::object, py::object) { guard.reset(); }
};
} // anon

PYBIND11_MODULE(ag, m) {
  m.doc() = "Minimal pybind11 bindings for ag::Variable core and ops";
  m.attr("__version__") = AG_BINDINGS_VERSION;

  // --- Grad mode ---
  m.def("is_grad_enabled", &ag::is_grad_enabled, "Return global grad mode");
  m.def("set_grad_enabled", &ag::set_grad_enabled, py::arg("enabled"),
        "Enable/disable global grad mode for new nodes");

  py::class_<PyNoGradCtx>(m, "nograd")
      .def(py::init<>())
      .def("__enter__", &PyNoGradCtx::enter, py::return_value_policy::reference_internal)
      .def("__exit__", &PyNoGradCtx::exit);

  // --- Variable type: expose constructor and useful accessors ---
  py::class_<ag::Variable>(m, "Variable")
    .def(py::init<const std::vector<float>&, const std::vector<std::size_t>&, bool>(),
         py::arg("value"), py::arg("shape"), py::arg("requires_grad") = false)
    .def_static("from_numpy", [](py::array_t<float, py::array::c_style | py::array::forcecast> arr, bool requires_grad){
        py::buffer_info info = arr.request();
        std::vector<float> data((float*)info.ptr, (float*)info.ptr + info.size);
        std::vector<std::size_t> shape(info.shape.begin(), info.shape.end());
        return ag::Variable(data, shape, requires_grad);
    }, py::arg("array"), py::arg("requires_grad") = false)
    .def("value", [](const ag::Variable& v){ return v.value(); })
    .def("grad", [](const ag::Variable& v){ return v.grad(); })
    .def("shape", [](const ag::Variable& v){ return v.shape(); })
    .def("requires_grad", &ag::Variable::requires_grad)
    .def("zero_grad", [](ag::Variable& v){ v.zero_grad(); })
    .def("backward", [](ag::Variable& v){ v.backward(); })
    ;

  // --- Graph helpers ---
  m.def("stop_gradient", &ag::stop_gradient, py::arg("x"));
  m.def("detach", &ag::detach, py::arg("x"));

  // --- Submodule for ops ---
  auto ops = m.def_submodule("ops", "Tensor ops that operate on ag::Variable");

  // Elementwise basic
  ops.def("add", &ag::add, "Elementwise add", py::arg("a"), py::arg("b"));
  ops.def("sub", &ag::sub, "Elementwise subtract", py::arg("a"), py::arg("b"));
  ops.def("mul", &ag::mul, "Elementwise multiply", py::arg("a"), py::arg("b"));
  ops.def("div", &ag::div, "Elementwise divide", py::arg("a"), py::arg("b"));
  ops.def("neg", &ag::neg, "Elementwise negate", py::arg("x"));

  // Elementwise trig/exp
  ops.def("sin", &ag::sinv, "Elementwise sin", py::arg("x"));
  ops.def("cos", &ag::cosv, "Elementwise cos", py::arg("x"));
  ops.def("exp", &ag::expv, "Elementwise exp", py::arg("x"));
  ops.def("pow", &ag::pow, "Elementwise power", py::arg("base"), py::arg("exponent"));

  // Activations
  ops.def("relu", &ag::relu, py::arg("x"));
  ops.def("sigmoid", &ag::sigmoid, py::arg("x"));
  ops.def("tanh", &ag::tanhv, py::arg("x"));
  ops.def("log",  &ag::logv,  py::arg("x"));
  ops.def("clamp", &ag::clamp, py::arg("x"), py::arg("lo"), py::arg("hi"));

  // Linalg
  ops.def("matmul", &ag::matmul, py::arg("A"), py::arg("B"));

  // Reduce / broadcast
  ops.def("reduce_sum", &ag::reduce_sum,
          py::arg("x"),
          py::arg("axes") = std::vector<int>{},
          py::arg("keepdims") = false);

  ops.def("reduce_mean", &ag::reduce_mean,
          py::arg("x"),
          py::arg("axes") = std::vector<int>{},
          py::arg("keepdims") = false);

  ops.def("broadcast_to", &ag::broadcast_to, py::arg("x"), py::arg("shape"));

  // Reshape
  ops.def("flatten", &ag::flatten, py::arg("x"), py::arg("start_dim") = std::size_t{1});
  ops.def("reshape", &ag::reshape, py::arg("x"), py::arg("new_shape"));

  // Small QoL aliases at top-level (optional)
  m.def("matmul", &ag::matmul, py::arg("A"), py::arg("B"));
  m.def("reshape", &ag::reshape, py::arg("x"), py::arg("new_shape"));
  m.def("flatten", &ag::flatten, py::arg("x"), py::arg("start_dim") = std::size_t{1});
}
