
// ag_bindings.cpp — single TU pybind11 bindings for ag::Variable and ops.
// Drop this in your bindings/ folder (or src/) and build with pybind11.
//
// It tries both "ag/..." and local includes so it works whether your headers
// live under include/ag/... or side-by-side with this file.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ---- flexible includes ----
#if __has_include("ag/core/variables.hpp")
  #include "ag/core/variables.hpp"
#else
  #include "variables.hpp"
#endif

#if __has_include("ag/ops/activations.hpp")
  #include "ag/ops/activations.hpp"
#else
  #include "activations.hpp"
#endif

#if __has_include("ag/ops/elementwise.hpp")
  #include "ag/ops/elementwise.hpp"
#else
  #include "elementwise.hpp"
#endif

#if __has_include("ag/ops/linalg.hpp")
  #include "ag/ops/linalg.hpp"
#else
  #include "linalg.hpp"
#endif

#if __has_include("ag/ops/reduce.hpp")
  #include "ag/ops/reduce.hpp"
#else
  #include "reduce.hpp"
#endif

#if __has_include("ag/ops/reshape.hpp")
  #include "ag/ops/reshape.hpp"
#else
  #include "reshape.hpp"
#endif

#if __has_include("ag/core/graph.hpp")
  #include "ag/core/graph.hpp"
#elif __has_include("ag/ops/graph.hpp")
  #include "ag/ops/graph.hpp"
#else
  #include "graph.hpp"
#endif

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

  // --- Opaque Variable type (constructed in C++; treated as handle in Python) ---
  // We don't expose internals here; you can add .value(), .grad(), etc. later if desired.
  py::class_<ag::Variable>(m, "Variable");

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
