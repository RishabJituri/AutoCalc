// ag_bindings.cpp — single TU pybind11 bindings for ag::Variable and m.
// Drop this in your bindings/ folder (or src/) and build with pybind11.
//
// It tries both "ag/..." and local includes so it works whether your headers
// live under include/ag/... or side-by-side with this file.
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// forward declare nn binder implemented in bindings/nn.cpp
void bind_nn(py::module_ &m);

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
  // auto m = m.def_submodule("Variable","This is the variable (tensor) class");
  // auto nn = m.def_submodule("nn");
  bind_nn(m);
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
    // matmul operator support: a @ b
    .def("__matmul__", [](const ag::Variable& a, const ag::Variable& b){ return ag::matmul(a,b); })
    // .T property returns a materialized transpose (last-two-dims swapped)
    .def_property_readonly("T", [](const ag::Variable& v){ return ag::transpose(v); })
    // at(begin,end) method
    .def("at", [](const ag::Variable& v, const std::vector<std::size_t>& begin, const std::vector<std::size_t>& end){
      return ag::at(v, begin, end);
    }, py::arg("begin"), py::arg("end"))
    // __getitem__ - accept tuple of slices/ints
    .def("__getitem__", [](const ag::Variable& v, py::object idx){
      std::vector<std::size_t> shape = v.shape();
      size_t r = shape.size();
      std::vector<std::size_t> begin(r), end(r);
      // default full slice
      for (size_t i=0;i<r;++i){ begin[i]=0; end[i]=shape[i]; }

      if (py::isinstance<py::tuple>(idx) || py::isinstance<py::list>(idx)) {
        py::sequence seq = idx.cast<py::sequence>();
        size_t provided = seq.size();
        if (provided > r) throw std::invalid_argument("too many indices");
        for (size_t i = 0; i < provided; ++i) {
          py::object it = seq[i];
          if (py::isinstance<py::int_>(it)) {
            long vidx = it.cast<long>();
            if (vidx < 0) throw std::invalid_argument("negative indices not supported");
            begin[i] = (size_t)vidx; end[i] = (size_t)vidx + 1;
          } else if (py::isinstance<py::slice>(it)) {
            py::slice s = it.cast<py::slice>();
            py::ssize_t start, stop, step, slicelength;
            if (!s.compute(shape[i], &start, &stop, &step, &slicelength)) throw std::invalid_argument("invalid slice");
            if (step != 1) throw std::invalid_argument("slice step != 1 not supported");
            begin[i] = (size_t)start; end[i] = (size_t)stop;
          } else {
            throw std::invalid_argument("unsupported index type");
          }
        }
      } else if (py::isinstance<py::int_>(idx)) {
        // single integer index applies to leading dimension (dim 0)
        long vidx = idx.cast<long>(); if (vidx < 0) throw std::invalid_argument("negative indices not supported");
        begin[0] = (size_t)vidx; end[0] = (size_t)vidx + 1;
      } else if (py::isinstance<py::slice>(idx)) {
        // single slice applies to leading dimension (dim 0)
        py::slice s = idx.cast<py::slice>();
        if (r == 0) throw std::invalid_argument("cannot slice scalar");
        py::ssize_t start, stop, step, slicelength;
        if (!s.compute(shape[0], &start, &stop, &step, &slicelength)) throw std::invalid_argument("invalid slice");
        if (step != 1) throw std::invalid_argument("slice step != 1 not supported");
        begin[0] = (size_t)start; end[0] = (size_t)stop;
      } else {
        throw std::invalid_argument("unsupported index type");
      }

      auto outv = ag::at(v, begin, end);
      // Collapse dims corresponding to integer indices to mimic numpy's rank reduction
      std::vector<std::size_t> new_shape;
      for (size_t i = 0; i < begin.size(); ++i) {
        if (!(end[i] == begin[i] + 1)) new_shape.push_back(end[i] - begin[i]);
      }
      if (new_shape.size() != begin.size()) {
        // reshape to remove unit-dims
        if (new_shape.empty()) {
          // return scalar as shape [1] -> reshape to {1}? we can return flattened vector
          return ag::reshape(outv, std::vector<std::size_t>{});
        } else {
          return ag::reshape(outv, new_shape);
        }
      }
      return outv;
    })
    ;

  // --- Graph helpers ---
  m.def("stop_gradient", &ag::stop_gradient, py::arg("x"));
  m.def("detach", &ag::detach, py::arg("x"));

  // auto m = m.def_submodule("m", "Tensor ops that operate on ag::Variable");

  // Elementwise basic
  m.def("add", &ag::add, "Elementwise add", py::arg("a"), py::arg("b"));
  m.def("sub", &ag::sub, "Elementwise subtract", py::arg("a"), py::arg("b"));
  m.def("mul", &ag::mul, "Elementwise multiply", py::arg("a"), py::arg("b"));
  m.def("div", &ag::div, "Elementwise divide", py::arg("a"), py::arg("b"));
  m.def("neg", &ag::neg, "Elementwise negate", py::arg("x"));

  // Elementwise trig/exp
  m.def("sin", &ag::sinv, "Elementwise sin", py::arg("x"));
  m.def("cos", &ag::cosv, "Elementwise cos", py::arg("x"));
  m.def("exp", &ag::expv, "Elementwise exp", py::arg("x"));
  m.def("pow", &ag::pow, "Elementwise power", py::arg("base"), py::arg("exponent"));

  // Activations
  m.def("relu", &ag::relu, py::arg("x"));
  m.def("sigmoid", &ag::sigmoid, py::arg("x"));
  m.def("tanh", &ag::tanhv, py::arg("x"));
  m.def("log",  &ag::logv,  py::arg("x"));
  m.def("clamp", &ag::clamp, py::arg("x"), py::arg("lo"), py::arg("hi"));

  // Linalg
  m.def("matmul", &ag::matmul, py::arg("A"), py::arg("B"));
  m.def("transpose", &ag::transpose, py::arg("A"));

  // Reduce / broadcast
  m.def("reduce_sum", &ag::reduce_sum,
          py::arg("x"),
          py::arg("axes") = std::vector<int>{},
          py::arg("keepdims") = false);

  m.def("reduce_mean", &ag::reduce_mean,
          py::arg("x"),
          py::arg("axes") = std::vector<int>{},
          py::arg("keepdims") = false);

  m.def("broadcast_to", &ag::broadcast_to, py::arg("x"), py::arg("shape"));

  // Reshape
  m.def("flatten", &ag::flatten, py::arg("x"), py::arg("start_dim") = std::size_t{1});
  m.def("reshape", &ag::reshape, py::arg("x"), py::arg("new_shape"));

  // Small QoL aliases at top-level (optional)
  m.def("matmul", &ag::matmul, py::arg("A"), py::arg("B"));
  m.def("reshape", &ag::reshape, py::arg("x"), py::arg("new_shape"));
  m.def("flatten", &ag::flatten, py::arg("x"), py::arg("start_dim") = std::size_t{1});
}
