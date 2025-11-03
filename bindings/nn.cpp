#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "ag/all.hpp"

namespace py = pybind11;

void bind_nn(py::module_ &m) {
  auto nn = m.def_submodule("nn", "Neural network modules and layers");

  // Trampoline to allow Python subclasses to override virtual forward() and provide _parameters
  struct PyModule : ag::nn::Module {
    using ag::nn::Module::Module;
    ag::Variable forward(const ag::Variable& x) override {
      PYBIND11_OVERLOAD_PURE(ag::Variable, ag::nn::Module, forward, x);
    }
    // default _parameters() returns empty list; Python subclasses may override if desired
    std::vector<ag::Variable*> _parameters() override { return {}; }
  };

  // Base Module (shared ptr) — allow Python subclassing via PyModule
  py::class_<ag::nn::Module, PyModule, std::shared_ptr<ag::nn::Module>>(nn, "Module")
    .def(py::init<>())
    // Python-aware parameters(): merge C++-registered params with any parameters found on the Python
    // instance __dict__ (including parameters from Python-defined child modules). This makes attribute
    // assignment from Python subclasses robust even when the C++ register hooks cannot obtain a
    // shared_ptr to the assigned Python object.
    .def("parameters", [](py::object self_obj) {
      std::vector<ag::Variable*> out;
      // Use builtins.id to track visited Python objects and avoid infinite recursion
      py::object builtins = py::module_::import("builtins");
      py::set visited;

      // Recursive lambda (C++14 compatible via std::function)
      std::function<void(py::object)> collect;
      collect = [&](py::object obj) {
        try {
          py::object oid = builtins.attr("id")(obj);
          if (visited.contains(oid)) return; // already processed
          visited.add(oid);
        } catch (...) {
          // if id() fails for some reason, continue without visited guard
        }

        // If obj is a backend Variable, add it
        try {
          if (py::isinstance<ag::Variable>(obj)) {
            ag::Variable &v = obj.cast<ag::Variable&>();
            out.push_back(&v);
            return;
          }
        } catch (...) {}

        // If obj is a module-like C++ Module, prefer native parameters() to avoid Python recursion
        try {
          if (py::isinstance<ag::nn::Module>(obj)) {
            try {
              // Attempt to cast to C++ reference; this will succeed for objects with a C++ holder
              ag::nn::Module &mcpp = obj.cast<ag::nn::Module&>();
              auto native = mcpp.parameters();
              out.insert(out.end(), native.begin(), native.end());
              return;
            } catch (...) {
              // Fall through to python-side exploration for pure-Python Module objects
            }
          }
        } catch (...) {}

        // Inspect Python __dict__ for Variables or child Modules and recurse
        try {
          if (py::hasattr(obj, "__dict__")) {
            py::dict d = obj.attr("__dict__");
            for (auto item : d) {
              py::handle h = item.second;
              py::object val = py::reinterpret_borrow<py::object>(h);
              try { collect(val); } catch (...) { /* ignore individual failures */ }
            }
          }
        } catch (...) {}
      };

      // Start with self: prefer native C++ params then fallback to Python inspection.
      try {
        // Try C++ native first
        ag::nn::Module &self_cpp = self_obj.cast<ag::nn::Module&>();
        try { auto native = self_cpp.parameters(); out.insert(out.end(), native.begin(), native.end()); } catch(...){}
      } catch(...){}

      // Then recursively collect from the Python instance (this will also visit pythonic children)
      collect(self_obj);

      // Deduplicate pointers while preserving order
      std::vector<ag::Variable*> uniq;
      for (auto p : out) {
        if (!p) continue;
        if (std::find(uniq.begin(), uniq.end(), p) == uniq.end()) uniq.push_back(p);
      }
      return uniq;
    }, py::return_value_policy::reference_internal)
    .def("named_parameters", [](ag::nn::Module& self, const std::string& prefix){ return self.named_parameters(prefix); })
    .def("zero_grad", [](ag::nn::Module& self){ self.zero_grad(); })
    .def("train", [](ag::nn::Module& self, bool mode){ if(mode) self.train(); else self.eval(); })
    .def("__call__", [](ag::nn::Module& self, const ag::Variable& x){ return self.forward(x); })
    // explicit register helpers (useful from Python)
    .def("register_module", [](ag::nn::Module& self, std::shared_ptr<ag::nn::Module> m){ self.register_module(m); })
    .def("register_module", [](ag::nn::Module& self, const std::string& name, std::shared_ptr<ag::nn::Module> m){ self.register_module(name, m); })
    .def("register_parameter", [](ag::nn::Module& self, const std::string& name, ag::Variable& v){ self.register_parameter(name, v); })
    // PyTorch-like __setattr__: operate on the Python instance and perform best-effort registration.
    .def("__setattr__", [](py::object self_obj, const std::string& name, py::object obj){
      try {
        // Ignore assigning Python modules
        if (py::isinstance<py::module_>(obj)) {
          py::dict d = self_obj.attr("__dict__");
          d[py::str(name)] = obj;
          return;
        }

        if (py::isinstance<ag::nn::Module>(obj)) {
          // Try to obtain a std::shared_ptr for the assigned module and register it with the C++ Module
          try {
            auto child_sp = obj.cast<std::shared_ptr<ag::nn::Module>>();
            ag::nn::Module &self_cpp = self_obj.cast<ag::nn::Module&>();
            self_cpp.register_module(name, child_sp);
          } catch (...) {
            // If cast to shared_ptr fails (e.g., pure-Python subclass without a C++ holder),
            // fall back to storing in __dict__ and rely on parameters() to discover parameters.
          }
          py::dict d = self_obj.attr("__dict__");
          d[py::str(name)] = obj;
          return;
        }

        if (py::isinstance<ag::Variable>(obj)) {
          try {
            ag::Variable &v = obj.cast<ag::Variable&>();
            ag::nn::Module &self_cpp = self_obj.cast<ag::nn::Module&>();
            // attempt to register parameter on the C++ side; ignore failures and fall back to __dict__ storage
            try { self_cpp.register_parameter(name, v); } catch(...) {}
          } catch (...) {}
          py::dict d = self_obj.attr("__dict__");
          d[py::str(name)] = obj;
          return;
        }
      } catch (...) {
        // fall back to normal attribute set
      }
      // default store in __dict__ to avoid infinite recursion
      py::dict d = self_obj.attr("__dict__");
      d[py::str(name)] = obj;
    });

  // Linear: Linear(in_features, out_features, bias=true, init_scale=0.02, seed=0xC0FFEE)
  py::class_<ag::nn::Linear, ag::nn::Module, std::shared_ptr<ag::nn::Linear>>(nn, "Linear")
    .def(py::init([](std::size_t in_f, std::size_t out_f, bool bias, float init_scale, unsigned long long seed){
      return std::make_shared<ag::nn::Linear>(in_f, out_f, bias, init_scale, seed);
    }), py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true, py::arg("init_scale") = 0.02f, py::arg("seed") = 0xC0FFEEULL)
    .def("forward", [](ag::nn::Linear& self, const ag::Variable& x){ return self.forward(x); });

  // Conv2d: use factory to map scalar kernel/strides/pads -> pair args
  py::class_<ag::nn::Conv2d, ag::nn::Module, std::shared_ptr<ag::nn::Conv2d>>(nn, "Conv2d")
    .def(py::init([](std::size_t in_ch, std::size_t out_ch,
                     std::size_t kh, std::size_t kw,
                     std::size_t sh, std::size_t sw,
                     std::size_t ph, std::size_t pw,
                     std::size_t dh, std::size_t dw,
                     bool bias, float init_scale, unsigned long long seed){
      return std::make_shared<ag::nn::Conv2d>(in_ch, out_ch,
                                              std::make_pair(kh,kw),
                                              std::make_pair(sh,sw),
                                              std::make_pair(ph,pw),
                                              std::make_pair(dh,dw),
                                              bias, init_scale, seed);
    }), py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_h"), py::arg("kernel_w"),
       py::arg("stride_h")=1, py::arg("stride_w")=1, py::arg("pad_h")=0, py::arg("pad_w")=0,
       py::arg("dilation_h")=1, py::arg("dilation_w")=1, py::arg("bias")=true, py::arg("init_scale")=0.02f, py::arg("seed")=0xC0FFEEULL)
    .def("forward", [](ag::nn::Conv2d& self, const ag::Variable& x){ return self.forward(x); });

  // Sequential (do not expose default constructor to Python for now)
  py::class_<ag::nn::Sequential, ag::nn::Module, std::shared_ptr<ag::nn::Sequential>>(nn, "Sequential")
    // Note: constructor intentionally not exposed; create instances via C++ or other factories
    .def("append", [](ag::nn::Sequential& s, std::shared_ptr<ag::nn::Module> m){ s.push(std::move(m)); })
    .def("forward", [](ag::nn::Sequential& s, const ag::Variable& x){ return s.forward(x); });

  // BatchNorm2d
  py::class_<ag::nn::BatchNorm2d, ag::nn::Module, std::shared_ptr<ag::nn::BatchNorm2d>>(nn, "BatchNorm2d")
    .def(py::init<std::size_t, float, float>(), py::arg("num_features"), py::arg("eps")=1e-5f, py::arg("momentum")=0.1f)
    .def("forward", [](ag::nn::BatchNorm2d& self, const ag::Variable& x){ return self.forward(x); });

  // Dropout
  py::class_<ag::nn::Dropout, ag::nn::Module, std::shared_ptr<ag::nn::Dropout>>(nn, "Dropout")
    .def(py::init<float, unsigned long long>(), py::arg("p")=0.5f, py::arg("seed")=0ULL)
    .def("forward", [](ag::nn::Dropout& self, const ag::Variable& x){ return self.forward(x); });

  // Pooling
  py::class_<ag::nn::MaxPool2d, ag::nn::Module, std::shared_ptr<ag::nn::MaxPool2d>>(nn, "MaxPool2d")
    .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>(),
         py::arg("kH"), py::arg("kW"), py::arg("sH")=0, py::arg("sW")=0, py::arg("pH")=0, py::arg("pW")=0)
    .def("forward", [](ag::nn::MaxPool2d& self, const ag::Variable& x){ return self.forward(x); });

  py::class_<ag::nn::AvgPool2d, ag::nn::Module, std::shared_ptr<ag::nn::AvgPool2d>>(nn, "AvgPool2d")
    .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>(),
         py::arg("kH"), py::arg("kW"), py::arg("sH")=0, py::arg("sW")=0, py::arg("pH")=0, py::arg("pW")=0)
    .def("forward", [](ag::nn::AvgPool2d& self, const ag::Variable& x){ return self.forward(x); });

  // LSTM
  py::class_<ag::nn::LSTM, ag::nn::Module, std::shared_ptr<ag::nn::LSTM>>(nn, "LSTM")
    .def(py::init<std::size_t, std::size_t, int, bool>(), py::arg("input_size"), py::arg("hidden_size"), py::arg("num_layers")=1, py::arg("bias")=true)
    .def("forward", [](ag::nn::LSTM& self, const ag::Variable& x){ return self.forward(x); });

  // Losses
  nn.def("cross_entropy", [](const ag::Variable& logits, const std::vector<std::size_t>& targets){
    return ag::nn::cross_entropy(logits, targets);
  }, py::arg("logits"), py::arg("targets"));

  // optim namespace and SGD wrapper
  auto optim = nn.def_submodule("optim", "Optimizers");
  py::class_<ag::nn::SGD>(optim, "SGD")
    .def(py::init<float, float, bool, float>(), py::arg("lr")=0.1f, py::arg("momentum")=0.0f, py::arg("nesterov")=false, py::arg("weight_decay")=0.0f)
    .def("step", [](ag::nn::SGD& self, ag::nn::Module& m){ self.step(m); });

  // Re-export optim.SGD as nn.SGD for convenience
  try {
    nn.attr("SGD") = optim.attr("SGD");
  } catch (const std::exception&) {
    // ignore if attr copy fails
  }
}
