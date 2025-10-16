#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "ag/all.hpp"

namespace py = pybind11;

void bind_nn(py::module_ &m) {
  auto nn = m.def_submodule("nn", "Neural network modules and layers");

  // Base Module (shared ptr)
  py::class_<ag::nn::Module, std::shared_ptr<ag::nn::Module>>(nn, "Module")
    .def("parameters", [](ag::nn::Module& self) -> std::vector<ag::Variable*> { return self.parameters(); },
         py::return_value_policy::reference_internal)
    .def("zero_grad", [](ag::nn::Module& self){ self.zero_grad(); })
    .def("train", [](ag::nn::Module& self, bool mode){ if(mode) self.train(); else self.eval(); });

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

  // Sequential
  py::class_<ag::nn::Sequential, ag::nn::Module, std::shared_ptr<ag::nn::Sequential>>(nn, "Sequential")
    .def(py::init<>())
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
}
