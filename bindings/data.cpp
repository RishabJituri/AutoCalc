// bindings/data.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <optional>
#include <memory>

#include "ag/all.hpp"
#include "ag/data/dataset.hpp"
#include "ag/data/dataloader.hpp"
#include "ag/data/transforms.hpp"

namespace py = pybind11;

namespace {

// Python trampoline for ag::data::Dataset so Python subclasses can override
struct PyDataset : public ag::data::Dataset {
  using ag::data::Dataset::Dataset;
  std::size_t size() const override {
    PYBIND11_OVERLOAD_PURE(std::size_t, ag::data::Dataset, size);
  }
  ag::data::Example get(std::size_t idx) const override {
    PYBIND11_OVERLOAD_PURE(ag::data::Example, ag::data::Dataset, get, idx);
  }
};

// Python iterator wrapper around DataLoader::next()
struct DataLoaderIter {
  std::shared_ptr<ag::data::DataLoader> loader;

  explicit DataLoaderIter(std::shared_ptr<ag::data::DataLoader> l)
  : loader(std::move(l)) {}

  ag::data::Batch next() {
    // Return next batch or raise StopIteration when there are no more batches.
    try {
      if (!loader->has_next()) throw py::stop_iteration();
    } catch (...) {
      // If has_next isn't available or throws, fall back to calling next() and checking size.
    }
    auto ex = loader->next();
    if (ex.size == 0) throw py::stop_iteration();
    return ex;
  }
};

} // anon

namespace ag_bindings {

void bind_data(py::module_& root) {
  py::module_ m = root.def_submodule("data", "Datasets, dataloaders, and transforms");

  // ag::data::Example
  py::class_<ag::data::Example>(m, "Example")
    // If your struct is aggregate-initializable, keep this; otherwise remove.
    .def(py::init<>())
    // ---- tweak these two accessors if your field names differ ----
    .def_readwrite("x", &ag::data::Example::x, "Input/value tensor")       // <-- adjust name if needed
    .def_readwrite("y", &ag::data::Example::y, "Optional target tensor")   // <-- adjust name if needed
    .def("__repr__", [](const ag::data::Example&){
      return "<ag.data.Example>";
    });

  // Batch (collated batch container)
  py::class_<ag::data::Batch>(m, "Batch")
    .def(py::init<>())
    .def_readwrite("x", &ag::data::Batch::x)
    .def_readwrite("y", &ag::data::Batch::y)
    .def_readwrite("size", &ag::data::Batch::size)
    .def("__repr__", [](const ag::data::Batch&){ return "<ag.data.Batch>"; });

  // Transform = std::function<Example(const Example&)>
  m.attr("Transform") = py::module_::import("typing").attr("Callable");

  // Compose
  py::class_<ag::data::Compose>(m, "Compose")
    .def(py::init<>())
    .def_readwrite("ts", &ag::data::Compose::ts, "Sequence of callables (Example->Example)")
    .def("__call__", [](const ag::data::Compose& c, const ag::data::Example& e){
      return c(e);
    });

  // Dataset base (abstract) - use trampoline so Python subclasses can override
  py::class_<ag::data::Dataset, PyDataset, std::shared_ptr<ag::data::Dataset>>(m, "Dataset")
    // allow Python subclasses to call super().__init__()
    .def(py::init<>())
    .def("size", &ag::data::Dataset::size)
    .def("__len__", &ag::data::Dataset::size)
    .def("get", &ag::data::Dataset::get, py::arg("index"))
    .def("__getitem__", &ag::data::Dataset::get, py::arg("index"));

  // TransformDataset wrapper
  py::class_<ag::data::TransformDataset, ag::data::Dataset, std::shared_ptr<ag::data::TransformDataset>>(m, "TransformDataset")
    .def(py::init<std::shared_ptr<ag::data::Dataset>, ag::data::Transform>(),
         py::arg("base"), py::arg("transform"))
    .def("size", &ag::data::TransformDataset::size)
    .def("get",  &ag::data::TransformDataset::get);

  // DataLoaderOptions (expose common knobs; collate can be a Python callable)
  py::class_<ag::data::DataLoaderOptions>(m, "DataLoaderOptions")
    .def(py::init<>())
    .def_readwrite("batch_size", &ag::data::DataLoaderOptions::batch_size)
    .def_readwrite("shuffle",    &ag::data::DataLoaderOptions::shuffle)
    .def_readwrite("drop_last",  &ag::data::DataLoaderOptions::drop_last)
    .def_readwrite("seed",       &ag::data::DataLoaderOptions::seed);

  // DataLoader (own a Dataset shared_ptr and options)
  py::class_<ag::data::DataLoader, std::shared_ptr<ag::data::DataLoader>>(m, "DataLoader")
    .def(py::init<std::shared_ptr<ag::data::Dataset>, ag::data::DataLoaderOptions>(),
         py::arg("dataset"), py::arg("options"))
    .def("reset",   &ag::data::DataLoader::reset)
    .def("rewind",  &ag::data::DataLoader::rewind)
    .def("indices", &ag::data::DataLoader::indices, py::return_value_policy::reference_internal)
    .def_property_readonly("options", &ag::data::DataLoader::options, py::return_value_policy::reference_internal)
    .def("next",    &ag::data::DataLoader::next,
         "Return the next batch (raises StopIteration at end)")

    // number of batches
    .def("__len__", [](ag::data::DataLoader& dl){
      const auto& opts = dl.options();
      std::size_t total = dl.size();
      if (opts.drop_last) return total / opts.batch_size;
      return (total + opts.batch_size - 1) / opts.batch_size;
    })

    // Python iterator protocol
    .def("__iter__", [](std::shared_ptr<ag::data::DataLoader> self){
      // restart each new iteration to be Pythonic; remove reset() if you don’t want that
      self->reset();
      return DataLoaderIter{self};
    })
    ;

  // Iterator binding
  py::class_<DataLoaderIter>(m, "DataLoaderIter")
    .def("__iter__", [](DataLoaderIter& self)->DataLoaderIter& { return self; },
         py::return_value_policy::reference_internal)
    .def("__next__", [](DataLoaderIter& it){
      try { return it.next(); }
      catch (const py::stop_iteration&) { throw; }
      catch (...) {
        // If your DataLoader::next signals end some other way, map it here.
        throw py::stop_iteration();
      }
    });
}

} // namespace ag_bindings
