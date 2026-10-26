#include <pybind11/stl/filesystem.h>

#include "convert.hpp"
#include "pyctr/config.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_config, m) {
  m.doc() = "Loading robots from YAML files (pyctr.yaml_to_model)";
  m.def("setup_tubes", &pyctr::setup_tubes, py::arg("config_path"),
        "Raw parameters of every tube as dicts, in the units of the files (pyctr.yaml_to_model.setup_tubes)");
  m.def(
      "load_tubes",
      [](const std::filesystem::path& config_path) {
        py::list out;
        for (const auto& t : pyctr::load_tubes(config_path)) out.append(pyctr::tube_to_python(t));
        return out;
      },
      py::arg("config_path"),
      "Tube parameter dicts in SI units with the conversions of pyctr.yaml_to_model.load_continous_ctcr_model");
}
