// Conversions between Python objects and the C++ types, shared by all binding files.
#pragma once

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "pyctr/core.hpp"

namespace pyctr {
namespace py = pybind11;

// TubeParameters from a dict of parameters or from an object with a `params` dict (the pyctr rod classes).
// Same keys as in Python. "g" may be a scalar (gravity -g along z, as in the parameter dicts) or a vector
// (as stored by CosseratRod after construction). "kappa" may be a scalar or the vector [0, kappa, 0].
inline TubeParameters tube_from_python(py::handle obj) {
  py::dict d = py::hasattr(obj, "params") ? obj.attr("params").cast<py::dict>() : py::reinterpret_borrow<py::dict>(obj);
  TubeParameters t;
  auto number = [&](const char* key, double& value) {
    if (d.contains(key)) value = py::float_(d[key]);
  };
  auto matrix = [&](const char* key, Mat3& value) {
    if (d.contains(key)) value = d[key].cast<Mat3>();
  };
  number("E", t.E);
  number("G", t.G);
  number("r_inner", t.r_inner);
  number("r_outer", t.r_outer);
  number("rho", t.rho);
  number("L", t.L);
  number("straight_length", t.straight_length);
  number("s", t.s);
  if (d.contains("g")) {
    py::array_t<double, py::array::c_style | py::array::forcecast> g(d["g"]);
    t.gravity = g.size() == 1 ? Vec3(0, 0, -g.data()[0]) : Vec3(Eigen::Map<const Vec3>(g.data()));
  }
  if (d.contains("kappa")) {
    py::array_t<double, py::array::c_style | py::array::forcecast> k(d["kappa"]);
    t.kappa = k.size() == 1 ? k.data()[0] : k.data()[1];
  }
  matrix("Bse", t.Bse);
  matrix("Bbt", t.Bbt);
  matrix("C", t.C);
  return t;
}

inline std::vector<TubeParameters> tubes_from_python(py::iterable tubes) {
  std::vector<TubeParameters> out;
  for (py::handle tube : tubes) out.push_back(tube_from_python(tube));
  return out;
}

// dict with the same keys as the Python parameters (scalars, g as scalar if it points along -z)
inline py::dict tube_to_python(const TubeParameters& t) {
  py::dict d;
  d["E"] = t.E;
  d["G"] = t.G;
  d["r_inner"] = t.r_inner;
  d["r_outer"] = t.r_outer;
  d["rho"] = t.rho;
  d["L"] = t.L;
  d["straight_length"] = t.straight_length;
  d["s"] = t.s;
  d["kappa"] = t.kappa;
  d["g"] = Eigen::Vector3d(t.gravity);
  d["A"] = t.A();
  d["I"] = t.I();
  d["J"] = t.J();
  d["Kse"] = t.Kse();
  d["Kbt"] = t.Kbt();
  d["Bse"] = t.Bse;
  d["Bbt"] = t.Bbt;
  d["C"] = t.C;
  return d;
}

}  // namespace pyctr
