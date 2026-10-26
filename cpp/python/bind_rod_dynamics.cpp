#include "convert.hpp"
#include "pyctr/rod_dynamics.hpp"

namespace py = pybind11;
using pyctr::DynamicCosseratRod;
using pyctr::Mat3;
using pyctr::Vec3;
using pyctr::Vec6;

PYBIND11_MODULE(_rod_dynamics, m) {
  py::class_<DynamicCosseratRod>(m, "DynamicCosseratRod",
                                 "Rod with clamped base (initial conditions p0, R0) and a tip wrench [n, m] (world "
                                 "frame) as boundary condition. Extra params: Bse, Bbt (material damping, 3x3), C "
                                 "(square law drag, 3x3). All default to zero.")
      .def(py::init([](py::object params, double dt, double bdf_alpha) {
             pyctr::TubeParameters t;
             if (params.is_none())
               t.kappa = 0.16;  // as CurvedCosseratRod without params
             else if (py::isinstance<py::dict>(params) || py::hasattr(params, "params"))
               t = pyctr::tube_from_python(params);
             else
               t = params.cast<pyctr::TubeParameters>();
             return DynamicCosseratRod(t, dt, bdf_alpha);
           }),
           py::arg("params") = py::none(), py::arg("dt") = 2e-3, py::arg("bdf_alpha") = -0.2)
      .def(
          "set_initial_conditions",
          [](DynamicCosseratRod& rod, py::array_t<double, py::array::c_style | py::array::forcecast> p0,
             const Mat3& R0) {
            if (p0.size() != 3) throw py::value_error("p0 must have 3 elements");  // Python passes (1, 3)
            rod.set_initial_conditions(Eigen::Map<const Vec3>(p0.data()), R0);
          },
          py::arg("p0"), py::arg("R0"))
      .def("init_static", &DynamicCosseratRod::init_static, py::arg("tip_wrench") = Vec6::Zero(),
           "Solve the static BVP and take it as rest state at t = 0.\n"
           ":return: positions (N, 3), orientations (N, 9), wrenches (N, 6)")
      .def("step", &DynamicCosseratRod::step, py::arg("tip_wrench") = Vec6::Zero(),
           "Advance one time step dt. Call init_static first.\n"
           ":return: positions (N, 3), orientations (N, 9), wrenches (N, 6). Velocities q, w are in y[:, 18:24]")
      // copies: writing into them does not change the rod
      .def_property_readonly("y", [](const DynamicCosseratRod& rod) { return rod.y; })
      .def_property_readonly("z", [](const DynamicCosseratRod& rod) { return rod.z; })
      .def_property_readonly("params", [](const DynamicCosseratRod& rod) { return pyctr::tube_to_python(rod.params); })
      .def_readonly("N", &DynamicCosseratRod::N)
      .def_readonly("dt", &DynamicCosseratRod::dt);
}
