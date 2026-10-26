#include "convert.hpp"
#include "pyctr/robot_dynamics.hpp"

namespace py = pybind11;
using pyctr::Mat3;
using pyctr::Vec3;
using pyctr::Vec6;
using pyctr::VecX;

namespace {

// The Python tubes stay attached: robot.tubes is [(i, rod)] as in ConcentricTubeContinuumRobot, and init_static and
// translate read the tube parameters from the rods again (Python reads L, s, ... of the rods at that time).
struct Robot : pyctr::DynamicConcentricTubeRobot {
  using DynamicConcentricTubeRobot::DynamicConcentricTubeRobot;
  py::list tubes_py;

  void refresh_tubes() {
    std::vector<pyctr::TubeParameters> t;
    for (py::handle item : tubes_py) t.push_back(pyctr::tube_from_python(item.cast<py::tuple>()[1]));
    tubes = std::move(t);
  }
  const Robot& initialized_or_raise() const {
    if (!initialized()) throw py::attribute_error("call init_static first");
    return *this;
  }
};

Vec3 vec3(py::handle a, const char* name) {
  auto v = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
  if (!v || v.size() != 3) throw py::value_error(std::string(name) + " must have 3 elements");
  return Eigen::Map<const Vec3>(v.data());
}

// numpy view on a member, writing into it changes the robot (as with the numpy attributes in Python)
py::array view(py::object self, double* data, std::vector<py::ssize_t> shape, std::vector<py::ssize_t> strides) {
  return py::array_t<double>(std::move(shape), std::move(strides), data, self);
}
py::array view(py::object self, VecX& v) { return view(self, v.data(), {v.size()}, {sizeof(double)}); }
py::array view(py::object self, Vec3& v) { return view(self, v.data(), {3}, {sizeof(double)}); }

}  // namespace

PYBIND11_MODULE(_robot_dynamics, m) {
  py::class_<Robot>(m, "DynamicConcentricTubeRobot",
                    "Concentric tube robot dynamics, Till et al. 2020, port of "
                    "pyctr.cosserat_rod_dynamics.DynamicConcentricTubeRobot (see there).")
      .def(py::init([](py::iterable tubes, double dt, double bdf_alpha, double contact_stiffness,
                       double contact_dissipation, const Mat3& R_init, py::object p_init) {
             py::list rods(tubes);
             Robot r(pyctr::tubes_from_python(rods), dt, bdf_alpha, contact_stiffness, contact_dissipation, R_init,
                     vec3(p_init, "p_init"));
             for (size_t i = 0; i < rods.size(); ++i) r.tubes_py.append(py::make_tuple(i, rods[i]));
             return r;
           }),
           py::arg("tubes"), py::arg("dt") = 2e-3, py::arg("bdf_alpha") = -0.2, py::arg("contact_stiffness") = 1e6,
           py::arg("contact_dissipation") = 0.0, py::arg("R_init") = Mat3(Mat3::Identity()),
           py::arg("p_init") = Vec3(Vec3::Zero()), "tubes: rods with params or dicts")
      .def("rotate", &Robot::rotate, py::arg("alphas"))
      .def(
          "translate",
          [](Robot& r, const VecX& betas) {
            r.refresh_tubes();
            r.translate(betas);
          },
          py::arg("betas"),
          "Change the betas. ValueError if tube 1 does not stick out or an inner tube would end inside an outer "
          "tube. The grid changes in the next step (remesh).")
      .def(
          "init_static",
          [](Robot& r, const Vec6& tip_wrench, const std::optional<pyctr::Contacts>& contacts) {
            r.refresh_tubes();
            return r.init_static(tip_wrench, contacts);
          },
          py::arg("tip_wrench") = Vec6(Vec6::Zero()), py::arg("contacts") = py::none(),
          "Solve the static BVP for the current alphas and betas and take it as rest state at t = 0\n"
          ":param contacts: None or (normals, points), both (M - 1, 3) in the world frame\n"
          ":return: positions (M, 3), orientations (M, 9), wrenches [n, m^b] (M, 6) at the arc lengths self.s")
      .def("step", &Robot::step, py::arg("tip_wrench") = Vec6(Vec6::Zero()), py::arg("contacts") = py::none(),
           "Advance one time step dt with the current alphas and betas (see rotate, translate, remesh). Call "
           "init_static first.\n"
           ":return: positions (M, 3), orientations (M, 9), wrenches [n, m^b] (M, 6) at the arc lengths self.s")
      .def("remesh", &Robot::remesh,
           "Grid for new betas (translate) with the same number of nodes, called by step. Nodes (start of Newton) "
           "and history terms are interpolated in s; the history of u_z and gamma of a tube only from where the tube "
           "was before (none for a tube that was fully retracted). Does nothing if the betas did not change.")
      .def("get_state", &Robot::get_state,
           "All values step() depends on besides the inputs, as one vector: y, z_old, z_h, alphas_old, alphas_h and "
           "the betas of the grid. The size does not change with the betas.")
      .def("set_state", &Robot::set_state, py::arg("x"),
           "Inverse of get_state for the same robot (after init_static), also sets the betas")
      .def("__deepcopy__",
           [](const Robot& r, py::dict memo) {
             Robot c(r);
             c.tubes_py = py::module_::import("copy").attr("deepcopy")(r.tubes_py, memo);
             return c;
           })
      .def_property_readonly("tubes", [](const Robot& r) { return r.tubes_py; })
      .def_property(
          "alphas", [](py::object self) { return view(self, self.cast<Robot&>().alphas); },
          [](Robot& r, const VecX& a) {
            if (a.size() != r.alphas.size()) throw py::value_error("one alpha per tube");
            r.alphas = a;
          })
      .def_property(
          "betas", [](py::object self) { return view(self, self.cast<Robot&>().betas); },
          [](Robot& r, const VecX& b) {
            if (b.size() != r.betas.size()) throw py::value_error("one beta per tube");
            r.betas = b;
          })
      .def_property(
          "p_init",  // (3, 1) as in Python
          [](py::object self) {
            return view(self, self.cast<Robot&>().p_init.data(), {3, 1}, {sizeof(double), sizeof(double)});
          },
          [](Robot& r, py::object p) { r.p_init = vec3(p, "p_init"); })
      .def_property(
          "R_init",  // Mat3 is column major
          [](py::object self) {
            return view(self, self.cast<Robot&>().R_init.data(), {3, 3}, {sizeof(double), 3 * sizeof(double)});
          },
          [](Robot& r, const Mat3& R) { r.R_init = R; })
      .def_property(
          "base_velocity",
          [](py::object self) {
            Robot& r = self.cast<Robot&>();
            return py::make_tuple(view(self, r.base_velocity.first), view(self, r.base_velocity.second));
          },
          [](Robot& r, py::sequence v) {
            if (v.size() != 2) throw py::value_error("base_velocity: (linear, angular)");
            r.base_velocity = {vec3(v[0], "linear velocity"), vec3(v[1], "angular velocity")};
          })
      .def_readonly("dt", &Robot::dt)
      .def_readwrite("contact_stiffness", &Robot::contact_stiffness)
      .def_readwrite("contact_dissipation", &Robot::contact_dissipation)
      // copies, AttributeError before init_static (ConcentricTubeRobotSystem checks hasattr(robot, "s"))
      .def_property_readonly("s", [](const Robot& r) { return r.initialized_or_raise().s; })
      .def_property_readonly("y", [](const Robot& r) { return r.initialized_or_raise().y; })
      .def_property_readonly("radius", [](const Robot& r) { return r.initialized_or_raise().radius; })
      .def_property_readonly("contact_force", [](const Robot& r) { return r.initialized_or_raise().contact_force; });
}
