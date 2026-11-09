// Optional single-service EdgeRM binding. requires Pi verification.
// Source pattern: robot-agent-edgerm/multimodal_robot_middleware.
// Requires EdgeRM's patched rclcpp/rclpy, not stock Jazzy.
#include <memory>
#include <string>
#include <pybind11/pybind11.h>
#include <rmw_edgerm/rmw_edgerm.hpp>
#include <fall_detection_msgs/srv/send_synchronized_frame.hpp>

namespace py = pybind11;
PYBIND11_MODULE(_service, module)
{
  py::class_<rclcpp::Middleware,
    std::unique_ptr<rclcpp::Middleware, py::nodelete>>(module, "RCLCPPMiddleware");
  py::class_<rmw_edgerm::Middleware<fall_detection_msgs::srv::SendSynchronizedFrame>,
    rclcpp::Middleware>(module, "SendSynchronizedFrame")
    .def(py::init<const std::string &, const std::string &, bool>())
    .def_readonly("index", &rclcpp::Middleware::index);
}
