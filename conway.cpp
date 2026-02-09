#include "conway.h"
#include <optional>
#include <torch/extension.h>

class GameOfLife {
public:
  void step(torch::Tensor grid_in, torch::Tensor grid_out1, torch::Tensor grid_out2,
            std::optional<torch::Stream> stream = std::nullopt) {
    game_of_life_step(grid_in, grid_out1, grid_out2, stream);
  }
};

PYBIND11_MODULE(conway, m) {
  py::class_<GameOfLife>(m, "GameOfLife")
      .def(py::init<>())
      .def("step", &GameOfLife::step, py::arg("grid_in"), py::arg("grid_out1"),
           py::arg("grid_out2"), py::arg("stream") = py::none());
}
