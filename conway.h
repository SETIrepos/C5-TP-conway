#pragma once
#include <torch/extension.h>

void game_of_life_step(torch::Tensor grid_in, torch::Tensor grid_out1, torch::Tensor grid_out2, 
                       std::optional<torch::Stream> stream);
