#pragma once
#include <torch/extension.h>

void game_of_life_step(torch::Tensor grid_in, torch::Tensor grid_out,
                       torch::Tensor mask_in, torch::Tensor mask_out,
                       std::optional<torch::Stream> stream);
