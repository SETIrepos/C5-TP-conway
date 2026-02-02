#include "conway.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

__global__ void game_of_life_kernel(unsigned char *grid, unsigned char *new_grid, int width,
                                    int height) {
  for (int block_start_x = blockIdx.x * blockDim.x; block_start_x < width;
       block_start_x += blockDim.x * gridDim.x) {

    for (int block_start_y = blockIdx.y * blockDim.y; block_start_y < height;
         block_start_y += blockDim.y * gridDim.y) {

      int x = block_start_x + threadIdx.x;
      int y = block_start_y + threadIdx.y;

      if (x >= width || y >= height)
        continue;

      int idx = y * width + x;

      // Calculate the number of alive neighbors
      int alive_neighbors = 0;
      for (int dy = -1; dy < 2; dy++) {
        for (int dx = -1; dx < 2; dx++) {
          if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height && (dy != 0 || dx != 0)) {
            alive_neighbors += grid[idx + dx + dy * width];
          }
        }
      }

      // Apply the Game of Life rules
      new_grid[idx] = 0;
      if (grid[idx] == 0 && alive_neighbors == 3 || (grid[idx] == 1 && (alive_neighbors == 4 || alive_neighbors == 3))) {
        new_grid[idx] = 1;
      }
      // else if (grid[idx] == 1 && (alive_neighbors == 4 || alive_neighbors == 3)) {
      //   new_grid[idx] = 1;
      // }

      // TODO(once you pass the conformance test): measure with nvprof, and
      // check for different ways of improving performance
    }
  }
}

void game_of_life_step(torch::Tensor grid_in, torch::Tensor grid_out,
                       std::optional<torch::Stream> stream) {
  int width = grid_in.size(1);
  int height = grid_in.size(0);
  assert(grid_in.sizes() == grid_out.sizes());

  cudaStream_t cudaStream = 0;
  if (stream.has_value()) {
    cudaStream = c10::cuda::CUDAStream(stream.value()).stream();
  }

  const dim3 blockSize(32, 32);
  const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(
      grid_in.data_ptr<unsigned char>(), grid_out.data_ptr<unsigned char>(), width, height);
}
