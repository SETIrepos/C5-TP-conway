#include "conway.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

// ==========================================
// LOOKUP TABLE LOGIC
// ==========================================

unsigned char* d_lut = nullptr; 

void precompute_lut() {
    if (d_lut != nullptr) return;

    // LUT compacte : on encode les 512 configurations 3x3 (2^9) en 64 octets
    // chaque octet contient 8 résultats (1 bit par configuration).
    // index_byte = pattern >> 3  (6 bits -> 0..63)
    // bit_in_byte = pattern & 0x7 (3 bits -> 0..7)
    unsigned char host_lut[64];
    for (int i = 0; i < 64; ++i) host_lut[i] = 0;

    // Pour chaque configuration 3x3 (bits 0..8, centre = bit 4)
    for (int pattern = 0; pattern < 512; ++pattern) {
        int self = (pattern >> 4) & 1; // centre
        int neighbors = 0;
        for (int b = 0; b < 9; ++b) {
            if (b == 4) continue;
            neighbors += (pattern >> b) & 1;
        }

        int alive = (neighbors == 3) || (self == 1 && neighbors == 2);
        if (alive) {
            int byte_idx = pattern >> 3;        // /8
            int bit_pos  = pattern & 0x7;       // %8
            host_lut[byte_idx] |= (1u << bit_pos);
        }
    }

    // Copier la LUT compacte sur le device (64 octets)
    cudaMalloc(&d_lut, sizeof(host_lut));
    cudaMemcpy(d_lut, host_lut, sizeof(host_lut), cudaMemcpyHostToDevice);
}

#define TILE_DIM 32
#define HALO_DIM (TILE_DIM + 2)

__global__ void game_of_life_kernel(unsigned char *grid, unsigned char *new_grid, int width, int height, unsigned char* lut) {
    __shared__ unsigned char tile[HALO_DIM][HALO_DIM];

    // Stratégie Grid Stride Loop avec Shared Memory et LUT
    // Chaque bloc produit une tuile de output de 32x32 pixels
    // avec 32x32 threads par bloc.
    
    int tx = threadIdx.x; // 0..31
    int ty = threadIdx.y; // 0..31
    
    // Coordonnées de sortie
    int gx_out = blockIdx.x * TILE_DIM + tx;
    int gy_out = blockIdx.y * TILE_DIM + ty;

    // --- Chargement Collaboratif ---
    // 1024 threads chargent 34x34 = 1156 elements.
    int tid = ty * blockDim.x + tx; 
    int num_threads = blockDim.x * blockDim.y; // 1024
    
    // Origine de lecture : Coin haut-gauche du bloc de 32x32 moins 1 (halo)
    int read_base_x = blockIdx.x * TILE_DIM - 1;
    int read_base_y = blockIdx.y * TILE_DIM - 1;

    for (int i = tid; i < HALO_DIM * HALO_DIM; i += num_threads) {
        int ly = i / HALO_DIM;
        int lx = i % HALO_DIM;
        
        int gx = read_base_x + lx;
        int gy = read_base_y + ly;

        unsigned char val = 0;
        if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
            val = grid[gy * width + gx];
        }
        tile[ly][lx] = val;
    }

    __syncthreads();

    // --- Calcul LUT ---
    // Chaque thread calcule 1 pixel de sortie.
    // L'indice (r, c) dans 'tile' pour le pixel (gx_out, gy_out) est (ty + 1, tx + 1).
    if (gx_out < width && gy_out < height) {
        int pattern = 0;
        int r = ty + 1;
        int c = tx + 1;

        #pragma unroll
        for (int dr = -1; dr <= 1; ++dr) {
            #pragma unroll
            for (int dc = -1; dc <= 1; ++dc) {
                if (tile[r + dr][c + dc]) {
                    pattern |= (1 << ((dr + 1) * 3 + (dc + 1)));
                }
            }
        }
        new_grid[gy_out * width + gx_out] = (lut[pattern >> 3] >> (pattern & 0x7)) & 1;
    }
}

void game_of_life_step(torch::Tensor grid_in, torch::Tensor grid_out,
                       std::optional<torch::Stream> stream) {
  
  precompute_lut();

  int width = grid_in.size(1);
  int height = grid_in.size(0);
  assert(grid_in.sizes() == grid_out.sizes());

  cudaStream_t cudaStream = 0;
  if (stream.has_value()) {
    cudaStream = c10::cuda::CUDAStream(stream.value()).stream();
  }

  // Threads par bloc (32x32), couvre une tuile de 32x32 pixels
  const dim3 blockSize(TILE_DIM, TILE_DIM);
  const dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM,
                      (height + TILE_DIM - 1) / TILE_DIM);

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(
      grid_in.data_ptr<unsigned char>(), grid_out.data_ptr<unsigned char>(), width, height, d_lut);
}


// TODO : faire en sorte que un thread fasse 8 pixels (2x4) on utilise tout les bit d'un octet de la LUT 