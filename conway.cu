#include "conway.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <omp.h>


unsigned char* d_lut = nullptr; 

void precompute_lut() {
    if (d_lut != nullptr) return; 

    unsigned char host_lut[65536];
    
    // Pour chaque configuration possible de 4x4 (16 bits)
    #pragma omp parallel for
    for (int i = 0; i < 65536; i++) {
        // On reconstruite une mini-grille temporaire
        int temp_grid[4][4];
        for (int bit = 0; bit < 16; bit++) {
            // Mapping : 0..3 -> ligne 0, 4..7 -> ligne 1...
            /*
            0  1  2  3
            4  5  6  7
            8  9  10 11
            12 13 14 15
            */
            int r = bit / 4;
            int c = bit % 4;
            temp_grid[r][c] = (i >> bit) & 1;
        }

        // On calcule le résultat pour le bloc 2x2 central
        // Correspondants aux indices (1,1), (1,2), (2,1), (2,2)
        unsigned char result_mask = 0;
        
        // Coordonnées relatives des 4 pixels cibles
        int targets[4][2] = {{1,1}, {1,2}, {2,1}, {2,2}};
        
        for (int k = 0; k < 4; k++) {
            int r = targets[k][0];
            int c = targets[k][1];
            
            // Compter voisins
            int neighbors = 0;
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    if (dr == 0 && dc == 0) continue;
                    neighbors += temp_grid[r + dr][c + dc];
                }
            }
            
            int self = temp_grid[r][c];
            int alive = (neighbors == 3) || (self == 1 && neighbors == 2);
            
            if (alive) {
                result_mask |= (1 << k);
            }
        }
        host_lut[i] = result_mask;
    }

    // Allocation GPU
    cudaMalloc(&d_lut, 65536 * sizeof(unsigned char));
    cudaMemcpy(d_lut, host_lut, 65536 * sizeof(unsigned char), cudaMemcpyHostToDevice);
}

#define BLOCK_DIM 32
#define HALO_DIM (BLOCK_DIM + 2) // 32 + 2 pour le halo de 1 uint16_t

// __global__ void compute_lut_kernel() {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= 65536) return;

//     // On reconstruite une mini-grille temporaire
//     int temp_grid[4][4];
//     for (int bit = 0; bit < 16; bit++) {
//         // Mapping lineaire : 0..3 -> ligne 0, 4..7 -> ligne 1...
//         int r = bit / 4;
//         int c = bit % 4;
//         temp_grid[r][c] = (idx >> bit) & 1;
//     }

//     // On calcule le résultat pour le bloc 2x2 central
//     // Correspondants aux indices (1,1), (1,2), (2,1), (2,2)
//     unsigned char result_mask = 0;
    
//     // Coordonnées relatives des 4 pixels cibles
//     int targets[4][2] = {{1,1}, {1,2}, {2,1}, {2,2}};
    
//     for (int k = 0; k < 4; k++) {
//         int r = targets[k][0];
//         int c = targets[k][1];
        
//         // Compter voisins
//         int neighbors = 0;
//         for (int dr = -1; dr <= 1; dr++) {
//             for (int dc = -1; dc <= 1; dc++) {
//                 if (dr == 0 && dc == 0) continue;
//                 neighbors += temp_grid[r + dr][c + dc];
//             }
//         }
        
//         int self = temp_grid[r][c];
//         int alive = (neighbors == 3) || (self == 1 && neighbors == 2);
        
//         if (alive) {
//             result_mask |= (1 << k);
//         }
//     }
//     host_lut[idx] = result_mask;

// }

__device__ inline bool get_mask_bit(const uint32_t* mask, int x, int y, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) return false;
    int idx = y * width + x;
    return (mask[idx / 32] >> (idx % 32)) & 1;
}

__device__ inline void set_mask_bit(uint32_t* mask, int x, int y, int width) {
    int idx = y * width + x;
    atomicOr(&mask[idx / 32], 1 << (idx % 32));
}

__device__ static inline uint64_t build_patch6x6(
    uint16_t NO, uint16_t N, uint16_t NE,
    uint16_t O,  uint16_t C, uint16_t E,
    uint16_t SO, uint16_t S, uint16_t SE)
{
/*
15  12 13 14 15  12

3   0  1  2  3   0
7   4  5  6  7   4
11  8  9  10 11  8
15  12 13 14 15  12

3   0  1  2  3   0
*/
    uint64_t patch = 0;
    // Build first row (i = 0) - row r-1 from NO, N, NE
    uint8_t left0  = (NO >> 15) & 1;
    uint8_t mid40  = (N  >> 12) & 0xF;
    uint8_t right0 = (NE >> 12) & 1;
    uint8_t row6_0 = left0 | (mid40 << 1) | (right0 << 5);
    patch |= (uint64_t)row6_0 << (6 * 0);

    // Unrolled central rows (i = 1..4) corresponding to ri = 0..3
    for (int ri = 0; ri < 4; ++ri) {
        uint8_t left1  = (O >> (4*ri + 3)) & 1;
        uint8_t mid41  = (C >> (4*ri)) & 0xF;
        uint8_t right1 = (E >> (4*ri)) & 1;
        uint8_t row6_1 = left1 | (mid41 << 1) | (right1 << 5);
        patch |= (uint64_t)row6_1 << (6 * (ri + 1));
    }

    // Last row (i = 5) - row r+4 from SO, S, SE
    uint8_t left5  = (SO >> 3) & 1;
    uint8_t mid45  = (S  >> 0) & 0xF;
    uint8_t right5 = (SE >> 0) & 1;
    uint8_t row6_5  = left5 | (mid45 << 1) | (right5 << 5);
    patch |= (uint64_t)row6_5 << (6 * 5);

    return patch;
}

// Extract a 4x4 index (16-bit) from the 6x6 patch.
/*
uint64_t 
0  1  2  3  4  5
6  7  8  9  10 11
12 13 14 15 16 17
18 19 20 21 22 23
24 25 26 27 28 29
30 31 32 33 34 35
xx ........... 63


uint64_t patch layout (6 rows of 6 bits = 36 bits, stored in a uint64_t):
           |             |                 |                 |                 |                 |
0 1 2 3 4 5|6 7 8 9 10 11|12 13 14 15 16 17|18 19 20 21 22 23|24 25 26 27 28 29|30 31 32 33 34 35| xx...xx
           |             |                 |                 |                 |                 |
*/
__device__ static inline uint16_t extract_4x4_idx_from_patch(uint64_t patch, int top_row, int left_col) {
    // Le but de cette méthode est d'extraire l'incide de 4x4 bits (16 bits) correspondant à une sous-région de 4x4 dans la patch 6x6. Cet indice sera utilisé pour faire une lookup dans la LUT pré-calculée.
    uint16_t idx = 0;
    for (int r = 0; r < 4; ++r) {
        uint8_t row6 = (patch >> (6 * (top_row + r))) & 0x3F;       // get 6-bit row
        uint8_t cols = (row6 >> left_col) & 0xF;                   // 4 contiguous bits
        idx |= (uint16_t)cols << (4 * r);                         // row-major 16-bit
    }
    return idx;
}

// Assemble 4 results (each 2x2 encoded as 4 bits: low 2 bits row0, high 2 bits row1) into a 4x4 uint16
__device__ static inline uint16_t assemble_from_quadrants(uint8_t tl, uint8_t tr, uint8_t bl, uint8_t br) {
    uint16_t out = 0;
    // TL -> rows 0..1, cols 0..1
    out |= ((uint16_t)(tl & 0x3)) << 0;       // row0 cols0-1 -> bits 0-1
    out |= ((uint16_t)((tl >> 2) & 0x3)) << 4; // row1 cols0-1 -> bits 4-5

    // TR -> rows 0..1, cols 2..3
    out |= ((uint16_t)(tr & 0x3)) << 2;       // row0 cols2-3 -> bits 2-3
    out |= ((uint16_t)((tr >> 2) & 0x3)) << 6; // row1 cols2-3 -> bits 6-7

    // BL -> rows 2..3, cols 0..1
    out |= ((uint16_t)(bl & 0x3)) << 8;       // row2 cols0-1 -> bits 8-9
    out |= ((uint16_t)((bl >> 2) & 0x3)) << 12;// row3 cols0-1 -> bits 12-13

    // BR -> rows 2..3, cols 2..3
    out |= ((uint16_t)(br & 0x3)) << 10;      // row2 cols2-3 -> bits 10-11
    out |= ((uint16_t)((br >> 2) & 0x3)) << 14;// row3 cols2-3 -> bits 14-15

    return out;
}

__global__ void game_of_life_kernel(uint16_t *grid, uint16_t *new_grid, int width, int height, unsigned char* lut, uint32_t* mask_in, uint32_t* mask_out) {
    __shared__ uint16_t tile[HALO_DIM][HALO_DIM];

    int tx = threadIdx.x; // 0..BLOCK_DIM-1
    int ty = threadIdx.y; // 0..BLOCK_DIM-1

    int tid = ty * BLOCK_DIM + tx;
    int num_threads = BLOCK_DIM * BLOCK_DIM;

    // Origine de lecture : Coin haut-gauche du bloc de uint16 moins 1 (halo)
    int read_base_x = blockIdx.x * BLOCK_DIM - 1;
    int read_base_y = blockIdx.y * BLOCK_DIM - 1;

    for (int i = tid; i < HALO_DIM * HALO_DIM; i += num_threads) {
        int ly = i / HALO_DIM;
        int lx = i % HALO_DIM;

        int gx = read_base_x + lx;
        int gy = read_base_y + ly;

        uint16_t val = 0;
        if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
            val = grid[gy * width + gx];
        }
        tile[ly][lx] = val;
    }

    __syncthreads();

    // Local position inside the tile (centered at +1,+1)
    int lx = tx + 1;
    int ly = ty + 1;

    // Global output coordinates (in units of uint16 blocks)
    int gid_x = blockIdx.x * BLOCK_DIM + tx;
    int gid_y = blockIdx.y * BLOCK_DIM + ty;

    // If outside the grid, nothing to do
    if (gid_x >= width || gid_y >= height) return;

    // Check if we need to update
    bool active = false;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
             if (get_mask_bit(mask_in, gid_x + dx, gid_y + dy, width, height)) {
                 active = true;
                 break;
             }
        }
        if (active) break;
    }

    if (!active) {
        // Shortcut -> la cellule et ses voisines n'ont pas changé
        new_grid[gid_y * width + gid_x] = tile[ly][lx];
        return;
    }

    // Load the 3x3 neighborhood from shared tile
    uint16_t NO = tile[ly - 1][lx - 1];
    uint16_t N  = tile[ly - 1][lx    ];
    uint16_t NE = tile[ly - 1][lx + 1];
    uint16_t O  = tile[ly    ][lx - 1];
    uint16_t C  = tile[ly    ][lx    ];
    uint16_t E  = tile[ly    ][lx + 1];
    uint16_t SO = tile[ly + 1][lx - 1];
    uint16_t S  = tile[ly + 1][lx    ];
    uint16_t SE = tile[ly + 1][lx + 1];

    uint16_t v = NO | N | NE | O | C | E | SO | S | SE;
    if (v == 0) {
        new_grid[gid_y * width + gid_x] = 0;
        return;
    }

    uint64_t patch = build_patch6x6(NO, N, NE, O, C, E, SO, S, SE);

    // get 4 indices for quadrants
    uint16_t idx_tl = extract_4x4_idx_from_patch(patch, 0, 0); // rows 0..3, cols 0..3
    uint16_t idx_tr = extract_4x4_idx_from_patch(patch, 0, 2); // rows 0..3, cols 2..5
    uint16_t idx_bl = extract_4x4_idx_from_patch(patch, 2, 0); // rows 2..5, cols 0..3
    uint16_t idx_br = extract_4x4_idx_from_patch(patch, 2, 2); // rows 2..5, cols 2..5

    uint8_t res_tl = lut[idx_tl];
    uint8_t res_tr = lut[idx_tr];
    uint8_t res_bl = lut[idx_bl];
    uint8_t res_br = lut[idx_br];

    uint16_t result = assemble_from_quadrants(res_tl, res_tr, res_bl, res_br);

    // Write result
    new_grid[gid_y * width + gid_x] = result;
    
    // Update mask if changed
    if (result != C) {
        set_mask_bit(mask_out, gid_x, gid_y, width);
    }
}

void game_of_life_step(torch::Tensor grid_in, torch::Tensor grid_out,
                       torch::Tensor mask_in, torch::Tensor mask_out,
                       std::optional<torch::Stream> stream) {
  
  precompute_lut();

  int width = grid_in.size(1);
  int height = grid_in.size(0);
  assert(grid_in.sizes() == grid_out.sizes());
  
  // Ensure mask sizes ?  
  mask_out.zero_();

  cudaStream_t cudaStream = 0;
  if (stream.has_value()) {
    cudaStream = c10::cuda::CUDAStream(stream.value()).stream();
  }

  // Launch configuration: BLOCK_DIM threads per block, grid covers uint16 blocks
  int uint16_width = width;   // number of uint16 per row
  int uint16_height = height; // number of uint16 per column

  const dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
  const dim3 gridSize((uint16_width + BLOCK_DIM - 1) / BLOCK_DIM,
                      (uint16_height + BLOCK_DIM - 1) / BLOCK_DIM);

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(
      grid_in.data_ptr<uint16_t>(), grid_out.data_ptr<uint16_t>(), uint16_width, uint16_height, d_lut,
      mask_in.data_ptr<uint32_t>(), mask_out.data_ptr<uint32_t>());
}


// TODO : faire une lookup table 3 par 3 et faire un indice de 512 bits (9 bits pour les voisins) pour calculer directement le résultat d'un pixel. chaque thread se deplace pour faire 2 par 2 cellules mais pourquoi faire cela pourquoi chaque thread ferais 2 par 2 cellules ?