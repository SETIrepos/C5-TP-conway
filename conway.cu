#include "conway.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <omp.h>


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

// Extract a 3x3 index (9 bits) from the 6x6 patch to compute the result of the cell at (row, col)
// row and col are coordinates within the 4x4 target block (0..3).
// In the 6x6 patch, the 4x4 block starts at (1,1).
// So cell (row, col) corresponds to 6x6 patch index (row+1, col+1).
// Its 3x3 neighborhood is from (row, col) to (row+2, col+2) in the patch.
__device__ static inline uint16_t extract_3x3_idx_from_patch(uint64_t patch, int row, int col) {
    uint16_t idx = 0;
    // We need 3 rows from the patch starting at 'row'
    for (int r = 0; r < 3; ++r) {
        uint8_t row6 = (patch >> (6 * (row + r))) & 0x3F;
        uint8_t cols = (row6 >> col) & 0x7; // 3 bits
        idx |= (uint16_t)cols << (3 * r);
    }
    return idx;
}

__device__ static inline uint8_t lut3x3_get(const unsigned char *shared_lut, uint16_t idx) {
    int byte_idx = idx >> 3;
    int bit_pos = idx & 0x7;
    return (shared_lut[byte_idx] >> bit_pos) & 1;
}

__global__ void game_of_life_kernel(uint16_t *grid, uint16_t *new_grid, int width, int height, unsigned char* lut) {
    __shared__ uint16_t tile[HALO_DIM][HALO_DIM];
    __shared__ unsigned char shared_lut[64];

    int tx = threadIdx.x; // 0..BLOCK_DIM-1
    int ty = threadIdx.y; // 0..BLOCK_DIM-1

    int tid = ty * BLOCK_DIM + tx;
    int num_threads = BLOCK_DIM * BLOCK_DIM;

    // Load LUT into shared memory
    if (tid < 64) {
        shared_lut[tid] = lut[tid];
    }

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

    uint64_t patch = build_patch6x6(NO, N, NE, O, C, E, SO, S, SE);

    // Compute the result for the 4x4 block (16 pixels)
    uint16_t result = 0;
    
    // Unroll manually or by compiler? 4x4 loop is small enough.
    // row 0
    #pragma unroll
    for(int r=0; r<4; ++r) {
        #pragma unroll
        for(int c=0; c<4; ++c) {
             uint16_t idx = extract_3x3_idx_from_patch(patch, r, c);
             uint8_t alive = lut3x3_get(shared_lut, idx);
             if(alive) {
                 result |= (1 << (r * 4 + c)); 
             }
        }
    }

    // Write result
    new_grid[gid_y * width + gid_x] = result;
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

  // Launch configuration: BLOCK_DIM threads per block, grid covers uint16 blocks
  int uint16_width = width;   // number of uint16 per row
  int uint16_height = height; // number of uint16 per column

  const dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
  const dim3 gridSize((uint16_width + BLOCK_DIM - 1) / BLOCK_DIM,
                      (uint16_height + BLOCK_DIM - 1) / BLOCK_DIM);

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(
      grid_in.data_ptr<uint16_t>(), grid_out.data_ptr<uint16_t>(), uint16_width, uint16_height, d_lut);
}


// TODO : faire une lookup table 3 par 3 et faire un indice de 512 bits (9 bits pour les voisins) pour calculer directement le résultat d'un pixel. chaque thread se deplace pour faire 2 par 2 cellules mais pourquoi faire cela pourquoi chaque thread ferais 2 par 2 cellules ?