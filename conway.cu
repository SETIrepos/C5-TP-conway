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
#define HALO_DIM (BLOCK_DIM + 4) // 32 + 4 to allow 2 steps of evolution (Radius 2)

// ... existing code ...
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

// __device__ static inline uint64_t build_patch6x6(
//     uint16_t NO, uint16_t N, uint16_t NE,
//     uint16_t O,  uint16_t C, uint16_t E,
//     uint16_t SO, uint16_t S, uint16_t SE)
// {
// /*
// 15  12 13 14 15  12

// 3   0  1  2  3   0
// 7   4  5  6  7   4
// 11  8  9  10 11  8
// 15  12 13 14 15  12

// 3   0  1  2  3   0
// */
//     uint64_t patch = 0;
//     // Build first row (i = 0) - row r-1 from NO, N, NE
//     uint8_t left0  = (NO >> 15) & 1;
//     uint8_t mid40  = (N  >> 12) & 0xF;
//     uint8_t right0 = (NE >> 12) & 1;
//     uint8_t row6_0 = left0 | (mid40 << 1) | (right0 << 5);
//     patch |= (uint64_t)row6_0 << (6 * 0);

//     // Unrolled central rows (i = 1..4) corresponding to ri = 0..3
//     for (int ri = 0; ri < 4; ++ri) {
//         uint8_t left1  = (O >> (4*ri + 3)) & 1;
//         uint8_t mid41  = (C >> (4*ri)) & 0xF;
//         uint8_t right1 = (E >> (4*ri)) & 1;
//         uint8_t row6_1 = left1 | (mid41 << 1) | (right1 << 5);
//         patch |= (uint64_t)row6_1 << (6 * (ri + 1));
//     }

//     // Last row (i = 5) - row r+4 from SO, S, SE
//     uint8_t left5  = (SO >> 3) & 1;
//     uint8_t mid45  = (S  >> 0) & 0xF;
//     uint8_t right5 = (SE >> 0) & 1;
//     uint8_t row6_5  = left5 | (mid45 << 1) | (right5 << 5);
//     patch |= (uint64_t)row6_5 << (6 * 5);

//     return patch;
// }

__device__ static inline uint64_t build_patch8x8(
    uint16_t NO, uint16_t N, uint16_t NE,
    uint16_t O,  uint16_t C, uint16_t E,
    uint16_t SO, uint16_t S, uint16_t SE)
{
/*
10 11   8  9  10 11   8  9
14 15   12 13 14 15   12 13

2  3    0  1  2  3    0  1
6  7    4  5  6  7    4  5
10 11   8  9  10 11   8  9
14 15   12 13 14 15   12 13

2  3    0  1  2  3    0  1
6  7    4  5  6  7    4  5
*/

    uint64_t patch = 0;
    // Build from NO, N, NE (Rows 0 and 1 of patch come from Rows 2 and 3 of neighbors)
    for (int i = 0; i < 2; ++i) { // i=0->Row2, i=1->Row3
        int src_row = 2 + i;
        uint8_t left0  = (NO >> (4*src_row + 2)) & 3; // Cols 2,3
        uint8_t mid40  = (N  >> (4*src_row)) & 0xF;   // Cols 0,1,2,3
        uint8_t right0 = (NE >> (4*src_row)) & 3;     // Cols 0,1
        uint8_t row8 = left0 | (mid40 << 2) | (right0 << 6);
        patch |= (uint64_t)row8 << (8 * i); // Patch Row i
    }

    // Build from O, C, E (Rows 2..5 of patch come from Rows 0..3 of neighbors)
    for (int i = 0; i < 4; ++i) {
        uint8_t left1  = (O >> (4*i + 2)) & 3;
        uint8_t mid41  = (C >> (4*i)) & 0xF;
        uint8_t right1 = (E >> (4*i)) & 3;
        uint8_t row8 = left1 | (mid41 << 2) | (right1 << 6);
        patch |= (uint64_t)row8 << (8 * (i + 2)); // Patch Row i+2
    }

    // Build from SO, S, SE (Rows 6 and 7 of patch come from Rows 0 and 1 of neighbors)
    for (int i = 0; i < 2; ++i) {
        uint8_t left5  = (SO >> (4*i + 2)) & 3;
        uint8_t mid45  = (S  >> (4*i)) & 0xF;
        uint8_t right5 = (SE >> (4*i)) & 3;
        uint8_t row8 = left5 | (mid45 << 2) | (right5 << 6);
        patch |= (uint64_t)row8 << (8 * (i + 6)); // Patch Row i+6
    }

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


uint64_t patch layout (6 rows of 6 bits = 36 bits, stored in a uint64_t):
           |             |                 |                 |                 |                 |
0 1 2 3 4 5|6 7 8 9 10 11|12 13 14 15 16 17|18 19 20 21 22 23|24 25 26 27 28 29|30 31 32 33 34 35| xx...xx
           |             |                 |                 |                 |                 |
*/
__device__ static inline uint16_t extract_4x4_idx_from_patch36(uint64_t patch, int top_row, int left_col) {
    uint16_t idx = 0;
    for (int r = 0; r < 4; ++r) {
        uint8_t row6 = (patch >> (6 * (top_row + r))) & 0x3F;       // get 6-bit row
        uint8_t cols = (row6 >> left_col) & 0xF;                   // 4 contiguous bits
        idx |= (uint16_t)cols << (4 * r);                         // row-major 16-bit
    }
    return idx;
}

__device__ static inline uint16_t extract_4x4_idx_from_patch64(uint64_t patch, int top_row, int left_col) {
    uint16_t idx = 0;
    for (int r = 0; r < 4; ++r) {
        uint8_t row8 = (patch >> (8 * (top_row + r))) & 0xFF;       // get 8-bit row
        uint8_t cols = (row8 >> left_col) & 0xF;                   // 4 contiguous bits
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

__device__ static inline uint64_t assemble64_from_quadrants(uint8_t q00, uint8_t q01, uint8_t q02, uint8_t q10, uint8_t q11, uint8_t q12, uint8_t q20, uint8_t q21, uint8_t q22) {
    uint64_t out = 0;
    // q00
    out |= ((uint64_t)(q00 & 0x3)) << 0;
    out |= ((uint64_t)((q00 >> 2) & 0x3)) << 6;
    // q01
    out |= ((uint64_t)(q01 & 0x3)) << 2;
    out |= ((uint64_t)((q01 >> 2) & 0x3)) << 8;
    // q02
    out |= ((uint64_t)(q02 & 0x3)) << 4;
    out |= ((uint64_t)((q02 >> 2) & 0x3)) << 10;


    // q10
    out |= ((uint64_t)(q10 & 0x3)) << 12;
    out |= ((uint64_t)((q10 >> 2) & 0x3)) << 18;
    // q11
    out |= ((uint64_t)(q11 & 0x3)) << 14;
    out |= ((uint64_t)((q11 >> 2) & 0x3)) << 20;
    // q12
    out |= ((uint64_t)(q12 & 0x3)) << 16;
    out |= ((uint64_t)((q12 >> 2) & 0x3)) << 22;


    // q20
    out |= ((uint64_t)(q20 & 0x3)) << 24;
    out |= ((uint64_t)((q20 >> 2) & 0x3)) << 30;
    // q21
    out |= ((uint64_t)(q21 & 0x3)) << 26;
    out |= ((uint64_t)((q21 >> 2) & 0x3)) << 32;
    // q22
    out |= ((uint64_t)(q22 & 0x3)) << 28;
    out |= ((uint64_t)((q22 >> 2) & 0x3)) << 34;


    return out;
}

__global__ void game_of_life_kernel(uint16_t *grid, uint16_t *new_grid1, uint16_t *new_grid2, int width, int height, unsigned char* lut) {
    // Need two buffers for double buffering logic? Or just one big initial tile and one intermediate tile.
    // T0 Tile: Radius 2 needed. Size 36x36.
    // T1 Tile: Radius 1 needed. Size 34x34. Can fit in 36x36 buffer.
    
    // We can't overwrite T0 while neighbors read it.
    // So we need distinct storage.
    __shared__ uint16_t tile0[HALO_DIM][HALO_DIM];
    __shared__ uint16_t tile1[HALO_DIM][HALO_DIM]; // Only needs [1..34][1..34] effectively

    int tx = threadIdx.x; // 0..BLOCK_DIM-1
    int ty = threadIdx.y; // 0..BLOCK_DIM-1

    int tid = ty * BLOCK_DIM + tx;
    int num_threads = BLOCK_DIM * BLOCK_DIM;

    // Load T0 (Radius 2)
    // Origin: Block start - 2
    int read_base_x = blockIdx.x * BLOCK_DIM - 2;
    int read_base_y = blockIdx.y * BLOCK_DIM - 2;

    for (int i = tid; i < HALO_DIM * HALO_DIM; i += num_threads) {
        int ly = i / HALO_DIM;
        int lx = i % HALO_DIM;

        int gx = read_base_x + lx;
        int gy = read_base_y + ly;

        uint16_t val = 0;
        if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
            val = grid[gy * width + gx];
        }
        tile0[ly][lx] = val;
    }

    __syncthreads();

    // Step 1: Compute T1 for Radius 1 region (Indices 1..34 in tile0)
    // We need to fill tile1[1..34][1..34]
    // Map threads to cover this area. Area size: (BLOCK_DIM+2)^2 = 34*34 = 1156.
    // num_threads = 1024.
    
    int t1_dim = BLOCK_DIM + 2;
    int t1_offset = 1; // start index in tile array
    
    for (int i = tid; i < t1_dim * t1_dim; i += num_threads) {
        int ry = i / t1_dim; // 0..33 relative to T1 region
        int rx = i % t1_dim;

        int ly = ry + t1_offset; // index in tile0
        int lx = rx + t1_offset;

        // Load 3x3 neighborhood from tile0 centered at ly, lx
        uint16_t NO = tile0[ly - 1][lx - 1];
        uint16_t N  = tile0[ly - 1][lx    ];
        uint16_t NE = tile0[ly - 1][lx + 1];
        uint16_t O  = tile0[ly    ][lx - 1];
        uint16_t C  = tile0[ly    ][lx    ];
        uint16_t E  = tile0[ly    ][lx + 1];
        uint16_t SO = tile0[ly + 1][lx - 1];
        uint16_t S  = tile0[ly + 1][lx    ];
        uint16_t SE = tile0[ly + 1][lx + 1];

        // Standard 1-step computation
        if ((NO | N | NE | O | C | E | SO | S | SE) == 0) {
            tile1[ly][lx] = 0;
            continue;
        }

        // We only need the central 4x4 result here (T+1)
        // Manual 4 quadrants lookup
        // We can reuse the build_patch8x8 but it is overkill? 
        // No, build_patch8x8 is for 2-step.
        // For 1-step, we just need the 16-bit result for 'C'.
        // Let's implement a lighter build_patch4x4+2 = 6x6 patch builder for 1 step?
        // Actually, we can just grab the 4 quadrants directly from the 3x3 uint16s.
        
        // Let's use build_patch8x8 because we have it, but maybe optimize later?
        // Wait, build_patch8x8 gives 8x8 bits.
        // To get T+1 for C (4x4), we need 6x6 bits.
        // build_patch8x8 is fine.
        
        uint64_t patch = build_patch8x8(NO, N, NE, O, C, E, SO, S, SE);
        
        // Extract 4 quadrants for T+1
        // The patch is 8x8. C corresponds to offsets 2..5.
        // We need 2x2 results for inputs at:
        // TL (0,0 in C) -> needs bits [1..4]x[1..4] from Patch relative to C? No.
        // C is at offset 2. C's range is [2..5].
        // To get result at C_row=0, C_col=0, we need bits centered there.
        // The LUT takes a 4x4 input.
        // TL (0,0) quadrant needs 4x4 input centered at intersection? No.
        // LUT: 4x4 input (0..3) -> 2x2 output (1..2).
        // 
        // We want output at C coordinates 0..1 (TL quadrant).
        // This corresponds to patch coordinates 2..3.
        // This output requires input patch coordinates 1..4.
        // So start_row=1, start_col=1.
        
        uint16_t idx_tl = extract_4x4_idx_from_patch64(patch, 1, 1);
        uint16_t idx_tr = extract_4x4_idx_from_patch64(patch, 1, 3);
        uint16_t idx_bl = extract_4x4_idx_from_patch64(patch, 3, 1);
        uint16_t idx_br = extract_4x4_idx_from_patch64(patch, 3, 3);

        uint8_t r_tl = lut[idx_tl];
        uint8_t r_tr = lut[idx_tr];
        uint8_t r_bl = lut[idx_bl];
        uint8_t r_br = lut[idx_br];

        tile1[ly][lx] = assemble_from_quadrants(r_tl, r_tr, r_bl, r_br);
    }

    __syncthreads();

    // Step 2 & Final Write
    // Global Coords for T2
    int gid_x = blockIdx.x * BLOCK_DIM + tx;
    int gid_y = blockIdx.y * BLOCK_DIM + ty;

    if (gid_x >= width || gid_y >= height) return;

    // For T2, we need neighborhood from tile1
    // Center is lx = tx + 2, ly = ty + 2 (Because tile mapping starts at -2)
    // tx=0 -> lx=2. Radius 1 lookup needs lx-1 = 1. Valid (tile1 filled from 1..34).
    int lx = tx + 2;
    int ly = ty + 2;

    uint16_t res1 = tile1[ly][lx];
    new_grid1[gid_y * width + gid_x] = res1;

    // Load 3x3 from tile1 for T2
    uint16_t NO = tile1[ly - 1][lx - 1];
    uint16_t N  = tile1[ly - 1][lx    ];
    uint16_t NE = tile1[ly - 1][lx + 1];
    uint16_t O  = tile1[ly    ][lx - 1];
    uint16_t C  = tile1[ly    ][lx    ];
    uint16_t E  = tile1[ly    ][lx + 1];
    uint16_t SO = tile1[ly + 1][lx - 1];
    uint16_t S  = tile1[ly + 1][lx    ];
    uint16_t SE = tile1[ly + 1][lx + 1];

    if ((NO | N | NE | O | C | E | SO | S | SE) == 0) {
        new_grid2[gid_y * width + gid_x] = 0;
        return;
    }

    uint64_t patch = build_patch8x8(NO, N, NE, O, C, E, SO, S, SE);
    
    // Same offsets for T2
    uint16_t idx_tl = extract_4x4_idx_from_patch64(patch, 1, 1);
    uint16_t idx_tr = extract_4x4_idx_from_patch64(patch, 1, 3);
    uint16_t idx_bl = extract_4x4_idx_from_patch64(patch, 3, 1);
    uint16_t idx_br = extract_4x4_idx_from_patch64(patch, 3, 3);

    uint8_t r_tl = lut[idx_tl];
    uint8_t r_tr = lut[idx_tr];
    uint8_t r_bl = lut[idx_bl];
    uint8_t r_br = lut[idx_br];

    new_grid2[gid_y * width + gid_x] = assemble_from_quadrants(r_tl, r_tr, r_bl, r_br);
}

void game_of_life_step(torch::Tensor grid_in, torch::Tensor grid_out1, torch::Tensor grid_out2, std::optional<torch::Stream> stream) {
  
  precompute_lut();

  int width = grid_in.size(1);
  int height = grid_in.size(0);
  assert(grid_in.sizes() == grid_out1.sizes());
  assert(grid_in.sizes() == grid_out2.sizes());

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

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(grid_in.data_ptr<uint16_t>(), grid_out1.data_ptr<uint16_t>(), grid_out2.data_ptr<uint16_t>(), uint16_width, uint16_height, d_lut);
}


// TODO : faire une lookup table 3 par 3 et faire un indice de 512 bits (9 bits pour les voisins) pour calculer directement le résultat d'un pixel. chaque thread se deplace pour faire 2 par 2 cellules mais pourquoi faire cela pourquoi chaque thread ferais 2 par 2 cellules ?