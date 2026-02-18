
# All Kernel

## noif-char

FPS=48.69 grid-size=32768

```c++
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
      new_grid[idx] = alive_neighbors == 3 || (grid[idx] == 1 && alive_neighbors == 2);
      // new_grid[idx] = 0;
      // if (grid[idx] == 0 && alive_neighbors == 3 || (grid[idx] == 1 && (alive_neighbors == 4 || alive_neighbors == 3))) {
      //   new_grid[idx] = 1;
      // }
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
```


## lookup
FPS=112.76
```c++
unsigned char* d_lut = nullptr; 

void precompute_lut() {
    if (d_lut != nullptr) return; 

    unsigned char host_lut[65536];
    
    // Pour chaque configuration possible de 4x4 (16 bits)
    for (int i = 0; i < 65536; i++) {
        // On reconstruite une mini-grille temporaire
        int temp_grid[4][4];
        for (int bit = 0; bit < 16; bit++) {
            // Mapping lineaire : 0..3 -> ligne 0, 4..7 -> ligne 1...
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

#define TILE_DIM 64
#define HALO_DIM (TILE_DIM + 2)
// Pour la version LUT, un thread traite 2x2 pixels, donc le block de thread est plus petit
#define LUT_THREAD_DIM (TILE_DIM / 2) 

__global__ void game_of_life_kernel(unsigned char *grid, unsigned char *new_grid, int width, int height, unsigned char* lut) {
    __shared__ unsigned char tile[HALO_DIM][HALO_DIM];

    // Stratégie Grid Stride Loop avec Shared Memory et LUT
    // Chaque bloc produit une tuile de output de 32x32 pixels
    // Mais on n'a que 16x16 threads par bloc.
    
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15
    
    // Coordonnées de sortie (coin haut gauche du bloc 2x2 traité par ce thread)
    int out_glob_x = blockIdx.x * TILE_DIM + 2 * tx;
    int out_glob_y = blockIdx.y * TILE_DIM + 2 * ty;

    // --- Chargement Collaboratif ---
    // 256 threads chargent 34x34 = 1156 elements. -> ~4.5 elements/thread
    int tid = ty * LUT_THREAD_DIM + tx; 
    int num_threads = LUT_THREAD_DIM * LUT_THREAD_DIM; // 256
    
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
    // Ce thread s'occupe des pixels locaux dans la tile :
    // (2*ty + 1, 2*tx + 1) -> haut gauche du 2x2
    // On doit extraire le voisinage 4x4 centré sur ce bloc.
    // Le bloc 4x4 commence à : row = (2*ty + 1) - 1 = 2*ty
    //                          col = (2*tx + 1) - 1 = 2*tx
    
    int tile_r = 2 * ty;
    int tile_c = 2 * tx;
    
    uint16_t state_idx = 0;
    
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            if (tile[tile_r + r][tile_c + c]) {
                // bit mapping doit correspondre à celui de precompute_lut
                // bit = r * 4 + c
                state_idx |= (1 << (r * 4 + c));
            }
        }
    }
    
    unsigned char res = lut[state_idx];

    // Ecriture des 4 pixels
    // Mapping: bit 0 -> (0,0), bit 1 -> (0,1), bit 2 -> (1,0), bit 3 -> (1,1) (relatifs au 2x2)
    // ATTENTION : dans precompute, j'ai utilisé: Targets {{1,1}, {1,2}, {2,1}, {2,2}}
    // qui correspondent (r,c) dans le bloc 4x4.
    // 1,1 -> top-left du 2x2. (bit 0)
    // 1,2 -> top-right du 2x2. (bit 1)
    // 2,1 -> bot-left du 2x2. (bit 2)
    // 2,2 -> bot-right du 2x2. (bit 3)
    
    if (out_glob_x < width && out_glob_y < height) 
        new_grid[out_glob_y * width + out_glob_x] = (res >> 0) & 1;
        
    if (out_glob_x + 1 < width && out_glob_y < height)
        new_grid[out_glob_y * width + out_glob_x + 1] = (res >> 1) & 1;
        
    if (out_glob_x < width && out_glob_y + 1 < height) 
        new_grid[(out_glob_y + 1) * width + out_glob_x] = (res >> 2) & 1;

    if (out_glob_x + 1 < width && out_glob_y + 1 < height) 
        new_grid[(out_glob_y + 1) * width + out_glob_x + 1] = (res >> 3) & 1;
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

  // Threads par bloc définis par LUT_THREAD_DIM, couvre TILE_DIM pixels
  const dim3 blockSize(LUT_THREAD_DIM, LUT_THREAD_DIM);
  const dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM,
                      (height + TILE_DIM - 1) / TILE_DIM);

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(
      grid_in.data_ptr<unsigned char>(), grid_out.data_ptr<unsigned char>(), width, height, d_lut);
}
```

## uint16
FPS=416.51

```c++
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


uint64_t patch layout (6 rows of 6 bits = 36 bits, stored in a uint64_t):
           |             |                 |                 |                 |                 |
0 1 2 3 4 5|6 7 8 9 10 11|12 13 14 15 16 17|18 19 20 21 22 23|24 25 26 27 28 29|30 31 32 33 34 35| xx...xx
           |             |                 |                 |                 |                 |
*/
__device__ static inline uint16_t extract_4x4_idx_from_patch(uint64_t patch, int top_row, int left_col) {
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

__global__ void game_of_life_kernel(uint16_t *grid, uint16_t *new_grid, int width, int height, unsigned char* lut) {
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
```
