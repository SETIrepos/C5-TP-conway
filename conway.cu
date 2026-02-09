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

__global__ void compute_lut_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) return;

    // On reconstruite une mini-grille temporaire
    int temp_grid[4][4];
    for (int bit = 0; bit < 16; bit++) {
        // Mapping lineaire : 0..3 -> ligne 0, 4..7 -> ligne 1...
        int r = bit / 4;
        int c = bit % 4;
        temp_grid[r][c] = (idx >> bit) & 1;
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
    host_lut[idx] = result_mask;

}

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


// TODO : faire une lookup table 3 par 3 et faire un indice de 512 bits (9 bits pour les voisins) pour calculer directement le résultat d'un pixel. chaque thread se deplace pour faire 2 par 2 cellules mais pourquoi faire cela pourquoi chaque thread ferais 2 par 2 cellules ?