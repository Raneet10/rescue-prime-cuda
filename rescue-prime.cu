#include <helper_cuda.h>
#include <iostream>
#include <cooperative_groups.h>
#include "rescue-prime.cuh"

#define TILE_WIDTH 32

using namespace std;
using namespace cooperative_groups;


template<size_t Fp>
__global__ Fp rescue_prime::rescuePrime(int64_t *input_sequence, int64_t N, int64_t m, int64_t *mds, int64_t *round_constants, int64_t alpha, int64_t alphainv, int64_t rate) {
    Fp *state;
    checkCudaErrors(cudaMalloc(&state, m * m * sizeof(Fp)));
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = blockDim.x * gridDim.x;

    // absorbing phase
    int64_t absorbing_index = 0;
    while (absorbing_index < sizeof(input_sequence)) {
        for ( int i = idx; i < rate; i += stridex) {
            state[i] += input_sequence[absorbing_index];
            absorbing_index++;
        }
        state = rescuePrimePermutation<Fp>(N, m, mds, round_constants, alpha, alphainv, state);
    }

    Fp *output_sequence;
    // squeezing phase
    for (int i = idx; i < rate; i += stridex) {
        output_sequence[i] = state[i];
    }
    
    printf("Output sequence: %d", output_sequence);
    return output_sequence;
}

template<size_t Fp>
__device__ Fp* rescue_prime::matrix_mult(Fp *a, Fp *b, int64_t N) {
    Fp *c;
    checkCudaErrors(cudaMalloc(&c, N * N * sizeof(Fp)));

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // sub-matrix of a
    int aStart = TILE_WIDTH * by * N;
    int aStep = TILE_WIDTH;
    int aEnd = aStart + N - 1; 

    // sub-matrix of b
    int bStart = TILE_WIDTH * bx;
    int bStep = TILE_WIDTH * N;


    for (int a = aStart, b = bStart; a <= aEnd; a += aStep, b += bStep) {
        Fp cSum = 0;
        __shared__ Fp aTile[TILE_WIDTH * TILE_WIDTH],
                     bTile[TILE_WIDTH * TILE_WIDTH];

        aTile[ty * TILE_WIDTH + tx] = a[a + N * ty + tx];
        bTile[ty * TILE_WIDTH + tx] = b[b + N * ty + tx];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            cSum += aTile[ty * TILE_WIDTH + k] * bTile[k * TILE_WIDTH + tx];
        }
        __syncthreads();

    }

    int c = N * TILE_WIDTH * by + TILE_WIDTH * bx ;
    c[c + N * ty + tx] = cSum;
    
    return c;
}

template<size_t Fp>
__device__ Fp rescue_prime::exp_mod(Fp b, Fp e) {
    Fp result = 1;
    while (e > 0) {
        if (e & 1) {
            #if Fp == sizeof(__uint128_t)
                result = mul128(result, b) % rescue_prime::prime_field_mod;
            #else
                result = result * b % rescue_prime::prime_field_mod;
            #endif        
        }
        e = e >> 1;
        #if Fp == sizeof(__uint128_t)
            b = mul128(b,b) % rescue_prime::prime_field_mod;
        #else
            b = b * b % rescue_prime::prime_field_mod;    
        #endif
    }

    return result;
}

__device__ __uint128_t rescue_prime::mul128(__uint128_t a, __uint128_t b) {
    __uint128_t prod = 0;
    uint64_t a_low = a & 0xFFFFFFFFFFFFFFFF;
    uint64_t a_high = a >> 64;
    uint64_t b_low = b & 0xFFFFFFFFFFFFFFFF;
    uint64_t b_high = b >> 64;

    __uint128_t z0 = a_low * b_low;
    __uint128_t z1 = (a_low * b_high) << 64;
    __uint128_t z2 =  (a_high * b_low) << 64;
    __uint128_t z3 = (a_high * b_high) << 128;

    prod = z0 + z1 + z2 + z3;
    return prod;
}

template<size_t Fp>
__device__ Fp* rescue_prime:: rescuePrimePermutation(int64_t N, int64_t m, int64_t *mds, int64_t *round_constants, int64_t alpha, int64_t alphainv, Fp *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int stridey = blockDim.y * gridDim.y;

    for (int i = idx; i < N; i += stridex) {
        // S-box layer
        for (int j = idy; j < m; j += stridey) {
            state[j] =  exp_mod<Fp>(state[j] , alpha);
        }

        __syncthreads();

        // MDS layer
        state = matrix_mult<Fp>(mds, state, m);

        // Round constants
        for (int j = idy; j < m; j += stridey) {
            state[j] += round_constants[i*2*m + j];
        }

        __syncthreads();

        // inverse S-box layer
        for (int j = idy; j < m; j += stridey) {
            state[j] = exp_mod<Fp>(state[j] , alphainv);
        }

        __syncthreads();

        // MDS layer
        state = matrix_mult<Fp>(mds, state, m);

        // Round constants
        for (size_t j = idy; j < m; j += stridey) {
            state[j] += round_constants[i *2* m + m + j];
        }

        __syncthreads();
    }

    return state;
}
