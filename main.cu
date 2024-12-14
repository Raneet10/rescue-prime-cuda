#include <iostream>
#include <helper_cuda.h>
#include "rescue-prime.cuh"

int main() {
    int64_t *cmds = nullptr, *cround_constants = nullptr
    checkCudaErrors(cudaMalloc(&cmds, sizeof(rescue_prime::mds)));
    checkCudaErrors(cudaMalloc(&cround_constants, sizeof(rescue_prime::round_constants)));
    checkCudaErrors(cudaMemcpy(cmds, mds, sizeof(rescue_prime::mds), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(cround_constants, rescue_prime::round_constants, sizeof(rescue_prime::round_constants), cudaMemcpyHostToDevice));
    dim3 gridDim(32*m*sizeof(__uint128_t), 32*m*sizeof(__uint128_t));
    dim3 blockDim(32, 32);
    rescue_prime::rescuePrime<__uint128_t><<<gridDim, blockDim>>>(input_sequence, N, m, cmds, cround_constants, alpha, alphainv, rate);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(cmds));
    checkCudaErrors(cudaFree(cround_constants));
    return 0;
}