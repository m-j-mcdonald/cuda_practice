#include <assert.h>
#include <stdio.h>
#include <vector>
#include "cuda_utils.cuh"

constexpr int threads_per_block_x = 32;

// Decompose matrices into sub-blocks, copy into shared memory, multiply sub-blocks, sum results.
__global__ void blockMult(float* A, float* B, float* product, int a_rows, int a_cols, int b_cols) {
    __shared__ float partial_A[threads_per_block_x][threads_per_block_x];
    __shared__ float partial_B[threads_per_block_x][threads_per_block_x];
    __shared__ float partial_out[threads_per_block_x][threads_per_block_x];
    
    // Assume blockDim.x cleanly divides the # of cols in A (# rows in B).
    int submats_per_block = a_cols / blockDim.x;
    partial_out[threadIdx.x][threadIdx.y] = 0.0f;
    __syncthreads();

    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < submats_per_block; i++) {
        int a_x = i * blockDim.x + threadIdx.x;
        partial_A[threadIdx.x][threadIdx.y] = A[y_idx * a_cols + a_x];

        int b_y = i * blockDim.y + threadIdx.y;
        partial_B[threadIdx.x][threadIdx.y] = B[b_y * b_cols + x_idx];
        __syncthreads();

        for (int j = 0; j < blockDim.x; j++) {
            partial_out[threadIdx.x][threadIdx.y] += partial_A[j][threadIdx.y] * partial_B[threadIdx.x][j];
        }
        __syncthreads();
    }

    product[y_idx * b_cols + x_idx] = partial_out[threadIdx.x][threadIdx.y];
}

void onDevice(float* h_a, float* h_b, float* h_out, int a_rows, int a_cols, int b_cols) {
    // For simplicity, require the matrices to break up evenly into cuda blocks.
    assert(!(a_rows % threads_per_block_x));
    assert(!(b_cols % threads_per_block_x));

    float* d_a;
    float* d_b;
    float* d_out;
    int n_bytes_a = a_rows * a_cols * sizeof(float);
    int n_bytes_b = a_cols * b_cols * sizeof(float);
    int n_bytes_out = a_rows * b_cols * sizeof(float);
    cudaMalloc(&d_a, n_bytes_a);
    cudaMalloc(&d_b, n_bytes_b);
    cudaMalloc(&d_out, n_bytes_out);
    cudaCheckErrors("memory issue");

    cudaMemcpy(d_a, h_a, n_bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_bytes_b, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy host to dev error");

    const dim3 threads_per_block(threads_per_block_x, threads_per_block_x, 1);
    const dim3 blocks_in_grid(b_cols / 32, a_rows / 32, 1);

    blockMult<<<blocks_in_grid, threads_per_block>>>(d_a, d_b, d_out, a_rows, a_cols, b_cols);
    cudaCheckErrors("kernel execution error");
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, n_bytes_out, cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy dev to host error");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaCheckErrors("device de-allocate error");
}


int test(float* h_a, float* h_b, float* h_out, int a_rows, int a_cols, int b_cols) {
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            float target_out = 0.0f;
            for (int x = 0; x < a_cols; x++) {
                target_out += h_a[i * a_cols + x] * h_b[x * b_cols + j];
            }
            if(h_out[i * b_cols + j] != target_out) {
                printf("Error in result: (%d, %d) should be %f but is %f\n", i, j, target_out, h_out[i * b_cols + j]);
                return -1;
            }
        }
    }

    printf("success!\n");
    return 0;
}


int onHost() {
    // a_cols == b_rows, since we assume the matrices are compatible.
    const int a_rows = 256;
    const int a_cols = 128;
    const int b_cols = 64;

    float* h_a;
    float* h_b;
    float* h_out;
    cudaHostAlloc(&h_a, a_rows * a_cols * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_b, a_cols * b_cols * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_out, a_rows * b_cols * sizeof(float), cudaHostAllocDefault);
    cudaCheckErrors("host allocate error");

    for (int i = 0; i < a_rows * a_cols; i++) {
        h_a[i] = (i % 16);
    }
    for (int i = 0; i < a_cols * b_cols; i++) {
        h_b[i] = (i % 16);
    }

    onDevice(h_a, h_b, h_out, a_rows, a_cols, b_cols);
    int res_code = test(h_a, h_b, h_out, a_rows, a_cols, b_cols);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_out);
    cudaCheckErrors("host de-allocate error");
    return res_code;
}

int main(int argc, char** argv) {
    int res_code = onHost();

    return res_code;
}
