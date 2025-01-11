#include <assert.h>
#include <stdio.h>
#include <vector>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


template<typename T>
__device__ T cube(T val) {
    return val * val * val;
}

__global__ void cubeKernel(float* d_out, float* d_in, int rows, int cols) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    int out_idx = y * rows + x;
    if (out_idx < rows * cols) {
        d_out[out_idx] = cube<float>(d_in[out_idx]);
    }
}

void onDevice(float* h_in, float* h_out, int mat_rows, int mat_cols, cudaStream_t* streams, int n_streams) {
    int nbytes = sizeof(float) * mat_rows * mat_cols;
    
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, nbytes);
    cudaMalloc(&d_out, nbytes);
    cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice);
    cudaCheckErrors("memory issue");


    const int n_chunks = 64;
    const dim3 threads(32, 32, 1);
    const int blockx = (mat_rows + 31) / 32;
    const int blocky = (mat_cols + 31) / 32;
    const dim3 blocks(blockx, blocky / n_chunks + 1, 1);
    const size_t bytes_per_chunk = static_cast<size_t>(mat_rows * mat_cols / n_chunks) * sizeof(float);

    // launch the kernel
    for (int i = 0; i < n_chunks; i++) {
        cudaStream_t stream = streams[i % n_streams];
        size_t start_idx = i * mat_rows * mat_cols / n_chunks;
        cudaMemcpyAsync(d_in+start_idx, 
                        h_in+start_idx, 
                        bytes_per_chunk, 
                        cudaMemcpyHostToDevice, 
                        stream);
        cudaCheckErrors("async host to dev issue");
        cubeKernel<<<blocks, threads, 0, stream>>>(d_out + start_idx, d_in + start_idx, mat_rows, mat_cols);
        cudaCheckErrors("kernel issue");
        cudaMemcpyAsync(h_out+start_idx,
                        d_out+start_idx, 
                        bytes_per_chunk, 
                        cudaMemcpyDeviceToHost, 
                        stream);
        cudaCheckErrors("async dev to host issue");
    }

    cudaDeviceSynchronize();
    cudaCheckErrors("streams execution error");
    cudaFree(d_in);
    cudaFree(d_out);
    cudaCheckErrors("deallocate error");
}

void test(float* h_in, float* h_out, int mat_size) 
    // print out the resulting array
    for (int i = 0; i < mat_size; i++) {
        assert(h_out[i] == (h_in[i] * h_in[i] * h_in[i]));
        printf("%f", h_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    printf("succes!\n");
}

void onHost() {
    const int cols = 64;
    const int rows = 64;
    const int n_streams = 16;

    float* h_in;
    float* h_out;
    cudaHostAlloc(&h_in, cols * rows * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_out, cols * rows * sizeof(float), cudaHostAllocDefault);
    cudaCheckErrors("allocate error");

    for (int i = 0; i < cols * rows; i++) {
        h_in[i] = float(i);
    }

    // Set up streams.
    cudaStream_t streams[n_streams];
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    cudaCheckErrors("stream creation error");

    onDevice(h_in, h_out, cols, rows, streams, n_streams);
    test(h_in, h_out, cols*rows);

    cudaFree(h_in);
    cudaFree(h_out);
}

int main(int argc, char** argv) {
    onHost();

    return 0;
}