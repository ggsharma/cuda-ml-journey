// Create by Gautam Sharma
// github: ggsharma

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../utils.h"

// COMMAND : nvcc -std=c++20 main.cu add.cu -O0 -o better-add
// RESULTS:
// Time taken for GPU Addition : 334532 microseconds
// Time taken for GPU Addition : 334.532 milliseconds
// Time taken for CPU Addition : 17871566 microseconds
// Time taken for CPU Addition : 17871.6 milliseconds


constexpr size_t N = 10000000000; // Longer vectors
// Your add function declaration
__global__ void add(int *a, int *b, int *c, size_t N);

void CPUAdditionExecution(int *a, int *b, int *c, size_t N){
    for(int i=0; i<N; i++){
        volatile int d = c[i];
        c[i] = a[i] + b[i];
    }
}

int main(){

    Timer tGPU("GPU Addition");
    
    // Your CUDA code here
    int* a = new int[N];
    int* b = new int[N]; 
    int* c = new int[N];
    
    for(int i=0; i<N; i++){
        a[i] = -i;
        b[i] = i*i;
    }

    tGPU.start();
    int* dev_a, *dev_b, *dev_c;
    
    cudaMalloc((void**)&dev_a, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_c, N*sizeof(int));
    
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    
    add<<<128,128>>>(dev_a, dev_b, dev_c, N); // Linear grid of 128 blocks with 128 threads
    cudaDeviceSynchronize(); // Important for accurate timing!
    
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    tGPU.stop();

    // CPU addition
    Timer tCPU("CPU Addition");

    tCPU.start();
    CPUAdditionExecution(a, b, c, N);
    tCPU.stop();

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}