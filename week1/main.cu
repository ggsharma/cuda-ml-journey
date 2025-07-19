// Create by Gautam Sharma
// github: ggsharma

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../utils.h"

// COMMAND : nvcc -std=c++20 main.cu add.cu multiply.cu -O0 -o add-multiply
// RESULTS:
// Time taken for GPU Addition : 299814 microseconds
// Time taken for GPU Addition : 299.814 milliseconds
// Time taken for GPU Multiplication : 25020 microseconds
// Time taken for GPU Multiplication : 25.02 milliseconds
// Time taken for CPU Addition : 320 microseconds
// Time taken for CPU Addition : 0.32 milliseconds
// Time taken for CPU Multiplication : 313 microseconds
// Time taken for CPU Multiplication : 0.313 milliseconds


constexpr int N = 100000;
// Your add function declaration
__global__ void add(int *a, int *b, int *c, int N);
__global__ void multiply(int *a, int *b, int *c, int N);



void CPUAdditionExecution(int *a, int *b, int *c, int N){
    for(int i=0; i<N; i++){
        volatile int d = c[i];
        c[i] = a[i] + b[i];
    }
}

void CPUMultiplicationExecution(int *a, int *b, int *c, int N){
    for(int i=0; i<N; i++){
        volatile int d = c[i];
        c[i] = a[i] * b[i];
    }
}

int main(){

    Timer tGPU("GPU Addition");
    
    // Your CUDA code here
    int a[N], b[N], c[N];

    
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
    
    add<<<N,1>>>(dev_a, dev_b, dev_c, N);
    cudaDeviceSynchronize(); // Important for accurate timing!
    
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    tGPU.stop();

    Timer tGPU2("GPU Multiplication");
    tGPU2.start();    
    cudaMalloc((void**)&dev_a, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_c, N*sizeof(int));
    
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    
    multiply<<<N,1>>>(dev_a, dev_b, dev_c, N);
    cudaDeviceSynchronize(); // Important for accurate timing!
    
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    tGPU2.stop();

    Timer tCPU("CPU Addition");

    tCPU.start();
    CPUAdditionExecution(a, b, c, N);
    tCPU.stop();

    Timer tCPU2("CPU Multiplication");

    tCPU2.start();
    CPUMultiplicationExecution(a, b, c, N);
    tCPU2.stop();
    
    return 0;
}