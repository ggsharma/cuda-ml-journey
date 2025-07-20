// Create by Gautam Sharma
// github: ggsharma


// COMMAND : nvcc -std=c++20 main.cu -O0 -o dot-product


#include <iostream>
#include <chrono>
#include <cuda_runtime.h>



#define imin(a,b) (a<b?a:b)

const int N = 33*1024;

const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+ threadsPerBlock - 1)/ threadsPerBlock);

__global__ void dot(float *a, float *b, float*c) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;

    while(tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be power of 2
    int i = blockDim.x / 2;

    while( i != 0){
        if(cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
            // __syncthreads(); => If you put suncthreads here watch what happens. You will experience a stall.
            // syncthreads should never be on divergent branch as it will make the program stall.
        }
        __syncthreads(); 
        i /= 2; 
    }

    if(cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}


int main(){
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_C;

    // Allocate memory on CPU
    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid*sizeof(float));


    cudaMalloc((void**)&dev_a, N*sizeof(float));

    cudaMalloc((void**)&dev_b, N*sizeof(float));


    cudaMalloc((void**)&dev_partial_C, blocksPerGrid*sizeof(float));


    for(int i=0; i<N; i++){
        a[i] = i;
        b[i] = i*2;
    }


    cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_C);


    cudaMemcpy(partial_c, dev_partial_C, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

    // Finish up on the CPU side
    c = 0;
    for(int i=0; i<blocksPerGrid; i++){
        c += partial_c[i];
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_C);
}


