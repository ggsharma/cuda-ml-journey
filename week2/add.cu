
__global__ void add(int*a, int *b , int*c, size_t N){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}
