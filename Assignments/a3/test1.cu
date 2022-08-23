#include <cuda.h>
#include <stdio.h>

__global__ void kernal(unsigned int* i){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < 64)
        atomicInc(i, 1000);  // atomic increment by 1, ignore second parameter.
    __syncthreads();
    printf("%d\n", i[0]);
}

int main(){
    unsigned int *a;
    cudaMalloc(&a,sizeof(unsigned int));
    kernal<<<2,33>>>(a);
    return 0;
}