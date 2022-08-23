#include <cuda.h>
#include <stdio.h>
#include <iostream>

int main(){
    int n = 10;
    int *a, *b;
    a = (int*)malloc(n*sizeof(int));
    cudaMalloc(&b,10*sizeof(int));
    cudaMemset(b,-1,10*sizeof(int));
    cudaMemcpy(a,b,10*sizeof(int),cudaMemcpyDeviceToHost);
    for(int i=0;i<10;i++){
        std::cout<<a[i]<<std::endl;
    }
}