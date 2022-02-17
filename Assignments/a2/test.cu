#include<stdio.h>
#include<cuda.h>
#include<iostream>
using namespace std;

__global__ void initialise(int *A){
	unsigned int id = threadIdx.x;
	A[id] = 0;
} 

__global__ void print(int *A){
	unsigned int id = threadIdx.x;
	printf("%d ",A[id]);
}

__global__ void prod1(int *C, int *D, int *X, int r){
	extern __shared__ int s[];
	unsigned int id1 = r * blockIdx.x;
	unsigned int id2 = r * threadIdx.x;
	for(int i=0;i<r;i++){
		s[i] = C[id1 + i];
	}
	__syncthreads();
	int temp = 0;
	for(int i=0;i<r;i++){
		temp += s[i] * D[id2 + i];
		// printf("%d,%d : %d\n",blockIdx.x,threadIdx.x,temp);
	}
	X[blockIdx.x * blockDim.x + threadIdx.x] = temp;
}

__global__ void prod2(int *A, int *B, int *X, int q){
	extern __shared__ int s[];
	unsigned int id1 = q * blockIdx.x;
	unsigned int id2 = threadIdx.x;
	for(int i=0;i<q;i++){
		s[i] = A[id1 + i];
	}
	__syncthreads();
	int temp = 0;
	for(int i=0;i<q;i++){
		temp += s[i] * B[id2 + i * blockDim.x];
	}
	X[blockIdx.x * blockDim.x + threadIdx.x] = temp;
}

int main(){
    int A[6];
    int B[6];
	int C[12];
    int *a, *b, *c;
    cudaMalloc(&a,6*sizeof(int));
    cudaMalloc(&b,8*sizeof(int));
	cudaMalloc(&c,12*sizeof(int));
	for(int i=0;i<6;i++) cin>>A[i];
	for(int i=0;i<8;i++) cin>>B[i];
	cudaMemcpy(a, A, 6*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b, B, 8*sizeof(int), cudaMemcpyHostToDevice);
	// initialise<<<1,4>>>(c);
	// prod1<<<3,4,3*sizeof(int)>>>(a,b,c,2);
	// q,s,r,r
	prod2<<<3,4,2*sizeof(int)>>>(a,b,c,2);
	// print<<<1,4>>>(c);
	cudaDeviceSynchronize();
	cout<<endl;
	cudaMemcpy(C, c, 12*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0;i<12;i++){
		cout<<C[i]<<endl;
	}
}