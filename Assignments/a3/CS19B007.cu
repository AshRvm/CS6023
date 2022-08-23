#include <stdio.h>
#include <cuda.h>
#include <iostream>

using namespace std;

__global__ void schedule(int n, int *executionTime, int *priority, int *result, int *priorityMap){
    __shared__ int core;
    __shared__ int minCore;
    __shared__ int currTime;
    __shared__ int id;

    unsigned int tid = threadIdx.x;
    int coreEndTime = 0;
    currTime = 0;
    id = 0;

    while(id < n){
        core = priorityMap[priority[id]];
        minCore = 1001;
        __syncthreads();

        if(core == -1){
            if(coreEndTime <= currTime){
                atomicMin(&minCore, tid);
            }
        }
        __syncthreads();

        if(core == -1){
            core = minCore;
            priorityMap[priority[id]] = minCore;
        }
        __syncthreads();

        if(tid == core){
            currTime = (currTime > coreEndTime) ? currTime : coreEndTime;
            coreEndTime = currTime + executionTime[id];
            result[id] = coreEndTime;
            id += 1;
        }
        __syncthreads();
    }
}

//Complete the following function
void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
    int *dExecutionTime, *dPriority, *dResult, *priorityMap;

    cudaMalloc(&dExecutionTime, n*sizeof(int));
    cudaMalloc(&dPriority, n*sizeof(int));
    cudaMalloc(&dResult, n*sizeof(int));
    cudaMalloc(&priorityMap, m*sizeof(int));

    cudaMemcpy(dExecutionTime, executionTime, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dPriority, priority, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dResult, result, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(priorityMap, -1, m*sizeof(int));

    schedule<<<1,m>>>(n, dExecutionTime, dPriority, dResult, priorityMap);

    cudaMemcpy(result, dResult, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dExecutionTime);
    cudaFree(dPriority);
    cudaFree(dResult);
    cudaFree(priorityMap);
}

int main(int argc,char **argv)
{
    int m,n;
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &m );      //scaning for number of cores
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of tasks
   
   //Taking execution time and priorities as input	
    int *executionTime = (int *) malloc ( n * sizeof (int) );
    int *priority = (int *) malloc ( n * sizeof (int) );
    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &executionTime[i] );
    }

    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &priority[i] );
    }

    //Allocate memory for final result output 
    int *result = (int *) malloc ( (n) * sizeof (int) );
    for ( int i=0; i<n; i++ )  {
        result[i] = 0;
    }
    
     cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================
	

	operations ( m, n, executionTime, priority, result ); 
	
    //===========================================================================================================
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    //Total time of each task: Final Result
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", result[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
    std::free(executionTime);
    std::free(priority);
    std::free(result);
    
    
    
}
