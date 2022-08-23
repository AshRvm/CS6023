#include <stdio.h>
#include <cuda.h>
#include <iostream>

using namespace std;

__global__ void schedule(int n, unsigned int *id, int *executionTime, int *priority, int *result, int *priorityMap, int *coreEndTime, int *coreCurrId){
    __shared__ int currTime;
    __shared__ int lockVar[1];
    __shared__ int flag;
    unsigned int tid = threadIdx.x;
    int core = -1;
    int old = 0;
    currTime = 0;
    lockVar[0] = 0;
    flag = 0;
    while(id[0] < n){
        do{
            old = atomicCAS(lockVar, 0, 1);
            if(old == 0){
                core = priorityMap[priority[id[0]]];
                if(core == -1){
                    int minCore = 1001;
                    for(int i=0;i<blockDim.x;i++){
                        if(coreEndTime[i] <= currTime){
                            minCore = (minCore < i) ? minCore : i;
                        }
                    }
                    priorityMap[priority[id[0]]] = minCore;
                    core = minCore;
                }
                if(coreCurrId[core] == -1){
                    coreCurrId[core] = id[0];
                    if(currTime < coreEndTime[core]){
                        currTime = coreEndTime[core];
                    }else{
                        coreEndTime[core] = currTime;
                    }
                    id[0] += 1;
                    printf("%d :: %d\n",id[0], currTime);
                }else{
                    flag = 1;
                }

                lockVar[0] = 0;
                old = 0;
            }
        } while(old != 0 && flag != 1);

        int tempId = coreCurrId[tid];
        if(tempId != -1){
            coreEndTime[tid] += executionTime[tempId];
            result[tempId] = coreEndTime[tid];
            coreCurrId[tid] = -1;
        }
    
        __syncthreads();
        flag = 0;
    }
}

//Complete the following function
void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
    int *dExecutionTime, *dPriority, *dResult, *priorityMap, *coreEndTime, *coreCurrId;
    unsigned int *id;

    cudaMalloc(&dExecutionTime, n*sizeof(int));
    cudaMalloc(&dPriority, n*sizeof(int));
    cudaMalloc(&dResult, n*sizeof(int));
    cudaMalloc(&priorityMap, m*sizeof(int));
    cudaMalloc(&coreEndTime, m*sizeof(int));
    cudaMalloc(&coreCurrId, m*sizeof(int));
    cudaMalloc(&id, sizeof(unsigned int));

    cudaMemcpy(dExecutionTime, executionTime, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dPriority, priority, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dResult, result, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(priorityMap, -1, m*sizeof(int));
    cudaMemset(coreEndTime, 0, m*sizeof(int));
    cudaMemset(coreCurrId, -1, m*sizeof(int));

    schedule<<<1,m>>>(n, id, dExecutionTime, dPriority, dResult, priorityMap, coreEndTime, coreCurrId);

    cudaMemcpy(result, dResult, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dExecutionTime);
    cudaFree(dPriority);
    cudaFree(dResult);
    cudaFree(priorityMap);
    cudaFree(coreEndTime);
    cudaFree(id);
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
