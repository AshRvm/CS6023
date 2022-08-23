#include <stdio.h>
#include <cuda.h>

using namespace std;

__global__ void multicore_sched(int n, int *e,int *p, int *r){
    extern __shared__ int assigned_core[];
    __shared__ int smallest_free_core;
    __shared__ int launch_time;
    int core_free_time;

    int task_p;
    int task_core;
    bool is_free = true;
    int id = threadIdx.x;
    assigned_core[id] = -1;
    launch_time = 0;
    core_free_time = 0;
    __syncthreads();
    
    for(int i = 0; i < n; i++){
        task_p = p[i];
        task_core = assigned_core[task_p];

        if(task_core == -1){
            smallest_free_core = 1001;
            __syncthreads();
            if(is_free){
                atomicMin(&smallest_free_core, id);
            }
            __syncthreads();
            task_core = smallest_free_core;
            assigned_core[task_p] = smallest_free_core;
        }
        __syncthreads();

        //assert(task_core != -1);
        if(id == task_core){
            is_free = false;
            atomicMax(&launch_time, core_free_time);    //Max is enough
            core_free_time = launch_time + e[i];
            r[i] = core_free_time;
        }
        __syncthreads();
        if(launch_time >= core_free_time){
            is_free = true;
        }
        __syncthreads();

    }
}

//Complete the following function
void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
    int *d_result,*d_execTime,*d_priority;
    cudaMalloc(&d_result,n*sizeof(int));
    cudaMalloc(&d_execTime,n*sizeof(int)); 
    cudaMalloc(&d_priority,n*sizeof(int));
    cudaMemcpy(d_execTime, executionTime, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, executionTime, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_priority, priority, n*sizeof(int), cudaMemcpyHostToDevice);

    //
    multicore_sched<<<1,m,(m+1)*sizeof(int)>>>(n,d_execTime,d_priority,d_result);
    //
    //cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaFree(d_priority);
    cudaFree(d_execTime);
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
    /*
    FILE *extrafp;
    FILE *actual_out;
    extrafp = fopen("extra.txt","w");
    actual_out = fopen("output10.txt","r");
    int temp;
    for ( int i=0; i<200; i++ )  {
        fscanf( actual_out, "%d", &temp);
        fprintf( extrafp, "i=%d\tP=%d\tT=%d\tmy_act=%d,%d\n", i,priority[i],executionTime[i], result[i],temp);
    }

    fclose( extrafp );
    fclose( actual_out );
    */
    free(executionTime);
    free(priority);
    free(result);
    
    
    
}
