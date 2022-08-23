#include <set>
#include <cuda.h>
#include <vector>
#include <stdio.h>
#include <iterator>
#include <stdlib.h>
#include <iostream>
using namespace std;

__global__ void kernel(int* capacities, int* trainSrc, int* trainDst, int* classNumStart,
                        int* trainNumbers, int* classNumbers, int* src, int* dst, int*seatNumbers, int* threadClassMap,
                        int* travelCapacities, int* successfulSeats, unsigned int* count, int numReq, int numTrains, int numClasses){
    
    __shared__ int isDoneReq[5000];

    unsigned int tid = threadIdx.x;
    int classNum = threadClassMap[tid];
    int trainNum = numTrains-1;
    int isBusy = 0;

    int reqNum, reqSrc, reqDst, reqSeats;

    for(int i=0;i<numReq;i++){
        isDoneReq[i] = 0;
    }
    for(int i=0;i<numTrains;i++){
        if(classNum < classNumStart[i]){
            trainNum = i-1;
            break;
        }
    }
    __syncthreads();

    while(count[0] < numReq){
        for(int i=0;i<numReq;i++){
            if(!isDoneReq[i]){
                int reqTrainNum = trainNumbers[i];
                int id = classNumStart[reqTrainNum] + classNumbers[i];
                if(!isBusy && (classNum == id)){
                    isDoneReq[i] = 1;
                    isBusy = 1;
                    reqNum = i;
                    reqSrc = src[i];
                    reqDst = dst[i];
                    reqSeats = seatNumbers[i];
                }
            }
        }

        if(isBusy){
            isBusy = 0;
            int temp1 = trainSrc[trainNum];
            int temp2 = trainDst[trainNum];
            int min = (temp1 < temp2) ? temp1 : temp2;
            int ind1 = reqSrc - min;
            int ind2 = reqDst - min;
            if(ind1 > ind2){
                ind2 = ind1 + ind2;
                ind1 = ind2 - ind1;
                ind2 = ind2 - ind1;
            }

            int success = 1;
            int tempInd = classNum * 50;
            for(int i=ind1;i<ind2;i++){
                if(travelCapacities[i+tempInd] < reqSeats){
                    success = 0;
                    break;
                }
            }

            if(success){
                for(int i=ind1;i<ind2;i++){
                    travelCapacities[i+tempInd] -= reqSeats;
                }
                __threadfence();
                successfulSeats[reqNum] += reqSeats*(ind2-ind1);
            }
            atomicInc(count, 5001);
        }
        __syncthreads();
    }
}

int main(int argc,char **argv){
    int numTrains, numBatches, totalNumClasses=0;
    cin>>numTrains;

    vector<int> tempVec;
    int* host_trainClassCount = (int*)malloc(numTrains * sizeof(int));
    int* host_trainSrc = (int*)malloc(numTrains * sizeof(int));
    int* host_trainDst = (int*)malloc(numTrains * sizeof(int));
    int* host_classNumStart = (int*)malloc(numTrains * sizeof(int));

    int* device_trainClassCount;
    int* device_trainSrc;
    int* device_trainDst;
    int* device_classNumStart;

    for(int i=0;i<numTrains;i++){
        int trainNum, classCount, trainSrc, trainDst;
        cin>>trainNum>>classCount>>trainSrc>>trainDst;

        host_trainClassCount[trainNum] = classCount;
        host_trainSrc[trainNum] = trainSrc;
        host_trainDst[trainNum] = trainDst;

        host_classNumStart[i] = totalNumClasses;
        totalNumClasses += host_trainClassCount[trainNum];
        
        for(int j=0;j<host_trainClassCount[trainNum];j++){
            int classNum, capacity;
            cin>>classNum>>capacity;
            tempVec.push_back(capacity);
        }
    }

    int* device_trainClassCapacities;
    int* host_trainClassCapacities = (int*)malloc(totalNumClasses * sizeof(int));
    for(int i=0;i<totalNumClasses;i++){
        host_trainClassCapacities[i] = tempVec[i];
    }

    int* device_trainTravelCapacities;
    int* host_trainTravelCapacities = (int*)malloc(totalNumClasses * 50 * sizeof(int));
    for(int i=0;i<totalNumClasses;i++){
        for(int j=0;j<50;j++){
            host_trainTravelCapacities[i*50 + j] = host_trainClassCapacities[i];
        }
    }

    cudaMalloc(&device_trainTravelCapacities, totalNumClasses * 50 * sizeof(int));
    cudaMalloc(&device_trainClassCapacities, totalNumClasses * sizeof(int));
    cudaMalloc(&device_trainClassCount, numTrains * sizeof(int));
    cudaMalloc(&device_trainSrc, numTrains * sizeof(int));
    cudaMalloc(&device_trainDst, numTrains * sizeof(int));
    cudaMalloc(&device_classNumStart, numTrains * sizeof(int));

    cudaMemcpy(device_trainTravelCapacities, host_trainTravelCapacities, totalNumClasses * 50 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_trainClassCapacities, host_trainClassCapacities, totalNumClasses * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_trainClassCount, host_trainClassCount, numTrains * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_trainSrc, host_trainSrc, numTrains * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_trainDst, host_trainDst, numTrains * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_classNumStart, host_classNumStart, numTrains * sizeof(int), cudaMemcpyHostToDevice);

    cin>>numBatches;
    for(int i=0;i<numBatches;i++){
        int numReq;
        cin>>numReq;

        int* host_trainNumbers = (int*)malloc(numReq * sizeof(int));
        int* host_classNumbers = (int*)malloc(numReq * sizeof(int));
        int* host_reqSrc = (int*)malloc(numReq * sizeof(int));
        int* host_reqDst = (int*)malloc(numReq * sizeof(int));
        int* host_seatNumbers = (int*)malloc(numReq * sizeof(int));
        int* host_successfulSeats = (int*)malloc(numReq * sizeof(int));

        int* device_trainNumbers;
        int* device_classNumbers;
        int* device_reqSrc;
        int* device_reqDst;
        int* device_seatNumbers;
        int* device_successfulSeats;
        unsigned int* device_count;

        cudaMalloc(&device_trainNumbers, numReq * sizeof(int));
        cudaMalloc(&device_classNumbers, numReq * sizeof(int));
        cudaMalloc(&device_reqSrc, numReq * sizeof(int));
        cudaMalloc(&device_reqDst, numReq * sizeof(int));
        cudaMalloc(&device_seatNumbers, numReq * sizeof(int));
        cudaMalloc(&device_successfulSeats, numReq * sizeof(int));
        cudaMalloc(&device_count, sizeof(unsigned int));

        set<int> uniqueClassNums;
        int* host_threadClassMap;
        int* device_threadClassMap;

        int prevEnd = -1;
        for(int j=0;j<numReq;j++){
            int reqNum, trainNum, classNum, reqSrc, reqDst, seatNumbers;
            cin>>reqNum>>trainNum>>classNum>>reqSrc>>reqDst>>seatNumbers;

            host_trainNumbers[j] = trainNum;
            host_classNumbers[j] = classNum;
            host_reqSrc[j] = reqSrc;
            host_reqDst[j] = reqDst;
            host_seatNumbers[j] = seatNumbers;

            uniqueClassNums.insert(classNum + host_classNumStart[trainNum]);
            if((uniqueClassNums.size() == 1024) || (j == numReq-1)){
                int setSize = uniqueClassNums.size();
                int tempNumReq = j - prevEnd;
                host_threadClassMap = (int*)malloc(setSize * sizeof(int));
                
                int k=0;
                set<int>::iterator itr;
                for(itr = uniqueClassNums.begin(); itr != uniqueClassNums.end(); itr++){
                    host_threadClassMap[k] = *itr;
                    k++;
                }

                cudaMalloc(&device_threadClassMap, setSize * sizeof(int));
                cudaMemcpy(device_threadClassMap, host_threadClassMap, setSize * sizeof(int), cudaMemcpyHostToDevice);

                cudaMemcpy(device_trainNumbers, host_trainNumbers+prevEnd+1, tempNumReq * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(device_classNumbers, host_classNumbers+prevEnd+1, tempNumReq * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(device_reqSrc, host_reqSrc+prevEnd+1, tempNumReq * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(device_reqDst, host_reqDst+prevEnd+1, tempNumReq * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(device_seatNumbers, host_seatNumbers+prevEnd+1, tempNumReq * sizeof(int), cudaMemcpyHostToDevice);

                cudaMemset(device_count, 0, sizeof(unsigned int));
                cudaMemset(device_successfulSeats, 0, numReq * sizeof(int));
                cudaDeviceSynchronize();

                kernel<<<1, setSize>>>(device_trainClassCapacities, device_trainSrc, device_trainDst, device_classNumStart,
                                        device_trainNumbers, device_classNumbers, device_reqSrc, device_reqDst, device_seatNumbers, device_threadClassMap,
                                        device_trainTravelCapacities, device_successfulSeats, device_count, tempNumReq, numTrains, setSize);
        
                cudaMemcpy(host_successfulSeats+prevEnd+1, device_successfulSeats, tempNumReq * sizeof(int), cudaMemcpyDeviceToHost);

                prevEnd = j;
                uniqueClassNums.clear();
                free(host_threadClassMap);
                cudaFree(device_threadClassMap);
            }
        }
        int totalSeats = 0;
        int successCount = 0;
        for(int j=0;j<numReq;j++){
            if(host_successfulSeats[j] > 0){
                cout<<"success"<<endl;
                successCount ++;
                totalSeats += host_successfulSeats[j];
            }else{
                cout<<"failure"<<endl;
            }
        }
        cout<<successCount<<" "<<numReq-successCount<<endl;
        cout<<totalSeats<<endl;

        free(host_trainNumbers);
        free(host_classNumbers);
        free(host_reqSrc);
        free(host_reqDst);
        free(host_seatNumbers);
        free(host_successfulSeats);

        cudaFree(device_trainNumbers);
        cudaFree(device_classNumbers);
        cudaFree(device_reqSrc);
        cudaFree(device_reqDst);
        cudaFree(device_seatNumbers);
        cudaFree(device_successfulSeats);

        cudaDeviceSynchronize();
    }

    free(host_trainClassCount);
    free(host_trainSrc);
    free(host_trainDst);
    free(host_classNumStart);
    free(host_trainClassCapacities);
    free(host_trainTravelCapacities);

    cudaFree(device_trainClassCount);
    cudaFree(device_trainSrc);
    cudaFree(device_trainDst);
    cudaFree(device_classNumStart);
    cudaFree(device_trainClassCapacities);
    cudaFree(device_trainTravelCapacities);

    return 0;
}