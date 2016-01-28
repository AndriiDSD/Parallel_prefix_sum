// Andrii Hlyvko
// Nadiia Chepurko
// Lucas Rivera

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

// Measure timinig 
#include "CycleTimer.h"

//#define SIZE 1000000  // size of the input array 
#define BLOCK_SIZE 256 // 256 threads per block
static int SIZE = 1000000;
using namespace std;

//////////////////////////////
/****************************
* Sequential functions:
*////////////////////////////


/*
* This function rounds up the input number to the next power of 2
*/
static inline int next_power2(int number)
{
    number--;
    number |= number >> 1;
    number |= number >> 2;
    number |= number >> 4;
    number |= number >> 8;
    number |= number >> 16;
    number++;
    return number;
}

/*
* This function finds repeats in the input array sequentially. 
* repeatIndex are all the indecies that were repeated. The output
* has all the repeating elements removed. The number of repeats is returned.
*/
static int find_repeats(int * array, int * repeatIndex, int *output, int size)
{
	if(size<=0)
		return -1;

	if(array == NULL || repeatIndex == NULL || output == NULL)
		return -1;

	int numRepeats = 0;
	
	for(int i=0; i<(size-1); i++)
	{
		if(array[i] == array[i+1])
		{
			repeatIndex[numRepeats] = i;
			numRepeats++; 
		}
		else
		{
			output[i-numRepeats] = array[i];
		}
	}
	output[size-numRepeats-1]=array[size-1];
	
	return numRepeats;
}

/*
* The sequential version of exclusive scan. This function sums up
* all values up to but not including input[i] and stores in output[i].
* Both arrays are of size int size.
*/
static void exclusive_scan(int * input, int * output, int size)
{
	if(size<=0)   // size is zero or negative
	return;

	if(input==NULL || output==NULL)  // check for null pointers
	return;

	output[0]=0;
	
	for (int i=1; i<size; i++)       // sum up the input values
	{
			output[i]= output[i-1]+input[i-1];
	}
}


/*//////////////////////////////////
* Cuda functions
*///////////////////////////////////

/**
* This function is used to add the auxilary array to get the scanned array.
* The auxilary array holds the sum from individual blocks. The input array is scanned in
* blocks, so the sum from the previous block has to be added to all elements of next block.
**/
__global__ void addAuxilary(int *input,int inputSize, int *auxilary, int auxSize)
{
	//each thread gets an element in the input array
	int threadId = threadIdx.x +blockIdx.x*blockDim.x;    // each thread adds the corresponding 
							     // entry in auxilary to every entry in input
		
	// threads in first block do not do work
	while(threadId < inputSize)
	{
		if(blockIdx.x!=0 && threadId < inputSize)// if not the first block and not over limit then add
		{
			input[threadId]+=auxilary[blockIdx.x];
		}
		threadId += gridDim.x*blockDim.x;
	}
}


/**
* This function performs prefix sum on the input array. The prefix sum is performed on
* blocks and the sum of individual blocks is stored in auxilary. Later the auxilary is added 
* to the input to get the final prefix sum.
**/
__global__ void exclusive_scan_cuda(int *input, int *auxilary, int n, int totalSize)
{
	__shared__ int cache[BLOCK_SIZE];
	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;  // calculate thread id
	int threadNum = threadIdx.x;
	int offset =1;

	// copy input into shared memory
	while(threadId < totalSize){
		if(threadNum<n)
		{
			cache[threadIdx.x] =input[threadId];
		
		}

		// upsweep phase
		for(int d = n>>1; d>0; d >>=1)
		{
			__syncthreads();
			if(threadNum < d)
			{
				int ai = offset*(2*threadNum+1)-1;
				int bi = offset*(2*threadNum+2)-1;
				if(bi<BLOCK_SIZE && ai< BLOCK_SIZE)
				cache[bi] += cache[ai];             
			}
			offset*=2;	
		}

		// store last element in auxilary before setting it to zero
		if(threadNum==0 && auxilary!=NULL)
		{
			auxilary[blockIdx.x]=cache[BLOCK_SIZE - 1];		
		}
		if(threadNum==0)
			cache[BLOCK_SIZE - 1]=0;

		// downsweep phase
		for(int d = 1; d < n; d*=2)
		{
			offset >>=1;
			__syncthreads();
			if(threadNum < d)
			{
				int ai = offset*(2*threadNum+1)-1;
				int bi = offset*(2*threadNum+2)-1;
				if(ai < BLOCK_SIZE && bi < BLOCK_SIZE)
				{
					int temp = cache[ai];              
					cache[ai] = cache[bi];
					cache[bi] += temp;
				}
			}
		}
		
		// synchronize threads and store result back to input
		__syncthreads();
		if(threadNum<n)
		{
			input[threadId] = cache[threadIdx.x];
		}
		threadId += blockDim.x*gridDim.x;
	}
}

/**
* This function performs exclusive scan on the input array.
* It launches the scan on blocks of data then the auxilary array is added to 
* corresponding blocks to compute total sum. First the input is scanned in blocks 
* and the block sums are stored in auxilary. Then the auxilary is scanned and its block
* sums are stored in auxilary2. Then auxilary2 is scanned and its elements are added 
* to corresponding blocks in auxilary. Finally, auxilary is added to input to compute the
* final prefix sum.
**/
void launch_exclusive_scan(int *input_device,int *auxilary, int *auxilary2, int length)
{
	int threadsPerBlock = BLOCK_SIZE;
	int blocksPerGrid = (SIZE+threadsPerBlock-1)/threadsPerBlock;

	cudaMemset(auxilary, 0, length*sizeof(int));
	cudaMemset(auxilary2, 0 , length*sizeof(int));

	exclusive_scan_cuda<<<blocksPerGrid,BLOCK_SIZE>>>(input_device,auxilary,threadsPerBlock, SIZE);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("scan input failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();

	// call to exclusive_scan_cuda on auxilary array
	exclusive_scan_cuda<<<blocksPerGrid,BLOCK_SIZE>>>(auxilary, auxilary2, threadsPerBlock, SIZE);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("scan auxilary failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();

	exclusive_scan_cuda<<<blocksPerGrid, BLOCK_SIZE>>>(auxilary2, NULL, threadsPerBlock, SIZE);
	
	// call to add the auxilary sum addAuxilary
	addAuxilary<<<blocksPerGrid, BLOCK_SIZE>>>(auxilary,SIZE, auxilary2, BLOCK_SIZE);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("add auxilary failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	addAuxilary<<<blocksPerGrid,BLOCK_SIZE>>>(input_device,SIZE,auxilary, threadsPerBlock);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("add auxilary failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();
}


/**
* This function sets up the device to perform exclusive scan. It allocates memory on the device
* and calles the function that launches kernels on device.
**/
double cudaExclusiveScan(int *array, int *output, int size)
{
	double startTime=0.0, endTime=0.0;
	
	int *output_device, *auxilary, *auxilary2; // memory on the device

	int padding = next_power2(size);   // input and output have to be a power of 2
	int t= next_power2((SIZE+BLOCK_SIZE-1)/BLOCK_SIZE);

	// allocate memory on device that is padded to next power of 2
	if(cudaSuccess != cudaMalloc((void**)&output_device, padding*sizeof(int)))
		printf("cudaMalloc error\n");
	if(cudaSuccess != cudaMalloc((void**)&auxilary, padding*sizeof(int)))
		printf("cudaMalloc error\n");	
	if(cudaSuccess != cudaMalloc((void**)&auxilary2, padding*sizeof(int)))
		printf("cudaMalloc error\n");

	cudaMemcpy(output_device, array, size*sizeof(int), cudaMemcpyHostToDevice);  //send input data to device

	startTime=CycleTimer::currentSeconds();
	launch_exclusive_scan(output_device,auxilary,auxilary2, padding);		// launch threads to compute scan
	endTime=CycleTimer::currentSeconds();

	cudaMemcpy(output, output_device, size*sizeof(int), cudaMemcpyDeviceToHost);  //copy back the result

	// free device memory
	cudaFree(output_device);
	cudaFree(auxilary);
	cudaFree(auxilary2);
	return (endTime-startTime);
}

////////////////////////////////////////////
//// find repeats cuda section
///////////////////////////////////////////

/**
* This function sums up the input array and stores the partial sums of individual 
* blocks in the output array.
**/
__global__ void sum(int *array, int *result, int size)
{
	__shared__ int cache[BLOCK_SIZE];
	int threadSum=0;

	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	while (threadId < size) // go through the array and add to individual sum
	{
		threadSum += array[threadId];
		threadId += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = threadSum;

	__syncthreads();

	int i = blockDim.x/2;

	while (i != 0) 
	{
		if(cacheIndex < i)
		{
			cache[cacheIndex] += cache[cacheIndex + i];
			__syncthreads();
		} 
		i = i/2;
	}

	if(cacheIndex == 0)  // the first thread stores back the partial sum
		result[blockIdx.x] = cache[0];

}


/*
* This function stores values at input for which the predicate is true at indesies given 
* by scanned predicate. If predicate at threadId is 1 then the input[threadId] is stored in result.
*/
__global__ void stream_compaction(int *input, int *predicate, int *scanned_predicate,int *no_repeats, int length)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int index = scanned_predicate[threadId];
	if(predicate[threadId] == 1 && threadId < length)
	{
		no_repeats[index] = input[threadId];	
	}
}

/*
* This function stores values of index at input for which the predicate is true at indesies given 
* by scanned predicate.
*/
__global__ void stream_compaction_index(int *input, int *predicate, int *scanned_predicate,int *repeats_i, int length)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int index = scanned_predicate[threadId];
	if(predicate[threadId] == 1 && threadId < length)
	{
		repeats_i[index] = threadId;//scanned_predicate[threadId];	
	}
}


/*
* This function scanns the input and fills the predicate based on the input.
* The value of predicate is true if input[i] != input[i+1]
*/
__global__ void cuda_predicate_no_repeats(int *input, int *predicate, int length)
{
	__shared__ int cache[BLOCK_SIZE+1];
	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;  // calculate thread id

	while(threadId < length){

		// copy input into shared memory
		if(threadId<length)
		{
			cache[threadIdx.x] =input[threadId];
			if((blockIdx.x != (blockDim.x-1)) && threadIdx.x == 0) // not the last block
			{ //first thread updated the last cache element 
				cache[blockDim.x]=input[blockDim.x*blockIdx.x + blockDim.x]; 
			}		
		}
	
		__syncthreads();

		// fill out predicate
		if(threadId != (length-1))
		{
			if(cache[threadIdx.x] != cache[threadIdx.x+1])
				predicate[threadId] = 1;
			else
				predicate[threadId] = 0;
		}
		else
		{
			predicate[threadId] =1;
		}
	
		threadId+=blockDim.x*gridDim.x;
	}		
}

/*
* This function scanns the input and fills the predicate based on the input.
* The value of predicate is true if input[i] == input[i+1]
*/
__global__ void cuda_predicate_repeats(int *input, int *predicate, int length)
{
	__shared__ int cache[BLOCK_SIZE+1];
	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;  // calculate thread id

	while(threadId < length){

		// copy input into shared memory
		if(threadId<length)
		{
			cache[threadIdx.x] =input[threadId];
			if((blockIdx.x != (blockDim.x-1)) && threadIdx.x == 0) // not the last block
			{ //first thread updated the last cache element 
				cache[blockDim.x]=input[blockDim.x*blockIdx.x + blockDim.x]; 
			}		
		}
	
		__syncthreads();

		// fill out predicate
		if(threadId != (length-1))
		{
			if(cache[threadIdx.x] == cache[threadIdx.x+1])
				predicate[threadId] = 1;
			else
				predicate[threadId] = 0;
		}
		else
		{
			predicate[threadId] =0;
		}
	
		threadId+=blockDim.x*gridDim.x;
	}		
}

/**
* This function is a wrapper to compute the repeating indexes and eliminate repeats 
* by using the cuda kernel functions. First the predicate of the input is computed
* to find which elements do not repeat. Then the prefix sum of the predicate is computed 
* to find out at which indexes to store elements of input that do not repeat. Then, the 
* no_repeats output is computed using the predicate and scanned predicate. 
* In the second stage the second predicate is true for values that do repeat. Its prefix sum is also 
* computed and is udes to store the indexes for which the input elements repeat.
**/
int launch_repeats(int *input, int *repeati, int *no_repeats, int length)
{
	int threadsPerBlock = BLOCK_SIZE;
	int blocksPerGrid = (SIZE+threadsPerBlock-1)/threadsPerBlock;

	int numRepeats=0;
	int *predicate, *scanned_predicate, *auxilary, *auxilary2, *partial_sum_device;

	if(cudaSuccess != cudaMalloc((void **)&predicate, length*sizeof(int)))
	{
		printf("cudaMalloc fail\n");
		return -1;
	}
	if(cudaSuccess != cudaMalloc((void **)&scanned_predicate, length*sizeof(int)))
	{
		printf("cudaMalloc fail\n");
		return -1;
	}
	if(cudaSuccess != cudaMalloc((void **)&auxilary, length*sizeof(int)))
	{
		printf("cudaMalloc fail\n");
		return -1;
	}
	if(cudaSuccess != cudaMalloc((void **)&auxilary2, length*sizeof(int)))
	{
		printf("cudaMalloc fail\n");
		return -1;
	}
	if(cudaSuccess != cudaMalloc((void **)&partial_sum_device, BLOCK_SIZE*sizeof(int)))
	{
		printf("cudaMalloc fail\n");
		return -1;
	}
	int *partial_sum = (int*)malloc(BLOCK_SIZE*sizeof(int));
	cudaMemset(predicate, 0, length*sizeof(int));
	cudaMemset(scanned_predicate, 0, length*sizeof(int));
	cudaMemset(auxilary, 0, length*sizeof(int));
	cudaMemset(auxilary2, 0, length*sizeof(int));
	
	// fill out predicate for elements that do not repeat
	cuda_predicate_no_repeats<<<blocksPerGrid, BLOCK_SIZE>>>(input, predicate, SIZE);
	cudaDeviceSynchronize();
	
	cudaMemcpy(scanned_predicate, predicate, SIZE*sizeof(int), cudaMemcpyDeviceToDevice);	

	//scan predicate
	exclusive_scan_cuda<<<blocksPerGrid,BLOCK_SIZE>>>(scanned_predicate, auxilary, threadsPerBlock, SIZE);
	cudaDeviceSynchronize();

	// scan the auxilary
	exclusive_scan_cuda<<<blocksPerGrid,BLOCK_SIZE>>>(auxilary, auxilary2, threadsPerBlock, SIZE);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("scan auxilary failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();

	// scan auxilary2
	exclusive_scan_cuda<<<blocksPerGrid, BLOCK_SIZE>>>(auxilary2, NULL, threadsPerBlock, SIZE);
	
	// call to add the auxilary sum addAuxilary
	addAuxilary<<<blocksPerGrid, BLOCK_SIZE>>>(auxilary,SIZE, auxilary2, BLOCK_SIZE);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("add auxilary failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	addAuxilary<<<blocksPerGrid,BLOCK_SIZE>>>(scanned_predicate,SIZE,auxilary, threadsPerBlock);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("add auxilary failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();
	
	// copy array based on predicate which will eliminate the repeats
	stream_compaction<<<blocksPerGrid, BLOCK_SIZE>>>(input, predicate, scanned_predicate, no_repeats, SIZE);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("stream compaction failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	////////////////////////////////////////////////
	// stage 2 finds all indexes for which elements do repeat
	cudaMemset(predicate, 0, length*sizeof(int));
	cudaMemset(scanned_predicate, 0, length*sizeof(int));
	cudaMemset(auxilary, 0, length*sizeof(int));
	cudaMemset(auxilary2, 0, length*sizeof(int));

	cuda_predicate_repeats<<<blocksPerGrid, BLOCK_SIZE>>>(input, predicate, SIZE);
	cudaDeviceSynchronize();
	
	cudaMemcpy(scanned_predicate, predicate, SIZE*sizeof(int), cudaMemcpyDeviceToDevice);

	// sum predicate to find how many repeats do we have
	sum<<<128, BLOCK_SIZE>>>(predicate, partial_sum_device,SIZE);
	cudaMemcpy(partial_sum, partial_sum_device, BLOCK_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<128; i++)
	{
		numRepeats +=partial_sum[i];
	}


	//scan predicate
	exclusive_scan_cuda<<<blocksPerGrid,BLOCK_SIZE>>>(scanned_predicate, auxilary, threadsPerBlock, SIZE);
	cudaDeviceSynchronize();

	exclusive_scan_cuda<<<blocksPerGrid,BLOCK_SIZE>>>(auxilary, auxilary2, threadsPerBlock, SIZE);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("scan auxilary failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();

	exclusive_scan_cuda<<<blocksPerGrid, BLOCK_SIZE>>>(auxilary2, NULL, threadsPerBlock, SIZE);
	
	// call to add the auxilary sum addAuxilary
	addAuxilary<<<blocksPerGrid, BLOCK_SIZE>>>(auxilary,SIZE, auxilary2, BLOCK_SIZE);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("add auxilary failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	addAuxilary<<<blocksPerGrid,BLOCK_SIZE>>>(scanned_predicate,SIZE,auxilary, threadsPerBlock);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("add auxilary failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();
	

	// copy array based on predicate
	stream_compaction_index<<<blocksPerGrid, BLOCK_SIZE>>>(input, predicate, scanned_predicate, repeati, SIZE);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
   		 printf("stream compaction failed: %s\n", cudaGetErrorString(cudaStatus));
	}


	free(partial_sum);
	cudaFree(predicate);
	cudaFree(scanned_predicate);
	cudaFree(auxilary);
	cudaFree(auxilary2);
	cudaFree(partial_sum_device);
	return numRepeats;

}

/**
* This function initializes some memory on the device and calls the wrapper to find the
* repeating elements. It is used to measure timing of launch_repeats function.
**/
double find_repeats_cuda(int *input, int *repeating_index, int *no_repeats, int length, int *output_length)
{
	double startTime=0.0, endTime=0.0;

	int finalLength = next_power2(length);
	
	int *input_device, *repeating_device, *no_repeats_device;

	// allocate device memory
	if(cudaSuccess!=cudaMalloc((void **)&input_device, finalLength*sizeof(int)))
	{
		printf("cudaMalloc fail\n");
		return -1.0;
	}
	if(cudaSuccess!=cudaMalloc((void **)&repeating_device, finalLength*sizeof(int)))
	{
		printf("cudaMalloc fail\n");
		return -1.0;
	}
	if(cudaSuccess != cudaMalloc((void **)&no_repeats_device, finalLength*sizeof(int)))
	{
		printf("cudaMalloc fail\n");
		return -1.0;
	}
	

	// copy to device
	cudaMemcpy(input_device, input, finalLength*sizeof(int), cudaMemcpyHostToDevice);


	// launch kernel
	startTime = CycleTimer::currentSeconds();
	*output_length = launch_repeats(input_device, repeating_device, no_repeats_device, finalLength);
	endTime = CycleTimer::currentSeconds();		
		
	//copy from device
	cudaMemcpy(repeating_index, repeating_device, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(no_repeats, no_repeats_device, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
			
	// cleanup
	cudaFree(input_device);
	cudaFree(repeating_device);
	cudaFree(no_repeats_device);

	return (endTime - startTime);
} 


/**
* This function prints out info about CUDA capable GPU
**/
static void cudaInfo()
{
	int numDevices=0;    // number of CUDA capable devices
	cudaDeviceProp prop;  // device properties
	

	cudaGetDeviceCount(&numDevices);   // get the number of cuda devices
	
	if(numDevices<=0)
	{
		cout <<"No CUDA devices found"<<endl;
		return;
	}

	

	for(int i=0; i<numDevices; i++)  // print out information for each device
	{
		cudaGetDeviceProperties(&prop, i);
		printf("---------------------------------\n");
		cout << "Device Number: " << i<<endl;
		printf("Device Name: %s\n",prop.name);
		printf("Computing capability: %d.%d\n",prop.major,prop.minor);
		printf("Total global memory: %ld\n",prop.totalGlobalMem);
		printf("Total shared memory per core: %ld\n",prop.sharedMemPerBlock);
		printf("Total constant memory: %ld\n",prop.totalConstMem);
		printf("Number of cores: %d\n",prop.multiProcessorCount);
		printf("Threads per warp: %d\n",prop.warpSize);
		printf("Max threads per block %d\n",prop.maxThreadsPerBlock);
		printf("Max thread dimensions: %d,%d,%d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
		printf("Max Grid Size: %d,%d,%d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
	}
}


int getArrayFromFile(int **arr, char *fileName)
{
	printf("file is:%s\n",fileName);
	if(fileName == NULL)
		return -1;

	//int *array =*arr;
	int size=0;

	FILE *fp = fopen(fileName, "r");
	if(fp==NULL)
		return -1;

	fscanf(fp,"%d",&size);
	if(size<=0)
		return -1;

	printf("file size is:%d\n",size);

	*arr=(int*)malloc(size*sizeof(int));
	int temp =0;

	for(int i=0; i<size; i++)
	{
		fscanf(fp, "%d", &temp);
		(*arr)[i]=temp;
	}
	


	return size;
}

int main(int argc, char *argv[])
{
	cudaInfo();

	int *array=NULL;//(int*)malloc(SIZE*sizeof(int)); // memory alligned input array

	int *scan = (int*)malloc((SIZE)*sizeof(int));
	int *scan_cuda=(int*)malloc((SIZE)*sizeof(int));

	int *repeating_index= (int*)malloc(SIZE*sizeof(int));
	int *repeating_index_cuda=(int*)malloc(SIZE*sizeof(int));

	int *no_repeats = (int*)malloc(SIZE*sizeof(int));
	int *no_repeats_cuda=(int*)malloc(SIZE*sizeof(int));


	int numberRepeats=0, numberRepeats_cuda=0;
	double startTime=0.0, endTime=0.0, serialRepeats=0.0, serialScan=0.0, cudaRepeats=0.0, cudaScan=0.0;
	

	unsigned int  seed =(unsigned int) time(NULL);  // generate a seed for random generator

	// initialize arrays
	if(argc == 1)
	{
		array =(int*)malloc(SIZE*sizeof(int));
		for(int i=0; i<SIZE;i++)
		{
			array[i]=(rand_r(&seed)%101);  //thread safe random generator
			repeating_index[i]=0;
			no_repeats[i]=0;
		}
	}
	else if(argc == 2)
	{
		SIZE = getArrayFromFile(&array,argv[1]);
	}
	else
	{
		printf("usage: ./assignment2  or ./assignment2 filename.txt\n");
		free(repeating_index);
		free(repeating_index_cuda);
		free(no_repeats);
		free(no_repeats_cuda);
		free(scan);
		free(scan_cuda);
		return 0;
	}

	if(array == NULL || SIZE <= 0)
	{
		printf("Array pointer is null\n");
		free(repeating_index);
		free(repeating_index_cuda);
		free(no_repeats);
		free(no_repeats_cuda);
		free(scan);
		free(scan_cuda);
		return 0;
	}


	// initialize arrays
	//for(int i=0; i<SIZE;i++)
	//{
	//	array[i]=(rand_r(&seed)%101);  //thread safe random generator
	//	repeating_index[i]=0;
	//	no_repeats[i]=0;
	//}

	startTime= CycleTimer::currentSeconds();
	exclusive_scan(array,scan,(SIZE));
	endTime=CycleTimer::currentSeconds();
	serialScan = endTime-startTime;



	startTime=CycleTimer::currentSeconds();
	numberRepeats= find_repeats(array, repeating_index, no_repeats, SIZE);
	endTime=CycleTimer::currentSeconds();
	serialRepeats=endTime-startTime;

	printf("-------------------------------\n");
	printf("Serial Part:\n");
	printf("Find Repeats: %1.5f,   Scan: %1.5f, number of repeats:%d\n",serialRepeats, serialScan,numberRepeats);
	printf("Last element of find repeats is:%d\n", array[repeating_index[numberRepeats-1]]);
	printf("i   array[i]    repeated i  no_repeats[i]   scan[i]\n");
	for(int i=0; i<10;i++)
	{
		printf("%d  %5d      %5d        %5d         %5d\n",i,array[i],repeating_index[i],no_repeats[i],scan[i]);
	}

	// run exclusive scan on gpu
	cudaScan = cudaExclusiveScan(array, scan_cuda, (SIZE));
	
	// do cuda find repeats
	cudaRepeats = find_repeats_cuda(array, repeating_index_cuda, no_repeats_cuda, SIZE, &numberRepeats_cuda);
	
	bzero(&repeating_index_cuda[numberRepeats_cuda],(SIZE-numberRepeats_cuda)*sizeof(int));
	bzero(&no_repeats_cuda[SIZE-numberRepeats_cuda],numberRepeats_cuda*sizeof(int));

	
	printf("-------------------------------\n");
	printf("\n\n-------------------------------\n");
	printf("Cuda implementation:\n");
	printf("CUDA Scan:%1.5f,   CUDA Find repeats:%1.5f, number of repeats:%d\n",cudaScan, cudaRepeats, numberRepeats_cuda);
	printf("Last element of find repeats is:%d\n", array[repeating_index_cuda[numberRepeats-1]]);
	printf("i   array[i]    repeated i  no_repeats[i]   scan[i]\n");
	for(int i=0; i<10;i++)
	{
		printf("%d  %5d        %5d        %5d          %5d\n",i,array[i],repeating_index_cuda[i],no_repeats_cuda[i],scan_cuda[i]);
	}

	//// write results to file
	FILE *A_scan_result = fopen("A_exlusive_scan_results.txt", "w+");
	FILE *B_repeated_index = fopen("B_repeated_index_results.txt", "w+"); 
	FILE *C_no_repeats = fopen("C_no_repeats_results.txt", "w+");
	
	if(A_scan_result == NULL || B_repeated_index == NULL || C_no_repeats == NULL)
		return 0;


	printf("All results were written to files\n");

	fprintf(A_scan_result,"Results of exclusive scan on array A of size %d using CPU and GPU\n", SIZE);
	fprintf(A_scan_result, "    i     A[i]        CPU_scan[i]          GPU_scan[i]\n");	

	for(int i=0; i<SIZE;i++)
	{
		fprintf(A_scan_result, "%7d   %4d      %10d       %10d\n",i,array[i],scan[i],scan_cuda[i]);
	}

	
	fprintf(B_repeated_index, "B array has the indexes of input for which A[i]=A[i+1]\n");
	fprintf(B_repeated_index,"    i      CPU B[i]          GPU B[i]\n");
	for(int i=0; i<numberRepeats; i++)
	{
		fprintf(B_repeated_index, "%7d      %7d       %7d\n",i,repeating_index[i], repeating_index_cuda[i]);
	}


	fprintf(C_no_repeats, "C array has the input array with repeats eliminated\n");
	fprintf(C_no_repeats,"     i      CPU C[i]          GPU C[i]\n");
	for(int i=0; i<(SIZE-numberRepeats); i++)
	{
		fprintf(C_no_repeats, "%7d      %4d       %4d\n",i,no_repeats[i], no_repeats_cuda[i]);
	}


	// cleanup
	fclose(A_scan_result);
	fclose(B_repeated_index);
	fclose(C_no_repeats);


	free(array);
	free(repeating_index);
	free(repeating_index_cuda);
	free(no_repeats);
	free(no_repeats_cuda);
	free(scan);
	free(scan_cuda);
	return 0;
}














