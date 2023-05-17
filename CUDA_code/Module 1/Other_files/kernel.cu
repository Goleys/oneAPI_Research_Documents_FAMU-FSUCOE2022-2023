#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
/*
//Take one argument, point to the integer array, and we transfer memory from host pointer to the device
__global__ void unique_idx_calc_threadIdx(int* input)
{
	//Store unique id for a threadc considering one thread block in a variable call tid
	int tid = threadIdx.x;
	printf("threadIdx: %d, value : %d \n", tid, input[tid]);
}

__global__ void unique_gid_calculation(int* input)
{
	int tid = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int gid = tid + offset;
	printf("blockIdx.x: %d, threadIdx.x: %d, gid : %d, value: %d \n",
			blockIdx.x, tid, gid, input[gid]);
}*/
__global__ void unique_gid_calculation_2d_2d(int* data)
{
	int tid = blockDim.x * threadIdx.y +threadIdx.x;
	int num_threads_in_a_block = blockDim.x + blockDim.y;
	int block_offset = blockIdx.x * num_threads_in_a_block;
	int 
	int row_offset = blockDim.x * gridDim.x * blockIdx.y;
	int gid = row_offset + block_offset + tid;
	printf("blockIdx.x: %d,blockIdx.y: %d,  threadIdx.x: %d, gid : %d, value: %d \n",
		blockIdx.x, blockIdx.y, tid, gid, data[gid]);
}
int main()
{
	int array_size = 16;	//Size of array 8
	int array_byte_size = sizeof(int) * array_size;	//4 bytes * array size --> 4 * 8 = 32 bytes
	int h_data[] = {23,9,4,53,65,12,1,33,22,43,56,4,76,81,94,32};		//Array with data 87,45,23,12,342,56,44,9
	//Printing the array above
	for (int i = 0;i < array_size; i++)
	{
		printf("%d ", h_data[i]);
	}
	printf("\n \n");	//Double space

	//Integer pointer 
	int* d_data;
	//Takes input from pointer created before and the array byte size
	cudaMalloc((void**)&d_data, array_byte_size);	//Malloc is to allocate memory in RAM?
	//Takes input from pointer, array of information, and total amount of memory
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	//Creating dimensional variables
	dim3 block(4);	//x y z	8 4
	dim3 grid(2,2);	//x y z 1 2
	//unique_idx_calc_threadIdx
	unique_gid_calculation_2d << < grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}