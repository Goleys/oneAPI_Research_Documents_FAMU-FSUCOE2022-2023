#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Add the .h .cpp and .cuh and .cu files through the actual folder !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#include "cuda_common.cuh"
#include <stdio.h>
#include "common.h"

//for random intialize
#include <stdlib.h>
#include <time.h>

//for memset
#include <cstring>

#include "common.h"

__global__ void sum_arrays_gpu(int* a, int* b, int* c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
		c[gid] = a[gid] + b[gid];
}

void sum_arrays_cpu(int* a, int* b, int* c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	int size = 10000;
	int block_size = 128;

	//number of bytes needed to hold element count
	int n_bytes = size * sizeof(int);	//Same as NO_bytes

	//Host Pointer
	int* h_a, * h_b, * gpu_results, * h_c;	//Add another pointer to hold array c

	//allocate memory for host size pointers
	h_a = (int*)malloc(n_bytes);
	h_b = (int*)malloc(n_bytes);
	gpu_results = (int*)malloc(n_bytes);
	h_c = (int*)malloc(n_bytes);

	//initialize h_a and h_b arrays randomly
	time_t t;
	srand((unsigned)time(&t));

	for (size_t i = 0; i < size; i++)
		h_a[i] = (int)(rand() & 0xFF);

	for (size_t i = 0; i < size; i++)
		h_b[i] = (int)(rand() & 0xFF);

	//Calculate array c in CPU
	sum_array_cpu(h_a, h_b, h_c, size);

	//Set gpu_results in memory
	memset(gpu_results, 0, n_bytes);


	//Device Pointers
	int* d_a, * d_b, * d_c;

	//allocate memory for device size pointers
	cudaMalloc((int**)&d_a, n_bytes);

	//Memory transfer from Host to Device
	cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);

	//Launching the grid
	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);

	sum_arrays_gpu << <grid, block> >> (d_a, d_b, d_c, size);
	cudaDeviceSynchronize();

	//Memory Tranfer back to host 
	cudaMemcpy(d_c, gpu_results, n_bytes, cudaMemcpyDeviceToHost);

	//array comparison
	for

		//We then reclaim the memory from host and device
		cudaFree(d_c); cudaFree(d_b); cudaFree(d_a);
	free(h_a); free(h_b); free(gpu_results);
}

