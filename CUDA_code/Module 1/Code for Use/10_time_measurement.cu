#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_common.cuh"
#include <stdio.h>

//for random intialize
#include <stdlib.h>
#include <time.h>

//for memset
#include <cstring>

__global__ void sum_arrays_gpu(int* a, int* b, int* c, int size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < size)
		c[index] = a[index] + b[index];
}

//compare two arrays
void compare_arrays(int* a, int* b, int size)
{
	bool eq = true;
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			bool eq = false;
		}
	}
	if (eq == false)
		printf("Arrays are different \n");
	else
		printf("Arrays are the same \n");
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
	int size = 1 << 25;
	int block_size = 128;
	cudaError error;	//We assign the variable to the functions we used

	//number of bytes needed to hold element count
	int n_bytes = size * sizeof(int);	//Same as NO_bytes

	//Host Pointer
	int* h_a, * h_b, * gpu_results, * h_c;	//Add another pointer to hold array c

	//allocate memory for host size pointers
	h_a = (int*)malloc(n_bytes);
	h_b = (int*)malloc(n_bytes);
	gpu_results = (int*)malloc(n_bytes);
	cpu_results = (int*)malloc(n_bytes);

	//initialize h_a and h_b arrays randomly
	time_t t;
	srand((unsigned)time(&t));

	for (size_t i = 0; i < size; i++)
		h_a[i] = (int)(rand() & 0xFF);

	for (size_t i = 0; i < size; i++)
		h_b[i] = (int)(rand() & 0xFF);

	//Set gpu_results in memory
	memset(gpu_results, 0, n_bytes);
	memset(cpu_results, 0, n_bytes);
	
	//Sumation of the CPU
	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	sum_arrays_cpu(h_a, h_b, cpu_results, size);
	cpu_end = clock();
	
	//Device Pointers
	int* d_a, * d_b, * d_c;

	//allocate memory for device size pointers
	cudaMalloc((int**)&d_a, n_bytes);
	cudaMalloc((int**)&d_b, n_bytes);
	cudaMalloc((int**)&d_c, n_bytes);

	//Launching the grid
	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);

	clock_t htod_start, htod_end;
	htod_start = clock();
	//Memory transfer from Host to Device
	cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);
	htod_end = clock();
	
	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	sum_arrays_gpu << < grid, block >> > (d_a, d_b, d_c, size);
	cudaDeviceSynchronize();
	gpu_end = clock();
	
	clock_t dtoh_start, dtoh_end;
	dtoh_start = clock();
	//Memory Tranfer back to host 
	cudaMemcpy(d_c, gpu_results, n_bytes, cudaMemcpyDeviceToHost);
	dtoh_end = clock();
	
	//array comparison
	compare_arrays(cpu_results, gpu_results, size);

	printf("Sum array CPU execution time: %4.6f \n",
			(double)((double)(cpu_end-cpu_start)/CLOCKS_PER_SEC));

	printf("Sum array GPU execution time: %4.6f \n",
			(double)((double)(gpu_end-gpu_start)/CLOCKS_PER_SEC));
			
	printf("htod mem transfer time: %4.6f \n",
			(double)((double)(htod_end-htod_start)/CLOCKS_PER_SEC));
			
	printf("dtoh mem transfer time: %4.6f \n",
			(double)((double)(dtoh_end-dtoh_start)/CLOCKS_PER_SEC));
	
	printf("Sum array GPU total execution time: %4.6f \n",
			(double)((double)(dtoh_end-htod_start)/CLOCKS_PER_SEC));
			
	//We then reclaim the memory from host and device
	cudaFree(d_c); cudaFree(d_b); cudaFree(d_a);
	free(h_a); free(h_b); free(gpu_results);
}

