#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_common.cuh"
#include <stdio.h>

//for random intialize
#include <stdlib.h>
#include <time.h>

//for memset
#include <cstring>

//Kernel that adds up three Arrays 
__global__ void sum_arrays_gpu(int* a, int* b, int * d, int* c, int size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < size)
		c[index] = a[index] + b[index] + d[index];
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

//function that adds up three Arrays in CPU 
void sum_arrays_cpu(int* a, int* b, int* d, int* c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i] + d[i];
	}
}

int main()
{
	int size = 1 << 22;
	int block_size = 64*8;	//2 4 8
	cudaError error;	//We assign the variable to the functions we used

	//number of bytes needed to hold element count
	int n_bytes = size * sizeof(int);	//Same as NO_bytes

	//Host Pointer
	int *h_a, *h_b, *h_d, *gpu_results, *cpu_results;	//Add another pointer to hold array c

	//allocate memory for host size pointers
	h_a = (int*)malloc(n_bytes);
	h_b = (int*)malloc(n_bytes);
	h_d = (int*)malloc(n_bytes);
	gpu_results = (int*)malloc(n_bytes);
	cpu_results = (int*)malloc(n_bytes);

	//initialize h_a, h_b, and h_d arrays randomly
	time_t t;
	srand((unsigned)time(&t));

	for (size_t i = 0; i < size; i++)
		h_a[i] = (int)(rand() & 0xFF);

	for (size_t i = 0; i < size; i++)
		h_b[i] = (int)(rand() & 0xFF);

	for (size_t i = 0; i < size; i++)
		h_d[i] = (int)(rand() & 0xFF);

	//Set gpu_results in memory
	memset(gpu_results, 0, n_bytes);
	memset(cpu_results, 0, n_bytes);

	//Sumation of the CPU 
	clock_t cpu_start, cpu_end;	//Initialize clock trackers for CPU execution
	cpu_start = clock();		//Time before sum_array happens
	sum_arrays_cpu(h_a, h_b, h_d, cpu_results, size);
	cpu_end = clock();			//Time after sum_array happened

	//Device Pointers
	int *d_a, *d_b, *d_d, *d_c;

	//allocate memory for device size pointers
	gpuErrchk(cudaMalloc((int**)&d_a, n_bytes));
	gpuErrchk(cudaMalloc((int**)&d_b, n_bytes));
	gpuErrchk(cudaMalloc((int**)&d_d, n_bytes));
	gpuErrchk(cudaMalloc((int**)&d_c, n_bytes));

	//Launching the grid
	dim3 block(block_size); 
	dim3 grid((size / block.x) + 1);

	clock_t htod_start, htod_end;	//Initialize clock trackers for data transfer of htod (host to device)
	htod_start = clock();	//Time before Memory transfer from Host to Device
	//Memory transfer from Host to Device
	gpuErrchk(cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_d, h_d, n_bytes, cudaMemcpyHostToDevice));
	htod_end = clock();		//Time after Memory transfer from Host to Device

	clock_t gpu_start, gpu_end;	//Initialize clock trackers for GPU execution
	gpu_start = clock();
	sum_arrays_gpu << < grid, block >> > (d_a, d_b,d_d, d_c, size);
	cudaDeviceSynchronize();
	gpu_end = clock();

	clock_t dtoh_start, dtoh_end;	//Initialize clock trackers for data transfer of dtoh (device to host)
	dtoh_start = clock();	//Time before Memory Tranfer back to host
	//Memory Tranfer back to host 
	gpuErrchk(cudaMemcpy(gpu_results, d_c, n_bytes, cudaMemcpyDeviceToHost));
	dtoh_end = clock();		//Time after Memory Tranfer back to host

	//array comparison
	compare_arrays(cpu_results, gpu_results, size);

	printf("Sum array CPU execution time: %4.6f \n",
		(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));

	printf("Sum array GPU execution time: %4.6f \n",
		(double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));

	printf("htod mem transfer time: %4.6f \n",
		(double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));

	printf("dtoh mem transfer time: %4.6f \n",
		(double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));

	printf("Sum array GPU total execution time: %4.6f \n",
		(double)((double)(dtoh_end - htod_start) / CLOCKS_PER_SEC));

	//We then reclaim the memory from host and device
	gpuErrchk(cudaFree(d_c)); gpuErrchk(cudaFree(d_b)); gpuErrchk(cudaFree(d_a));
	free(h_a); free(h_b); free(gpu_results);
}

