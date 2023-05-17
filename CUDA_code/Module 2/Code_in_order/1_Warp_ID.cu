#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void print_details_of_warps()
{
	int gid = blockIdx.y * gridDim.x * blockDim.x		//Y-offset
			  + blockIdx.x * blockDim.x + threadIdx.x;	//X-offset & actual Thread Idx

	int warp_id = threadIdx.x / 32;	//Lets' see this one | threadIdx.x / 32

	int gbid = blockIdx.y * gridDim.x + blockIdx.x;	//Block Index of y * Grid dim X (2) + block Index | Offset Y Block + Block X Idx

	printf("tid : %d, bid.x : %d, bid.y : %d, gid : %d, warp_id : %d, gbid : %d \n",
			threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
}

int main(int argc , char** argv)
{
	dim3 block_size(42);
	dim3 grid_size(2,2);

	print_details_of_warps << <grid_size,block_size >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return EXIT_SUCCESS;
}