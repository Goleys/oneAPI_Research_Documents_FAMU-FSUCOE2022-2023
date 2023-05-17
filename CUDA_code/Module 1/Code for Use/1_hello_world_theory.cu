//Allows us to use keywords defined in the CUDA runtime
#include "cuda_runtime.h"	
#include "device_launch_parameters.h"
//Allows us to use printf function to output text to the console
#include <stdio.h>

//Creating a kernel
__global__ void hello_cuda()
{
	printf("Hello CUDA world \n");	
}

//In this example, host code will be the main function
int main()
{
	int nx, ny;
	nx = 16;
	ny = 4;
	dim3 block(8,2);	// x y z
	dim3 grid(nx/block.x,ny/block.y);
	
	hello_cuda << <grid, block >> > ();	

	cudaDeviceSynchronize();	//guarantees that host code to wait at this point untill all previous kernels finish execution

	//reset the device
	cudaDeviceReset();

	return 0;
}