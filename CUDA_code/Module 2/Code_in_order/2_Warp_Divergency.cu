#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_common.cuh"

__global__ void code_without_divergence()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;    //X dimensional Grid

    float a, b;
    a = b = 0;

    int warp_id = gid / 32; //Calculate Warp Id by dividing GID by 32 (remember that a warp has 32 threads inside)

    if (warp_id % 2 == 0)
    {
        a = 100.0;
        b = 50.0;
    }
    else
    {
        a = 200.0;
        b = 75.0;
    }
}

__global__ void divergence_code()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;    //X dimensional Grid

    float a, b;
    a = b = 0;

    //When calculating branch efficiency, it will take the portions of code inside both if and else blocks
    //In consideration. Also, the CUDA compiler will still optimize some aspects of our code regardless of 
	// us turning off the compiler
    if (gid % 2 == 0)   //Assigning Values based on Thread Id's   
    {
        a = 100.0;
        b = 50.0;
    }
    else
    {
        a = 200.0;
        b = 75.0;
    }
}

int main(int argc, char** argv)
{
    printf("\n-----------------------WARP DIVERGENCE EXAMPLE------------------------ \n\n");

    int size = 1 << 22;

    dim3 block_size(128);
    dim3 grid_size((size + block_size.x - 1) / block_size.x);

    code_without_divergence << <grid_size, block_size >> > ();
    cudaDeviceSynchronize();

    divergence_code << <grid_size, block_size >> > ();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}