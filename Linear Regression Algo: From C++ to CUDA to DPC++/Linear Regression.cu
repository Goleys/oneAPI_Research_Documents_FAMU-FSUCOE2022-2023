//If you run this code in a regular cpp compiler, eliminate the CUDA .h files
//If you run this code in a CUDA suported environment, keep the CUDA .h files
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// C++ program to implement
#include <iostream>
#include <cstdio>
#include <vector>
using namespace std;

//Kernel that adds up two Arrays 
__global__ void vector_initialization(float* a, float* b, float *sxy, float* sx, float *sy, float *sxs,float* sys, int size)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index < size)
    {
        sxy[index] = a[index] * b[index];   // Contains sum of product of all (i-th x) and (i-th y)
        sx[index] = a[index];    // Contains sum of all (i-th x)
        sy[index] = b[index];    // Contains sum of all (i-th y)
        sxs[index] = a[index] * a[index]; //Contains sum of square of all (i-th x)
        sys[index] = b[index] * b[index]; // Contains sum of square of all (i-th y)
        //printf("a: %f, b: %f \n", a[index], b[index]);
    }
}

//Kernel that adds up two Arrays 
__global__ void squareErr(float* a, float* b, float *c, float coe,float ct, int size)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size)
        c[index] = (((coe * a[index] + ct) - b[index]) * ((coe * a[index] + ct) - b[index]));
}

//Prototypes
float sumAll(float*,int);
void PrintBestFittingLine(float&, float&, float, float, float, float, int);
float predict(float,float&, float& );

// Driver code
int main(int argc, char** argv)
{
    // Number of pairs of (xi, yi)
    freopen("large_input.txt", "r", stdin); // freopen("large_input.txt", "r", stdin);
    int s;
    cin >> s;
    int block_size = 64;
    int n_bytes = s * sizeof(float);
    float coeff = 0, constTerm = 0;

    //Host Pointers
    float  * h_a, * h_b, * sum_xy,
           * sum_x, * sum_y, * sum_x_square, * sum_y_square,
           * sqc;	//Add another pointer to hold array c

    //allocate memory for host size pointers
    h_a = (float*)malloc(n_bytes); h_b = (float*)malloc(n_bytes); sum_xy = (float*)malloc(n_bytes); sum_x = (float*)malloc(n_bytes);
    sum_y = (float*)malloc(n_bytes); sum_x_square = (float*)malloc(n_bytes); sum_y_square = (float*)malloc(n_bytes);
    sqc = (float*)malloc(n_bytes);



    for (int i = 0; i < s; i++) {
        // In a csv file all the values of
        // xi and yi are separated by commas
        char comma;
        cin >> h_a[i] >> comma >> h_b[i];
        sum_xy[i] = 0; sum_x[i] = 0;sum_y[i] = 0;
        sum_x_square[i] = 0; sum_y_square[i] = 0; 
    }

    //Device Pointers
    float* d_a, * d_b, * sxy, * sx, * sy, * sxs, * sys, *sq;

    //allocate memory for device size pointers
    cudaMalloc((float**)&d_a, n_bytes);cudaMalloc((float**)&d_b, n_bytes);
    cudaMalloc((float**)&sxy, n_bytes);cudaMalloc((float**)&sx, n_bytes);
    cudaMalloc((float**)&sy, n_bytes); cudaMalloc((float**)&sxs, n_bytes);
    cudaMalloc((float**)&sys, n_bytes); cudaMalloc((float**)&sq, n_bytes);

    //Launching the grid
    dim3 block(block_size);
    dim3 grid((s / block.x) + 1);

    //Memory transfer from Host to Device
    cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice); cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(sxy, sum_xy, n_bytes, cudaMemcpyHostToDevice); cudaMemcpy(sx, sum_x, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(sy, sum_y, n_bytes, cudaMemcpyHostToDevice); cudaMemcpy(sxs, sum_x_square, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(sys, sum_y_square, n_bytes, cudaMemcpyHostToDevice);

    vector_initialization << < grid, block >> > (d_a, d_b, sxy,sx,sy,sxs,sys, s);
    cudaDeviceSynchronize();

    //Memory Tranfer back to host 
    cudaMemcpy(sum_xy, sxy, n_bytes, cudaMemcpyDeviceToHost); cudaMemcpy(sum_x, sx, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_y, sy, n_bytes, cudaMemcpyDeviceToHost); cudaMemcpy(sum_x_square, sxs, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_y_square, sys, n_bytes, cudaMemcpyDeviceToHost);

    PrintBestFittingLine(coeff, constTerm, sumAll(sum_x_square,s), sumAll(sum_xy,s), sumAll(sum_x,s), sumAll(sum_y,s), s);

    cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice); cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);
    squareErr << < grid, block >> > (d_a, d_b, sq, coeff, constTerm, s);
    cudaMemcpy(sqc, sq, n_bytes, cudaMemcpyDeviceToHost);

    cout << "The errorSquared  = " << sumAll(sqc, s) << endl;
}

float sumAll(float* lol, int yes)
{
    float olo = 0;
    for (int i = 0; i < yes; i++) olo += lol[i];
    return olo;
}

// Function that print the best fitting line
void PrintBestFittingLine(float& coe, float& ct, float sxs, float sxy, float sx, float sy, int size)
{
    if (coe == 0 && ct == 0) {
        coe = (size * sxy - sx * sy) / (size * sxs - sx * sx); // Calculate the coefficient slope of the best fitting line
        ct = (sy * sxs - sx * sxy) / (size * sxs - sx * sx);  // Calculate the constant term of the best fitting line
    }
    cout << "The best fitting line is y = " << coe << "x + " << ct << endl;
}

//Predicts the respective y value for an specific x value
float predict(float x, float& coe, float& ct)
{
    return coe * x + ct;
}