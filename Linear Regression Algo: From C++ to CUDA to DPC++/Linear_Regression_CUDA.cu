//If you run this code in a regular cpp compiler, eliminate the CUDA .h files
//If you run this code in a CUDA suported environment, keep the CUDA .h files
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// C++ program to implement
#include <iostream>
#include <cstdio>
#include <vector>
#include <ctime>
using namespace std;

//Kernel that adds up two Arrays
__global__ void vector_initialization(float* a, float* b, float* sxy, float* sx,
    float* sy, float* sxs, float* sys, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

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
__global__ void squareErr(float* a, float* b, float* c, float coe, float ct, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
        c[index] = (((coe * a[index] + ct) - b[index]) * ((coe * a[index] + ct) - b[index]));
}

//Prototypes
float sumAll(float*, int);
void PrintBestFittingLine(float&, float&, float, float, float, float, int);
float predict(float, float&, float&);

// Driver code
int main(int argc, char** argv)
{
    // Number of pairs of (xi, yi)
    freopen("very_large_input.txt", "r", stdin); // freopen("large_input.txt", "r", stdin);
    int s;
    cin >> s;
    cout << s << endl;
    int n_bytes = s * sizeof(float);
    float coeff = 0, constTerm = 0;

    //Host Pointers
    float* h_a, * h_b, * sum_xy,
        * sum_x, * sum_y, * sum_x_square, * sum_y_square,
        * sqc;	//Add another pointer to hold array c

    //allocate memory for host size pointers
    h_a = (float*)malloc(n_bytes); h_b = (float*)malloc(n_bytes);
    sum_xy = (float*)malloc(n_bytes); sum_x = (float*)malloc(n_bytes);
    sum_y = (float*)malloc(n_bytes); sum_x_square = (float*)malloc(n_bytes);
    sum_y_square = (float*)malloc(n_bytes); sqc = (float*)malloc(n_bytes);

    for (int i = 0; i < s; i++) {
        // In a csv file all the values of
        // xi and yi are separated by commas
        char comma;
        cin >> h_a[i] >> comma >> h_b[i];
        sum_xy[i] = 0; sum_x[i] = 0; sum_y[i] = 0;
        sum_x_square[i] = 0; sum_y_square[i] = 0;
    }

    //Device Pointers
    float* d_a, * d_b, * sxy, * sx, * sy, * sxs, * sys, * sq;

    //allocate memory for device size pointers
    cudaMalloc(&d_a, n_bytes); cudaMalloc(&d_b, n_bytes);
    cudaMalloc(&sxy, n_bytes); cudaMalloc(&sx, n_bytes);
    cudaMalloc(&sy, n_bytes); cudaMalloc(&sxs, n_bytes);
    cudaMalloc(&sys, n_bytes); cudaMalloc(&sq, n_bytes);

    clock_t htod_start, htod_end;	//Initialize clock trackers for data transfer of htod (host to device)
    htod_start = clock();	//Time before Memory transfer from Host to Device
    //Memory transfer from Host to Device
    cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(sxy, sum_xy, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(sx, sum_x, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(sy, sum_y, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(sxs, sum_x_square, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(sys, sum_y_square, n_bytes, cudaMemcpyHostToDevice);
    htod_end = clock();		//Time after Memory transfer from Host to Device

    clock_t gpu_start, gpu_end;	//Initialize clock trackers for GPU execution
    gpu_start = clock();
    vector_initialization << < 100, s/100 >> > (d_a, d_b, sxy, sx, sy, sxs, sys, s);
    cudaDeviceSynchronize();
    gpu_end = clock();

    clock_t dtoh_start, dtoh_end;	//Initialize clock trackers for data transfer of dtoh (device to host)
    dtoh_start = clock();	//Time before Memory Tranfer back to host
    //Memory Tranfer back to host
    cudaMemcpy(sum_xy, sxy, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_x, sx, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_y, sy, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_x_square, sxs, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_y_square, sys, n_bytes, cudaMemcpyDeviceToHost);
    dtoh_end = clock();		//Time after Memory Tranfer back to host

    PrintBestFittingLine(coeff, constTerm, sumAll(sum_x_square, s), sumAll(sum_xy, s), sumAll(sum_x, s), sumAll(sum_y, s), s);

    clock_t htod_start2, htod_end2;	//Initialize clock trackers for data transfer of htod (host to device)
    htod_start2 = clock();	//Time before Memory transfer from Host to Device
    cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice); cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);
    htod_end2 = clock();		//Time after Memory transfer from Host to Device

    clock_t gpu_start2, gpu_end2;	//Initialize clock trackers for GPU execution
    gpu_start2 = clock();
    squareErr << <  100, s / 100 >> > (d_a, d_b, sq, coeff, constTerm, s);
    gpu_end2 = clock();

    clock_t dtoh_start2, dtoh_end2;	//Initialize clock trackers for data transfer of dtoh (device to host)
    dtoh_start2 = clock();	//Time before Memory Tranfer back to host
    cudaMemcpy(sqc, sq, n_bytes, cudaMemcpyDeviceToHost);
    dtoh_end2 = clock();		//Time after Memory Tranfer back to host

    cout << "The errorSquared  = " << sumAll(sqc, s) << endl;

    //For Parameter Inialization
    printf("Host to Device Memory Transfer Time for parameters: %4.6f \n",
        (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));

    printf("Time to calculate right values parameters for Linear Regression: %4.6f \n",
        (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));

    printf("Device to Host Memory Transfer Time for parameters: %4.6f \n",
        (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));

    //For MSE
    printf("Host to Device Memory Transfer Time for MSE: %4.6f \n",
        (double)((double)(htod_end2 - htod_start2) / CLOCKS_PER_SEC));

    printf("Time to calculate Mean Square Error: %4.6f \n",
        (double)((double)(gpu_end2 - gpu_start2) / CLOCKS_PER_SEC));

    printf("Device to Host Memory Transfer Time for MSE: %4.6f \n",
        (double)((double)(dtoh_end2 - dtoh_start2) / CLOCKS_PER_SEC));

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