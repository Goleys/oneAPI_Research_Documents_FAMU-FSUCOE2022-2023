//If you run this code in a regular cpp compiler, eliminate the CUDA .h files
//If you run this code in a CUDA suported environment, keep the CUDA .h files
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
// C++ program to implement
#include <iostream>
#include <cstdio>
#include <vector>
#include <ctime>
#include <time.h>

using namespace std;

//Kernel that adds up two Arrays
void vector_initialization(float* a, float* b, float* sxy, float* sx,
    float* sy, float* sxs, float* sys, int size, sycl::nd_item<3> item_ct1)
{
    int index = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);

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
void squareErr(float* a, float* b, float* c, float coe, float ct, int size,
               sycl::nd_item<3> item_ct1)
{
    int index = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
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
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
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
    d_a = sycl::malloc_device<float>(s, q_ct1);
    d_b = sycl::malloc_device<float>(s, q_ct1);
    sxy = sycl::malloc_device<float>(s, q_ct1);
    sx = sycl::malloc_device<float>(s, q_ct1);
    sy = sycl::malloc_device<float>(s, q_ct1);
    sxs = sycl::malloc_device<float>(s, q_ct1);
    sys = sycl::malloc_device<float>(s, q_ct1);
    sq = sycl::malloc_device<float>(s, q_ct1);


    clock_t htod_start, htod_end;	//Initialize clock trackers for data transfer of htod (host to device)
    /*
    DPCT1008:0: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    htod_start = clock(); // Time before Memory transfer from Host to Device
    //Memory transfer from Host to Device
    q_ct1.memcpy(d_a, h_a, n_bytes);
    q_ct1.memcpy(d_b, h_b, n_bytes);
    q_ct1.memcpy(sxy, sum_xy, n_bytes);
    q_ct1.memcpy(sx, sum_x, n_bytes);
    q_ct1.memcpy(sy, sum_y, n_bytes);
    q_ct1.memcpy(sxs, sum_x_square, n_bytes);
    q_ct1.memcpy(sys, sum_y_square, n_bytes).wait();
    /*
    DPCT1008:1: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    htod_end = clock(); // Time after Memory transfer from Host to Device

    clock_t gpu_start, gpu_end;	//Initialize clock trackers for GPU execution
    /*
    DPCT1008:2: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    gpu_start = clock();
    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 100) *
                                             sycl::range<3>(1, 1, s / 100),
                                         sycl::range<3>(1, 1, s / 100)),
                       [=](sycl::nd_item<3> item_ct1) {
                           vector_initialization(d_a, d_b, sxy, sx, sy, sxs,
                                                 sys, s, item_ct1);
                       });
    dev_ct1.queues_wait_and_throw();
    /*
    DPCT1008:4: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    gpu_end = clock();

    clock_t dtoh_start, dtoh_end;	//Initialize clock trackers for data transfer of dtoh (device to host)
    /*
    DPCT1008:5: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    dtoh_start = clock(); // Time before Memory Tranfer back to host
    //Memory Tranfer back to host
    q_ct1.memcpy(sum_xy, sxy, n_bytes);
    q_ct1.memcpy(sum_x, sx, n_bytes);
    q_ct1.memcpy(sum_y, sy, n_bytes);
    q_ct1.memcpy(sum_x_square, sxs, n_bytes);
    q_ct1.memcpy(sum_y_square, sys, n_bytes).wait();
    /*
    DPCT1008:6: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    dtoh_end = clock(); // Time after Memory Tranfer back to host

    PrintBestFittingLine(coeff, constTerm, sumAll(sum_x_square, s), sumAll(sum_xy, s), sumAll(sum_x, s), sumAll(sum_y, s), s);

    clock_t htod_start2, htod_end2;	//Initialize clock trackers for data transfer of htod (host to device)
    /*
    DPCT1008:7: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    htod_start2 = clock(); // Time before Memory transfer from Host to Device
    q_ct1.memcpy(d_a, h_a, n_bytes); q_ct1.memcpy(d_b, h_b, n_bytes).wait();
    /*
    DPCT1008:8: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    htod_end2 = clock(); // Time after Memory transfer from Host to Device

    clock_t gpu_start2, gpu_end2;	//Initialize clock trackers for GPU execution
    /*
    DPCT1008:9: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    gpu_start2 = clock();
    /*
    DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 100) *
                                             sycl::range<3>(1, 1, s / 100),
                                         sycl::range<3>(1, 1, s / 100)),
                       [=](sycl::nd_item<3> item_ct1) {
                           squareErr(d_a, d_b, sq, coeff, constTerm, s,
                                     item_ct1);
                       });
    /*
    DPCT1008:11: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    gpu_end2 = clock();

    clock_t dtoh_start2, dtoh_end2;	//Initialize clock trackers for data transfer of dtoh (device to host)
    /*
    DPCT1008:12: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    dtoh_start2 = clock(); // Time before Memory Tranfer back to host
    q_ct1.memcpy(sqc, sq, n_bytes).wait();
    /*
    DPCT1008:13: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    dtoh_end2 = clock(); // Time after Memory Tranfer back to host

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