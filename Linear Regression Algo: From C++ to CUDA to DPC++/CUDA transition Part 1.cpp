//If you run this code in a regular cpp compiler, eliminate the CUDA .h files
//If you run this code in a CUDA suported environment, keep the CUDA .h files
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// C++ program to implement
#include <iostream>
#include <cstdio>
#include <vector>
using namespace std;

//Prototypes
void PrintBestFittingLine(float&, float&, vector<float>&, float&, float&, float&, float&);
void squareErr(vector<float>, vector<float>, float, float);
float predict(float, float&, float&);

// Driver code
int main(int argc, char** argv)
{
    //Initialize clock trackers for CPU execution
    clock_t cpu_start_in, cpu_start_pro, cpu_start_cal, cpu_start_pre, cpu_start_sq,
        cpu_end_in, cpu_end_pro, cpu_end_cal, cpu_end_pre, cpu_end_sq;
    //Variable Declaration
    vector<float> x; vector<float> y; //Dynamic Arrays: Contain all (i-th x) and all (i-th y)
    float coeff = 0;    // Store the coefficient/slope in the best fitting line
    float constTerm = 0;    // Store the constant term in the best fitting line
    float sum_xy = 0;   // Contains sum of product of all (i-th x) and (i-th y)
    float sum_x = 0;    // Contains sum of all (i-th x)
    float sum_y = 0;    // Contains sum of all (i-th y)
    float sum_x_square = 0; //Contains sum of square of all (i-th x)
    float sum_y_square = 0; // Contains sum of square of all (i-th y)

    float xi[200], float yi[200];

    freopen("large_input.txt", "r", stdin);

    // Number of pairs of (xi, yi)
    // in the dataset
    int size;
    cin >> size;

    cpu_start_in = clock();
    for (int i = 0; i < size; i++) {
        // In a csv file all the values of
        // xi and yi are separated by commas
        char comma;
        cin >> xi[i] >> comma >> yi[i];
        //cout << "Values for x and y " << xi[i] << " " << yi[i] << "\n";
    }
    cpu_end_in = clock();

    cpu_start_pro = clock();
    for (int i = 0; i < size; i++) {
        // In a csv file all the values of
        // xi and yi are separated by commas
        sum_xy += xi[i] * yi[i];
        sum_x += xi[i];
        sum_y += yi[i];
        sum_x_square += xi[i] * xi[i];
        sum_y_square += yi[i] * yi[i];
        x.push_back(xi[i]);
        y.push_back(yi[i]);
    }
    cpu_end_pro = clock();

    cpu_start_cal = clock();
    PrintBestFittingLine(coeff, constTerm, x, sum_x_square, sum_xy, sum_x, sum_y);
    cpu_end_cal = clock();

    cpu_start_pre = clock();
    cout << "Predicted value at 2060 = " << predict(2060, coeff, constTerm) << endl;
    cpu_end_pre = clock();
    // Printing the best fitting line
    cpu_start_sq = clock();
    squareErr(x, y, coeff, constTerm);
    cpu_end_sq = clock();

    //Displaying the times to complete each block of instruction.
    /*
    cout << "Time it took to Transfer x and y values from file:"
         << static_cast<double>(static_cast<double>(cpu_end_in - cpu_start_in) / CLOCKS_PER_SEC) << "\n";
    cout << "Time it took to Process Linear Regression Variables :"
         << static_cast<double>(static_cast<double>(cpu_end_pro - cpu_start_pro) / CLOCKS_PER_SEC) << "\n";
    cout << "Time it took to Predict the Best Fitting Line:"
         << static_cast<double>(static_cast<double>(cpu_end_cal - cpu_start_cal) / CLOCKS_PER_SEC) << "\n";
    cout << "Time it took to Predict a value from a new number:"
         << static_cast<double>(static_cast<double>(cpu_end_pre - cpu_start_pre) / CLOCKS_PER_SEC) << "\n";
    cout << "Time it took to calculate the Square Error:"
         << static_cast<double>(static_cast<double>(cpu_end_sq - cpu_start_sq) / CLOCKS_PER_SEC) << "\n";*/
    printf("Time it took to Transfer x and y values from file: %4.6f \n",
        (double)((double)(cpu_end_in - cpu_start_in) / CLOCKS_PER_SEC));
    printf("Time it took to Process Linear Regression Variables : %4.6f \n",
        (double)((double)(cpu_end_pro - cpu_start_pro) / CLOCKS_PER_SEC));
    printf("Time it took to Predict the Best Fitting Line: %4.6f \n",
        (double)((double)(cpu_end_cal - cpu_start_cal) / CLOCKS_PER_SEC));
    printf("Time it took to Predict a value from a new number: %4.6f \n",
        (double)((double)(cpu_end_pre - cpu_start_pre) / CLOCKS_PER_SEC));
    printf("Time it took to calculate the Square Error: %4.6f \n",
        (double)((double)(cpu_end_sq - cpu_start_sq) / CLOCKS_PER_SEC));
}

// Function that print the best fitting line
void PrintBestFittingLine(float& coe, float& ct, vector<float>& x, float& sxs, float& sxy, float& sx, float& sy)
{
    if (coe == 0 && ct == 0) {
        coe = (x.size() * sxy - sx * sy) / (x.size() * sxs - sx * sx); // Calculate the coefficient slope of the best fitting line
        ct = (sy * sxs - sx * sxy) / (x.size() * sxs - sx * sx);  // Calculate the constant term of the best fitting line
    }
    cout << "The best fitting line is y = " << coe << "x + " << ct << endl;
}

//display square error, and error of a certain number inside the vector x
void squareErr(vector<float> x, vector<float> y, float coe, float ct)
{
    float se = 0;   //Square error
    for (int i = 0; i < x.size(); i++) {
        se += (((coe * x[i] + ct) - y[i]) * ((coe * x[i] + ct) - y[i]));
    }
    cout << "The errorSquared  = " << se << endl;
}
//Predicts the respective y value for an specific x value
float predict(float x, float& coe, float& ct)
{
    return coe * x + ct;
}