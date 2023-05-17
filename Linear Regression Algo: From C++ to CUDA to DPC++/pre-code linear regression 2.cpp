
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// C++ program to implement
#include <iostream>
#include <cstdio>
#include <vector>
using namespace std;

//Prototypes

void PrintBestFittingLine(float&, float&, vector<float>& , float&, float&, float&, float&);
void calculateCoefficient(float&, vector<float>&, float&, float&, float&, float&);
// Calculate the constant term of the best fitting line
void calculateConstantTerm(float&, vector<float>&, float&, float&, float&, float&);
float errorSquare(vector<float>& , vector<float>& , float& , float& );
float errorIn(float, vector<float>&, vector<float>&, float&, float&);
float predict(float,float&, float& );

// Driver code
int main(int argc, char** argv)
{
    //Variable Declaration
    vector<float> x;vector<float> y; //Dynamic Arrays: Contain all (i-th x) and all (i-th y)
    float coeff = 0;    // Store the coefficient/slope in the best fitting line
    float constTerm = 0;    // Store the constant term in the best fitting line
    float sum_xy = 0;   // Contains sum of product of all (i-th x) and (i-th y)
    float sum_x = 0;    // Contains sum of all (i-th x)
    float sum_y = 0;    // Contains sum of all (i-th y)
    float sum_x_square = 0; //Contains sum of square of all (i-th x)
    float sum_y_square = 0; // Contains sum of square of all (i-th y)

    freopen("input.txt", "r",stdin);

    // Number of pairs of (xi, yi)
    // in the dataset
    int size;
    cin >> size;

    for (int i = 0; i < size; i++) {
        // In a csv file all the values of
        // xi and yi are separated by commas
        char comma;
        float xi;
        float yi;
        cin >> xi >> comma >> yi;
        sum_xy += xi * yi;
        sum_x += xi;
        sum_y += yi;
        sum_x_square += xi * xi;
        sum_y_square += yi * yi;
        x.push_back(xi);
        y.push_back(yi);
    }

    PrintBestFittingLine(coeff, constTerm, x, sum_x_square, sum_xy, sum_x, sum_y);
    
    // Printing the best fitting line
    cout << "Predicted value at 2060 = " << predict(2060, coeff, constTerm) << endl;
    cout << "The errorSquared  = " << errorSquare(x, y, coeff, constTerm) << endl;
    cout << "Error in 2050 = " << errorIn(2050, x, y, coeff, constTerm) << endl;
}

// Function that calculate the coefficient slope of the best fitting line
void calculateCoefficient(float &coe, vector<float>& x, float &sxs, float &sxy,float &sx, float &sy)
{
    float N = x.size();
    float numerator = (N * sxy - sx * sy);
    float denominator = (N * sxs - sx * sx);
    coe = numerator / denominator;
}
// Calculate the constant term of the best fitting line
void calculateConstantTerm(float& ct, vector<float>& x, float& sxs, float& sxy, float& sx, float& sy){
    float N = x.size();
    float numerator = (sy * sxs - sx * sxy);
    float denominator = (N * sxs - sx * sx);
    ct = numerator / denominator;
}

// Function that print the best fitting line
void PrintBestFittingLine(float& coe, float& ct, vector<float>& x, float& sxs, float& sxy, float& sx, float& sy)
{
    if (coe == 0 && ct == 0) {
        calculateCoefficient(coe, x, sxs,sxy,sx,sy);
        calculateConstantTerm(ct,x,sxs,sxy,sx,sy);
    }
    cout << "The best fitting line is y = " << coe << "x + " << ct << endl;
}

// Returns overall sum of square of errors   
float errorSquare(vector<float>& x, vector<float>& y, float& coe, float& ct)
{
    float ans = 0;
    for (int i = 0;i < x.size(); i++) {
        ans += (((coe * x[i] + ct) - y[i])
            * ((coe * x[i] + ct) - y[i]));
    }
    return ans;
}
// Return the error i.e the difference between the actual value and value predicted by our model
float errorIn(float num, vector<float>& x, vector<float>& y, float& coe, float& ct)
{
    for (int i = 0; i < x.size(); i++) {
        if (num == x[i]) {
            return (y[i] - (coe * x[i] + ct));
        }
    }
    return 0;
}

float predict(float x, float& coe, float& ct)
{
    return coe * x + ct;
}