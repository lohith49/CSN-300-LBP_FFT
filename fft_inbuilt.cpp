#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "data_double.h"
#include <fftw3.h>
#include <fstream>
using namespace std;
using namespace std::chrono;

vector<double> multiply_polynomials(const vector<double>& a, const vector<double>& b) {
    int n = 1;
    while (n < a.size() + b.size()) n <<= 1;  // Next power of 2

    // Allocate FFTW complex arrays
    fftw_complex* A = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex* B = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex* C = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    // Initialize A and B
    for (int i = 0; i < n; ++i) {
        A[i][0] = (i < a.size()) ? a[i] : 0;
        A[i][1] = 0;
        B[i][0] = (i < b.size()) ? b[i] : 0;
        B[i][1] = 0;
    }

    // Create FFTW plans
    fftw_plan pa = fftw_plan_dft_1d(n, A, A, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pb = fftw_plan_dft_1d(n, B, B, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute forward FFTs
    fftw_execute(pa);
    fftw_execute(pb);

    // Point-wise multiplication
    for (int i = 0; i < n; ++i) {
        C[i][0] = A[i][0] * B[i][0] - A[i][1] * B[i][1];
        C[i][1] = A[i][0] * B[i][1] + A[i][1] * B[i][0];
    }

    // Inverse FFT
    fftw_plan pc = fftw_plan_dft_1d(n, C, C, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(pc);

    // Normalize and extract result
    vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = C[i][0] / n;
    }

    // Cleanup
    fftw_destroy_plan(pa);
    fftw_destroy_plan(pb);
    fftw_destroy_plan(pc);
    fftw_free(A);
    fftw_free(B);
    fftw_free(C);

    return result;
}

int main() {
    auto start = high_resolution_clock::now();
    vector<double> result = multiply_polynomials(A, B);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Execution Time: " << duration.count() << " ms" << endl;

    ofstream file("output_inbuilt.txt");
    for (double num : result) {
        file << num << " "; 
    }
    file.close();

    cout << endl;
}
