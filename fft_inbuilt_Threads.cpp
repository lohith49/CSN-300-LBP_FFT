//This is inbuilt fft function with threads
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "data_double.h"
#include <fftw3.h>
#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;




std::vector<double> multiply_polynomials(const std::vector<double>& a, const std::vector<double>& b) {
    int n = 1;
    while (n < a.size() + b.size()) n <<= 1;  
    
    fftw_complex* A = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex* B = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex* C = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        A[i][0] = (i < a.size()) ? a[i] : 0;
        A[i][1] = 0;
        B[i][0] = (i < b.size()) ? b[i] : 0;
        B[i][1] = 0;
    }

    fftw_plan pa = fftw_plan_dft_1d(n, A, A, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pb = fftw_plan_dft_1d(n, B, B, FFTW_FORWARD, FFTW_ESTIMATE);
    
    std::thread t1([&]() { fftw_execute(pa); });
    std::thread t2([&]() { fftw_execute(pb); });
    t1.join();
    t2.join();

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        C[i][0] = A[i][0] * B[i][0] - A[i][1] * B[i][1];
        C[i][1] = A[i][0] * B[i][1] + A[i][1] * B[i][0];
    }

    fftw_plan pc = fftw_plan_dft_1d(n, C, C, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(pc);
    std::vector<double> result(n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        result[i] = C[i][0] / n;
    }

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

    ofstream file("output_inbuilt_Threads.txt");

    for (double num : result) {
        file << num << " "; 
    }
    
    file.close();
    cout << endl;
}
