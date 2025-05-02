//iterative algo optimized with openMp


#include <bits/stdc++.h>
#include <chrono>
#include <omp.h> // OpenMP header
#include "data.h"
using namespace std;
using namespace std::chrono;

using cd = complex<double>;
const double PI = acos(-1);


void bitReverse(vector<cd>& a) {
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
}



void fft_omp(vector<cd>& a, bool invert) {
    int n = a.size();
    bitReverse(a);

    vector<cd> twiddle(n / 2);
    #pragma omp parallel for
    for (int i = 0; i < n / 2; i++) {
        double ang = 2 * PI * i / n * (invert ? -1 : 1);
        twiddle[i] = cd(cos(ang), sin(ang));
    }

    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        int step = n / len;
        #pragma omp parallel for
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half; j++) {
                cd w = twiddle[j * step];
                cd u = a[i + j];
                cd v = a[i + j + half] * w;
                a[i + j] = u + v;
                a[i + j + half] = u - v;
            }
        }
    }

    if (invert) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            a[i] /= n;
        }
    }
}

vector<int> multiply_omp(const vector<int>& a, const vector<int>& b) {
    int n = 1;
    while (n < a.size() + b.size()) n <<= 1;

    vector<cd> fa(n, 0), fb(n, 0);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < a.size(); i++) fa[i] = a[i];
        }
        #pragma omp section
        {
            for (int i = 0; i < b.size(); i++) fb[i] = b[i];
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        fft_omp(fa, false);
        #pragma omp section
        fft_omp(fb, false);
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        fa[i] *= fb[i];
    }
    fft_omp(fa, true);

    vector<int> result(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        result[i] = round(fa[i].real());
    }

    return result;
}


signed main() {



    auto start = high_resolution_clock::now();
    vector<int> c = multiply_omp(A, B);
    auto stop = high_resolution_clock::now();
    auto duration_withOpenmp = duration_cast<milliseconds>(stop - start);
    cout << "Execution Time with OpenMP: " << duration_withOpenmp.count() << " ms" << endl;
        


    ofstream file("output_OpenMP.txt");

    for (int num : c) {
        file << num << " "; 
    }
    
    file.close();

    return 0;
}

