//iterative algo optimized with threads

#include <bits/stdc++.h>
#include <chrono>
#include <thread>
#include "data.h"
#include <omp.h> 
#include <immintrin.h>
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

void fft_threads(vector<cd>& a, bool invert) {
    int n = a.size();
    bitReverse(a);

    vector<cd> twiddle(n / 2);
    for (int i = 0; i < n / 2; i++) {
        double ang = 2 * PI * i / n * (invert ? -1 : 1);
        twiddle[i] = cd(cos(ang), sin(ang));
    }

    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half; j++) {
                cd w = twiddle[j * (n / len)];
                cd u = a[i + j];
                cd v = a[i + j + half] * w;
                a[i + j] = u + v;
                a[i + j + half] = u - v;
            }
        }
    }

    if (invert) {
        for (auto& x : a) x /= n;
    }
}



void parallel_multiply_threads(vector<cd>& fa, const vector<cd>& fb, int n) {
    int num_threads = thread::hardware_concurrency();
    if (num_threads < 2) num_threads = 2;
    int chunk_size = n / num_threads;

    vector<thread> threads;
    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? n : start + chunk_size;
        threads.emplace_back([&fa, &fb, start, end]() {
            for (int i = start; i < end; i++) {
                fa[i] *= fb[i];
            }
        });
    }
    for (auto& t : threads) t.join();
}





vector<int> multiply_threads(const vector<int>& a, const vector<int>& b) {
    int n = 1;
    while (n < a.size() + b.size()) n <<= 1;

    vector<cd> fa(n, 0), fb(n, 0);
    for (int i = 0; i < a.size(); i++) fa[i] = a[i];
    for (int i = 0; i < b.size(); i++) fb[i] = b[i];

    thread t1(fft_threads, ref(fa), false);
    thread t2(fft_threads, ref(fb), false);
    t1.join();
    t2.join();

    parallel_multiply_threads(fa, fb, n);

    fft_threads(fa, true);

    vector<int> result(n);
    for (int i = 0; i < n; i++) {
        result[i] = round(fa[i].real());
    }
    return result;
}

signed main() {

    auto start = high_resolution_clock::now();
    vector<int> c = multiply_threads(A, B);
    auto stop = high_resolution_clock::now();
    auto duration_withThreads = duration_cast<milliseconds>(stop - start);
    cout << "Execution Time with threads: " << duration_withThreads.count() << " ms" << endl;
        


    ofstream file("output_multiThread.txt");

    for (int num : c) {
        file << num << " "; 
    }
    
    file.close();



    return 0;
}

