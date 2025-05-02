//iterative algo optimized with threads

#include <bits/stdc++.h>
#include <chrono>
#include <thread>
#include "data.h"
#include <omp.h> 
#include <immintrin.h>
#include <cufft.h>







#include <vector>
#include <complex>
#include <thread>
#include <future>
#include <cmath>
#include <memory>
#include <atomic>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <algorithm>


using namespace std;
using namespace std::chrono;

using cd = complex<double>;
const double PI = acos(-1);



// Bit reversal for radix-4
// void bitReverseRadix4(vector<cd>& a) {
//     int n = a.size();
//     int bits = 0;
//     while ((1 << bits) < n) bits++;
    
//     // Ensure n is a power of 4
//     if (bits % 2 != 0) {
//         throw runtime_error("Size must be a power of 4 for radix-4 FFT");
//     }
    
//     vector<int> rev(n, 0);
//     for (int i = 0; i < n; i++) {
//         int val = 0;
//         for (int j = 0; j < bits; j += 2) {
//             val <<= 2;
//             val |= (i >> j) & 3;
//         }
//         rev[i] = val;
//     }
    
//     for (int i = 0; i < n; i++) {
//         if (i < rev[i]) {
//             swap(a[i], a[rev[i]]);
//         }
//     }
// }

// void fftRadix4(vector<cd>& a, bool invert) {
//     int n = a.size();
    
//     // Check if n is a power of 4
//     int log4n = 0;
//     int temp = n;
//     while (temp > 1) {
//         temp >>= 2;
//         log4n++;
//     }
    
//     if ((1 << (2 * log4n)) != n) {
//         throw runtime_error("Size must be a power of 4 for radix-4 FFT");
//     }
    
//     bitReverseRadix4(a);
    
//     // Precompute twiddle factors
//     vector<vector<cd>> twiddle(log4n);
//     for (int step = 0; step < log4n; step++) {
//         int len = 1 << (2 * (step + 1));
//         twiddle[step].resize(len);
//         for (int k = 0; k < len; k++) {
//             double ang = 2 * PI * k / len * (invert ? -1 : 1);
//             twiddle[step][k] = cd(cos(ang), sin(ang));
//         }
//     }
    
//     for (int step = 0; step < log4n; step++) {
//         int len = 1 << (2 * (step + 1));
//         int quarter = len / 4;
        
//         for (int i = 0; i < n; i += len) {
//             for (int j = 0; j < quarter; j++) {
//                 cd w0 = twiddle[step][0 * j];
//                 cd w1 = twiddle[step][1 * j];
//                 cd w2 = twiddle[step][2 * j];
//                 cd w3 = twiddle[step][3 * j];
                
//                 cd a0 = a[i + j];
//                 cd a1 = a[i + j + quarter] * w1;
//                 cd a2 = a[i + j + 2 * quarter] * w2;
//                 cd a3 = a[i + j + 3 * quarter] * w3;
                
//                 cd t0 = a0 + a2;
//                 cd t1 = a0 - a2;
//                 cd t2 = a1 + a3;
//                 cd t3 = cd(0, 1) * cd(invert ? -1.0 : 1.0) * (a1 - a3);
                
//                 a[i + j] = t0 + t2;
//                 a[i + j + quarter] = t1 + t3;
//                 a[i + j + 2 * quarter] = t0 - t2;
//                 a[i + j + 3 * quarter] = t1 - t3;
//             }
//         }
//     }
    
//     if (invert) {
//         for (auto& x : a) x /= n;
//     }
// }

// vector<int> multiplyRadix4(const vector<int>& a, const vector<int>& b) {
//     int n = 1;
//     while (n < a.size() + b.size()) n <<= 1;
    
//     // Make n a power of 4 by padding if necessary
//     while (n & (n - 1)) n += n & -n;
//     if (n & (n >> 1)) n <<= 1;
    
//     vector<cd> fa(n, 0), fb(n, 0);
//     for (int i = 0; i < a.size(); i++) fa[i] = a[i];
//     for (int i = 0; i < b.size(); i++) fb[i] = b[i];
    
//     fftRadix4(fa, false);
//     fftRadix4(fb, false);
    
//     for (int i = 0; i < n; i++) {
//         fa[i] *= fb[i];
//     }
    
//     fftRadix4(fa, true);
    
//     vector<int> result(n);
//     for (int i = 0; i < n; i++) {
//         result[i] = round(fa[i].real());
//     }
    
//     // Trim trailing zeros
//     while (!result.empty() && result.back() == 0) result.pop_back();
    
//     return result;
// }
































void bitReverseRadix4_threads(vector<cd>& a) {
    int n = a.size();
    int bits = 0;
    while ((1 << bits) < n) bits++;
    
    // Ensure n is a power of 4
    if (bits % 2 != 0) {
        throw runtime_error("Size must be a power of 4 for radix-4 FFT");
    }
    
    vector<int> rev(n, 0);
    for (int i = 0; i < n; i++) {
        int val = 0;
        for (int j = 0; j < bits; j += 2) {
            val <<= 2;
            val |= (i >> j) & 3;
        }
        rev[i] = val;
    }
    
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            swap(a[i], a[rev[i]]);
        }
    }
}

void fftRadix4_threads(vector<cd>& a, bool invert) {
    int n = a.size();
    
    int log4n = 0;
    int temp = n;
    while (temp > 1) {
        temp >>= 2;
        log4n++;
    }
    
    if ((1 << (2 * log4n)) != n) {
        throw runtime_error("Size must be a power of 4 for radix-4 FFT");
    }
    
    bitReverseRadix4_threads(a);
    vector<vector<cd>> twiddle(log4n);
    for (int step = 0; step < log4n; step++) {
        int len = 1 << (2 * (step + 1));
        twiddle[step].resize(len);
        for (int k = 0; k < len; k++) {
            double ang = 2 * PI * k / len * (invert ? -1 : 1);
            twiddle[step][k] = cd(cos(ang), sin(ang));
        }
    }
    
    for (int step = 0; step < log4n; step++) {
        int len = 1 << (2 * (step + 1));
        int quarter = len / 4;
        
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < quarter; j++) {
                cd w0 = twiddle[step][0 * j];
                cd w1 = twiddle[step][1 * j];
                cd w2 = twiddle[step][2 * j];
                cd w3 = twiddle[step][3 * j];
                
                cd a0 = a[i + j];
                cd a1 = a[i + j + quarter] * w1;
                cd a2 = a[i + j + 2 * quarter] * w2;
                cd a3 = a[i + j + 3 * quarter] * w3;
                
                cd t0 = a0 + a2;
                cd t1 = a0 - a2;
                cd t2 = a1 + a3;
                cd t3 = cd(0, 1) * cd(invert ? -1.0 : 1.0) * (a1 - a3);
                
                a[i + j] = t0 + t2;
                a[i + j + quarter] = t1 + t3;
                a[i + j + 2 * quarter] = t0 - t2;
                a[i + j + 3 * quarter] = t1 - t3;
            }
        }
    }
    
    if (invert) {
        for (auto& x : a) x /= n;
    }
}

void parallel_multiply_radix4(vector<cd>& fa, const vector<cd>& fb, int n) {
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

vector<int> multiplyRadix4_threads(const vector<int>& a, const vector<int>& b) {
    int n = 1;
    while (n < a.size() + b.size()) n <<= 1;
    
    // Make n a power of 4 by padding if necessary
    while (n & (n - 1)) n += n & -n;
    if (n & (n >> 1)) n <<= 1;
    
    vector<cd> fa(n, 0), fb(n, 0);
    for (int i = 0; i < a.size(); i++) fa[i] = a[i];
    for (int i = 0; i < b.size(); i++) fb[i] = b[i];

    
    thread t1(fftRadix4_threads, ref(fa), false);
    thread t2(fftRadix4_threads, ref(fb), false);
    t1.join();
    t2.join();
    
    parallel_multiply_radix4(fa, fb, n);
    
    fftRadix4_threads(fa, true);
    
    vector<int> result(n);
    for (int i = 0; i < n; i++) {
        result[i] = round(fa[i].real());
    }
    \
    while (!result.empty() && result.back() == 0) result.pop_back();
    
    return result;
}





signed main() {

    auto start = high_resolution_clock::now();
    vector<int> c = multiplyRadix4_threads(A, B);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Execution Time of radix4 with threads: " << duration.count() << " ms" << endl;





    ofstream file("output_radix4.txt");

    for (int num : c) {
        file << num << " "; 
    }

    file.close();
    return 0;
}
