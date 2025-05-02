#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <fstream>
#include <thread>
#include <omp.h>

using namespace std;
using namespace std::chrono;
using cd = complex<double>;
const double PI = acos(-1);

//Without Thread

void bitReverse(vector<cd>& a) {
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
}

void fft(vector<cd>& a, bool invert) {
    int n = a.size();
    bitReverse(a);
    vector<cd> twiddle(n / 2);
    for (int i = 0; i < n / 2; i++) {
        double ang = 2 * PI * i / n * (invert ? -1 : 1);
        twiddle[i] = cd(cos(ang), sin(ang));
    }

    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        int step = n / len;
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
        for (auto& x : a) x /= n;
    }
}

vector<int> multiply(const vector<int>& a, const vector<int>& b) {
    int n = 1;
    while (n < a.size() + b.size()) n <<= 1;

    vector<cd> fa(n, 0), fb(n, 0);
    for (int i = 0; i < a.size(); i++) fa[i] = a[i];
    for (int i = 0; i < b.size(); i++) fb[i] = b[i];

    fft(fa, false);
    fft(fb, false);

    for (int i = 0; i < n; i++) {
        fa[i] *= fb[i];
    }

    fft(fa, true);

    vector<int> result(n);
    for (int i = 0; i < n; i++) {
        result[i] = round(fa[i].real());
    }
    return result;
}















//Openmp



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

















//Combination of Threads and openmp


void fft_threads(vector<cd>& a, bool invert) {
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
        #pragma omp parallel for
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
        #pragma omp parallel for
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
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        result[i] = round(fa[i].real());
    }
    return result;
}










// Benchmarking
void generate_test_data(int size, vector<int>& a, vector<int>& b) {
    a.resize(size);
    b.resize(size);
    for (int i = 0; i < size; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }
}

int main() {
    vector<int> sizes = {64, 256, 1024, 4096, 16384, 65536, 123443,1000000, 10000000};
    
    ofstream csv("results.csv");
    csv << "Size,Single_Thread,OpenMp,OpenMpWithThreads\n";
    
    for (int size : sizes) {
        vector<int> a, b;
        generate_test_data(size, a, b);
        
        // Warm-up runs
        // multiply_radix2_threads(a, b);
        // multiply_radix4(a, b);
        // multiply_radix4_threads(a, b);
        
        // Radix-2 timing
        auto start = high_resolution_clock::now();
        auto result1 = multiply(a, b);
        auto end = high_resolution_clock::now();
        double t1 = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        // Radix-4 Sequential timing
        start = high_resolution_clock::now();
        auto result2 = multiply_omp(a, b);
        end = high_resolution_clock::now();
        double t2 = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        // Radix-4 Parallel timing
        start = high_resolution_clock::now();
        auto result3 = multiply_threads(a, b);
        end = high_resolution_clock::now();
        double t3 = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        csv << size << "," << t1 << "," << t2 << "," << t3 << "\n";
        cout << "Size " << size << ": R2=" << t1 << "ms, R4S=" << t2 << "ms, R4P=" << t3 << "ms\n";
    }
    
    csv.close();
    
    // Verify correctness
    vector<int> a, b;
    generate_test_data(64, a, b);
    auto r1 = multiply(a, b);
    auto r2 = multiply_omp(a, b);
    auto r3 = multiply_threads(a, b);
    bool correct = true;
    for (int i = 0; i < r1.size(); i++) {
        if(r1[i]!= r2[i] || r2[i]!=r3[i]){
            cout<<"Mismatch at: "<<i<<" "<<r1[i]<<" "<<r2[i]<<" "<<r3[i]<<endl;
            correct=false;
        }
    }
    cout << "Results match: " << (correct ? "Yes" : "No") << endl;
    
    return 0;
}
