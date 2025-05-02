//iterative algo un-optimized

#include <bits/stdc++.h>
#include <chrono>
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

signed main() {
    // cout<<A.size()<<"   "<<B.size()<<endl;
    auto start = high_resolution_clock::now();
    vector<int> c = multiply(A, B);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Execution Time with single thread: " << duration.count() << " ms" << endl;


    ofstream file("output_singleThread.txt");

    for (int num : c) {
        file << num << " "; 
    }
    
    file.close();
    return 0;
}

