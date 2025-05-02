# FFT Implementations in C++

This repository contains multiple C++ implementations of the Fast Fourier Transform (FFT), showcasing various optimization strategies and parallelism models such as manual threading, OpenMP, and usage of the FFTW3 library.

---

## Prerequisites

Make sure your system has the following:

- A C++ compiler (g++ with OpenMP support)
- FFTW3 library installed (for the inbuilt FFT versions)

To install FFTW3 on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install libfftw3-dev
```

---

## Compilation and Execution

Below are the commands to compile and run each FFT implementation.

---

### 1. Single-threaded FFT

**Files Used:** fft_singleThread.cpp, data.cpp

```bash
g++ fft_singleThread.cpp data.cpp -o fft_single
./fft_single
```

---

### 2. Multithreaded FFT (Manual Threading)

**Files Used:** fft_multiThread.cpp, data.cpp

```bash
g++ fft_multiThread.cpp data.cpp -o fft_multiple
./fft_multiple
```

---

### 3. FFT Using OpenMP

**Files Used:** fft_openmp.cpp, data.cpp

```bash
g++ fft_openmp.cpp data.cpp -o fft_openmp -fopenmp
./fft_openmp
```

---

### 4. FFT Using FFTW3 (Inbuilt Library)

**Files Used:** fft_inbuilt.cpp, data_double.cpp

```bash
g++ fft_inbuilt.cpp data_double.cpp -o fft_in -lfftw3
./fft_in
```

---

### 5. FFT Using FFTW3 with OpenMP

**Files Used:** fft_inbuilt_Threads.cpp, data_double.cpp

```bash
g++ fft_inbuilt_Threads.cpp data_double.cpp -o fft_in_Threads -fopenmp -lfftw3
./fft_in_Threads
```

---

### 6. Radix-4 FFT

**Files Used:** fft_radix4.cpp, data.cpp

```bash
g++ fft_radix4.cpp data.cpp -o fft_r4
./fft_r4
```

---

### 7. FFT with Threads and OpenMP

**Files Used:** fft_Thread_openmp.cpp, data.cpp

```bash
g++ fft_Thread_openmp.cpp data.cpp -o fft_open_Thread -fopenmp
./fft_open_Thread
```

---

### 8. Optimized FFT Build (AVX2, FMA, O3)

**Files Used:** fft_Thread_openmp.cpp, data.cpp

```bash
g++ fft_Thread_openmp.cpp data.cpp -o fft_t -mavx2 -mfma -O3
./fft_t
```
