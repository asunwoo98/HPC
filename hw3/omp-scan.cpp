#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  long step;
  long threads;
  prefix_sum[0] = 0;
  #pragma omp parallel
  {
    threads = omp_get_num_threads();
    step = n/threads;
    if (n % threads != 0) step = step + 1;
    long tid = omp_get_thread_num();
    if (tid != 0) prefix_sum[tid*step] = A[tid*step-1];

    for (long y = tid*step+1; y< (tid+1)*step; y++) {
      if(y<n)  prefix_sum[y] = prefix_sum[y-1] + A[y-1];
    }
  }

  //sequential correction
  for (long i = 1; i< threads; i++) {
    for (long j = i*step; j< (i+1)*step; j++) {
      if(j<n) prefix_sum[j] = prefix_sum[i*step-1] + prefix_sum[j];
    }
  }
}

int main(int argc, char** argv) {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  const int threads = read_option<int>("-t", argc, argv, "4");
  #if defined(_OPENMP)
  omp_set_num_threads(threads);
  #endif

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);
  free(A);
  free(B0);
  free(B1);
  return 0;
}
