// adapted from gpu03.cu
// $ nvcc -arch=sm_61 gpu03.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
//k-means algorithm:
//  initialization: pick k random data points as start points
//  1. calculate distance to centroids for every point
//  2. pick the minimum distance
//  3. calculate new centroid
//  terminate conditions: iterations/centroids dont move
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string>

double L2 (double* v1, double* v2, long len) {
  double sum = 0;
  for (int i = 0; i < len; i++){
    sum += (v1-v2) * (v1-v2)
  }
}

double L2 (long* v1, long* v2, long len) {
  double sum = 0;
  for (int i = 0; i < len; i++){
    sum += (v1-v2) * (v1-v2)
  }
}
//initiate centroids as unique random data points from input
void init_random(double* c1, double* c2, double* c1_ref, double* c2_ref,const double* x1, const double* x2, long K, long N){
  long* temp = (long*) malloc (k * sizeof(long));
  for (int i = 0; i < K; i++) {
    boolean nodupe;
    do {
      nodupe = false;
      long new = rand() % N;
      temp[i] = new;
      for (int j = 0; j < i) {
        if (temp[j] == new) nodupe = true;
      }
    } while(nodupe);
    c1[i] = x1[temp[i]];
    c2[i] = x2[temp[i]];
    c1_ref[i] = x1[temp[i]];
    c2_ref[i] = x2[temp[i]];
  }
  free(temp);
}

void expectation (double* c1, double* c2, const double* x1, const double* x2, const long* z, const double* dist, long K, long N){
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < K; j++) {
      dist[j] = (x1[i]-c1[j]) * (x1[i]-c1[j]) + (x2[i]-c2[j]) * (x2[i]-c2[j]);
    }

    z[i] = 0;
    for (long j = 1; j < K; j++) {
      if (dist[j] < dist[z[i]]) z[i] = j;
    }
  }
}

void maximization (double* c1, double* c2, const double* x1, const double* x2, const long* z, long K, long N){
  double* count = (double*) malloc(K * sizeof(double));
  for (long i = 0; i < K; i++){
    c1[i] = 0;
    c2[i] = 0;
    count[i] = 0;
  }
  for (long i = 0; i < N; i++){
    c1[z[i]] += x1[i];
    c2[z[i]] += x2[i];
    count[z[i]] += 1;
  }

  for (long i = 0; i < K; i++){
    c1[i] /= count[i];
    c2[i] /= count[i];
  }
  free(count);
}

__global__
void kmeans_kernel(double* c, const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //__shared__ double temp;
  //temp = 0;
  for (long i = 0; i < N; i++) {
    //if (idx < N) c[i] += a[idx] * b[i*N+idx]; //this version produces errors.
    if (idx < N) c[idx] += a[i] * b[idx*N+i];
  }
  //c[idx] = temp;
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main(int argc, char** argv) {
  //long N = (1UL<<25); // 2^25
  long N = read_option<int>("-n", argc, argv, "5000");
  long K = read_option<long>("-k", argc, argv, "15");
  long max_iter = read_option<long>("-itr", argc, argv, "1000");
  double tol = read_option<double>("-tol", argc, argv, "1e-5");
  double* x1 = (double*) malloc(N * sizeof(double));
  double* x2 = (double*) malloc(N * sizeof(double));
  double* c1 = (double*) malloc(K * sizeof(double));
  double* c2 = (double*) malloc(K * sizeof(double));
  double* c1_prev = (double*) malloc(K * sizeof(double));
  double* c2_prev = (double*) malloc(K * sizeof(double));
  double* c1_ref = (double*) malloc(K * sizeof(double));
  double* c2_ref = (double*) malloc(K * sizeof(double));
  double* dist = (double*) malloc(K * sizeof(double));
  long* z = (long*) malloc(N * sizeof(long));
  long* z_prev = (long*) malloc(N * sizeof(long));
  long* z_ref = (long*) malloc(N * sizeof(long));

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x1[i] = i+2;
    x2[i] = i+1;
    z[i] = 0;
    z_ref[i] = 0;
    for (long j = 0; j < K; j++) {

    }
  }

  //ifstream data ("s1.txt");
  // if(data.is_open()){
  //   for(int i = 0 ; i < N; i++){
  //     String line, dim1, dim2;
  //     getline(data,line);
  //     line >> dim1 >> dim2;
  //     x1[i] = stod(dim1);
  //     x2[i] = stod(dim2);
  //   }
  //   data.close();
  // }
  // else printf("Unable to open data file.\n", );
  FILE* data;
  data = fopen("s1.txt","r");
  for(int i = 0 ; i < N; i++){
    fscanf(data, "%f", x1[i]);
    fscanf(data, "%f", x2[i]);
  }
  fclose(data);
  }


  //initialize cluster centers
  init_random (c1, c2, c1_ref, c2_ref, x1, x2, K, N);

  double tt = omp_get_wtime();
  //printf("running inner_prod\n");
  //means(z_ref, x, y, N);
  for (int i = 0; i < max_iter; i++) {
    expectation(c1_ref, c2_ref, x1, x2, z_ref, dist, K, N);
    maximization(c1_ref, c2_ref, x1, x2, z_ref, K, N);

    if (L2(z_prev, z, N) < tol && L2(c1_prev,c1, K)+L2(c2_prev,c2, K) < tol){
      printf("Done at iteration %d\n", i);
      break;
    }
    copy (z_ref, z_ref + N, z_prev);
    copy (c1_ref, c1_ref + N, c1_prev);
    copy (c2_ref, c2_ref + N, c2_prev);
  }
  //printf("exiting inner_prod\n");
  //printf("CPU Bandwidth = %f GB/s\n", 3/1e9/ (omp_get_wtime()-tt)*N*N*sizeof(double)) ;
  printf("Time taken = %f s\n", omp_get_wtime()-tt) ;



  double *x_d, *y_d, *z_d;
  cudaMalloc(&x_d, N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&y_d, N*N*sizeof(double));
  Check_CUDA_Error("malloc y failed");
  cudaMalloc(&z_d, N*sizeof(double));
  Check_CUDA_Error("malloc z failed");

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  //Check_CUDA_Error("memcpy x failed");
  cudaMemcpy(y_d, y, N*N*sizeof(double), cudaMemcpyHostToDevice);
  //Check_CUDA_Error("memcpy y failed");
  kmeans_kernel<<<N/1024,1024>>>(z_d, x_d, y_d, N);
  //Check_CUDA_Error("kernel");
  cudaDeviceSynchronize();
  cudaMemcpy(z, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);
  printf("GPU Bandwidth = %f GB/s\n", 3/1e9/ (omp_get_wtime()-tt)*N*N*sizeof(double)) ;

  double err = 0;
  for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
  printf("Error = %f\n", err);

  //printf("%f,  %f", z_ref[0], z_ref[1]);
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);

  free(x);
  free(y);
  free(z);
  free(z_ref);

  return 0;
}
