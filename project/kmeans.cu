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
#include <time.h>
#include "utils.h"


double L2 (double* v1, double* v2, long len) {
  double sum = 0;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < len; i++){
    sum += (v1[i]-v2[i]) * (v1[i]-v2[i]);
  }
  return sum;
}

double L2 (int* v1, int* v2, long len) {
  double sum = 0;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < len; i++){
    sum += (v1[i]-v2[i]) * (v1[i]-v2[i]);
  }
  return sum;
}
//initiate centroids as unique random data points from input
void init_random(double* c1, double* c2, double* c1_ref, double* c2_ref, const double* x1, const double* x2, long K, long N){
   srand(645135);
  //srand(time(NULL));
  long* temp = (long*) malloc (K * sizeof(long));
  for (int i = 0; i < K; i++) {
    bool dupe;
    do {
      dupe = false;
      long newc = rand() % N;
      temp[i] = newc;
      for (int j = 0; j < i; j++) {
        if (temp[j] == newc) dupe = true;
      }
    } while(dupe);
    c1[i] = x1[temp[i]];
    c2[i] = x2[temp[i]];
    c1_ref[i] = x1[temp[i]];
    c2_ref[i] = x2[temp[i]];
  }
  free(temp);
}

void expectation (double* c1, double* c2, const double* x1, const double* x2, int* z, long K, long N){
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    double bestdist, currdist;
    bestdist = DBL_MAX;
    for (long j = 0; j < K; j++) {
      currdist = (x1[i]-c1[j]) * (x1[i]-c1[j]) + (x2[i]-c2[j]) * (x2[i]-c2[j]);
      if (currdist < bestdist) {
        z[i] = j;
        bestdist = currdist;
      }
    }
  }
}

void maximization (double* c1, double* c2, const double* x1, const double* x2, const int* z, long K, long N){
  double* count = (double*) malloc(K * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < K; i++){
    c1[i] = 0;
    c2[i] = 0;
    count[i] = 0;
  }
  // #pragma omp parallel for schedule(static) reduction(+:c1[:N],c2[:N])
  for (long i = 0; i < N; i++){
    c1[z[i]] += x1[i];
    c2[z[i]] += x2[i];
    count[z[i]] += 1;
  }
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < K; i++){
    c1[i] /= count[i];
    c2[i] /= count[i];
  }
  free(count);
}

__global__
void expectation_kernel(double* c1, double* c2, const double* __restrict__ x1, const double* __restrict__ x2, int* z, long K, long N){
  double bestdist, currdist;
  int besti;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const double lx1= x1[idx], lx2 = x2[idx];
  // extern __shared__ double lc1[];
  // double* lc2=&lc1[K];
  // for(int i = 0 ; i< K; i++){
  //   lc1[i] = c1[i];
  //   lc2[i] = c2[i];
  // }
  // __syncthreads();
  if (idx < N){
    bestdist = DBL_MAX;
    for (long j = 0; j < K; j++) {
      currdist = (lx1-c1[j]) * (lx1-c1[j]) + (lx2-c2[j]) * (lx2-c2[j]);
      if (currdist < bestdist) {
        besti = j;
        bestdist = currdist;
      }
    }
  }
  z[idx] = besti;
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
  long N = read_option<int>("-n", argc, argv, "100000");
  long K = read_option<long>("-k", argc, argv, "100");
  long max_iter = read_option<long>("-itr", argc, argv, "10000");
  double tol = read_option<double>("-tol", argc, argv, "1e-5");
  bool fast_finish = false;
  double* x1 = (double*) malloc(N * sizeof(double));
  double* x2 = (double*) malloc(N * sizeof(double));
  double* c1 = (double*) malloc(K * sizeof(double));
  double* c2 = (double*) malloc(K * sizeof(double));
  double* c1_prev = (double*) malloc(K * sizeof(double));
  double* c2_prev = (double*) malloc(K * sizeof(double));
  double* c1_ref = (double*) malloc(K * sizeof(double));
  double* c2_ref = (double*) malloc(K * sizeof(double));
  int* z = (int*) malloc(N * sizeof(int));
  int* z_prev = (int*) malloc(N * sizeof(int));
  int* z_ref = (int*) malloc(N * sizeof(int));

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
  data = fopen("birch1.txt","r");
  for(int i = 0 ; i < N; i++){
    fscanf(data, "%lf", &x1[i]);
    fscanf(data, "%lf", &x2[i]);
  }
  fclose(data);
  printf("import successful\n");


  //initialize cluster centers
  init_random (c1, c2, c1_ref, c2_ref, x1, x2, K, N);
  // printf("Initial centroids:\n");
  // for(int i = 0; i< K; i++){
  //   printf("%f, %f\n",c1_ref[i],c2_ref[i] );
  //   // printf("diff: %f, %f\n",c1_ref[i]-c1[i],c2_ref[i]-c2[i] );
  // }


  printf("begin CPU kmeans:\n");
  double tt = omp_get_wtime();
  //printf("running inner_prod\n");
  //means(z_ref, x, y, N);
  int iter = 0;
  for (iter = 0; iter < max_iter; iter++) {
    // printf("begin expectation\n");
    expectation(c1_ref, c2_ref, x1, x2, z_ref, K, N);
    // printf("begin maximization\n");
    maximization(c1_ref, c2_ref, x1, x2, z_ref, K, N);
    // printf("begin tol check\n");
    // printf("z change: %f\n",L2(z_prev,z_ref, N));
    // printf("centroid change: %f\n",L2(c1_prev,c1_ref, K)+L2(c2_prev,c2_ref, K));

    if(fast_finish){
      if (L2(z_prev, z_ref, N) < tol && L2(c1_prev,c1_ref, K)+L2(c2_prev,c2_ref, K) < tol){
        printf("Done at iteration %d\n", iter);
        break;
      }
      // printf("begin copy\n");
      std::copy (z_ref, z_ref + N, z_prev);
      std::copy (c1_ref, c1_ref + K, c1_prev);
      std::copy (c2_ref, c2_ref + K, c2_prev);
      // z_temp = z_prev; z_prev = z_ref; z_ref = z_temp;
      // c1_temp = c1_prev; c1_prev = c1_ref; c1_ref = c1_temp;
      // c2_temp = c2_prev; c2_prev = c2_ref; c2_ref = c2_temp;
    }

  }
  //printf("exiting inner_prod\n");
  //printf("CPU Bandwidth = %f GB/s\n", 3/1e9/ (omp_get_wtime()-tt)*N*N*sizeof(double)) ;
  printf("Time taken CPU = %f s\n", omp_get_wtime()-tt) ;
  printf("Finished in %d iterations.\n", iter) ;

  // printf("final centroids:\n" );
  // for(int i = 0; i< K; i++){
  //   printf("%f, %f\n",c1_ref[i],c2_ref[i] );
  // }


  // printf("assignments: \n");
  // for(int i = 0; i< N; i++){
  //   printf("%d\n",z_ref[i]);
  // }

  printf("\n\n\n" );
  double *x1_d, *x2_d;
  double *c1_d, *c2_d;
  int *z_d;
  cudaMalloc(&x1_d, N*sizeof(double));
  Check_CUDA_Error("cuda malloc x1 failed");
  cudaMalloc(&x2_d, N*sizeof(double));
  Check_CUDA_Error("cuda malloc x2 failed");
  cudaMalloc(&c1_d, K*sizeof(double));
  Check_CUDA_Error("cuda malloc c1 failed");
  cudaMalloc(&c2_d, K*sizeof(double));
  Check_CUDA_Error("cuda malloc c2 failed");
  cudaMalloc(&z_d, N*sizeof(int));
  Check_CUDA_Error("cuda malloc z failed");

  printf("begin GPU kmeans:\n");
  tt = omp_get_wtime();
  cudaMemcpy(x1_d, x1, N*sizeof(double), cudaMemcpyHostToDevice);
  Check_CUDA_Error("memcpy x1 failed");
  cudaMemcpy(x2_d, x2, N*sizeof(double), cudaMemcpyHostToDevice);
  Check_CUDA_Error("memcpy x2 failed");
  // cudaMemcpy(c1_d, c1, K*sizeof(double), cudaMemcpyHostToDevice);
  // Check_CUDA_Error("memcpy c1 failed");
  // cudaMemcpy(c2_d, c2, K*sizeof(double), cudaMemcpyHostToDevice);
  // Check_CUDA_Error("memcpy c2 failed");
  cudaMemcpy(z_d, z, N*sizeof(int), cudaMemcpyHostToDevice);
  Check_CUDA_Error("memcpy z failed");
  iter = 0;
  for (iter = 0; iter < max_iter; iter++) {
    // printf("begin iter %d\n",iter);
    cudaMemcpy(c1_d, c1, K*sizeof(double), cudaMemcpyHostToDevice);
    Check_CUDA_Error("memcpy c1 failed");
    cudaMemcpy(c2_d, c2, K*sizeof(double), cudaMemcpyHostToDevice);
    Check_CUDA_Error("memcpy c2 failed");
    expectation_kernel<<<N/1024,1024>>>(c1_d,c2_d,x1_d, x2_d, z_d, K, N);
    Check_CUDA_Error("kernel");
    cudaDeviceSynchronize();
    cudaMemcpy(z, z_d, N*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(c1, c1_d, K*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(c2, c2_d, K*sizeof(double), cudaMemcpyDeviceToHost);
    // printf("begin maximization\n");
    maximization(c1, c2, x1, x2, z, K, N);

    // printf("z change: %f\n",L2(z_prev,z, N));
    // printf("centroid change: %f\n",L2(c1_prev,c1, K)+L2(c2_prev,c2, K));
    // printf("begin tol check\n");

    if(fast_finish){
      if (L2(z_prev, z, N) < tol && L2(c1_prev,c1, K)+L2(c2_prev,c2, K) < tol){
        printf("Done at iteration %d\n", iter);
        break;
      }
      // printf("begin copy\n");
      std::copy (z, z + N, z_prev);
      std::copy (c1, c1 + K, c1_prev);
      std::copy (c2, c2 + K, c2_prev);
      // z_temp = z_prev; z_prev = z_ref; z_ref = z_temp;
      // c1_temp = c1_prev; c1_prev = c1_ref; c1_ref = c1_temp;
      // c2_temp = c2_prev; c2_prev = c2_ref; c2_ref = c2_temp;
    }
  }
  // printf("GPU Bandwidth = %f GB/s\n", 3/1e9/ (omp_get_wtime()-tt)*N*N*sizeof(double)) ;
  printf("Time taken GPU = %f s\n", omp_get_wtime()-tt) ;
  printf("Finished in %d iterations.\n", iter) ;

  // printf("final centroids:\n" );
  // for(int i = 0; i< K; i++){
  //   printf("%f, %f\n",c1[i],c2[i] );
  // }

  // for(int i = 0; i<N; i++){
  //   int min_index=i+1;
  //   for(int j = i+1; j< N ;j++){
  //     if(z[j]<z[min_index]){
  //       min_index = j;
  //     }
  //   }
  //   std::swap(z[i],z[min_index]);
  // }
  // for(int i = 0; i<N; i++){
  //   int min_index=i+1;
  //   for(int j = i+1; j< N; j++){
  //     if(z_ref[j]<z_ref[min_index]){
  //       min_index = j;
  //     }
  //   }
  //   std::swap(z_ref[i],z_ref[min_index]);
  // }
  // double err = 0;
  // for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
  // double err = L2(c1,c1_ref,K)+L2(c2,c2_ref,K);
  std::sort(z,z+N);
  std::sort(z_ref,z+N);
  double err = L2(z,z_ref,N);
  printf("Error = %f\n", (err));

  // printf("final centroids:\n" );
  // for(int i = 0; i< K; i++){
  //   printf("%f, %f\n",c1[i],c2[i] );
  // }
  // printf("final centroids:\n" );
  // for(int i = 0; i< K; i++){
  //   printf("%f, %f\n",c1_ref[i],c2_ref[i] );
  // }

  //printf("%f,  %f", z_ref[0], z_ref[1]);
  cudaFree(x1_d);
  cudaFree(x2_d);
  cudaFree(c1_d);
  cudaFree(c2_d);
  cudaFree(z_d);

  free(x1);
  free(x2);
  free(c1);
  free(c2);
  free(c1_prev);
  free(c2_prev);
  free(c1_ref);
  free(c2_ref);
  free(z);
  free(z_prev);
  free(z_ref);

  return 0;
}
