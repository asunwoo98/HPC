// adapted from gpu03.cu
// $ nvcc -arch=sm_61 gpu03.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void inner_prod(double* c, const double* a, const double* b, long N){
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N; j++) {
      c[i] += a[j] * b[i*N+j];
    }
  }
}

__global__
void inner_prod_kernel(double* c, const double* a, const double* b, long N){
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

int main() {
  //long N = (1UL<<25); // 2^25
  long N = (1UL<<12); //2^15
  double* x = (double*) malloc(N * sizeof(double));
  double* y = (double*) malloc(N * N * sizeof(double));
  double* z = (double*) malloc(N * sizeof(double));
  double* z_ref = (double*) malloc(N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = i+2;
    
    z[i] = 0;
    z_ref[i] = 0;
    for (long j = 0; j < N; j++) {
      y[j*N+i] = i+j+1;
    }
  }

  double tt = omp_get_wtime();
  //printf("running inner_prod\n");
  inner_prod(z_ref, x, y, N);
  //printf("exiting inner_prod\n");
  printf("CPU Bandwidth = %f GB/s\n", 3/1e9/ (omp_get_wtime()-tt)*N*N*sizeof(double)) ;

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
  inner_prod_kernel<<<N/1024,1024>>>(z_d, x_d, y_d, N);
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

