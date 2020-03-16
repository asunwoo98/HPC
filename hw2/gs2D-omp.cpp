#include <stdio.h>
#include "utils.h"
#include <limits.h>
#include <math.h>
#include <omp.h>

//calculates residual without A since it is not really needed to be saved
double residual(double* u, double* f, long N, double *temp){
  //double *temp = (double*) malloc((N+2) * (N+2) * sizeof(double));
  double res = 0;
  #pragma omp parallel
  {//start parallel
  #pragma omp for collapse(2)
  for (int i = 1; i < N+1; i++) {
    for (int j = 1; j < N+1; j++) {
      temp[i*(N+2)+j] = -f[i*(N+2)+j];
      temp[i*(N+2)+j] += 4*u[i*(N+2)+j] + u[(i-1)*(N+2)+j] + u[i*(N+2)+(j-1)] + u[(i+1)*(N+2)+j] + u[i*(N+2)+(j+1)];
    }
  }

  #pragma omp for collapse(2) reduction(+:res)
  for (int i = 1; i < N+1; i++){
    for (int j = 1; j< N+1; j++){
      res += temp[i*(N+2)+j]*temp[i*(N+2)+j];
    }
  }
  }//end prallel
  //free(temp);
  return sqrt(res);
}

void iterategs(double* u, double* f, long N,double *newu){
  //double *newu = (double*) malloc((N+2) * (N+2) * sizeof(double));
  #pragma omp parallel
  {//start parallel
  #pragma omp for collapse(2)
  for(int i = 1; i< N+1; i++){
    for(int j = 1; j< N+1; j++){
      if((i+j)%2 == 0){
        newu[i*(N+2)+j]=f[i*(N+2)+j]/(N+1)/(N+1) + u[(i-1)*(N+2)+j] + u[i*(N+2)+(j-1)] + u[(i+1)*(N+2)+j] + u[i*(N+2)+(j+1)];
        newu[i*(N+2)+j]=newu[i*(N+2)+j]/4;
      }
    }
  }
  

  #pragma omp for collapse(2)
  for(int i = 1; i< N+1; i++){
    for(int j = 1; j< N+1; j++){
      if((i+j)%2 == 1){
        newu[i*(N+2)+j]=f[i*(N+2)+j]/(N+1)/(N+1) + newu[(i-1)*(N+2)+j] + newu[i*(N+2)+(j-1)] + newu[(i+1)*(N+2)+j] + newu[i*(N+2)+(j+1)];
        newu[i*(N+2)+j]=newu[i*(N+2)+j]/4;
      }
    }
  }
  
  #pragma omp for collapse(2)
  for(int i = 1; i< N+1; i++){
    for(int j = 1; j< N+1; j++){
      u[i*(N+2)+j]=newu[i*(N+2)+j];
    }
  }

  
  }//end parallel
  //free(newu);
}

int main(int argc, char** argv) {
  const long itr = read_option<long>("-itr", argc, argv, "1000");
  const long N = read_option<long>("-n", argc, argv, "100");
  const int threads = read_option<int>("-t", argc, argv, "4");
  //const double hh = 0.1; //h^2
  #if defined(_OPENMP)
  omp_set_num_threads(threads);
  #endif

  //allocate
  //double* a = (double*) malloc(N * N * sizeof(double));
  double* u = (double*) malloc((N+2) * (N+2) * sizeof(double));
  double* f = (double*) malloc((N+2) * (N+2) * sizeof(double));
  double* temp = (double*) malloc((N+2) * (N+2) * sizeof(double));

  //initialize
  for (int i = 0; i < N+2; i++) {
    for(int j = 0; j < N+2; j++) {
      //u
      u[i*(N+2)+j] = 0;

      //f
      f[i*(N+2)+j] = 1;

      //a
      //a[i*N+j] = 0;
      //a[i*N+j] = 4/(N+1)/(N+1);
      //if(x - 1 == y || x + 1 == y) a[i*N+j] = -1/(N+1)/(N+1);
    }
  }
  
  double res0 = residual(u,f,N,temp);
  int currItr = 0;
  double res = std::numeric_limits<double>::max();
  Timer t;
  printf("Itr#   Residual\n");
  t.tic();
  while (currItr < itr && res> res0/1e6) {
    iterategs(u,f,N,temp);
    res = residual(u,f,N,temp);
    printf("%4d %10f\n", currItr+1, res);
    currItr++;
  }
  printf("Runtime for %d iterations: %10fs", itr, t.toc());
  //printf("%d",69%2);
  //free(a);
  free(u);
  free(f);
  free(temp);
  return 0;
}