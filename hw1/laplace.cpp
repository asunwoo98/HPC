#include <stdio.h>
#include "utils.h"
#include <limits.h>
#include <math.h>

double residual(double* a, double* u, double* f, long N, double* temp){
  for (int x = 0; x < N; x++) {
  temp[x] = -f[x];
    for (int y = 0; y < N; y++) {
      temp[x] += a[x*N+y]*u[y];
    }
  }

  double res = 0;
  for (int x = 0; x < N; x++){
    res += temp[x]*temp[x];
  }
  return sqrt(res);
}

void iteratejacobi(double* a, double* u, double* f, long N, double* newu){
  for(int i = 0; i< N; i++){
    newu[i]=f[i];
    for(int j = 0; j< N; j++) {
      if(j!=i) newu[i]-=a[i*N+j]*u[j];
    }
    newu[i]=newu[i]/a[i*N+i];
  }
  
  for(int x=0;x < N; x++){
    u[x]=newu[x];
  }
}

void iterategauss(double* a, double* u, double* f, long N, double* newu){
  for(int i = 0; i< N; i++){
    newu[i]=f[i];
    for(int j = 0; j< N; j++) {
      if(j<i) newu[i]-=a[i*N+j]*newu[j];
      else if(j>i) newu[i]-=a[i*N+j]*u[j];
    }
    newu[i]=newu[i]/a[i*N+i];
  }
  
  for(int x=0;x < N; x++){
    u[x]=newu[x];
  }
}

int main(int argc, char** argv) {
  const long itr = read_option<long>("-itr", argc, argv, "5000");
  const long N = read_option<long>("-n", argc, argv, "100");
  const double hh = 0.1; //h^2

  //allocate
  double* a = (double*) malloc(N * N * sizeof(double));
  double* u = (double*) malloc(N * sizeof(double));
  double* f = (double*) malloc(N * sizeof(double));
  double* temp = (double*) malloc(N * sizeof(double));

  //initialize
  for (int x = 0; x < N; x++) {
    u[x] = 0;
    f[x] = 1;
    for (int y = 0; y < N; y++) {
      a[x+y*N] = 0;
      if(x == y) a[x+y*N] = 2/hh;
      if(x - 1 == y || x + 1 == y) a[x+y*N] = -1/hh;
    }
  }
  
  double res0 = residual(a,u,f,N,temp);
  int currItr = 0;
  double res = std::numeric_limits<double>::max();
  Timer t;
  printf("Itr#   Residual\n");
  t.tic();
  while (currItr < itr && res> res0/1e6) {
    iterategauss(a,u,f,N,temp);
    res = residual(a,u,f,N,temp);
    printf("%4d %10f\n", currItr+1, res);
    currItr++;
  }
  printf("Runtime for %d iterations: %10fs", itr, t.toc());
  free(a);
  free(u);
  free(f);
  free(temp);
  return 0;
}