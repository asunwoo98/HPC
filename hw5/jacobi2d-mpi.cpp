/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, double* f, int lN, double invhsq){
  //int i;
  double tmp, gres = 0.0, lres = 0.0;

  // for (i = 1; i <= lN; i++){
  //   tmp = ((2.0*lu[i] - lu[i-1] - lu[i+1]) * invhsq - 1);
  //   lres += tmp * tmp;
  // }
  for (int i = 1; i < lN+1; i++) {
    for (int j = 1; j < lN+1; j++) {
      tmp = -f[i*(lN+2)+j] + (4*u[i*(lN+2)+j] + u[(i-1)*(lN+2)+j] + u[i*(lN+2)+(j-1)] + u[(i+1)*(lN+2)+j] + u[i*(lN+2)+(j+1)])*invhsq;
      lres += tmp*tmp
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i, p, N, lN, iter, max_iters;
  MPI_Status status, status1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  lN = N / sqrt(p);
  if ((N % sqrt(p) != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of sqrt(p)\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lf = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lutemp;
  //initialize lf
  for (int i = 0; i < lN+2; i++) {
    for(int j = 0; j < lN+2; j++) {
      //f
      lf[i*(lN+2)+j] = 1;
    }
  }

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lf, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    /* Jacobi step for local points */
    // for (i = 1; i <= lN; i++){
    //   lunew[i]  = 0.5 * (hsq + lu[i - 1] + lu[i + 1]);
    // }
    for(int i = 1; i< lN+1; i++){
      for(int j = 1; j< lN+1; j++){
        lunew[i*(lN+2)+j]=f[i*(lN+2)+j]*hsq + u[(i-1)*(lN+2)+j] + u[i*(lN+2)+(j-1)] + u[(i+1)*(lN+2)+j] + u[i*(lN+2)+(j+1)];
        lunew[i*(lN+2)+j]=newu[i*(lN+2)+j]/4;
      }
    }

    /* communicate ghost values */
    /* left */
    if (mpirank % lN != 0) {
      /* If not the last process, send/recv bdry values to the right */
      for (int i = 1; i < lN+1; i++){
        MPI_Send(&(lunew[1+i*(lN+2)]), 1, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
        MPI_Recv(&(lunew[0+i*(lN+2)]), 1, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      }
    }
    /* right */
    if (mpirank % lN != 3) {
      /* If not the first process, send/recv bdry values to the left */
        for (int i = 1; i < lN+1; i++){
          MPI_Send(&(lunew[lN+i*(lN+2)]), 1, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
          MPI_Recv(&(lunew[lN+1+i*(lN+2)]), 1, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status);
        }
    }

    /* up */
    if( mpirank <= p - lN ) {
      MPI_Send(&(lunew[1+(lN)*(lN+2)]), lN, MPI_DOUBLE, mpirank+4, 120, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[1+(lN+1)*(lN+2)]), lN, MPI_DOUBLE, mpirank+4, 121, MPI_COMM_WORLD, &status);
    }

    /* down */
    if( mpirank >= lN ) {
      MPI_Send(&(lunew[1+lN+2]), lN, MPI_DOUBLE, mpirank-4, 121, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[1]), lN, MPI_DOUBLE, mpirank-4, 120, MPI_COMM_WORLD, &status);
    }


    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lf);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
