/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <algorithm>
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
      tmp = -f[i*(lN+2)+j] + (4*lu[i*(lN+2)+j] + lu[(i-1)*(lN+2)+j] + lu[i*(lN+2)+(j-1)] + lu[(i+1)*(lN+2)+j] + lu[i*(lN+2)+(j+1)]);//*invhsq;
      lres += tmp*tmp;
    }
  }
  // printf("going into allreduce\n" );
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

  sscanf(argv[1], "%d", &lN);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  N = lN * sqrt(p);
  if ((N % (int)sqrt(p) != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of sqrt(p)\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc( (lN + 2)*(lN + 2),sizeof(double));
  double * lf = (double *) calloc((lN + 2)*(lN + 2),sizeof(double));
  double * lunew = (double *) calloc((lN + 2)*(lN + 2),sizeof(double));
  double * lutemp;
  //initialize lf
  for (int i = 0; i < lN+2; i++) {
    for(int j = 0; j < lN+2; j++) {
      //f
      lf[i*(lN+2)+j] = 1;
      // lu[i*(lN+2)+j] = 0;
      // lunew[i*(lN+2)+j] = 0;
    }
  }
  // printf("lN = %d\n", lN);
  // printf("memory alloc success, %f\n", lu[(lN+2)*(lN+2)-1]);

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lf, lN, invhsq);
  gres = gres0;
  // printf("initial residual sucess\n");

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    /* Jacobi step for local points */
    // for (i = 1; i <= lN; i++){
    //   lunew[i]  = 0.5 * (hsq + lu[i - 1] + lu[i + 1]);
    // }
    for(int i = 1; i< lN+1; i++){
      for(int j = 1; j< lN+1; j++){
        lunew[i*(lN+2)+j]=lf[i*(lN+2)+j]*hsq + lu[(i-1)*(lN+2)+j] + lu[i*(lN+2)+(j-1)] + lu[(i+1)*(lN+2)+j] + lu[i*(lN+2)+(j+1)];
        lunew[i*(lN+2)+j]=lunew[i*(lN+2)+j]/4;
      }
    }

    // printf("jacobistep success\n");
    /* communicate ghost values */
    /* left */
    int sqrtp = sqrt(p);
    if ((mpirank % sqrtp != 0) && p>1) {
      /* If not the last process, send/recv bdry values to the right */
      for (int i = 1; i < lN+1; i++){
        MPI_Send(&(lunew[1+i*(lN+2)]), 1, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
        MPI_Recv(&(lunew[0+i*(lN+2)]), 1, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
      }
    }
    // printf("left communicate success\n");
    /* right */
    if ((mpirank % sqrtp != sqrtp-1) && p>1) {
      /* If not the first process, send/recv bdry values to the left */
        for (int i = 1; i < lN+1; i++){
          MPI_Send(&(lunew[lN+i*(lN+2)]), 1, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
          MPI_Recv(&(lunew[lN+1+i*(lN+2)]), 1, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status);
        }
    }
    // printf("right communicate success\n");
    /* up */
    if( (mpirank < p - sqrtp) && p>1) {
      MPI_Send(&(lunew[1+(lN)*(lN+2)]), lN, MPI_DOUBLE, mpirank+sqrtp, 120, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[1+(lN+1)*(lN+2)]), lN, MPI_DOUBLE, mpirank+sqrtp, 121, MPI_COMM_WORLD, &status);
    }
    // printf("up communicate success\n");
    /* down */
    if( (mpirank >= sqrtp) && p>1) {
      MPI_Send(&(lunew[1+lN+2]), lN, MPI_DOUBLE, mpirank-sqrtp, 121, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[1]), lN, MPI_DOUBLE, mpirank-sqrtp, 120, MPI_COMM_WORLD, &status);
    }

    // printf("ghost value exchange success\n");
    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    // std::copy(lunew,lunew+(lN + 2)*(lN + 2),lu);
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lf, lN, invhsq);
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
