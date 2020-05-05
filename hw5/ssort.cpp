// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include "utils.h"

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = read_option<int>("-n", argc, argv, "10e4");;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  clock_t t;
  MPI_Barrier(MPI_COMM_WORLD);
  t=clock();

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* loc_split = (int*) malloc ((p-1)*sizeof(int));
  for (int i = 0; i< p-1; i++) {
    loc_split[i] = vec[(i+1)*N/p];
  }

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* root_split = (int*) malloc (p*(p-1)*sizeof(int));
  MPI_Gather(loc_split, p-1, MPI_INT, root_split, p*(p-1), MPI_INT, 0, MPI_COMM_WORLD);

  printf("mpi gather complete\n");
  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  if (rank == 0){
    std::sort(root_split, root_split+p*(p-1));
    for (int i = 0; i< p-1; i++) {
      loc_split[i] = root_split[p/2+i*p];
    }
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast (loc_split, p-1, MPI_INT, 0, MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  //int* scount = (int*) malloc ((p))*sizeof(int));
  int* sdispls = (int*) malloc ((p)*sizeof(int));
  sdispls[0] = 0;
  for (int i = 0; i < p-1; i++) {
    sdispls[i+1] = std::lower_bound(vec, vec+N, sdispls[i]) - vec;
  }
  int* sendcounts = (int*) malloc ((p)*sizeof(int));
  for (int i = 0; i < p-1; i++) {
    sendcounts[i] = sdispls[i+1]-sdispls[i];
  }
  sendcounts[p-1] = N - sdispls[p-1];

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  int* recvcounts = (int*) malloc ((p)*sizeof(int));
  MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

  int* rdispls = (int*) malloc ((p)*sizeof(int));
  rdispls[0] = 0;
  for (int i = 0; i < p-1; i++) {
    rdispls[i+1] = rdispls[i] + recvcounts[i];
  }
  int  totalcount = 0;
  for (int i = 0; i < p; i++) totalcount += recvcounts[i];

  int* newvec = (int*)malloc(totalcount*sizeof(int));
  MPI_Alltoallv(vec, sendcounts, sdispls, MPI_INT, newvec, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

  // do a local sort of the received data
  std::sort(newvec,newvec+totalcount);

  MPI_Barrier(MPI_COMM_WORLD);
  t = clock() -t;
  if(rank == 0){
    printf("ssort on lN = %d finished in %f s\n",N, (float)t );
  }
  // every process writes its result to a file
  // FILE* output;
  // output= fopen("output.txt","w");
  // for (int i = 0; i < p; i++) {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   if (rank == i) {
  //     fprintf(output,"process %d ==> ", rank);
  //     for (long k = 0; k < totalcount; k++) {
  //       fprintf(output,"%4ld ", newvec[k]);
  //     }
  //     fprintf(output,"\n");
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }
  // fclose(output);
  FILE* output;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  output= fopen(filename,"w+");
  fprintf(output,"process %d has elements ==> \n", rank);
  for (long k = 0; k < totalcount; k++) {
    fprintf(output,"%4ld ", newvec[k]);
  }
  fclose(output);


  free(vec);
  free(loc_split);
  free(root_split);
  free(sdispls);
  free(sendcounts);
  free(recvcounts);
  free(rdispls);
  free(newvec);
  MPI_Finalize();
  return 0;
}
