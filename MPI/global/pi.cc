#include <iostream>
#include <cmath>
#include <cstdlib>
#include <mpi.h>
double drand() {
    const double fac = 1.0/(RAND_MAX-1.0);
    return fac*random();
}

int main(int argc, char** argv) {
  int rank, size;
  MPI_Status status;

  MPI_Init (&argc, &argv);/* starts MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);/* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);/* get number of processes */

  srandom(rank*11+13);

  const long N = 100000000;
  long sum = 0;
  for (long i=rank; i<N; i+=size) {
    double x = 2.0*(drand()-0.5); // Random value in [-1,1]
    double y = 2.0*(drand()-0.5); // Random value in [-1,1]
    double rsq = x*x + y*y;
    if (rsq < 1.0) sum++;
  }

  if (rank == 0) {
    for (int i=1; i<size; i++) {
      long sum2;
      MPI_Recv(&sum2, 1, MPI_LONG, i, 1, MPI_COMM_WORLD, &status);
      sum+=sum2;
      
    }
     double pi = (4.0*sum)/N;
     std::cout.precision(8);
     std::cout << pi <<std::endl;
  }
  else {
    MPI_Send(&sum, 1, MPI_LONG, 0, 1, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}

