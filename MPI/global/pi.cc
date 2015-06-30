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

  const long N = 10000000000;
  long sum = 0;
  for (long i=rank; i<N; i+=size) {
    double x = 2.0*(drand()-0.5); // Random value in [-1,1]
    double y = 2.0*(drand()-0.5); // Random value in [-1,1]
    double rsq = x*x + y*y;
    if (rsq < 1.0) sum++;
  }

  long total;
  MPI_Reduce(&sum, &total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  double pi = (4.0*total)/N;
  std::cout.precision(8);
  std::cout << pi <<std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}

