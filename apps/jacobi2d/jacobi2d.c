/* 2D Jacobi stencil with a 2D Cartesian decomposition and halo exchange.
 * usage: jacobi2d [N] [iters]   (N x N global grid, default 1024, 100 iters)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define U(i, j) u[(i) * (ny + 2) + (j)]
#define V(i, j) v[(i) * (ny + 2) + (j)]

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = argc > 1 ? atoi(argv[1]) : 1024;
  int iters = argc > 2 ? atoi(argv[2]) : 100;

  int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
  MPI_Dims_create(size, 2, dims);
  MPI_Comm cart;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);
  MPI_Cart_coords(cart, rank, 2, coords);

  int up, down, left, right;
  MPI_Cart_shift(cart, 0, 1, &up, &down);
  MPI_Cart_shift(cart, 1, 1, &left, &right);

  int nx = N / dims[0], ny = N / dims[1]; /* local interior size */
  double *u = calloc((size_t)(nx + 2) * (ny + 2), sizeof(double));
  double *v = calloc((size_t)(nx + 2) * (ny + 2), sizeof(double));
  double *sendl = malloc(nx * sizeof(double)), *sendr = malloc(nx * sizeof(double));
  double *recvl = malloc(nx * sizeof(double)), *recvr = malloc(nx * sizeof(double));

  /* boundary condition: hot top edge of the global domain */
  if (coords[0] == 0)
    for (int j = 0; j < ny + 2; j++) U(0, j) = V(0, j) = 100.0;

  MPI_Datatype rowtype;
  MPI_Type_contiguous(ny, MPI_DOUBLE, &rowtype);
  MPI_Type_commit(&rowtype);

  double t0 = MPI_Wtime(), resid = 0.0;
  for (int it = 0; it < iters; it++) {
    MPI_Request req[8];
    int nr = 0;

    for (int i = 0; i < nx; i++) {
      sendl[i] = U(i + 1, 1);
      sendr[i] = U(i + 1, ny);
    }
    MPI_Irecv(&U(0, 1), 1, rowtype, up, 0, cart, &req[nr++]);
    MPI_Irecv(&U(nx + 1, 1), 1, rowtype, down, 1, cart, &req[nr++]);
    MPI_Irecv(recvl, nx, MPI_DOUBLE, left, 2, cart, &req[nr++]);
    MPI_Irecv(recvr, nx, MPI_DOUBLE, right, 3, cart, &req[nr++]);
    MPI_Isend(&U(nx, 1), 1, rowtype, down, 0, cart, &req[nr++]);
    MPI_Isend(&U(1, 1), 1, rowtype, up, 1, cart, &req[nr++]);
    MPI_Isend(sendr, nx, MPI_DOUBLE, right, 2, cart, &req[nr++]);
    MPI_Isend(sendl, nx, MPI_DOUBLE, left, 3, cart, &req[nr++]);
    MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);

    if (left != MPI_PROC_NULL) for (int i = 0; i < nx; i++) U(i + 1, 0) = recvl[i];
    if (right != MPI_PROC_NULL) for (int i = 0; i < nx; i++) U(i + 1, ny + 1) = recvr[i];

    double local = 0.0;
    for (int i = 1; i <= nx; i++)
      for (int j = 1; j <= ny; j++) {
        V(i, j) = 0.25 * (U(i - 1, j) + U(i + 1, j) + U(i, j - 1) + U(i, j + 1));
        double d = V(i, j) - U(i, j);
        local += d * d;
      }
    double *t = u; u = v; v = t;

    MPI_Allreduce(&local, &resid, 1, MPI_DOUBLE, MPI_SUM, cart);
  }
  double elapsed = MPI_Wtime() - t0, maxt;
  MPI_Reduce(&elapsed, &maxt, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
  if (rank == 0)
    printf("jacobi2d: N=%d iters=%d ranks=%d (%dx%d) residual=%g time=%.3fs\n",
           N, iters, size, dims[0], dims[1], resid, maxt);

  MPI_Type_free(&rowtype);
  free(u); free(v); free(sendl); free(sendr); free(recvl); free(recvr);
  MPI_Comm_free(&cart);
  MPI_Finalize();
  return 0;
}
