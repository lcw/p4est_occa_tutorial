#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <occa.h>
#include <p4est.h>

void run(MPI_Comm comm, occaDevice device) {
  int entries = 5;

  float *a = (float *)malloc(entries * sizeof(float));
  float *b = (float *)malloc(entries * sizeof(float));
  float *ab = (float *)malloc(entries * sizeof(float));

  for (int i = 0; i < entries; ++i) {
    a[i] = (float)i;
    b[i] = (float)(1 - i);
    ab[i] = NAN;
  }

  occaMemory o_a =
      occaDeviceTypedMalloc(device, entries, occaDtypeFloat, NULL, occaDefault);
  occaMemory o_b =
      occaDeviceTypedMalloc(device, entries, occaDtypeFloat, NULL, occaDefault);
  occaMemory o_ab =
      occaDeviceTypedMalloc(device, entries, occaDtypeFloat, NULL, occaDefault);

  occaProperties props = occaCreateProperties();
  occaPropertiesSet(props, "defines/TILE_SIZE", occaInt(10));
  occaKernel add_vectors =
      occaDeviceBuildKernel(device, "kernels.okl", "add_vectors", props);

  occaCopyPtrToMem(o_a, a, occaAllBytes, 0, occaDefault);
  occaCopyPtrToMem(o_b, b, occaAllBytes, 0, occaDefault);
  occaKernelRun(add_vectors, occaInt(entries), o_a, o_b, o_ab);
  occaCopyMemToPtr(ab, o_ab, occaAllBytes, 0, occaDefault);

  for (int i = 0; i < entries; ++i) {
    printf("%g\n", ab[i]);
  }

  for (int i = 0; i < entries; ++i) {
    assert(ab[i] == (a[i] + b[i]));
  }

  free(a);
  free(b);
  free(ab);

  occaFree(&props);
  occaFree(&add_vectors);
  occaFree(&o_a);
  occaFree(&o_b);
  occaFree(&o_ab);
}

int main(int argc, char **argv) {
  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;

  SC_CHECK_MPI(MPI_Init(&argc, &argv));
  SC_CHECK_MPI(MPI_Comm_size(comm, &size));
  SC_CHECK_MPI(MPI_Comm_rank(comm, &rank));
  sc_init(comm, 1, 1, NULL, SC_LP_DEFAULT);
  p4est_init(NULL, SC_LP_DEFAULT);

  const char *device_string = "mode: 'Serial'";

#if 0
  const char *device_string = "mode: 'OpenMP', schedule: 'compact', chunk: 10";
  const char *device_string = "mode: 'OpenCL', device_id: 0, platform_id: 0";
  const char *device_string = "mode: 'CUDA', device_id: 0";
#endif

  occaDevice device = occaCreateDeviceFromString(device_string);

  run(comm, device);

  occaFree(&device);

  sc_finalize();
  SC_CHECK_MPI(MPI_Finalize());

  return 0;
}
