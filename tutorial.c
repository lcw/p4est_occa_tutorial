#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <occa.h>
#include <p4est.h>

// {{{ PCG32
// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
// Copied from: http://www.pcg-random.org/
// Copied from: https://github.com/imneme/pcg-c-basic

typedef struct {
  uint64_t state;
  uint64_t inc;
} pcg32_random_t;

void pcg32_srandom_r(pcg32_random_t *rng, uint64_t initstate, uint64_t initseq);

uint32_t pcg32_boundedrand_r(pcg32_random_t *rng, uint32_t bound);

uint32_t pcg32_random_r(pcg32_random_t *rng) {
  uint64_t oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
  uint32_t rot = (uint32_t)(oldstate >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

void pcg32_srandom_r(pcg32_random_t *rng, uint64_t initstate,
                     uint64_t initseq) {
  rng->state = 0U;
  rng->inc = (initseq << 1u) | 1u;
  pcg32_random_r(rng);
  rng->state += initstate;
  pcg32_random_r(rng);
}

uint32_t pcg32_boundedrand_r(pcg32_random_t *rng, uint32_t bound) {
  // To avoid bias, we need to make the range of the RNG a multiple of
  // bound, which we do by dropping output less than a threshold.
  // A naive scheme to calculate the threshold would be to do
  //
  //     uint32_t threshold = 0x100000000ull % bound;
  //
  // but 64-bit div/mod is slower than 32-bit div/mod (especially on
  // 32-bit platforms).  In essence, we do
  //
  //     uint32_t threshold = (0x100000000ull-bound) % bound;
  //
  // because this version will calculate the same modulus, but the LHS
  // value is less than 2^32.

  uint32_t threshold = -bound % bound;

  // Uniformity guarantees that this loop will terminate.  In practice, it
  // should usually terminate quickly; on average (assuming all bounds are
  // equally likely), 82.25% of the time, we can expect it to require just
  // one iteration.  In the worst case, someone passes a bound of 2^31 + 1
  // (i.e., 2147483649), which invalidates almost 50% of the range.  In
  // practice, bounds are typically small and only a tiny amount of the range
  // is eliminated.
  for (;;) {
    uint32_t r = pcg32_random_r(rng);
    if (r >= threshold)
      return r % bound;
  }
}

// }}}

void run(MPI_Comm comm, occaDevice device) {
  int entries = 500;

  float *a = (float *)calloc(entries, sizeof(float));
  float *b = (float *)calloc(entries, sizeof(float));
  float *ab = (float *)calloc(entries, sizeof(float));

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
  occaPropertiesSet(props, "defines/blockSize", occaInt(256));
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
  MPI_Comm comm = MPI_COMM_WORLD;

  SC_CHECK_MPI(MPI_Init(&argc, &argv));
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
