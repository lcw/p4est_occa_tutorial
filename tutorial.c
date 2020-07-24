#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <occa.h>
#include <p4est_extended.h>
#include <p4est_vtk.h>

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

// {{{ OCCA Hello world
void run_occa(MPI_Comm comm, occaDevice device) {
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
// }}}

typedef struct mesh {
  MPI_Comm comm;
  occaDevice device;

  p4est_connectivity_t *connectivity;
  p4est_t *p4est;

  double *x;
  double *y;
} mesh_t;

typedef struct quad_info {
  p4est_locidx_t numquads;

  int8_t *EToL;
  p4est_topidx_t *EToT;

  p4est_qcoord_t *EToX;
  p4est_qcoord_t *EToY;

} quad_info_t;

void mesh_iterate_volume(p4est_iter_volume_info_t *info, void *user_data) {
  quad_info_t *quad_info = user_data;

  p4est_tree_t *tree = p4est_tree_array_index(info->p4est->trees, info->treeid);
  const p4est_locidx_t e = tree->quadrants_offset + info->quadid;

  quad_info->EToL[e] = info->quad->level;
  quad_info->EToT[e] = info->treeid;
  quad_info->EToX[e] = info->quad->x;
  quad_info->EToY[e] = info->quad->y;
}

quad_info_t *quad_info_new(p4est_t *p4est) {
  quad_info_t *quad_info = calloc(1, sizeof(quad_info_t));

  quad_info->numquads = p4est->local_num_quadrants;
  quad_info->EToL = calloc(p4est->local_num_quadrants, sizeof(int8_t));
  quad_info->EToT = calloc(p4est->local_num_quadrants, sizeof(p4est_topidx_t));
  quad_info->EToX = calloc(p4est->local_num_quadrants, sizeof(p4est_qcoord_t));
  quad_info->EToY = calloc(p4est->local_num_quadrants, sizeof(p4est_qcoord_t));

  p4est_iterate(p4est, NULL, quad_info, mesh_iterate_volume, NULL, NULL);

  return quad_info;
}

void mesh_write(mesh_t *mesh, const char *filename) {
  p4est_vtk_context_t *context = p4est_vtk_context_new(mesh->p4est, filename);
  p4est_vtk_context_set_scale(context, 1.0);

  context = p4est_vtk_write_header(context);

  const p4est_locidx_t numquads = mesh->p4est->local_num_quadrants;

  sc_array_t viewx, viewy;
  sc_array_init_data(&viewx, mesh->x, sizeof(double),
                     numquads * P4EST_CHILDREN);
  sc_array_init_data(&viewy, mesh->y, sizeof(double),
                     numquads * P4EST_CHILDREN);

  context = p4est_vtk_write_point_dataf(context, 2, 0, "our_x", &viewx, "our_y",
                                        &viewy, context);

  p4est_vtk_write_footer(context);
}

mesh_t *mesh_new(MPI_Comm comm, occaDevice device) {
  mesh_t *mesh = calloc(1, sizeof(mesh_t));

  const int level = 3;

  mesh->comm = comm;
  mesh->device = device;
  mesh->connectivity = p4est_connectivity_new_disk(0, 0);
  mesh->p4est =
      p4est_new_ext(comm, mesh->connectivity, 0, level, 1, 0, NULL, mesh);

  p4est_locidx_t numquads = mesh->p4est->local_num_quadrants;
  mesh->x = calloc(numquads * P4EST_CHILDREN, sizeof(double));
  mesh->y = calloc(numquads * P4EST_CHILDREN, sizeof(double));

  quad_info_t *quad_info = quad_info_new(mesh->p4est);

  // copy quad_info to GPU
  // copy tree_to_vertices, vertices from the connectivity

  for (p4est_locidx_t e = 0; e < mesh->p4est->local_num_quadrants; ++e) {
    const p4est_topidx_t tree = quad_info->EToT[e];
    const int8_t level = quad_info->EToL[e];

    const double cr = (double)quad_info->EToX[e] / (double)P4EST_ROOT_LEN;
    const double cs = (double)quad_info->EToY[e] / (double)P4EST_ROOT_LEN;

    const double h2 = .5 / (double)(1 << level);

    const p4est_topidx_t v00 =
        mesh->connectivity->tree_to_vertex[tree * P4EST_CHILDREN + 0];
    const p4est_topidx_t v01 =
        mesh->connectivity->tree_to_vertex[tree * P4EST_CHILDREN + 1];
    const p4est_topidx_t v10 =
        mesh->connectivity->tree_to_vertex[tree * P4EST_CHILDREN + 2];
    const p4est_topidx_t v11 =
        mesh->connectivity->tree_to_vertex[tree * P4EST_CHILDREN + 3];

    const double x00 = mesh->connectivity->vertices[v00 * 3 + 0];
    const double x01 = mesh->connectivity->vertices[v01 * 3 + 0];
    const double x10 = mesh->connectivity->vertices[v10 * 3 + 0];
    const double x11 = mesh->connectivity->vertices[v11 * 3 + 0];

    const double y00 = mesh->connectivity->vertices[v00 * 3 + 1];
    const double y01 = mesh->connectivity->vertices[v01 * 3 + 1];
    const double y10 = mesh->connectivity->vertices[v10 * 3 + 1];
    const double y11 = mesh->connectivity->vertices[v11 * 3 + 1];

    for (int j = 0; j < 2; ++j) {
      const double s = cs + h2 * 2. * j;
      for (int i = 0; i < 2; ++i) {
        const double r = cr + h2 * 2. * i;

        const double w0 = (1 - r) * (1 - s);
        const double w1 = r * (1 - s);
        const double w2 = (1 - r) * s;
        const double w3 = r * s;

        const double x = w0 * x00 + w1 * x01 + w2 * x10 + w3 * x11;
        const double y = w0 * y00 + w1 * y01 + w2 * y10 + w3 * y11;

        const p4est_locidx_t id = i + 2 * j + e * P4EST_CHILDREN;
        mesh->x[id] = x;
        mesh->y[id] = y;
      }
    }

  } /* done loop over quadrants */

  //
  // build x and y in kernel
  //
  // destroy quad_info

  return mesh;
}

void mesh_destroy(mesh_t *mesh) {
  free(mesh->x);
  free(mesh->y);

  p4est_destroy(mesh->p4est);
  p4est_connectivity_destroy(mesh->connectivity);
}

void run(MPI_Comm comm, occaDevice device) {
  mesh_t *mesh = mesh_new(comm, device);

  p4est_vtk_write_file(mesh->p4est, NULL, "first_try_mesh");

  mesh_write(mesh, "second_try_mesh");

  mesh_destroy(mesh);
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
