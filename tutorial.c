#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <occa.h>
#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_iterate.h>
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

// {{{ run_occa
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

#define ADAPT_COARSEN 0
#define ADAPT_NONE 1
#define ADAPT_REFINE 2
#define ADAPT_TOUCHED 3

typedef struct quad_data {
  int8_t old_level;
  int8_t adapt_flag;
} quad_data_t;

typedef struct mesh {
  MPI_Comm comm;
  occaDevice device;

  p4est_connectivity_t *connectivity;
  p4est_t *p4est;

  int8_t *EToA; // quadrant to level change
  double *x;
  double *y;
} mesh_t;

typedef struct quad_info {
  p4est_locidx_t numquads;
  int8_t *EToL;         // quadrant to p4est level
  p4est_topidx_t *EToT; // quadrant to p4est treeid
  p4est_qcoord_t *EToX; // quadrant to p4est x-qcoord
  p4est_qcoord_t *EToY; // quadrant to p4est y-qcoord
} quad_info_t;

void mesh_iterate_volume(p4est_iter_volume_info_t *info, void *user_data) {
  quad_info_t *quad_info = user_data;

  p4est_tree_t *tree = p4est_tree_array_index(info->p4est->trees, info->treeid);
  const size_t e = tree->quadrants_offset + info->quadid;

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

void quad_info_destroy(quad_info_t *quad_info) {
  free(quad_info->EToL);
  free(quad_info->EToT);
  free(quad_info->EToX);
  free(quad_info->EToY);

  free(quad_info);
}

mesh_t *mesh_new(MPI_Comm comm, occaDevice device) {
  mesh_t *mesh = calloc(1, sizeof(mesh_t));

  mesh->comm = comm;
  mesh->device = device;

  mesh->connectivity = p4est_connectivity_new_byname("star");

  int mesh_start_level = 4;
  mesh->p4est = p4est_new_ext(comm, mesh->connectivity, 0, mesh_start_level, 1,
                              sizeof(quad_data_t), NULL, NULL);

  mesh->EToA = calloc(mesh->p4est->local_num_quadrants, sizeof(int8_t));

  quad_info_t *quad_info = quad_info_new(mesh->p4est);

  mesh->x =
      calloc(mesh->p4est->local_num_quadrants * P4EST_CHILDREN, sizeof(double));
  mesh->y =
      calloc(mesh->p4est->local_num_quadrants * P4EST_CHILDREN, sizeof(double));

  for (p4est_locidx_t e = 0; e < mesh->p4est->local_num_quadrants; ++e) {
    const p4est_topidx_t tree = quad_info->EToT[e];
    const int8_t level = quad_info->EToL[e];

    // Corner of the element we are updating
    const double cr = (double)quad_info->EToX[e] / P4EST_ROOT_LEN;
    const double cs = (double)quad_info->EToY[e] / P4EST_ROOT_LEN;

    // size of the element (in the tree reference space).
    const double h = 1 / (double)(1 << (level + 1));

    const p4est_topidx_t v00 =
        mesh->connectivity->tree_to_vertex[tree * P4EST_CHILDREN + 0];
    const p4est_topidx_t v10 =
        mesh->connectivity->tree_to_vertex[tree * P4EST_CHILDREN + 1];
    const p4est_topidx_t v01 =
        mesh->connectivity->tree_to_vertex[tree * P4EST_CHILDREN + 2];
    const p4est_topidx_t v11 =
        mesh->connectivity->tree_to_vertex[tree * P4EST_CHILDREN + 3];

    const double x00 = mesh->connectivity->vertices[v00 * 3 + 0];
    const double x10 = mesh->connectivity->vertices[v10 * 3 + 0];
    const double x01 = mesh->connectivity->vertices[v01 * 3 + 0];
    const double x11 = mesh->connectivity->vertices[v11 * 3 + 0];

    const double y00 = mesh->connectivity->vertices[v00 * 3 + 1];
    const double y10 = mesh->connectivity->vertices[v10 * 3 + 1];
    const double y01 = mesh->connectivity->vertices[v01 * 3 + 1];
    const double y11 = mesh->connectivity->vertices[v11 * 3 + 1];

    for (int j = 0; j < 2; ++j) {
      const double s = cs + h * 2 * j;
      for (int i = 0; i < 2; ++i) {
        const double r = cr + h * 2 * i;

        const double w0 = (1 - r) * (1 - s);
        const double w1 = r * (1 - s);
        const double w2 = (1 - r) * s;
        const double w3 = r * s;

        const double x = w0 * x00 + w1 * x10 + w2 * x01 + w3 * x11;
        const double y = w0 * y00 + w1 * y10 + w2 * y01 + w3 * y11;

        const p4est_locidx_t id = i + 2 * j + e * P4EST_CHILDREN;
        mesh->x[id] = x;
        mesh->y[id] = y;
      }
    }
  }

  quad_info_destroy(quad_info);

  return mesh;
}

void mesh_destroy(mesh_t *mesh) {
  free(mesh->EToA);
  free(mesh->x);
  free(mesh->y);

  p4est_destroy(mesh->p4est);

  p4est_connectivity_destroy(mesh->connectivity);

  free(mesh);
}

void mark_quadrants_rand(mesh_t *mesh) {
  static pcg32_random_t rng;
  static int init = 0;

  int rank, size;
  SC_CHECK_MPI(MPI_Comm_rank(mesh->comm, &rank));
  SC_CHECK_MPI(MPI_Comm_size(mesh->comm, &size));

  if (!init) {
#ifdef NONDETERMINISTIC_REFINEMENT
    pcg32_srandom_r(&rng, time(NULL) ^ (intptr_t)&printf, (intptr_t)&exit);
#else
    pcg32_srandom_r(&rng, 42u, 64u);
#endif
    init = 1;
  }

  int a = 1;
  int h = 0;
  p4est_gloidx_t gfp = mesh->p4est->global_first_quadrant[rank];

  for (p4est_gloidx_t e = 0; e < mesh->p4est->global_num_quadrants; ++e) {
    if (e % a == 0) {
      a = pcg32_boundedrand_r(&rng, 12) + 3;
      h = pcg32_boundedrand_r(&rng, 3);
    }

    if (e >= gfp && e < gfp + mesh->p4est->local_num_quadrants)
      mesh->EToA[e - gfp] = (int8_t)h;
  }
}

void replace_quads(p4est_t *p4est, p4est_topidx_t which_tree, int num_outgoing,
                   p4est_quadrant_t *outgoing[], int num_incoming,
                   p4est_quadrant_t *incoming[]) {
  const quad_data_t *outd = (quad_data_t *)outgoing[0]->p.user_data;

  for (int i = 0; i < num_incoming; ++i) {
    quad_data_t *ind = (quad_data_t *)incoming[i]->p.user_data;
    ind->old_level = outd->old_level;
    ind->adapt_flag = ADAPT_TOUCHED;
  }
}

int refine_quads(p4est_t *p4est, p4est_topidx_t which_tree,
                 p4est_quadrant_t *q) {
  const quad_data_t *d = (quad_data_t *)q->p.user_data;

  if (d->adapt_flag == ADAPT_REFINE)
    return 1;
  else
    return 0;
}

int coarsen_quads(p4est_t *p4est, p4est_topidx_t which_tree,
                  p4est_quadrant_t *children[]) {
  int retval = 1;

  for (int i = 0; i < P4EST_CHILDREN; ++i) {
    const quad_data_t *d = (quad_data_t *)children[i]->p.user_data;

    if (d->adapt_flag != ADAPT_COARSEN)
      retval = 0;
  }

  return retval;
}

void fill_user_data(p4est_iter_volume_info_t *info, void *user_data) {
  mesh_t *mesh = user_data;

  p4est_tree_t *tree = p4est_tree_array_index(info->p4est->trees, info->treeid);
  const size_t e = tree->quadrants_offset + info->quadid;

  quad_data_t *d = (quad_data_t *)info->quad->p.user_data;

  d->old_level = info->quad->level;
  d->adapt_flag = mesh->EToA[e];
}

void mesh_write(mesh_t *mesh, const char *filename) {
  p4est_vtk_context_t *context = p4est_vtk_context_new(mesh->p4est, filename);
  p4est_vtk_context_set_scale(context, 1.0);
  p4est_vtk_context_set_continuous(context, 0);

  context = p4est_vtk_write_header(context);
  SC_CHECK_ABORT(context != NULL,
                 P4EST_STRING "_vtk: Error writing vtk header");

  p4est_locidx_t numquads = mesh->p4est->local_num_quadrants;

  sc_array_t *x = sc_array_new_size(sizeof(double), numquads * P4EST_CHILDREN);
  sc_array_t *y = sc_array_new_size(sizeof(double), numquads * P4EST_CHILDREN);

  for (p4est_locidx_t i = 0; i < numquads * P4EST_CHILDREN; ++i) {
    double *xi = (double *)sc_array_index(x, i);
    double *yi = (double *)sc_array_index(y, i);
    xi[0] = mesh->x[i];
    yi[0] = mesh->y[i];
  }

  context = p4est_vtk_write_point_dataf(context, 2, 0, "our_x", x, "our_y", y,
                                        context);
  SC_CHECK_ABORT(context != NULL,
                 P4EST_STRING "_vtk: Error writing point data");

  sc_array_destroy(x);
  sc_array_destroy(y);

  int retval = p4est_vtk_write_footer(context);
  SC_CHECK_ABORT(!retval, P4EST_STRING "_vtk: Error writing footer");
}

void run(MPI_Comm comm, occaDevice device) {
  mesh_t *mesh = mesh_new(comm, device);

  mesh_write(mesh, "mesh_initial");

  // adapt the mesh
  mark_quadrants_rand(mesh);

  p4est_iterate(mesh->p4est, NULL, mesh, fill_user_data, NULL, NULL);

  p4est_refine_ext(mesh->p4est, 0, -1, refine_quads, NULL, replace_quads);
  p4est_coarsen_ext(mesh->p4est, 0, 0, coarsen_quads, NULL, replace_quads);
  p4est_balance_ext(mesh->p4est, P4EST_CONNECT_FULL, NULL, replace_quads);

  p4est_locidx_t *EToOldE =
      calloc(mesh->p4est->local_num_quadrants, sizeof(p4est_locidx_t));

  // {{{
  {
    p4est_locidx_t old_e = 0, e = 0;
    for (p4est_topidx_t t = mesh->p4est->first_local_tree;
         t <= mesh->p4est->last_local_tree; ++t) {
      p4est_tree_t *tree = p4est_tree_array_index(mesh->p4est->trees, t);
      sc_array_t *tquadrants = &tree->quadrants;

      const p4est_locidx_t Q = (p4est_locidx_t)tquadrants->elem_count;
      for (p4est_locidx_t q = 0; q < Q;) {
        p4est_quadrant_t *quad = p4est_quadrant_array_index(tquadrants, q);
        const quad_data_t *d = (quad_data_t *)quad->p.user_data;

        if (quad->level > d->old_level) {
          // refined
          mesh->EToA[old_e] = ADAPT_REFINE;
          for (int i = 0; i < P4EST_CHILDREN; ++i) {
            EToOldE[e + i] = old_e;
          }

          q += P4EST_CHILDREN;
          e += P4EST_CHILDREN;
          ++old_e;
        } else if (quad->level < d->old_level) {
          // coarsened
          mesh->EToA[old_e] = ADAPT_COARSEN;
          EToOldE[e] = old_e;

          ++q;
          ++e;
          old_e += P4EST_CHILDREN;
        } else {
          // nothing
          mesh->EToA[old_e] = ADAPT_NONE;
          EToOldE[e] = old_e;

          ++q;
          ++e;
          ++old_e;
        }
      }
    }
  }
  // }}}

  double *old_x = mesh->x;
  double *old_y = mesh->y;

  mesh->x =
      calloc(mesh->p4est->local_num_quadrants * P4EST_CHILDREN, sizeof(double));
  mesh->y =
      calloc(mesh->p4est->local_num_quadrants * P4EST_CHILDREN, sizeof(double));

  for (p4est_locidx_t e = 0; e < mesh->p4est->local_num_quadrants;) {
    p4est_locidx_t old_e = EToOldE[e];
    int8_t a = mesh->EToA[old_e];
    if (a == ADAPT_REFINE) {
      const double x00 = old_x[0 + P4EST_CHILDREN * old_e];
      const double x10 = old_x[1 + P4EST_CHILDREN * old_e];
      const double x01 = old_x[2 + P4EST_CHILDREN * old_e];
      const double x11 = old_x[3 + P4EST_CHILDREN * old_e];

      const double y00 = old_y[0 + P4EST_CHILDREN * old_e];
      const double y10 = old_y[1 + P4EST_CHILDREN * old_e];
      const double y01 = old_y[2 + P4EST_CHILDREN * old_e];
      const double y11 = old_y[3 + P4EST_CHILDREN * old_e];

      mesh->x[0 + P4EST_CHILDREN * e] = x00;
      mesh->x[1 + P4EST_CHILDREN * e] = 0.5 * x00 + 0.5 * x10;
      mesh->x[2 + P4EST_CHILDREN * e] = 0.5 * x00 + 0.5 * x01;
      mesh->x[3 + P4EST_CHILDREN * e] =
          0.25 * x00 + 0.25 * x10 + 0.25 * x01 + 0.25 * x11;

      mesh->x[4 + P4EST_CHILDREN * e] = 0.5 * x00 + 0.5 * x10;
      mesh->x[5 + P4EST_CHILDREN * e] = x10;
      mesh->x[6 + P4EST_CHILDREN * e] =
          0.25 * x00 + 0.25 * x10 + 0.25 * x01 + 0.25 * x11;
      mesh->x[7 + P4EST_CHILDREN * e] = 0.5 * x10 + 0.5 * x11;

      mesh->x[8 + P4EST_CHILDREN * e] = 0.5 * x00 + 0.5 * x01;
      mesh->x[9 + P4EST_CHILDREN * e] =
          0.25 * x00 + 0.25 * x10 + 0.25 * x01 + 0.25 * x11;
      mesh->x[10 + P4EST_CHILDREN * e] = x01;
      mesh->x[11 + P4EST_CHILDREN * e] = 0.5 * x01 + 0.5 * x11;

      mesh->x[12 + P4EST_CHILDREN * e] =
          0.25 * x00 + 0.25 * x10 + 0.25 * x01 + 0.25 * x11;
      mesh->x[13 + P4EST_CHILDREN * e] = 0.5 * x10 + 0.5 * x11;
      mesh->x[14 + P4EST_CHILDREN * e] = 0.5 * x01 + 0.5 * x11;
      mesh->x[15 + P4EST_CHILDREN * e] = x11;

      mesh->y[0 + P4EST_CHILDREN * e] = y00;
      mesh->y[1 + P4EST_CHILDREN * e] = 0.5 * y00 + 0.5 * y10;
      mesh->y[2 + P4EST_CHILDREN * e] = 0.5 * y00 + 0.5 * y01;
      mesh->y[3 + P4EST_CHILDREN * e] =
          0.25 * y00 + 0.25 * y10 + 0.25 * y01 + 0.25 * y11;

      mesh->y[4 + P4EST_CHILDREN * e] = 0.5 * y00 + 0.5 * y10;
      mesh->y[5 + P4EST_CHILDREN * e] = y10;
      mesh->y[6 + P4EST_CHILDREN * e] =
          0.25 * y00 + 0.25 * y10 + 0.25 * y01 + 0.25 * y11;
      mesh->y[7 + P4EST_CHILDREN * e] = 0.5 * y10 + 0.5 * y11;

      mesh->y[8 + P4EST_CHILDREN * e] = 0.5 * y00 + 0.5 * y01;
      mesh->y[9 + P4EST_CHILDREN * e] =
          0.25 * y00 + 0.25 * y10 + 0.25 * y01 + 0.25 * y11;
      mesh->y[10 + P4EST_CHILDREN * e] = y01;
      mesh->y[11 + P4EST_CHILDREN * e] = 0.5 * y01 + 0.5 * y11;

      mesh->y[12 + P4EST_CHILDREN * e] =
          0.25 * y00 + 0.25 * y10 + 0.25 * y01 + 0.25 * y11;
      mesh->y[13 + P4EST_CHILDREN * e] = 0.5 * y10 + 0.5 * y11;
      mesh->y[14 + P4EST_CHILDREN * e] = 0.5 * y01 + 0.5 * y11;
      mesh->y[15 + P4EST_CHILDREN * e] = y11;

      e += 4;
    } else if (a == ADAPT_COARSEN) {
      for (int i = 0; i < P4EST_CHILDREN; ++i) {
        mesh->x[i + P4EST_CHILDREN * e] = old_x[5 * i + P4EST_CHILDREN * old_e];
        mesh->y[i + P4EST_CHILDREN * e] = old_y[5 * i + P4EST_CHILDREN * old_e];
      }
      e += 1;
    } else {
      for (int i = 0; i < P4EST_CHILDREN; ++i) {
        mesh->x[i + P4EST_CHILDREN * e] = old_x[i + P4EST_CHILDREN * old_e];
        mesh->y[i + P4EST_CHILDREN * e] = old_y[i + P4EST_CHILDREN * old_e];
      }
      e += 1;
    }
  }

  free(old_x);
  free(old_y);
  free(mesh->EToA);
  mesh->EToA = calloc(mesh->p4est->local_num_quadrants, sizeof(int8_t));

  mesh_write(mesh, "mesh_adapted");

  free(EToOldE);

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
