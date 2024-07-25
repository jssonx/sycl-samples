#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "minitest.h"

double **lptr;
double **rptr;
double **pptr;
double **pptr2;
double **pptr3;
double **pptr4;

/* Parameters governing the size of the test */
#define        N 40000000      /* size of the data arrays used */
#define        NITER 1         /* number of iterations performed by each thread */

size_t nn = N;
int niter = NITER;
int omp_num_t;


int
main(int argc, char **argv, char **envp)
{
  /* setup_run -- parse the arguments, to reset N and NITER, as requested */
  setup_run(argc, argv);

  /* check number and accessibility of GPU devices */
  initgpu();

  /* set thread count to one */
  omp_num_t = 1;
  fprintf(stderr, "    [%d] This test will use a single CPU thread with data array size = %ld; for %d iterations\n\n",
    thispid, nn, niter );

  /* Allocate and initialize data */

  /* allocate pointer arrays for the threads */
  rptr = (double **) calloc(omp_num_t, sizeof(double *) );
  lptr = (double **) calloc(omp_num_t, sizeof(double *) );
  pptr = (double **) calloc(omp_num_t, sizeof(double *) );
  pptr2 = (double **) calloc(omp_num_t, sizeof(double *) );
  pptr3 = (double **) calloc(omp_num_t, sizeof(double *) );
  pptr4 = (double **) calloc(omp_num_t, sizeof(double *) );

  for ( int k = 0; k < omp_num_t; k++) {
    /* allocate and initialize the l and r arrays */
    lptr[k] = (double *) malloc (nn * sizeof(double) );
    if(lptr[k] == NULL) {
      fprintf(stderr, "[%d] Allocation for lptr[%d] failed; aborting\n", thispid, k);
      abort();
    }
    init(lptr[k], nn);

    rptr[k] = (double *) malloc (nn * sizeof(double) );
    if(rptr[k] == NULL) {
      fprintf(stderr, "[%d] Allocation for rptr[%d] failed; aborting\n", thispid, k);
      abort();
    }
    init(rptr[k], nn);

    /* allocate and clear the result array */
    pptr[k] = (double *) calloc(nn, sizeof(double) );
    if(pptr[k] == NULL) {
      fprintf(stderr, "[%d] Allocation for pptr[%d] failed; aborting\n", thispid, k);
      abort();
    }

    pptr2[k] = (double *) calloc(nn, sizeof(double) );
    if(pptr2[k] == NULL) {
      fprintf(stderr, "[%d] Allocation for pptr2[%d] failed; aborting\n", thispid, k);
      abort();
    }

    pptr3[k] = (double *) calloc(nn, sizeof(double) );
    if(pptr3[k] == NULL) {
      fprintf(stderr, "[%d] Allocation for pptr2[%d] failed; aborting\n", thispid, k);
      abort();
    }

    pptr4[k] = (double *) calloc(nn, sizeof(double) );
    if(pptr4[k] == NULL) {
      fprintf(stderr, "[%d] Allocation for pptr2[%d] failed; aborting\n", thispid, k);
      abort();
    }
  }

  /* perform the number of iterations requested */
  fprintf(stderr, "  [%d] start %d iteration%s\n", thispid, niter, (niter ==1 ? "" : "s") );
  for (int k = 0; k < niter; k++) {
#if 0
    fprintf(stderr, "    [%d] start iteration %d\n", thispid, k);
#endif
    {
      twork(k, 0 );
      twork2(k, 0 );
      twork3(k, 0 );
    }
    mpisync();
#if 0
    fprintf(stderr, "  [%d] end     iteration %d\n", thispid, k);
#endif
  }
  fprintf(stderr, "  [%d] end %d iteration%s\n", thispid, niter,  (niter ==1 ? "" : "s") );

  /* write out various elements in each thread's result array */
  // for (int k = 0; k < omp_num_t; k++) {
  //   output(k, pptr[k], nn, "result p array");
  //   checkdata(k, pptr[k], nn );
  // }

  // for (int k = 0; k < omp_num_t; k++) {
  //   output(k, pptr2[k], nn, "result p2 array");
  //   checkdata(k, pptr2[k], nn );
  // }

  // for (int k = 0; k < omp_num_t; k++) {
  //   output(k, pptr3[k], nn, "result p3 array");
  //   checkdata(k, pptr3[k], nn );
  // }

  // for (int k = 0; k < omp_num_t; k++) {
  //   output(k, pptr4[k], nn, "result p4 array");
  //   checkdata(k, pptr4[k], nn );
  // }

  teardown_run();

  return 0;
}

#include "maincommon.cc"
