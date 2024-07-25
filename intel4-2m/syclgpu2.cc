//==============================================================
// This code performs GPU offloading using sycl.
// =============================================================
#include "minitest.h"
#include <vector>
#include <iostream>
#include <string>


#define kkmax 2000
// sycl::default_selector d_selector;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

void
twork2( int iter, int threadnum)
{
  hrtime_t starttime = gethrtime();

  int nelements = nn;
  double *l1 = lptr[threadnum];
  double *r1 = rptr[threadnum];
  double *p1 = pptr4[threadnum];

  buffer<double, 1> a(l1, nn);
  buffer<double, 1> b(r1, nn);
  buffer<double, 1> c(p1, nn);

  // run kernel 10 times
  for (int i = 0; i < 1; i++) {
    q4.submit([&](handler &h) {
      accessor d_l1(a, h, read_only);
      accessor d_r1(b, h, read_only);
      accessor d_p1(c, h, write_only);
      h.parallel_for(nelements, [=](auto i) {
        for (int kk = 0 ; kk < kkmax ; kk++ ) {
          d_p1[i] = d_p1[i] + d_l1[nelements - kk] / double(kkmax) + d_r1[kk] / double(kkmax);
        }
      } );
    } );
  }

  for (int k = 0; k < omp_num_t; k++) {
    output(k, pptr4[k], nn, "result p4 array");
    checkdata(k, pptr4[k], nn );
  }

  q4.submit([&](handler &h) {
    accessor d_l1(a, h, read_only);
    accessor d_r1(b, h, read_only);
    accessor d_p1(c, h, write_only);
    h.parallel_for(nelements, [=](auto i) {
      for (int kk = 0 ; kk < kkmax ; kk++ ) {
        d_p1[i] = d_p1[i] + d_l1[nelements - kk] / double(kkmax) + d_r1[kk] / double(kkmax);
      }
    } );
  } );

  for (int k = 0; k < omp_num_t; k++) {
    output(k, pptr4[k], nn, "result p4 array");
    checkdata(k, pptr4[k], nn );
  }


  hrtime_t endtime = gethrtime();
  double  tempus =  (double) (endtime - starttime) / (double)1000000000.;
#if 1
 fprintf(stderr, "    [%d] Completed iteration %d, thread %d in %13.9f s.\n",
   thispid, iter, threadnum, tempus);
#endif
  spacer(50, true);

}


void
twork3( int iter, int threadnum)
{
  hrtime_t starttime = gethrtime();

  int nelements = nn;
  double *l1 = lptr[threadnum];
  double *r1 = rptr[threadnum];
  double *p1 = pptr4[threadnum];

  buffer<double, 1> a(l1, nn);
  buffer<double, 1> b(r1, nn);
  buffer<double, 1> c(p1, nn);

  // run kernel 10 times
  for (int i = 0; i < 1; i++) {
    q4.submit([&](handler &h) {
      accessor d_l1(a, h, read_only);
      accessor d_r1(b, h, read_only);
      accessor d_p1(c, h, write_only);
      h.parallel_for(nelements, [=](auto i) {
        for (int kk = 0 ; kk < kkmax ; kk++ ) {
          d_p1[i] = d_p1[i] + d_l1[nelements - kk] / double(kkmax) + d_r1[kk] / double(kkmax);
        }
      } );
    } );
  }

  for (int k = 0; k < omp_num_t; k++) {
    output(k, pptr4[k], nn, "result p4 array");
    checkdata(k, pptr4[k], nn );
  }

  q4.submit([&](handler &h) {
    accessor d_l1(a, h, read_only);
    accessor d_r1(b, h, read_only);
    accessor d_p1(c, h, write_only);
    h.parallel_for(nelements, [=](auto i) {
      for (int kk = 0 ; kk < kkmax ; kk++ ) {
        d_p1[i] = d_p1[i] + d_l1[nelements - kk] / double(kkmax) + d_r1[kk] / double(kkmax);
      }
    } );
  } );

  for (int k = 0; k < omp_num_t; k++) {
    output(k, pptr4[k], nn, "result p4 array");
    checkdata(k, pptr4[k], nn );
  }


  hrtime_t endtime = gethrtime();
  double  tempus =  (double) (endtime - starttime) / (double)1000000000.;
#if 1
 fprintf(stderr, "    [%d] Completed iteration %d, thread %d in %13.9f s.\n",
   thispid, iter, threadnum, tempus);
#endif
  spacer(50, true);

}
