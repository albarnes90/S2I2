//
// author: Ed Valeev (eduard@valeyev.net)
// date  : July 9, 2014
// the use of this software is permitted under the conditions GNU General Public License (GPL) version 2
//

#include <cassert>
#ifdef HAVE_MKL
#  include <mkl_cblas.h>
#else
#  include <cblas.h>
#endif
// Eigen library Core capabilities
#ifdef HAVE_EIGEN
#  include <Eigen/Core>
#endif

#include "gemm_kernels.h"

void dgemm(const double* a, const double* b, double* c, size_t n,
           size_t nrepeats) {

  for (int r = 0; r < nrepeats; ++r) {

    size_t ij = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j, ++ij) {

        double v = 0.0;
        size_t ik = i * n;
        size_t kj = j;

        // play with various compiler pragmas here e.g.
// #pragma ivdep
        for (int k = 0; k < n; ++k, ++ik, kj += n) {
          v += a[ik] * b[kj];
        }

        c[ij] = v;
      }
    }

  }

}

void dgemm_blocked(const double* a, const double* b, double* c, size_t n,
                   size_t nrepeats, size_t bsize) {

  // number of blocks
  auto nb = n / bsize;
  assert(n % bsize == 0);

  for (int r = 0; r < nrepeats; ++r) {
  
    size_t ij = 0;
    for (int Ib = 0; Ib < nb; ++Ib) {
      for (int Jb = 0; Jb < nb; ++Jb) {
        for (int Kb = 0; Kb < nb; ++Kb) {

          const int istart = Ib * bsize;
          const int ifence = istart + bsize;
          for (int i = istart; i < ifence; ++i) {

            const int jstart = Jb * bsize;
            const int jfence = jstart + bsize;
            size_t ij = i * n + jstart;
            for (int j = jstart; j < jfence; ++j, ++ij) {

              double v = 0.0;

              const int kstart = Kb * bsize;
              const int kfence = kstart + bsize;
              size_t ik = i * n + kstart;
              size_t kj = kstart * n + j;

#pragma ivdep
              for (int k = kstart; k < kfence; ++k, ++ik, kj += n) {
                v += a[ik] * b[kj];
              }

              c[ij] = v;
            }
          }

        }
      }
    }

  }
}
//Here we are implimenting the strassen algorithm. This requires dividing the matrix up into 4 equal blocks
void dgemm_strassen(const double* a, const double* b, double* c, size_t n,
                   size_t nrepeats, size_t bsize) {

  // number of blocks
  auto nb = 2;
  auto bsize = n/nb;
  auto bsize2 = bsize*bsize;
  assert(n % bsize == 0);

// we need to calculate elements of m1 - m7 from the blocks of a and b

  for (int r = 0; r < nrepeats; ++r) {
  double m1 = 0.0;
  double m2 = 0.0;
  double m3 = 0.0;
  double m4 = 0.0;
  double m5 = 0.0;
  double m6 = 0.0;
  double m7 = 0.0;

  constant int i1 = 0;
  constant int i2 = i1 + bsize;
    size_t ij = 0;
    for (int i = 0; i < bsize-1; ++i, ++i1, ++i2) {
      constant int j1 = 0;
      constant int j2 = j1 + bsize; 
      size_t ij11 = i1 * n + j1;
      size_t ij12 = i1 * n + j2;
      size_t ij21 = i2 * n + j1;
      size_t ij22 = i2 * n + j2;   
      for (int j = 0; j < bsize-1; ++j, ++ij11, ++ij12, ++ij21, ++ij22) {

        for (int k = 0; k < bsize-1; ++k) {

              int k1 = k; // need to move all of this outside of the for loop
              int k2 = k1 + bsize;
              size_t ik11 = i1 * n + k1;
              size_t ik12 = i1 * n + k2;
              size_t ik21 = i2 * n + k1;
              size_t ik22 = i2 * n + k2;
              size_t kj11 = k1 * n + j1;
              size_t kj12 = k1 * n + j2;
              size_t kj21 = k2 * n + j1;
              size_t kj22 = k2 * n + j2;

#pragma ivdep
                m1 += (a[ik11]+a[ik22]) * (b[kj11]+b[kj22]);
                m2 += (a[ik21]+a[ik22]) * b[kj11];
                m3 += a[ik11] * (b[kj12]-b[kj22]);
                m4 += a[ik22] * (b[kj21]-b[kj11]);
                m5 += (a[ik11]+a[ik12])*b[kj22];
                m6 += (a[ik21]-a[ik11])*(b[kj11]+b[kj12]);
                m7 += (a[ik12]-a[ik22])*(b[kj21]+b[kj22]);
              }

              c[ij11] = m1+m4-m5+m7;
              c[ij12] = m3+m5;
              c[ij21] = m2+m4;
              c[ij22] = m1-m2+m3+m6;
        }
      }
    }

  }
}

void dgemm_blas(const double* a, const double* b, double* c, size_t n,
                size_t nrepeats) {

  for (int r = 0; r < nrepeats; ++r) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n,
                b, n, 1.0, c, n);
  }

}

#ifdef HAVE_EIGEN
void dgemm_eigen(const double* a, const double* b, double* c, size_t n,
                 size_t nrepeats) {

  using namespace Eigen;
  typedef Eigen::Matrix<double,
                        Eigen::Dynamic,
                        Eigen::Dynamic,
                        Eigen::RowMajor> Matrix; // row-major dynamically-sized matrix of double
  Eigen::Map<const Matrix> aa(a, n, n);
  Eigen::Map<const Matrix> bb(b, n, n);
  Eigen::Map<Matrix> cc(c, n, n);
  for(size_t i = 0; i < nrepeats; ++i) {
    cc = aa * bb;
  }
}
#endif
