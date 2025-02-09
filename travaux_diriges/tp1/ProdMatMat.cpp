#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

// namespace {
// void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
//                    const Matrix& A, const Matrix& B, Matrix& C) {
//   #pragma omp parallel for collapse(2)
//   for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
//     for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
//       for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
//         C(i, j) += A(i, k) * B(k, j);
// }
// const int szBlock = 32;
// }  // namespace

// Matrix operator*(const Matrix& A, const Matrix& B) {
//   Matrix C(A.nbRows, B.nbCols, 0.0);
//   prodSubBlocks(0, 0, 0, std::max({A.nbRows, B.nbCols, A.nbCols}), A, B, C);
//   return C;
// }

/*
Test for 1023: OMP_NUM_THREADS=2 res=1.04964s
Test for 1023: OMP_NUM_THREADS=4 res=0.552998s
Test for 1023: OMP_NUM_THREADS=8 res=0.340979s
*/

namespace {

const int szBlock = 64; // 块大小可以调整以优化性能

void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  #pragma omp parallel
  {
    Matrix C_local(A.nbRows, B.nbCols, 0.0); // 每个线程的局部累加器

    #pragma omp for collapse(2) nowait
    for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i) {
      for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); ++k) {
        for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); ++j) {
          C_local(i, j) += A(i, k) * B(k, j);
        }
      }
    }

    #pragma omp critical
    for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
      for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); ++j)
        C(i, j) += C_local(i, j);
  }
}

void blockMatrixMultiply(const Matrix& A, const Matrix& B, Matrix& C) {
  #pragma omp parallel for collapse(3) schedule(dynamic)
  for (int i = 0; i < A.nbRows; i += szBlock) {
    for (int j = 0; j < B.nbCols; j += szBlock) {
      for (int k = 0; k < A.nbCols; k += szBlock) {
        prodSubBlocks(i, j, k, szBlock, A, B, C);
      }
    }
  }
}

}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  blockMatrixMultiply(A, B, C);
  return C;
}

/*
Q5
Test for 1024: OMP_NUM_THREADS=2 res=1.1271s
Test for 1024: OMP_NUM_THREADS=4 res=0.588318s
Test for 1024: OMP_NUM_THREADS=8 res=0.370747s

Q7
Test for 1024: OMP_NUM_THREADS=2 res=1.38966s
Test for 1024: OMP_NUM_THREADS=4 res=0.731067s
Test for 1024: OMP_NUM_THREADS=8 res=0.733767s
*/