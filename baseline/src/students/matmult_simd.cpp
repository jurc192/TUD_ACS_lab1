// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../matmult.hpp"

// Intel intrinsics for SSE/AVX:
#include <immintrin.h>

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O1")
/*************************************/

typedef union _avxd {
  __m256d val;
  double arr[4];
} avxd;


Matrix<float> multiplyMatricesSIMD(Matrix<float> a, Matrix<float> b) {

  // a.cols == b.rows !!! always true
  auto rows = a.rows;
  auto cols = b.columns;
  auto result = Matrix<float>(rows, cols);

  // Transpose the matrix B -> better memory access
  auto bT = Matrix<float>(b.columns, b.rows);
  for(size_t r = 0; r < bT.rows; r++) {
    for(size_t c = 0; c < bT.columns; c++) {
      bT(r, c) = b(c, r);
    }
  }


  for(size_t r = 0; r < rows; r++) {
    for(size_t c = 0; c < cols; c++) {

      float tmp = 0.0;
      for(size_t i = 0; i < a.columns/8; i++) {

        // AVX load vectors
        __m256 vec_a = _mm256_loadu_ps(&a(r, i*8));
        __m256 vec_b = _mm256_loadu_ps(&bT(c, i*8));

        // Multiplication
        __m256 vec_r = _mm256_mul_ps(vec_a, vec_b);

        // Addition - the stupid way
        // Tukaj moram vse elemente vektorja vec_r seštet in rezultat prištet v tmp
        float *r = (float *)&vec_r;
        tmp += (r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7]);

      }

      // Do the remainder
      auto rem = b.rows % 8;
      if (rem != 0) {
        
        // Vector mask: pos numbers - zeroes, neg numbers - remainder
        __m256i vec_mask = _mm256_set_epi32(7-rem, 6-rem, 5-rem, 4-rem, 3-rem, 2-rem, 1-rem, 0-rem);

        // AVX load vectors
        __m256 vec_a = _mm256_maskload_ps(&a(r, a.columns-rem), vec_mask);
        __m256 vec_b = _mm256_maskload_ps(&bT(c, bT.columns-rem), vec_mask);

        // Multiplication
        __m256 vec_r = _mm256_mul_ps(vec_a, vec_b);

        // Addition - the stupid way
        // Tukaj moram vse elemente vektorja vec_r seštet in rezultat prištet v tmp
        float *r = (float *)&vec_r;
        tmp += (r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7]);

      }

      result(r, c) = tmp;
    }
  }

  return result;
}



Matrix<double> multiplyMatricesSIMD(Matrix<double> a, Matrix<double> b) {
  
  // a.cols == b.rows !!! always true
  auto rows = a.rows;
  auto cols = b.columns;
  auto result = Matrix<double>(rows, cols);

  // Transpose the matrix B -> better memory access
  auto bT = Matrix<double>(b.columns, b.rows);
  for(size_t r = 0; r < bT.rows; r++) {
    for(size_t c = 0; c < bT.columns; c++) {
      bT(r, c) = b(c, r);
    }
  }


  for(size_t r = 0; r < rows; r++) {
    for(size_t c = 0; c < cols; c++) {

      double tmp = 0.0;
      for(size_t i = 0; i < a.columns/4; i++) {

        // AVX load vectors
        __m256d vec_a = _mm256_loadu_pd(&a(r, i*4));
        __m256d vec_b = _mm256_loadu_pd(&bT(c, i*4));    

        // Multiplication
        __m256d vec_r = _mm256_mul_pd(vec_a, vec_b);

        // Addition - the stupid way
        // Tukaj moram vse elemente vektorja vec_r seštet in rezultat prištet v tmp
        double *r = (double *)&vec_r;
        tmp += (r[0] + r[1] + r[2] + r[3]);
        
      }

      // Do the remainder
      auto rem = b.rows % 4;
      if (rem != 0) {
        
        // Vector mask: pos numbers - zeroes, neg numbers - remainder
        __m256i vec_mask = _mm256_set_epi32(3-rem, 3-rem, 2-rem, 2-rem, 1-rem, 1-rem, 0-rem, 0-rem);

        // AVX load vectors
        __m256d vec_a = _mm256_maskload_pd(&a(r, a.columns-rem), vec_mask);
        __m256d vec_b = _mm256_maskload_pd(&bT(c, bT.columns-rem), vec_mask);

        // Multiplication
        __m256d vec_r = _mm256_mul_pd(vec_a, vec_b);

        // Addition - the stupid way
        // Tukaj moram vse elemente vektorja vec_r seštet in rezultat prištet v tmp
        double *r = (double *)&vec_r;
        tmp += (r[0] + r[1] + r[2] + r[3]);
      }

      result(r, c) = tmp;
    }
  }

  return result;

}

/*************************************/
#pragma GCC pop_options
/*************************************/
