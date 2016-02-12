#include<string.h>
#include<stdbool.h> 
#include<stdio.h>
#include<xmmintrin.h> // Contain the SSE compiler intrinsics

#define UNROLL4

const char* dgemm_desc = "dgemm with sse.";

#define get_even(M) (((M)%2)?((M)+1):(M))
#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. 
 * but the storage is another story, here A and C have 'number of rows' = block_size_row
 * and B has 'number of rows' = block_size_inner
 * we need them to access the real value in storage
 */
static void do_block (int M, int N, int K, int block_size_row, int block_size_inner, double* A, double* B, double* C)
{
  // registers for SSE computation
  __m128d c0, c1, a0, a1, b0, b1, b2, b3;
  // floor the K so that we could perform 4 computations of C at a time
  // if one of two rows of K are left, we did them in the end.
  int K_floor = K;
  if (K%4 == 2 || K%4 == 1) {
    K_floor = K-2;
  }
  // more registers used when unrolling
  __m128d a2, a3, b4, b5, b6, b7;
  __m128d c2, c3, a4, a5, a6, a7;
  int k = 0;
  // unrolling to 4
  // these codes are a bit complex -- it is helpful to work through the code without unrolling first
  // i.e. codes after this loop
  for (; k < K_floor; k+=4) {
    for (int j = 0; j < N; j+=2) {
      b0 = _mm_load1_pd(B+k+j*block_size_inner);
      b1 = _mm_load1_pd(B+k+1+j*block_size_inner);
      b2 = _mm_load1_pd(B+k+(j+1)*block_size_inner);
      b3 = _mm_load1_pd(B+k+1+(j+1)*block_size_inner);

      b4 = _mm_load1_pd(B+k+2+j*block_size_inner);
      b5 = _mm_load1_pd(B+k+3+j*block_size_inner);
      b6 = _mm_load1_pd(B+k+2+(j+1)*block_size_inner);
      b7 = _mm_load1_pd(B+k+3+(j+1)*block_size_inner);
      // if row is also unrolled to 4
      if (M%4 == 0 || M%4 == 3) {
        for (int i = 0; i < M; i+=4) {
          c0 = _mm_load_pd(C+i+j*block_size_row);
          c1 = _mm_load_pd(C+i+(j+1)*block_size_row);
          c2 = _mm_load_pd(C+i+2+j*block_size_row);
          c3 = _mm_load_pd(C+i+2+(j+1)*block_size_row);
          
          a0 = _mm_load_pd(A+i+k*block_size_row);
          a1 = _mm_load_pd(A+i+(k+1)*block_size_row);
          a2 = _mm_load_pd(A+i+(k+2)*block_size_row);
          a3 = _mm_load_pd(A+i+(k+3)*block_size_row);
          
          a4 = _mm_load_pd(A+i+2+k*block_size_row);
          a5 = _mm_load_pd(A+i+2+(k+1)*block_size_row);
          a6 = _mm_load_pd(A+i+2+(k+2)*block_size_row);
          a7 = _mm_load_pd(A+i+2+(k+3)*block_size_row);
          
          c0 = _mm_add_pd(c0, _mm_mul_pd(a0,b0));
          c1 = _mm_add_pd(c1, _mm_mul_pd(a0,b2));
          c2 = _mm_add_pd(c2, _mm_mul_pd(a4,b0));
          c3 = _mm_add_pd(c3, _mm_mul_pd(a4,b2));
          
          c0 = _mm_add_pd(c0, _mm_mul_pd(a1,b1));
          c1 = _mm_add_pd(c1, _mm_mul_pd(a1,b3));
          c2 = _mm_add_pd(c2, _mm_mul_pd(a5,b1));
          c3 = _mm_add_pd(c3, _mm_mul_pd(a5,b3));

          c0 = _mm_add_pd(c0, _mm_mul_pd(a2,b4));
          c1 = _mm_add_pd(c1, _mm_mul_pd(a2,b6));
          c2 = _mm_add_pd(c2, _mm_mul_pd(a6,b4));
          c3 = _mm_add_pd(c3, _mm_mul_pd(a6,b6));

          c0 = _mm_add_pd(c0, _mm_mul_pd(a3,b5));
          c1 = _mm_add_pd(c1, _mm_mul_pd(a3,b7));
          c2 = _mm_add_pd(c2, _mm_mul_pd(a7,b5));
          c3 = _mm_add_pd(c3, _mm_mul_pd(a7,b7));
        
          _mm_store_pd(C+i+j*block_size_row, c0);
          _mm_store_pd(C+i+(j+1)*block_size_row, c1);
          _mm_store_pd(C+i+2+j*block_size_row, c2);
          _mm_store_pd(C+i+2+(j+1)*block_size_row, c3);
        }
      } else {
        // does not unroll to 4, compute 2 at a time
        for (int i = 0; i < M; i+=2) {
          c0 = _mm_load_pd(C+i+j*block_size_row);
          c1 = _mm_load_pd(C+i+(j+1)*block_size_row);
          
          a0 = _mm_load_pd(A+i+k*block_size_row);
          a1 = _mm_load_pd(A+i+(k+1)*block_size_row);

          a2 = _mm_load_pd(A+i+(k+2)*block_size_row);
          a3 = _mm_load_pd(A+i+(k+3)*block_size_row);

          
          c0 = _mm_add_pd(c0, _mm_mul_pd(a0,b0));
          c1 = _mm_add_pd(c1, _mm_mul_pd(a0,b2));
          c0 = _mm_add_pd(c0, _mm_mul_pd(a1,b1));
          c1 = _mm_add_pd(c1, _mm_mul_pd(a1,b3));

          c0 = _mm_add_pd(c0, _mm_mul_pd(a2,b4));
          c1 = _mm_add_pd(c1, _mm_mul_pd(a2,b6));
          c0 = _mm_add_pd(c0, _mm_mul_pd(a3,b5));
          c1 = _mm_add_pd(c1, _mm_mul_pd(a3,b7));
        
          _mm_store_pd(C+i+j*block_size_row, c0);
          _mm_store_pd(C+i+(j+1)*block_size_row, c1);
        }
      }
    }
  }
  // compute 2 at a time. only used for rows that are left after unrolling to 4
  // this is an easier version
  // it also works if we don't unroll to 4 at all, but that's a bit slower
  // loop k is pulled outside to make computation faster
  // loding b takes more time than loding c and writing back
  for (;k < K; k+=2) {
    for (int j = 0; j < N; j+=2) {
      b0 = _mm_load1_pd(B+k+j*block_size_inner);
      b1 = _mm_load1_pd(B+k+1+j*block_size_inner);
      b2 = _mm_load1_pd(B+k+(j+1)*block_size_inner);
      b3 = _mm_load1_pd(B+k+1+(j+1)*block_size_inner);

      for (int i = 0; i < M; i+=2) {
        c0 = _mm_load_pd(C+i+j*block_size_row);
        c1 = _mm_load_pd(C+i+(j+1)*block_size_row);

        a0 = _mm_load_pd(A+i+k*block_size_row);
        a1 = _mm_load_pd(A+i+(k+1)*block_size_row);

        c0 = _mm_add_pd(c0, _mm_mul_pd(a0,b0));
        c1 = _mm_add_pd(c1, _mm_mul_pd(a0,b2));
        c0 = _mm_add_pd(c0, _mm_mul_pd(a1,b1));
        c1 = _mm_add_pd(c1, _mm_mul_pd(a1,b3));
      
        _mm_store_pd(C+i+j*block_size_row, c0);
        _mm_store_pd(C+i+(j+1)*block_size_row, c1);
      }
    }
  }
}

// straight forward C version 
/*static void do_block (int M, int N, int K, int block_size_row, int block_size_inner, double* A, double* B, double* C)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) 
    {
      double cij = C[i+j*block_size_row];
      for (int k = 0; k < K; ++k)
        cij += A[i+k*block_size_row] * B[k+j*block_size_inner];
      C[i+j*block_size_row] = cij;
    }
  }
}*/

void print(double *mat, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", mat[i+j*N]);
    }
    printf("\n");
  }
  printf("\n");
}

void copy_block(double *dst, double *src, int n_dst, int n_src, int n_row, int n_col, bool padding_zero) {
  for (int j = 0; j < n_col; j++) {
    memcpy(dst+j*n_dst, src+j*n_src, n_row*sizeof(double));
  }
  if (padding_zero) {
    // pad zero if dimension is odd
    if (n_col % 2 == 1) {
      memset(dst+n_col*n_dst, 0, n_row*sizeof(double));
    }
    // pad the row
    if (n_row %2 == 1) {
      for(int j = 0; j < n_col; j++) {
        dst[n_row+j*n_dst] = 0;
      }
      if (n_col % 2 == 1) {
        dst[n_row+n_col*n_dst] = 0;
      }
    }
  }
}

/* This routine performs a dgemm operation
*  C := C + A * B
*  block_size are parameters that specify how we divide matrices into blocks.
*  blocks of C are of size block_size_row x block_size_col,
*  blocks of A are of size block_size_row x block_size_inner,
*  blocks of B are of size block_size_inner x block_size_col,
*/
void square_block_dgemm (int lda, double* A, double* B, double* C, int block_size_row, int block_size_col, int block_size_inner) {
  double *C_block;
  double *B_block;
  double *A_block;
  // to make use of SSE efficiently, we need memory that is aligned to address 
  // I'm not quite sure why we need 64 here, people tend to use 16 more often, but 64 gives better performance
  // when we copy_block later, we always make sure the dimension is multiple of 2, for sse usage, so we add get_even here
  posix_memalign((void **)&C_block, 64, block_size_row * block_size_col * sizeof(double));
  posix_memalign((void **)&A_block, 64, block_size_row * get_even(lda) * sizeof(double));
  posix_memalign((void **)&B_block, 64, get_even(lda) * block_size_col * sizeof(double));

  for (int i = 0; i < lda; i += block_size_row)
  {
    int M = min (block_size_row, lda-i);
    // we copy block of A into memory, and this speeds up inner loops
    bool padding_zero = true;
    copy_block(A_block, A+i, block_size_row, lda, M, lda, padding_zero);
    for (int j = 0; j < lda; j += block_size_col)
    {
      int N = min (block_size_col, lda-j);
      // the same to B as we have done for A
      bool padding_zero = true;
      copy_block(B_block, B+j*lda, get_even(lda), lda, lda, N, padding_zero);
      
      // block of C to update, because we are computing C := C + A * B here
      copy_block(C_block, C+i+j*lda, block_size_row, lda, M, N, padding_zero);

      for (int k = 0; k < lda; k += block_size_inner)
      {
        int K = min (block_size_inner, lda-k);
        // perform block matrix multiplication
        do_block(M, N, K, block_size_row, get_even(lda), A_block+k*block_size_row, B_block+k, C_block);
      }
      // copy back the block of C. we don't want to pad zero to C matrix.
      padding_zero = false;
      copy_block(C+i+j*lda, C_block, lda, block_size_row, M, N, padding_zero);
    }
  }
  free(C_block);
  free(B_block);
  free(A_block);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. 
 * block_size are optimized on hopper via a automatic script that sweep through
 * search space of cartesian product of {2,4,8,16,32,64,128,256}
 */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  int block_size_row = 256;
  int block_size_col = 16;
  int block_size_inner = 64;
  square_block_dgemm(lda, A, B, C, block_size_row, block_size_col, block_size_inner);
}
