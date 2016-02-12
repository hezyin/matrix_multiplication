#include <stdlib.h>
#include <stdio.h>
const char* dgemm_desc = "Multiple transpose blocked dgemm, 16, 42, 4.";

/*code does not yet work; there's some error on memory allocation*/
/*#if !defined(BLOCK_SIZE)*/
#define BLOCK_SIZE_L1 42
#define BLOCK_SIZE_L2 98
#define t_block_size 4
/*#endif*/
/*original block_size 41*/
#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k)
	cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}

static void do_block1 (int lda, int M, int N, int K, double* A, double* t_B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k)
	cij += A[i+k*lda] * t_B[j+k*lda];
      C[i+j*lda] = cij;
    }
}
inline void transpose_scalar_block(float *A, float *B, const int lda, const int ldb, const int block_size) {
    /*pragma omp parallel for*/
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
            B[j*ldb + i] = A[i*lda +j];
        }
    }
}

inline void transpose_block1(float *A, float *B, const int n, const int m, const int lda, const int ldb, const int block_size) {
    /*pragma omp parallel for*/
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            transpose_scalar_block(&A[i*lda +j], &B[j*ldb + i], lda, ldb, block_size);
        }
    }
}

static void transpose_block(int lda, double* B, double* t_B){
	for(int i=0; i<t_block_size;i++){
		for(int j=0;j<t_block_size;j++){
			t_B[j+i*lda] = B[i+j*lda];
		}
	}
}
static void transpose (int lda, double* B, double* t_B)
{
	/*unroll later*/
	for(int n=0; n<lda; n+=t_block_size){
		for(int m=0; m<lda; m+= t_block_size){
			transpose_block(lda, B+n+m*lda, t_B+m+n*lda);
		}
	}
}

static void transpose_naive(int lda, double* B, double* t_B){
	for(int i=0; i<lda;i++){
		for(int j=0;j<lda;j++){
			t_B[j+i*lda] = B[i+j*lda];
		}
	}
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm1 (int lda, double* A, double* B, double* C)
{

  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE_L2)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE_L2)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE_L2)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE_L2, lda-i);
	int N = min (BLOCK_SIZE_L2, lda-j);
	int K = min (BLOCK_SIZE_L2, lda-k);
		for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE_L1)
			for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE_L1)
				for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE_L1)
				{
					int X = min (BLOCK_SIZE_L1, M-i0);
					int Y = min (BLOCK_SIZE_L1, N-j0);
					int Z = min (BLOCK_SIZE_L1, K-k0);
					/* Perform individual block dgemm */
						do_block1(lda, X, Y, Z, A + (i0+i) + (k0+k)*lda, B + (k0+k) + (j0+j)*lda, C + (i0+i) + (j0+j)*lda);
				}
      }
}

void square_dgemm (int lda, double* A, double* B, double* C)
{
	double* mem = NULL;
	mem = (double*) malloc(lda*lda*sizeof(double*));
	/*printf("whoo");*/
	double* t_B = mem + 0;
	transpose_naive(lda, B, t_B);
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE_L2)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE_L2)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE_L2)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE_L2, lda-i);
	int N = min (BLOCK_SIZE_L2, lda-j);
	int K = min (BLOCK_SIZE_L2, lda-k);
		for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE_L1)
			for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE_L1)
				for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE_L1)
				{
					int X = min (BLOCK_SIZE_L1, M-i0);
					int Y = min (BLOCK_SIZE_L1, N-j0);
					int Z = min (BLOCK_SIZE_L1, K-k0);
					/* Perform individual block dgemm */
						do_block1(lda, X, Y, Z, A + (i0+i) + (k0+k)*lda, t_B + (j0+j) + (k0+k)*lda, C + (i0+i) + (j0+j)*lda);
				}
      }
}
