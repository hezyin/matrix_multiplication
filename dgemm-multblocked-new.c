const char* dgemm_desc = "Multiple blocked dgemm, l1 36, l298.";

/*#if !defined(BLOCK_SIZE)*/
#define BLOCK_SIZE_L1 36
#define BLOCK_SIZE_L2 98
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


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
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
						do_block(lda, X, Y, Z, A + (i0+i) + (k0+k)*lda, B + (k0+k) + (j0+j)*lda, C + (i0+i) + (j0+j)*lda);
				}
      }
}
