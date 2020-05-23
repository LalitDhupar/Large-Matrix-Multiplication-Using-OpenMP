/**
 * Concurrent Programming
 * Assignment-2 --> Efficient Large Matrix Multiplication in OpenMP
 */

#include <stdio.h>
#include <omp.h> //OpenMp library
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

int size = 1024; // matrix size
int th = 32; // openMP thread count
int thold = 128; //Strassen algorithm lower limit

// defining matrices
double **A, **B, **C, **CC;

void strassenalgorithm( double**, double**, double**, int, int, int, int, int, int, int);
void normalmatrixMultiplication(double**, double**, double**, int,int,int,int,int,int,int);

// Matrix should be square in size and divisible by 2

void matrixaddition( double **p, double **q, double **r, int sz, int r1, int c1, int r2, int c2) {
  // r = p + q
  //r and c defines the rows and columns of the matrix and sz defines the size of the matrix
  int i,j;
  #pragma omp parallel shared(p, q, r, r1, c1, r2, c2, sz) private(i, j) num_threads(th)
  {
    #pragma omp for schedule(static)
    for(i = 0; i < sz; i++) {
      for(j = 0; j < sz; j++) {
        r[i][j] = p[i + r1][j + c1] + q[i + r2][j + c2];
      }
    }
  }
}

void matrixaddition1(double **p, double **q, double **r, int sz, int r1, int c1, int r2, int c2, int r3, int c3) {
  // r = p + q
  //r and c defines the rows and columns of the matrix and sz defines the size of the matrix
  int i,j;
  #pragma omp parallel shared(p, q, r, r1, c1, r2, c2, r3, c3, sz) private(i, j) num_threads(th)
  {
    #pragma omp for schedule(static)
    for(i = r3; i < sz + r3; i++) {
      for(j = r3; j < sz + r3; j++) {
        r[i][j] = p[i - r3 + r1][j - c3 + c1] + q[i - r3 + r2][j - c3 + c2];
      }
    }
  }
}

void matrixaddition2(double **p, double **q, double **r, int sz, int r1, int c1, int r2, int c2) {
  // r = p + q
  //r and c defines the rows and columns of the matrix and sz defines the size of the matrix
  int i,j;
  #pragma omp parallel shared(p, q, r, r1, c1, r2, c2, sz) private (i, j) num_threads(th)
  {
    #pragma omp for schedule(static)
    for(i = r2; i < sz + r2; i++) {
      for(j = c2; j < sz + c2; j++) {
        r[i][j] = p[i - r2 + r1][j - c2 + c1] + q[i - r2][j - c2];
      }
    }
  }
}

void matrixsubtraction(double **p, double **q, double **r, int sz, int r1, int c1, int r2, int c2) {
  // r = p - q
  //r and c defines the rows and columns of the matrix and sz defines the size of the matrix
  int i, j;
  #pragma omp parallel shared(p, q, r, r1, c1, r2, c2, sz) private(i , j) num_threads(th)
  {
    #pragma omp for schedule(static)
    for(i = 0; i < sz; i++) {
      for(j = 0; j < sz; j++) {
        r[i][j] = p[i + r1][j + c1] - q[i + r2][j + c2];
      }
    }
  }
}

void matrixsubtraction1(double **p, double **q, double **r, int sz, int r1, int c1, int r2, int c2, int r3, int c3) {
  // r = p - q
  //r and c defines the rows and columns of the matrix and sz defines the size of the matrix
  int i, j;
  #pragma omp parallel shared(p, q, r, r1, c1, r2, c2, r3, c3, sz) private(i, j) num_threads(th)
  {
    #pragma omp for schedule(static)
    for(i = r3; i < sz + r3; i++) {
      for(j = c3; j < sz + c3; j++) {
        r[i][j] = p[i - r3 + r1][j - c3 + c1] - q[i - r3 + r2][j - c3 + c2];
      }
    }
  }
}

void matrixsubtraction2(double **p, double **q, double **r, int sz, int r1, int c1, int r2, int c2) {
  // r = p - q
  //r and c defines the rows and columns of the matrix and sz defines the size of the matrix
  int i, j;
  #pragma omp parallel shared(p, q, r, r1, c1, r2, c2, sz) private(i, j) num_threads(th)
  {
    #pragma omp for schedule(static)
    for(i = r2; i < sz + r2; i++) {
      for(j = c2; j < sz; j++) {
        r[i][j] = p[i -r2 + r1][j - c2 + c1] - q[i - r2][j - c2];
      }
    }
  }
}
// Multiply the two matrices: r = p * q
void normalmatrixMultiplication(double **p, double **q, double **r, int sz, int r1, int c1, int r2, int c2, int r3, int c3) {
  int i, j, k;
  #pragma omp parallel shared(p, q, r, sz, r1, c1, r2, c2,r3, c3) private(i, j, k) num_threads(th)
  {
    #pragma omp for schedule(static)
    for(i = r3; i < sz + r3; i++) {
      for(j = c3; j < sz; j++) {
        r[i][j] = 0.0;
        for(k = 0; k < sz; k++) {
          r[i][j] += p[i - r3 + r1][k + c1] * q[k + r2][j - c3 + c2];
        }
      }
    }
  }
}

// Strassen algorithm for matrix multiplication
void strassenalgorithm(double **a, double **b, double **c, int sz, int r1, int c1, int r2, int c2, int r3, int c3) {
  double **z1, **z2;
  int  newsz = sz/2;
  int i;

  if(sz >= thold) { // compares the size of the matrix with the threshold value
    // memory allocation
    z1 = (double**) malloc(sizeof(double*)*newsz);
    z2 = (double**) malloc(sizeof(double*)*newsz);

    for(i = 0; i < newsz; i++) {
      z1[i] = (double*) malloc(sizeof(double)*newsz);
      z2[i] = (double*) malloc(sizeof(double)*newsz);
    }

    // calculate M1 for the strassen alogorithm
    matrixaddition(a, a, z1, newsz, 0, 0, newsz, newsz);
    matrixaddition(b, b, z2, newsz, 0, 0, newsz, newsz);
    strassenalgorithm(z1, z1, c, newsz, 0, 0, 0 , 0, newsz, 0);

    // calculate M2 for the strassen alogorithm
    matrixaddition1(c, c, c, newsz, 0, 0, newsz, 0, 0, 0);
    matrixaddition1(c, c, c, newsz, newsz, 0, newsz, newsz, newsz, newsz);
    matrixaddition(a, a, z1, newsz, newsz, 0, newsz, newsz);
    strassenalgorithm(z1, b, c, newsz, 0, 0, 0, 0, newsz, 0);

    // calculate M3 for the strassen alogorithm
    matrixsubtraction(b, b, z2, newsz, 0, newsz, newsz, newsz);
    strassenalgorithm(a, z2, c, newsz, 0, 0, 0, 0, 0, newsz);

    // calculate M4 for the strassen alogorithm
    matrixsubtraction1(c, c, c, newsz, newsz, newsz, newsz, 0, newsz, newsz);
    matrixaddition1(c, c, c, newsz, newsz, newsz, 0, newsz, newsz, newsz);
    matrixsubtraction(b, b, z2, newsz, newsz, 0, 0, 0);
    strassenalgorithm(a, z2, z1, newsz, newsz, newsz, 0, 0, 0, 0);

    // calculate M5 for the strassen alogorithm
    matrixaddition2(c, z1, c, newsz, 0, 0, 0, 0);
    matrixaddition2(c, z1, c, newsz, newsz, 0, newsz, 0);
    matrixaddition(a, a, z1, newsz, 0, 0, 0, newsz);
    strassenalgorithm(z1, b, z2, newsz, 0, 0, newsz, newsz, 0, 0);

    matrixsubtraction2(c, z2, c, newsz, 0, 0, 0, 0);
    matrixaddition2(c, z2, c, newsz, 0, newsz, 0, newsz);

    // calculate M6 for the strassen alogorithm
    matrixsubtraction(a, a, z1, newsz, newsz, 0, 0, 0);
    matrixaddition(b, b, z2, newsz, 0, 0, 0, newsz);
    strassenalgorithm(z1, z2, c, newsz, 0, 0, 0, 0, newsz, newsz);

    // calculate M7 for the strassen alogorithm
    matrixsubtraction(a, a, z1, newsz, 0, newsz, newsz, newsz);
    matrixaddition(b, b, z2, newsz, newsz, 0, newsz, newsz);
    strassenalgorithm(z1, z2, c, newsz, 0, 0, 0, 0, 0, 0);
    // memeory deallocation
    free(z1);
    free(z2);
  }
  else {
    normalmatrixMultiplication(a, b, c, sz, r1, c1, r2, c2, r3, c3);
  }
}

// main method for the program
int main(int argc, char *argv[]) {
  if(argc > 1)
    size = atoi(argv[1]);
  if(argc > 2)
    th = atoi(argv[2]);
  if(argc > 3)
    thold = atoi(argv[3]);

double starttime = 0.0;
double endtime = 0.0;
double totaltime = 0.0;
int i, j, k;

A = (double**) malloc(sizeof(double*)*size);
for(i = 0; i < size; i++) {
  A[i] = (double*) malloc(sizeof(double)*size);
}

B = (double**) malloc(sizeof(double*)*size);
for(i = 0; i < size; i++) {
  B[i] = (double*) malloc(sizeof(double)*size);
}

C = (double**) malloc(sizeof(double*)*size);
for(i = 0; i < size; i++) {
  C[i] = (double*) malloc(sizeof(double)*size);
}

CC = (double**) malloc(sizeof(double*)*size);
for(i = 0; i < size; i++) {
  CC[i] = (double*) malloc(sizeof(double)*size);
}

for(i = 0; i < size; i++) {
  for(j = 0; j < size; j++) {
    A[i][j] = (i + j) * 1.0;
    B[i][j] = (i + j) * 1.0;
    C[i][j] = 0;
    CC[i][j] = 0;
  }
}


printf("Computing Sequential matrix multiplication:\n");
starttime = omp_get_wtime();
for(i = 0; i < size; i++) {
  for(j = 0; j < size; j++) {
    CC[i][j] = 0;
    for(k = 0; k < size; k++) {
      CC[i][j] += A[i][j] * B[i][j];
    }
  }
}

endtime = omp_get_wtime();
//printf("Sequential matrix multiplication completed\n");
totaltime = endtime - starttime;
printf("Sequantial Time taken = %0.3f \n", totaltime);
printf("Computing Strassen matrix multiplication:\n");
starttime = omp_get_wtime();
strassenalgorithm(C, B, C, size, 0, 0, 0, 0, 0, 0);
endtime = omp_get_wtime();
totaltime = endtime - starttime;
//printf("Strassen matrix multiplication compleyted\n");
printf("Strassen Time taken = %0.3f \n", totaltime);
printf("Number of threads used = %d\n", th);
}
