/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    for (int i = 0; i < n; i++) {
        y[i] = 0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    for (int i = 0; i < n; i++) {
        y[i] = 0;
        for (int j = 0; j < m; j++) {
            y[i] += A[i * m + j] * x[j];
        }
    }
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    for (int i = 0; i < n; i++) x[i] = 0;

    double* D = (double *) malloc(sizeof(double)*(n*n));
    double* R = (double *) malloc(sizeof(double)*(n*n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                D[i*n+j] = A[i*n+j];
                R[i*n+j] = 0;
            } else {
                D[i*n+j] = 0;
                R[i*n+j] = A[i*n+j];
            }
        }
    }

    for (int iter = 0; iter < max_iter; iter++) {
        //calculate 2 norm
        double* sol = (double *) malloc(sizeof(double)*n);
        matrix_vector_mult(n, A, x, sol);
        for (int i = 0; i < n; i++) {
            sol[i] -= b[i];
        }
        double sol2;
        matrix_vector_mult(1, n, sol, sol, &sol2);
        double _2norm = sqrt(sol2);
        free(sol);

        if (_2norm > l2_termination) {
            double* b_Rx = (double *) malloc(sizeof(double)*n);
            matrix_vector_mult(n, R, x, b_Rx);
            for(int i = 0; i < n; i++){
                x[i] = (b[i] - b_Rx[i])/D[i*n+i];
            }
            free(b_Rx);
        } else {
            break;
        }
    }

    free(D);
    free(R);
}
