/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>

/*
 * TODO: Implement your solutions here
 */


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    // obtain the necessary parameters required in thie method
    int rank;
    MPI_Comm_rank(comm, &rank);
    int p;
    MPI_Comm_size(comm, &p);
    int q = sqrt(p);
    int dims[2];
    int coords[2];
    int periods[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    int sourceRank;
    int sourceCoord[2] = {0, 0};
    MPI_Cart_rank(comm, sourceCoord, &sourceRank);

    //initialization of sendcounts and displs
    int *sendcounts = (int*) malloc(sizeof(int) * p);
    int *displs = (int*) malloc(sizeof(int) * p);
    if (rank == sourceRank) {
        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                if (j != 0) {
                    sendcounts[i * q + j] = 0;
                } else {
                    sendcounts[i * q + j] = block_decompose(n, q, i);
                }
                if (i == 0 && j == 0) {
                    displs[0] = 0;
                } else {
                    displs[i * q + j] = displs[i * q + j - 1] + sendcounts[i * q + j - 1];
                }
            }
        }
    }

    //perform the distribution using scatterv, and store the result in local_vector
    int count_first_col = 0;
    if (coords[1] == 0) { //the first column
        count_first_col = block_decompose(n, q, coords[0]);
    }
    *local_vector = (double*) malloc(sizeof(double) * count_first_col);
    MPI_Scatterv(input_vector, sendcounts, displs, MPI_DOUBLE, *local_vector, count_first_col, MPI_DOUBLE, sourceRank, comm);

    // free the useless memory
    free(sendcounts);
    free(displs);
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    // obtain the necessary parameters required in thie method
    int rank;
    MPI_Comm_rank(comm, &rank);
    int p;
    MPI_Comm_size(comm, &p);
    int q = sqrt(p);
    int dims[2];
    int coords[2];
    int periods[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    int sourceRank;
    int sourceCoord[2] = {0, 0};
    MPI_Cart_rank(comm, sourceCoord, &sourceRank);

    //initialization of sendcounts and displs
    int *count_first_col = (int*) malloc(sizeof(int) * p);
    int *displs = (int*) malloc(sizeof(int) * p);
    if (rank == sourceRank) {
        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                if (j != 0) {
                    count_first_col[i * q + j] = 0;
                } else {
                    count_first_col[i * q + j] = block_decompose(n, q, i);
                }
                if (i == 0 && j == 0) {
                    displs[0] = 0;
                } else {
                    displs[i * q + j] = displs[i * q + j - 1] + count_first_col[i * q + j - 1];
                }
            }
        }
    }

    // perform the gather operation using Gatherv, and store the result in output_vector
    int sendcounts = 0;
    if (coords[1] == 0) {
        sendcounts = block_decompose(n, q, coords[0]);
    }
    MPI_Gatherv(local_vector, sendcounts, MPI_DOUBLE, output_vector, count_first_col, displs, MPI_DOUBLE, sourceRank, comm);

    // free the useless memory
    free(count_first_col);
    free(displs);
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // obtain the necessary parameters required in thie method
    int rank;
    MPI_Comm_rank(comm, &rank);
    int p;
    MPI_Comm_size(comm, &p);
    int q = sqrt(p);
    int dims[2];
    int coords[2];
    int periods[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    int sourceRank;
    int sourceCoord[2] = {0, 0};
    MPI_Cart_rank(comm, sourceCoord, &sourceRank);

    //initialization of the dimention size in each processor
    int count_dim_0 = block_decompose_by_dim(n, comm, 0);
    int count_dim_1 = block_decompose_by_dim(n, comm, 1);

    //initialization of sendcounts and displs
    int *sendcounts = (int*) malloc(sizeof(int) * p);
    int *displs = (int*) malloc(sizeof(int) * p);
    displs[0] = 0;
    if (rank == sourceRank) {
        for (int i = 0; i < q; i++) {
            if (i != 0) {
                displs[i * q] = displs[i * q - 1] + sendcounts[i * q - 1] +
                    (block_decompose(n, q, i - 1) - 1) * n;
            }
            sendcounts[i * q] = (block_decompose(n, q, i) == 0) ? 0 : block_decompose(n, q, 0); //这里已经改了一些
            for (int j = 1; j < q; j++) {
                sendcounts[i * q + j] = (block_decompose(n, q, i) == 0) ? 0 : block_decompose(n, q, j);
                displs[i * q + j] = displs[i * q + j - 1] + sendcounts[i * q + j - 1];
            }
        }
    }

    //perform the matrix distribution for the first n/p lines in each processor
    *local_matrix = (double*) malloc(sizeof(double) * (count_dim_1 * count_dim_0));
    //In each iteration, send n/q lines of data in input matrix to the target place
    int rec_count = (count_dim_0 == 0) ? 0 : count_dim_1;
    for (int iter = 0; iter < n / q; iter++) {
        double *sendbuff = input_matrix + n * iter;
        double *rec_buff = *local_matrix + rec_count * iter;
        MPI_Scatterv(sendbuff, sendcounts, displs, MPI_DOUBLE, rec_buff, rec_count, MPI_DOUBLE, sourceRank, comm);
    }

    //deal with the leftover lines in the processors whose has the value: n % q != 0
    if (n % q != 0) {
        if (rank == 0) {
            for (int i = n % q; i < q; i++) {
                for (int j = 0; j < q; j++) {
                    sendcounts[i * q + j] = 0;
                    displs[i * q + j] = 0;
                }
            }
        }
        if (coords[0] >= n % q) rec_count = 0;
        int iter = n / q;
        double *sendbuff = input_matrix + n * iter;
        double *rec_buff = *local_matrix + rec_count * iter;
        MPI_Scatterv(sendbuff, sendcounts, displs, MPI_DOUBLE, rec_buff, rec_count, MPI_DOUBLE, sourceRank, comm);
    }

    // free the useless memory
    free(sendcounts);
    free(displs);
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    // obtain the necessary parameters required in thie method
    int rank;
    MPI_Comm_rank(comm, &rank);
    int p;
    MPI_Comm_size(comm, &p);
    int q = sqrt(p);
    int dims[2];
    int coords[2];
    int periods[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    MPI_Status stat;

    //first column send to diagonal processors
    int send_rec_count = block_decompose(n, q, coords[0]);
    if (coords[1] == 0) {
        MPI_Send(col_vector, send_rec_count, MPI_DOUBLE, (coords[0] * (q + 1)), 0, comm);
    }
    if (coords[0] == coords[1]) {
        MPI_Recv(row_vector, send_rec_count, MPI_DOUBLE, coords[0] * q, 0, comm, &stat);
    }

    //broadcast columnwise
    MPI_Comm n_comm;
    MPI_Comm_split(comm, coords[1], rank, &n_comm);
    MPI_Bcast(row_vector, block_decompose(n, q, coords[1]), MPI_DOUBLE, coords[1], n_comm);

    // free the useless memory
    MPI_Comm_free(&n_comm);
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // obtain the necessary parameters required in thie method
    int rank;
    MPI_Comm_rank(comm, &rank);
    int p;
    MPI_Comm_size(comm, &p);
    int q = sqrt(p);
    int dims[2];
    int coords[2];
    int periods[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    //distribute x using transpose method
    double* local_sol = (double*) malloc(sizeof(double) * block_decompose(n, q, coords[1]));
    transpose_bcast_vector(n, local_x, local_sol, comm);

    //calculate locally on each processor
    int rowBound = block_decompose(n, q, coords[0]);
    int colBound = block_decompose(n, q, coords[1]);
    double* temp_sol = (double*) malloc(sizeof(double) * rowBound);
    for (int i = 0; i < rowBound; i++) {
        temp_sol[i] = 0;
    }
    for(int i = 0; i < rowBound; i++){
        for(int j = 0; j < colBound;j++){
            temp_sol[i] += local_A[i * colBound + j] * local_sol[j];
        }
    }

    //reduce sum of rows to get the solution
    MPI_Comm n_comm;
    MPI_Comm_split(comm, coords[0], rank, &n_comm);
    MPI_Reduce(temp_sol, local_y, rowBound, MPI_DOUBLE, MPI_SUM, 0, n_comm);

    // free the useless memory
    MPI_Comm_free(&n_comm);
    free(temp_sol);
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // obtain the necessary parameters required in thie method
    int rank;
    MPI_Comm_rank(comm, &rank);
    int p;
    MPI_Comm_size(comm, &p);
    int q = sqrt(p);
    int dims[2];
    int coords[2];
    int periods[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    MPI_Status stat;
    int num_iter = 0; //num_iter counts the numebr of iterations in jacobi iteration
    int sign = 1; //sign defines whether to continue iteration in jacobi iteration
    MPI_Comm n_comm;
    double Ax_b_square; //stores the intermediate value of (Ax - b) ^ 2
    double norm; //norm stores the l2-norm value
    //initialization of the dimention size in each processor:
    int count_dim_0 = block_decompose_by_dim(n, comm, 0);
    int count_dim_1 = block_decompose_by_dim(n, comm, 1);

    //initialization of D, R
    double *local_D = (double*) malloc(sizeof(double) * count_dim_0);
    double *local_R = (double*) malloc(sizeof(double) * (count_dim_0 * count_dim_1));
    for (int i = 0; i < count_dim_0; i++) {
        for (int j = 0; j < count_dim_1; j++) {
            local_R[count_dim_1 * i + j] = local_A[count_dim_1 * i + j];
        }
    }
    if (coords[0] == coords[1]) {
        for (int i = 0; i < count_dim_0; i++) {
            local_D[i] = local_A[count_dim_1 * i + i];
            local_R[count_dim_1 * i + i] = 0;
        }
    }

    //initialization of x
    for (int i = 0; i < count_dim_0; i++) {
        local_x[i] = 0;
    }

    // diagonal processors send D to first column, A and R are in each processor, D is in the
    // first column
    if (coords[0] == coords[1]) {
        MPI_Send(local_D, count_dim_0, MPI_DOUBLE, coords[0] * q, 0, comm);
    }
    if (coords[1] == 0) { //coords[0] != 0 is not required, don't nned to avoid dead lock
        MPI_Recv(local_D, count_dim_0, MPI_DOUBLE, (coords[0] * (q + 1)), 0, comm, &stat);
    }

    //split the first colum as a communicator, and all other processors as another comminicator
    int color = 1;
    if (coords[1] == 0) {
        color = 0;
    }
    MPI_Comm_split(comm, color, rank, &n_comm);

    // local_Ax and local_Rx stores the Ax and Rx values
    double *local_Ax = (double*) malloc(sizeof(double) * count_dim_0);
    double *local_Rx = (double*) malloc(sizeof(double) * count_dim_0);

    //jacobi iteration
    while(sign == 1) {
        //calculate Ax and stores the solution in the first column
        distributed_matrix_vector_mult(n, local_A, local_x, local_Ax, comm);
        //compute the 2nd norm of Ax - b in the first column
        if (coords[1] == 0) {
            Ax_b_square = 0;
            for (int i = 0; i < count_dim_0; i++) {
                local_Ax[i] -= local_b[i];
                Ax_b_square += local_Ax[i] * local_Ax[i];
            }
        }
        // calculate the square of the second norm using reduce on first column processors,
        // and store the solution in the first processor
        MPI_Reduce(&Ax_b_square, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, n_comm);

        //calculate norm and update sign & num_iter in the first processor
        if (rank == 0) {
            norm = sqrt(norm);
            if (norm < l2_termination || num_iter > max_iter) {
                sign = 0;
            }
        }
        num_iter++;

        //Broadcast sign to all the processors to avoid infinite loop
        MPI_Bcast(&sign, 1, MPI_INT, 0, comm);

        //update x in the first column
        if (sign == 1) {
            //calculate Rx, and store it in the first column
            distributed_matrix_vector_mult(n, local_R, local_x, local_Rx, comm);
            //calculate x = (D^-1)(b - Rx) in the first column
            if (coords[1] == 0) {
                for (int i = 0; i < count_dim_0; i++) {
                    local_Rx[i] = local_b[i] - local_Rx[i];
                    local_x[i] = local_Rx[i] / local_D[i];
                }
            }
        }
    }
    // free the useless memory
    MPI_Comm_free(&n_comm);
    free(local_Ax);
    free(local_Rx);
    free(local_D);
    free(local_R);
}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
