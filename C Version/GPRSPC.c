/*
Author: SHUAIWEN CUI
E-mail: SHUAIWEN001@e.ntu.edu.sg
Data: Aug 02 2023
Description: This programme is used to perform Gaussian Process Regression (GPR) and Stochastic Process Control (SPC) on the given dataset. More details can be found in the comments.
*/

// note: this program requires the GNU Scientific Library (GSL) to be installed. Please install GSL before running this program.

// assuming you are using a Mac or Linux system, you can use the following commands to compile and execute the code
// to compile the code
//     gcc -o GPRSPC GPRSPC.c -lgsl -lgslcblas
// to execute the code
//     ./GPRSPC

//// dependencies
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

//// constants
#define MAX_DIMENSION 500

//// helper functions / definitions / data types / func prototypes
// helper - data type - input data
typedef struct
{
    float temperature;
    float max_disp_1;
} InputData;

// helper - data type - output data
typedef struct
{
    float max_disp_2;
} OutputData;

// helper - function -  load input data from CSV file
void LoadInputData(const char *filename, InputData *inputdata, int *records)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening file.\n");
        return;
    }
    int read = 0;
    *records = 0;
    do
    {
        read = fscanf(file, "%f,%f\n", &inputdata[*records].temperature, &inputdata[*records].max_disp_1);
        if (read == 2)
            (*records)++;
        if (read != 2 && !feof(file))
        {
            printf("File format incorrect.\n");
            fclose(file);
            return;
        }
        if (ferror(file))
        {
            printf("Error reading file.\n");
            fclose(file);
            return;
        }
    } while (!feof(file));
    fclose(file);
}

// helper - function -  load output data from CSV file
void LoadOutputData(const char *filename, OutputData *outputdata, int *records)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening file.\n");
        return;
    }
    int read = 0;
    *records = 0;
    do
    {
        read = fscanf(file, "%f\n", &outputdata[*records].max_disp_2);
        if (read == 1)
            (*records)++;
        if (read != 1 && !feof(file))
        {
            printf("File format incorrect.\n");
            fclose(file);
            return;
        }
        if (ferror(file))
        {
            printf("Error reading file.\n");
            fclose(file);
            return;
        }
    } while (!feof(file));
    fclose(file);
}

// helper - function - kernel value - ARD squared exponential kernel
double ARD_SE_Kernel(double x1[], double x2[], double Length_Scales[], double SigmaFactor, int dim)
{
    double sum = 0.0;
    for (int i = 0; i < dim; i++)
    {
        double diff = x1[i] - x2[i];
        sum += (diff * diff) / (2 * Length_Scales[i] * Length_Scales[i]);
    }
    double KernelValue = SigmaFactor * SigmaFactor * exp(-sum);
    return KernelValue;
}

//// main function
int main(void)
{
    //// configuration

    // feedback mode
    int feedback_mode = 1;

    // verbose mode
    int verbose_mode = 1;

    // hyperparameters for the Gaussian Progress Regression (GPR) model
    double SigmaFactor = 1.0;
    double Length_Scales[] = {1.0, 0.01};
    double alpha = 0.05;

    //// I - Load Data
    // print current task
    if (feedback_mode == 1)
    {
        printf("\n============================================= I - Load Data\n");
    }

    // load data from files
    InputData inputdata_train[MAX_DIMENSION];
    InputData inputdata_test[MAX_DIMENSION];
    OutputData outputdata_train[MAX_DIMENSION];
    OutputData outputdata_testtrue[MAX_DIMENSION];

    // variables used in loading
    int records_input_training = 0;
    int records_output_training = 0;
    int records_input_test = 0;
    int records_output_testtrue = 0;

    // load data required
    LoadInputData("x_train.csv", inputdata_train, &records_input_training);
    LoadOutputData("y_train.csv", outputdata_train, &records_output_training);
    LoadInputData("x_test.csv", inputdata_test, &records_input_test);
    LoadOutputData("y_testtrue.csv", outputdata_testtrue, &records_output_testtrue);

    // new matrices for data transfer using gsl matrix
    gsl_matrix *x_train = gsl_matrix_alloc(records_input_training, 2);
    gsl_matrix *y_train = gsl_matrix_alloc(records_output_training, 1);
    gsl_matrix *x_test = gsl_matrix_alloc(records_input_test, 2);
    gsl_matrix *y_testtrue = gsl_matrix_alloc(records_output_testtrue, 1);
    gsl_matrix *y_testpred = gsl_matrix_alloc(records_output_testtrue, 1); // same size with ground truth data

    // transfer data
    for (int i = 0; i < records_input_training; i++)
    {
        gsl_matrix_set(x_train, i, 0, inputdata_train[i].temperature);
        gsl_matrix_set(x_train, i, 1, inputdata_train[i].max_disp_1);
    }
    for (int i = 0; i < records_output_training; i++)
    {
        gsl_matrix_set(y_train, i, 0, outputdata_train[i].max_disp_2);
    }
    for (int i = 0; i < records_input_test; i++)
    {
        gsl_matrix_set(x_test, i, 0, inputdata_test[i].temperature);
        gsl_matrix_set(x_test, i, 1, inputdata_test[i].max_disp_1);
    }
    for (int i = 0; i < records_output_testtrue; i++)
    {
        gsl_matrix_set(y_testtrue, i, 0, outputdata_testtrue[i].max_disp_2);
    }

    // print data loaded
    if (feedback_mode == 1)
    {
        // print progress
        printf("\n[Progress Report]>>  Data successfully loaded!\n");
        printf("records_input_training: %d\n", records_input_training);
        printf("records_output_training: %d\n", records_output_training);
        printf("records_input_test: %d\n", records_input_test);
        printf("records_output_testtrue: %d\n", records_output_testtrue);
        printf("\n");
        // print data loaded
        if (verbose_mode == 1)
        {
            printf("\n[Progress Report]>>  Print the data.\n");

            printf("\n  x_train: \n\n");
            for (int i = 0; i < records_input_training; i++)
            {
                printf("%16.11f, %16.11f\n", gsl_matrix_get(x_train, i, 0), gsl_matrix_get(x_train, i, 1));
            }
            printf("\n  y_train: \n\n");
            for (int i = 0; i < records_output_training; i++)
            {
                printf("%16.11f\n", gsl_matrix_get(y_train, i, 0));
            }
            printf("\n  x_test: \n\n");
            for (int i = 0; i < records_input_test; i++)
            {
                printf("%16.11f, %16.11f\n", gsl_matrix_get(x_test, i, 0), gsl_matrix_get(x_test, i, 1));
            }
            printf("\n  y_testtrue: \n\n");
            for (int i = 0; i < records_output_testtrue; i++)
            {
                printf("%16.11f\n", gsl_matrix_get(y_testtrue, i, 0));
            }
            printf("\n");
        }
    }

    //// II - Gaussian Process Regression (GPR)

    if (feedback_mode == 1)
    {
        printf("\n============================================= II - Gaussian Process Regression\n");
    }

    // parameters for GPR
    int num_train_points = records_input_training;
    int num_test_points = records_input_test;
    int num_features = x_train->size2;

    // calculate the mean of the data
    double mean_y_train = 0.0;
    for (int i = 0; i < num_train_points; i++)
    {
        mean_y_train += gsl_matrix_get(y_train, i, 0);
    }
    mean_y_train /= num_train_points;

    // calculate the standard deviation of the data
    double std_y_train = 0.0;
    for (int i = 0; i < num_train_points; i++)
    {
        std_y_train += (gsl_matrix_get(y_train, i, 0) - mean_y_train) * (gsl_matrix_get(y_train, i, 0) - mean_y_train);
    }
    std_y_train /= num_train_points;
    std_y_train = sqrt(std_y_train);

    // calculate the variance of the data
    double var_y_train = std_y_train * std_y_train;

    // print progress
    if (feedback_mode == 1)
    {
        printf("\n[Progress Report]>>  Mean and standard deviation of the training data calculated.\n\n");
        printf("mean_y_train: %16.11f\n", mean_y_train);
        printf("std_y_train:  %16.11f\n", std_y_train);
        printf("var_y_train:  %16.11f\n", var_y_train);
    }

    // instructions on how to performan GPR
    // by GPR theory, we have:
    // y_testpred = K_test_train * (K_train_train + var_y_train * I)^(-1) * y_train
    // var_y_testpred = diag(K_test_test + var_y_train * I - K_test_train * (K_train_train + var_y_train * I)^(-1) * K_train_test)
    // if we denote K_train_train + var_y_train * I as K_train_train_plus, then we have:
    // y_testpred = K_test_train * K_train_train_plus^(-1) * y_train
    // var_y_testpred = diag(K_test_test + var_y_train * I - K_test_train * K_train_train_plus^(-1) * K_train_test)

    // helper matrices
    gsl_matrix *K_train_train = gsl_matrix_alloc(num_train_points, num_train_points);
    gsl_matrix *K_train_train_plus = gsl_matrix_alloc(num_train_points, num_train_points);
    gsl_matrix *K_train_train_plus_inv = gsl_matrix_alloc(num_train_points, num_train_points);
    gsl_matrix *K_train_test = gsl_matrix_alloc(num_train_points, num_test_points);
    gsl_matrix *K_test_train = gsl_matrix_alloc(num_test_points, num_train_points);
    gsl_matrix *K_test_test = gsl_matrix_alloc(num_test_points, num_test_points);
    gsl_matrix *I_train = gsl_matrix_alloc(num_train_points, num_train_points);

    // calculate the kernel matrices

    // K_train_train
    for (int i = 0; i < num_train_points; i++) // upper triangular matrix
    {
        for (int j = i; j < num_train_points; j++)
        {
            double x1[] = {gsl_matrix_get(x_train, i, 0), gsl_matrix_get(x_train, i, 1)};
            double x2[] = {gsl_matrix_get(x_train, j, 0), gsl_matrix_get(x_train, j, 1)};
            double KernelValue = ARD_SE_Kernel(x1, x2, Length_Scales, SigmaFactor, num_features);
            gsl_matrix_set(K_train_train, i, j, KernelValue);
        }
    }
    for (int i = 0; i < num_train_points; i++) // lower triangular matrix
    {
        for (int j = 0; j < i; j++)
        {
            gsl_matrix_set(K_train_train, i, j, gsl_matrix_get(K_train_train, j, i));
        }
    }

    // K_train_train_plus
    for (int i = 0; i < num_train_points; i++)
    {
        for (int j = 0; j < num_train_points; j++)
        {
            if (i == j)
            {
                gsl_matrix_set(K_train_train_plus, i, j, gsl_matrix_get(K_train_train, i, j) + var_y_train);
            }
            else
            {
                gsl_matrix_set(K_train_train_plus, i, j, gsl_matrix_get(K_train_train, i, j));
            }
        }
    }

    // K_train_train_plus_inv - calculate the inverse of K_train_train_plus
    gsl_matrix_memcpy(K_train_train_plus_inv, K_train_train_plus);
    int s;
    gsl_permutation *p = gsl_permutation_alloc(num_train_points);
    gsl_linalg_LU_decomp(K_train_train_plus_inv, p, &s);
    gsl_linalg_LU_invert(K_train_train_plus_inv, p, K_train_train_plus_inv);
    gsl_permutation_free(p);

    // K_train_test
    for (int i = 0; i < num_train_points; i++)
    {
        for (int j = 0; j < num_test_points; j++)
        {
            double x1[] = {gsl_matrix_get(x_train, i, 0), gsl_matrix_get(x_train, i, 1)};
            double x2[] = {gsl_matrix_get(x_test, j, 0), gsl_matrix_get(x_test, j, 1)};
            double KernelValue = ARD_SE_Kernel(x1, x2, Length_Scales, SigmaFactor, num_features);
            gsl_matrix_set(K_train_test, i, j, KernelValue);
        }
    }

    // K_test_train - use the symmetry
    for (int i = 0; i < num_test_points; i++)
    {
        for (int j = 0; j < num_train_points; j++)
        {
            gsl_matrix_set(K_test_train, i, j, gsl_matrix_get(K_train_test, j, i));
        }
    }

    // K_test_test
    for (int i = 0; i < num_test_points; i++)
    {
        for (int j = 0; j < num_test_points; j++)
        {
            double x1[] = {gsl_matrix_get(x_test, i, 0), gsl_matrix_get(x_test, i, 1)};
            double x2[] = {gsl_matrix_get(x_test, j, 0), gsl_matrix_get(x_test, j, 1)};
            double KernelValue = ARD_SE_Kernel(x1, x2, Length_Scales, SigmaFactor, num_features);
            gsl_matrix_set(K_test_test, i, j, KernelValue);
        }
    }

    // I_train
    for (int i = 0; i < num_train_points; i++)
    {
        for (int j = 0; j < num_train_points; j++)
        {
            if (i == j)
            {
                gsl_matrix_set(I_train, i, j, 1.0);
            }
            else
            {
                gsl_matrix_set(I_train, i, j, 0.0);
            }
        }
    }

    // calculate the mean and variance of the prediction
    gsl_matrix *mean_y_testpred = gsl_matrix_alloc(num_test_points, 1);
    gsl_matrix *var_y_testpred = gsl_matrix_alloc(num_test_points, 1);
    gsl_matrix *std_y_testpred = gsl_matrix_alloc(num_test_points, 1);

    // mean_y_testpred = K_test_train * K_train_train_plus^(-1) * y_train
    // intermediate matrix - K_test_trainxK_train_train_plus_inv - K_test_train * K_train_train_plus^(-1)
    gsl_matrix *K_test_trainxK_train_train_plus_inv = gsl_matrix_alloc(num_test_points, num_train_points);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, K_test_train, K_train_train_plus_inv, 0.0, K_test_trainxK_train_train_plus_inv);
    // mean_y_testpred = K_test_trainxK_train_train_plus_inv * y_train
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, K_test_trainxK_train_train_plus_inv, y_train, 0.0, mean_y_testpred);

    // var_y_testpred = diag(K_test_test + var_y_train * I - K_test_train * K_train_train_plus^(-1) * K_train_test)
    // intermediate matrix - var_y_trainxI - var_y_train * I
    gsl_matrix *var_y_trainxI = gsl_matrix_alloc(num_test_points, num_test_points);
    for (int i = 0; i < num_test_points; i++)
    {
        for (int j = 0; j < num_test_points; j++)
        {
            if (i == j)
            {
                gsl_matrix_set(var_y_trainxI, i, j, var_y_train);
            }
            else
            {
                gsl_matrix_set(var_y_trainxI, i, j, 0.0);
            }
        }
    }

    // intermediate matrix - K_test_trainxK_train_train_plus_invxK_train_test - K_test_train * K_train_train_plus^(-1) * K_train_test
    gsl_matrix *K_test_trainxK_train_train_plus_invxK_train_test = gsl_matrix_alloc(num_test_points, num_test_points);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, K_test_trainxK_train_train_plus_inv, K_train_test, 0.0, K_test_trainxK_train_train_plus_invxK_train_test);

    // intermediate matrix - K_test_test_and_var_y_trainxI - K_test_test + var_y_train * I
    gsl_matrix *K_test_test_and_var_y_trainxI = gsl_matrix_alloc(num_test_points, num_test_points);
    for (int i = 0; i < num_test_points; i++)
    {
        for (int j = 0; j < num_test_points; j++)
        {
            gsl_matrix_set(K_test_test_and_var_y_trainxI, i, j, gsl_matrix_get(K_test_test, i, j) + gsl_matrix_get(var_y_trainxI, i, j));
        }
    }

    // var_y_testpred = diag(K_test_test_and_var_y_trainxI - K_test_trainxK_train_train_plus_invxK_train_test)
    gsl_matrix *var_matrix = gsl_matrix_alloc(num_test_points, num_test_points);
    for (int i = 0; i < num_test_points; i++)
    {
        for (int j = 0; j < num_test_points; j++)
        {
            gsl_matrix_set(var_matrix, i, j, gsl_matrix_get(K_test_test_and_var_y_trainxI, i, j) - gsl_matrix_get(K_test_trainxK_train_train_plus_invxK_train_test, i, j));
        }
    }

    // var_y_testpred = diag(var_matrix)
    for (int i = 0; i < num_test_points; i++)
    {
        gsl_matrix_set(var_y_testpred, i, 0, gsl_matrix_get(var_matrix, i, i));
    }

    // std_y_testpred = sqrt(var_y_testpred)
    for (int i = 0; i < num_test_points; i++)
    {
        gsl_matrix_set(std_y_testpred, i, 0, sqrt(gsl_matrix_get(var_y_testpred, i, 0)));
    }

    // print progress
    if (feedback_mode == 1)
    {
        printf("\n[Progress Report]>>  Mean and variance of the prediction calculated.\n\n");
        if (verbose_mode == 1)
        {
            printf("\n  mean_y_testpred: \n\n");
            for (int i = 0; i < num_test_points; i++)
            {
                printf("%16.11f\n", gsl_matrix_get(mean_y_testpred, i, 0));
            }
            printf("\n  std_y_testpred: \n\n");
            for (int i = 0; i < num_test_points; i++)
            {
                printf("%16.11f\n", gsl_matrix_get(std_y_testpred, i, 0));
            }
            printf("\n  var_y_testpred: \n\n");
            for (int i = 0; i < num_test_points; i++)
            {
                printf("%16.11f\n", gsl_matrix_get(var_y_testpred, i, 0));
            }
        }
    }

    // output result to file

    // output mean_y_testpred
    FILE *file_mean_y_testpred = fopen("y_testpred.csv", "w");
    if (file_mean_y_testpred == NULL)
    {
        printf("Error opening file.\n");
        return 0;
    }
    for (int i = 0; i < num_test_points; i++)
    {
        fprintf(file_mean_y_testpred, "%16.11f\n", gsl_matrix_get(mean_y_testpred, i, 0));
    }
    fclose(file_mean_y_testpred);

    // output std_y_testpred
    FILE *file_std_y_testpred = fopen("y_testpred_std.csv", "w");
    if (file_std_y_testpred == NULL)
    {
        printf("Error opening file.\n");
        return 0;
    }
    for (int i = 0; i < num_test_points; i++)
    {
        fprintf(file_std_y_testpred, "%16.11f\n", gsl_matrix_get(std_y_testpred, i, 0));
    }
    fclose(file_std_y_testpred);

    // output var_y_testpred
    FILE *file_var_y_testpred = fopen("y_testpred_var.csv", "w");
    if (file_var_y_testpred == NULL)
    {
        printf("Error opening file.\n");
        return 0;
    }
    for (int i = 0; i < num_test_points; i++)
    {
        fprintf(file_var_y_testpred, "%16.11f\n", gsl_matrix_get(var_y_testpred, i, 0));
    }
    fclose(file_var_y_testpred);

    // print progress
    if (feedback_mode == 1)
    {
        printf("\n[Progress Report]>>  Mean, stdandard deviation and variance of the prediction output to file.\n\n");
    }

    //// III - Stochastic Process Control (SPC)
    // print progress
    if (feedback_mode == 1)
    {
        printf("\n============================================= III - Stochastic Process Control\n");
    }

    // calculate the residual of the prediction - y_res = y_testtrue - y_testpred
    gsl_matrix *y_res = gsl_matrix_alloc(num_test_points, 1);
    for (int i = 0; i < num_test_points; i++)
    {
        gsl_matrix_set(y_res, i, 0, gsl_matrix_get(y_testtrue, i, 0) - gsl_matrix_get(mean_y_testpred, i, 0));
    }

    // calculate the upper control limit (UCL) and lower control limit (LCL) of the residual
    // first to calculate the average of the standard deviation of the prediction
    double avg_y_testpred_std = 0.0;
    for (int i = 0; i < num_test_points; i++)
    {
        avg_y_testpred_std += gsl_matrix_get(std_y_testpred, i, 0);
    }
    avg_y_testpred_std /= num_test_points;



    // UCL =  1.96 * avg_y_testpred_std
    double UCL =  1.96 * avg_y_testpred_std;
    // LCL = - 1.96 * avg_y_testpred_std
    double LCL =  -1.96 * avg_y_testpred_std;

    // take the minimum and maximum of the residual
    double min_y_res = gsl_matrix_get(y_res, 0, 0);
    for (int i = 0; i < num_test_points; i++)
    {
        if (gsl_matrix_get(y_res, i, 0) < min_y_res)
        {
            min_y_res = gsl_matrix_get(y_res, i, 0);
        }
    }
    double max_y_res = gsl_matrix_get(y_res, 0, 0);
    for (int i = 0; i < num_test_points; i++)
    {
        if (gsl_matrix_get(y_res, i, 0) > max_y_res)
        {
            max_y_res = gsl_matrix_get(y_res, i, 0);
        }
    }

    // updatea UCL and LCL
    if (UCL < max_y_res)
    {
        UCL = max_y_res;
    }
    if (LCL > min_y_res)
    {
        LCL = min_y_res;
    }

    // print progress
    if (feedback_mode == 1)
    {
        printf("\n[Progress Report]>>  UCL and LCL of the residual calculated.\n\n");
        printf("UCL: %16.11f\n", UCL);
        printf("LCL: %16.11f\n", LCL);
    }

    // identify the out-of-control points
    int num_outofcontrol_points = 0;
    for (int i = 0; i < num_test_points; i++)
    {
        if (gsl_matrix_get(y_res, i, 0) > UCL || gsl_matrix_get(y_res, i, 0) < LCL)
        {
            num_outofcontrol_points++;
            // print the number of the out-of-control points
            printf("the point number #%d is out of control.\n", i + 1);
            // print the max displacement of the out-of-control points
            printf("the max displacement of the point number #%d is %16.11f.\n", i + 1, gsl_matrix_get(y_testtrue, i, 0));
        }
    }
    if (num_outofcontrol_points == 0)
    {
        printf("No out-of-control points found.\n");
    }

    // print progress
    if (feedback_mode == 1)
    {
        printf("\n GPR&SPC Programme successfully finished! \n\n");
    }

    return 0;
}
