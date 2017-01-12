/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.c
 * Author: tioa
 *
 * Created on January 7, 2016, 1:35 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mpi.h"

// ---------- CONSTANTS -------------------
#define VERSION "v1.1"
#define MASK_SIZE 3
#define IMAGE_GRAYSCALE_HUES 256

#define TAG_SPLIT_IMAGE 0
#define TAG_COLLECT_CONV 1

// ---------- GLOBAL VARIABLES ------------
// EDGE RECOGNITION MASK for Image Convolution calculation
int EDGE_MASK[][MASK_SIZE] = {
    {0, 1, 0},
    {1, -4, 1},
    {0, 1, 0}
};

int **Image, // the total image
**AugImage, // the augmented image
**ImgConv, // image convolution
**taskAugImage, // task's part of augmented image
**taskConv; // task's part of convolution

// ---------- FUNCTION PROTOTYPES for main() ----------------
void alloc_matrix(int ***data_ptr, int n, int m);
void free_matrix(int ***data_ptr, int n);
void readInputData(char*, int, int, int**);
void writeOutputData(char*, int, int, int**);
void flipHorizontal(int*, int, int, int*);
void flipVertical(int*, int, int, int*);
void print2DArray(int**, int, int);
void augmentImage(int**, int, int, int**);
void calcImgConv(int **AugImage, int **ImgConv, int rows, int cols);
int rankChunk(int rank, int imagesize, int tasks);

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "SYNTAX: %s <imagesize> <inputfile> <outputfile>\n", argv[0]);
        exit(1);        
    }
    int imagesize = atoi(argv[1]); // read image size
    char *inputfile = argv[2]; // read image filename
    char *outputfile = argv[3]; // read filename for convolution
    int rank, tasks, rc, chunk;
    double parallelCompEnd_t, 
            parallelCommStart_t, parallelCommEnd_t,
            parallelCommEnd2_t,
            end_t, start_t;
    MPI_Status status;
    MPI_Request request;
    int i, j, chunk_i;

    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &tasks); // get num of tasks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my task ID

    if (rank == 0) {
        printf("OpenMPI Image (%s) %dx%d Convolution - Tasks %d\n", VERSION, imagesize, imagesize, tasks);
        // start timestamp 
        start_t = MPI_Wtime();
    }
    // Flip EDGE_MASK horizontally and vertically
    int FLIPPED_HOR[MASK_SIZE][MASK_SIZE];
    flipHorizontal(EDGE_MASK, MASK_SIZE, MASK_SIZE, FLIPPED_HOR);
    flipVertical(FLIPPED_HOR, MASK_SIZE, MASK_SIZE, EDGE_MASK);

    // calculate task's chunk size
    chunk = rankChunk(rank, imagesize, tasks);
    // allocate memory for taskImage, AugImage and taskConv
    alloc_matrix(&taskAugImage, chunk + 2, imagesize + 2);    
    // task 0 is responsible for reading the image from the inputfile
    // and split the work to all tasks
    if (rank == 0) {
        // allocate memory for Image
        alloc_matrix(&Image, imagesize, imagesize);
        // read Image from file
        readInputData(inputfile, imagesize, imagesize, Image);
        // allocate memory for augmented image
        alloc_matrix(&AugImage, imagesize + 2, imagesize + 2);
        // augment image
        augmentImage(Image, imagesize, imagesize, AugImage);
        // destroy image matrix
        free_matrix(&Image, imagesize);
        // distribute chunks to tasks
        parallelCommStart_t = MPI_Wtime();
        int startrow = 0, endrow;
        for (i = 0; i < tasks; i++) {
            // calculate chuck for task-i and add 2
            chunk_i = rankChunk(i, imagesize, tasks) + 2;
            // calculate last row to copy to destination
            endrow = startrow + chunk_i - 1;
            if (i == 0) {
                for (j = startrow; j <= endrow; j++) {
                    for (int k=0; k<imagesize+2; k++) {
                        taskAugImage[j][k] = AugImage[j][k];
                    }
                }
                    
            } else
                for (j = startrow; j <= endrow; j++) {
                    MPI_Send(&AugImage[j][0], imagesize + 2, MPI_INT, i, TAG_SPLIT_IMAGE, MPI_COMM_WORLD);
                }
            startrow = --endrow;
        }
        free_matrix(&AugImage, imagesize + 2);
        parallelCommEnd_t = MPI_Wtime();
    } else {
        for (i = 0; i < chunk + 2; i++) {
            MPI_Recv(&taskAugImage[i][0], imagesize + 2, MPI_INT, 0, TAG_SPLIT_IMAGE, MPI_COMM_WORLD, &status);
        }
    }
    // calculate task's convolution
    // allocate taskConv
    alloc_matrix(&taskConv, chunk, imagesize);    
    // calculate task's convolution
    calcImgConv(taskAugImage, taskConv, chunk, imagesize);
    // release taskAugImage
    free_matrix(&taskAugImage, chunk + 2);
    
    // wait for all tasks to complete their calculation of convolution
    MPI_Barrier(MPI_COMM_WORLD);
    
    // task 0 is responsible for gathering all taskConvs and
    // merge them to ImgConv
    if (rank == 0) {
        parallelCompEnd_t = MPI_Wtime();
        // allocate memory for ImgConv
        alloc_matrix(&ImgConv, imagesize, imagesize);    
        // merge taskConvs to ImgConv
        int startrow = 0, endrow;
        for (i = 0; i < tasks; i++) {
            // calculate chuck for task-i and add 2
            chunk_i = rankChunk(i, imagesize, tasks);
            // calculate last row to copy to destination
            endrow = startrow + chunk_i - 1;
            if (i == 0) {
                for (j = startrow; j <= endrow; j++) {
                    for (int k=0; k<imagesize+2; k++) {
                        ImgConv[j][k] = taskConv[j][k];
                    }
                }
                    
            } else
                for (j = startrow; j <= endrow; j++) {
                    MPI_Recv(&ImgConv[j][0], imagesize, MPI_INT, i, TAG_COLLECT_CONV, MPI_COMM_WORLD, &status);
                }
            startrow = endrow + 1;
        }
    } else {
        for (i = 0; i < chunk; i++) {
            MPI_Send(&taskConv[i][0], imagesize, MPI_INT, 0, TAG_COLLECT_CONV, MPI_COMM_WORLD);
        }
    }
    // wait for all tasks to merge their taskConv to ImgConv
    MPI_Barrier(MPI_COMM_WORLD);
    // free taskConv
    free(taskConv);
    // task 0 prints the ImgConv
    if (rank == 0) {
        parallelCommEnd2_t = MPI_Wtime();
        writeOutputData(outputfile, imagesize, imagesize, ImgConv);
        // free ImgConv
        free(ImgConv);
        // end timestamp 
        end_t = MPI_Wtime();
    }
    MPI_Finalize();
    if (rank == 0) {
        printf("\nTotal duration:\t%0.2f msecs", (end_t-start_t)*1000);
        printf("\nParallel (comm - scatter) duration:\t%0.2f msecs", (parallelCommEnd_t-parallelCommStart_t)*1000);    
        printf("\nParallel (comp - convolution) duration:\t%0.2f msecs", (parallelCompEnd_t-parallelCommEnd_t)*1000);    
        printf("\nParallel (comm - gather) duration:\t%0.2f msecs", (parallelCommEnd2_t-parallelCompEnd_t)*1000);  
        printf("\n");
    }
    return (EXIT_SUCCESS);
}

void alloc_matrix(int ***data_ptr, int n, int m) {
    int row, i, j;
    int **data;
    data = (int **) malloc(n * sizeof (int *));
    for (row = 0; row < n; row++)
        data[row] = (int *) malloc(m * sizeof (int));
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            data[i][j] = i * j;
    *data_ptr = data;
}

void free_matrix(int ***data_ptr, int n) {
    int row;
    int **data = *data_ptr;
    for (row = 0; row < n; row++)
        free(data[row]);
    free(data);
}

void readInputData(char* file, int rows, int cols, int **image) {
    int row, col;
    FILE *fp;
    // open file for reading
    fp = fopen(file, "rb");
    if (fp == NULL) {
        return;
    }
    for (row = 0; row < rows; row++)
        fread(&image[row][0], sizeof(int)*cols, 1, fp);
    fclose(fp);
}

void writeOutputData(char* file, int rows, int cols, int **image) {
    int row, col;
    FILE *fp;
    // open file for writing
    fp = fopen(file, "wb");
    if (fp == NULL) {
        return;
    }
    for (row = 0; row < rows; row++)
        fwrite(&image[row][0], sizeof(int)*cols, 1, fp);
    fflush(fp);
    fclose(fp);
}

void flipHorizontal(int *arr, int rows, int cols, int *fliparr) {
    int row, col, colsmid = cols / 2;
    for (row = 0; row < rows; row++)
        for (col = 0; col <= colsmid; col++) {
            *(fliparr + row * cols + col) = *(arr + row * cols + cols - 1 - col);
            *(fliparr + row * cols + cols - 1 - col) = *(arr + row * cols + col);
        }
}

void flipVertical(int *arr, int rows, int cols, int *fliparr) {
    int row, col, rowssmid = rows / 2;
    for (col = 0; col < cols; col++)
        for (row = 0; row <= rowssmid; row++) {
            *(fliparr + row * cols + col) = *(arr + (rows - 1 - row) * cols + col);
            *(fliparr + (rows - 1 - row) * cols + col) = *(arr + row * cols + col);
        }
}

void print2DArray(int **arr, int rows, int cols) {
    int row, col, len;
    len = sizeof(char) * (rows*cols + rows)*5 + 1;
    char *s = malloc(len);
    *s = '\0';
    char t[100];
    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            sprintf(t, "%4d ", arr[row][col]);
            strcat(s, t);
        }
        strcat(s,"\n");
    }
    fprintf(stderr, "%s", s);
    free(s);
}

void augmentImage(int **image, int rows, int cols, int **augimage) {
    int row, col;
    int rowsaug = rows + 2;
    int colsaug = cols + 2;

    // initialize to 0 the 1st and last columns of augimage
    for (row = 0; row < rowsaug; row++) {
        augimage[row][0] = 0;
        augimage[row][colsaug - 1] = 0;
    }
    // initialize to 0 the 1st and last rows of augimage
    for (col = 0; col < colsaug; col++) {
        augimage[0][col] = 0;
        augimage[rowsaug - 1][col] = 0;
    }
    // copy image rows to augimage rows
    for (row = 1; row < rowsaug - 1; row++)
        for (col = 1; col < colsaug - 1; col++)
            augimage[row][col] = image[row - 1][col - 1];
}

void calcImgConv(int **AugImage, int **ImgConv, int rows, int cols) {
    for (int x = 0; x < rows; x++)
        for (int y = 0; y < cols; y++) {
            ImgConv[x][y] = 0;
            for (int k = 0; k < 3; k++)
                for (int j = 0; j < 3; j++)
                    ImgConv[x][y] += EDGE_MASK[k][j] * AugImage[x + k][y + j];
        }
}

int rankChunk(int rank, int imagesize, int tasks) {
    /* calculate task's chunk size
     * Initially all tasks get an equal share, but
     * the last task gets also the remaining lines
     * */
    int chunk = imagesize / tasks;
    if (rank == (tasks - 1))
        chunk += (imagesize % tasks);
    return chunk;
}
