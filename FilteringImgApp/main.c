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
#include <time.h>
#include <omp.h>

// ---------- CONSTANTS -------------------
#define VERSION "v1.1"
#define MASK_SIZE 3
#define IMAGE_GRAYSCALE_HUES 256

// ---------- GLOBAL VARIABLES ------------
// EDGE RECOGNITION MASK for Image Convolution calculation
int EDGE_MASK[][MASK_SIZE] = {
    {0,  1,  0},
    {1, -4,  1},
    {0,  1,  0}
};

int **Image, // the total image
**AugImage, // the augmented image
**ImgConv; // image convolution

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

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "SYNTAX: %s <imagesize> <inputfile> <outputfile>\n", argv[0]);
        exit(1);        
    }
    int imagesize = atoi(argv[1]); // read image size
    char *inputfile = argv[2]; // read image filename
    char *outputfile = argv[3]; // read filename for convolution

    // get start time
    printf("Serial Image (%s) %dx%d Convolution\n\n", VERSION, imagesize, imagesize);
    double parallelStart_t, parallelEnd_t, end_t, start_t = omp_get_wtime()*1000;
    
    int FLIPPED_HOR[MASK_SIZE][MASK_SIZE];
    // flip EDGE_MASK horizontally to FLIPPED_HOR
    flipHorizontal(EDGE_MASK, MASK_SIZE, MASK_SIZE, FLIPPED_HOR);
    // flip FLIPPED_HOR vertically to EDGE_MASK
    flipVertical(FLIPPED_HOR, MASK_SIZE, MASK_SIZE, EDGE_MASK);
    // allocate memory for Image
    alloc_matrix(&Image, imagesize, imagesize);
    // read Image from file
    readInputData(inputfile, imagesize, imagesize, Image);
    // allocate memory for augmented image
    alloc_matrix(&AugImage, imagesize + 2, imagesize + 2);
    // augment Image
    augmentImage(Image, imagesize, imagesize, AugImage);
    // destroy image matrix
    free_matrix(&Image, imagesize);
    // get parallel start time
    parallelStart_t = omp_get_wtime()*1000;
    // allocate memory for ImgConv
    alloc_matrix(&ImgConv, imagesize, imagesize);    
    // calculate convolution
    calcImgConv(AugImage, ImgConv, imagesize, imagesize);
    // get parallel end time
    parallelEnd_t = omp_get_wtime()*1000;
    // destroy augmented image
    free_matrix(&AugImage, imagesize + 2);
    // write Image to file
    writeOutputData(outputfile, imagesize, imagesize, ImgConv);
    // destroy ImgConv
    free_matrix(&ImgConv, imagesize);
    // get end time
    end_t = omp_get_wtime()*1000;
    printf("\nTotal duration:\t%0.2f msecs", (end_t-start_t));
    printf("\nConvolution calculation duration:\t%0.2f msecs", (parallelEnd_t-parallelStart_t));
    printf("\n");
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