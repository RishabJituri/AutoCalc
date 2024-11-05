#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix{
    
    double * value;
    int length;

}matrix;

matrix * create_matrix(int* size);

#endif

