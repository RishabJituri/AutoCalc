#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix{
    
    double * value;
    
    int length;

}matrix;

matrix * __create_matrix__(int* size, int len);

void __zeros__(matrix** mat);

void __rand__(matrix**mat, double min, double max);

#endif

