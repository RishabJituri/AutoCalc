#include "./matrix.h"
#include <math.h>

matrix * __create_matrix__(int* size, int len){
    
    matrix * mat = (matrix *)malloc(sizeof(matrix));
    
    mat->length = 1;
    
    for(int i = 0; i<len; i++){
        
        mat->length = mat->length*size[i];
    
    }

    mat->value = (double*)malloc(sizeof(double)*mat->length);

    return mat;
}

void __zeros__(matrix** mat){

    matrix * matr  = *mat;

    for(int i = 0; i<matr->length; i++){
        
        matr->value[i] = 0;
    
    }

}

void __rand__(matrix** mat, double min, double max){
    
    matrix * matr  = *mat;

    int scaled_max = max*1000;

    int scaled_min = min*1000;

    for(int i = 0; i<matr->length; i++){
        
        matr->value[i] = (double)(scaled_min + rand() % (scaled_max - scaled_min + 1))/1000;
    
    }

}

