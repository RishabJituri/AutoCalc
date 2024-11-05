#include "./matrix.h"

matrix * create_matrix(int* size, int len){
    
    matrix * mat = (matrix *)malloc(sizeof(matrix));
    
    mat->length = 1;
    
    for(int i = 0; i<len; i++){
        mat->length = mat->length*size[i];
    }

    mat->value = (double*)malloc(sizeof(double)*mat->length);

    return mat;
}

void zeros(matrix** mat){

    matrix * matr  = *mat;

    for(int i = 0; i<matr->length; i++){
        
        matr->value[i] = 0;
    
    }

}

void random(matrix** mat, double min, double max){
    
    matrix * matr  = *mat;
    
    for(int i = 0; i<matr->length; i++){
        
        matr->value[i] = 0;
    
    }

}
