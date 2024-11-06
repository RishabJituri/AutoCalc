#include <vector>
#include "matrix.h"
#include "tensor.hpp"

double tensor::operator[](std::vector<int> index){\
    if (index.size() != this->dim){
        //throw exception
    }
    int location = 0;
    int interval_size = this->mat->length;
    for(int i = 0; i<index.size(); i++){
        interval_size = interval_size/this->size[i];
        location = location + interval_size*index[i];
    }

    return this->mat->value[location];
    
    
}

tensor tensor::random(std::vector<int> size, double min, double max){
    
    tensor new_tensor = tensor(size);
    
    matrix * mat = new_tensor.mat;
    
    __rand__(&mat,min,max);
    
}

tensor tensor::zeros(std::vector<int> size){
    
    tensor new_tensor = tensor(size);
    
    matrix * mat = new_tensor.mat;
    
    __zeros__(&mat); 
    
}

tensor tensor::matmul(tensor A, tensor B, std::vector<int> dims){

}


