#include <iostream>
#include <stdexcept>
#include "./variable.hpp"


forward_variable forward_variable::operator+(const variable& other) const{
    forward_variable yield(this->item + other.item,this->grad_type || other.item);
    yield.grad = this->grad + other.grad;
    return yield;
}

forward_variable forward_variable::operator*(const variable& other) const{
    forward_variable yield(this->item * other.item,this->grad_type || other.item);
    yield.grad = this->grad *other.item +  other.grad * this->item;
    return yield;
}


    
    







