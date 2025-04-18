#include <iostream>
#include "./variable.hpp"
#include <math.h>

forward_variable forward_variable::operator+(const variable& other) const{

    forward_variable yield(this->item + other.item,this->grad_type || other.item);
    yield.grad = this->grad + other.grad;
    return yield;
}

forward_variable forward_variable::operator*(const variable& other) const{
    
    forward_variable yield(this->item * other.item,this->grad_type || other.item);
    yield.grad = this->grad * other.item +  other.grad * this->item;
    return yield;
}

forward_variable forward_variable::operator^(const variable& other) const{
    
    double exponent_yield = pow(this->item,other.item);
    forward_variable yield(exponent_yield,this->grad_type || other.item);
    yield.grad = pow(this->grad * other.item,other.item-1);
    return yield;
}

forward_variable forward_variable::operator+(const double& other) const{
    
    forward_variable yield(this->item + other,this->grad_type);
    yield.grad = this->grad;
    return yield;
}

forward_variable forward_variable::operator*(const double& other) const{
    
    forward_variable yield(this->item * other,this->grad_type);
    yield.grad = this->grad * other;
    return yield;
}

forward_variable forward_variable::operator^(const double& other) const{
    
    double exponent_yield = pow(this->item,other);
    forward_variable yield(exponent_yield,this->grad_type);
    yield.grad = this->grad * other * pow(this->item,other-1);
    return yield;
}



    
    







