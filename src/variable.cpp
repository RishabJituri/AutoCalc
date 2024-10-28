#include <iostream>
#include <stdexcept>
#include "./variable.hpp"

variable variable::__add__(variable one, variable two){
    variable new_var(one.item + two.item,one.grad_type || two.grad_type);
    new_var.grad = one.grad + two.grad;
    return new_var;
}

variable variable::__multiply__(variable one, variable two){
    double yield = one.item * two.item;
    grad_type = one.grad_type || two.grad_type;
    double gradient= 0;
    if (grad_type){gradient= one.grad * two.item + one.item * two.grad;}
    variable new_var(yield,grad_type);
    new_var.grad = gradient;
    return new_var;
}


    
    







