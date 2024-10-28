#include <iostream>
#include <stdexcept>

class variable {
public: 
    double item;
    bool grad_type;
    double grad;
    variable(double value, bool gt){
        item = value;
        grad_type = gt;
        grad = 0;
    }

    variable __add__(variable one, variable two){}
};

variable variable::__add__(variable one, variable two){
    variable new_var(one.item + two.item,one.grad_type || two.grad_type);
    new_var.grad = one.grad + two.grad
    return new_var;

    }
    
    

}





