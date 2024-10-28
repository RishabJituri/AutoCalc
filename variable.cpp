

class variable {
public: 
    double item;
    int grad_type;
    double grad;
    variable(double value, int gt){
        item = value;
        grad_type = gt;
    }

    variable __add__(variable one, variable two){}
};

variable variable::__add__(variable one, variable two){
    return 
    

}





