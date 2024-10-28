class variable {
public: 
    double item;
    bool grad_type;
    double grad;
    variable(double value, bool gt){
        item = value;
        grad_type = gt;
        grad = 0;
        if (gt){grad = 1;}
    }

    variable __add__(variable one, variable two){}
    variable __multiply__(variable one, variable two){}
};