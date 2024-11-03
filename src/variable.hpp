class variable {
public: 
    double item;
    bool grad_type;
    double grad;
    
    variable(double value, bool gt):item(value), grad_type(gt), grad(0) {
        if (gt) {
            grad = 1;
        }
    }
};

class forward_variable:public variable {
public:
    forward_variable(double value, bool gt) : variable(value, gt) {}

    forward_variable operator+(const variable& other) const;
    forward_variable operator*(const variable& other) const;
    forward_variable operator^(const variable& other) const;
};
