#include <vector>
#include <random>
#include "matrix.h"

class tensor{
    public:
        std::vector<int> size;
        int dim;
        matrix* mat;

        tensor(std::vector<int> size):size(size){
            dim = (size).size();
            mat = create_matrix(size.data());
            
        }

        static tensor random(std::vector<int> size, double min, double max){}
        


        

        





};