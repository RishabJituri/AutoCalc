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
            mat = __create_matrix__(size.data(),dim);
            
        }

        double operator[](std::vector<int> index);

        static tensor random(std::vector<int> size, double min, double max);

        static tensor zeros(std::vector<int> size);

        static tensor matmul(tensor A, tensor B, std::vector<int> dims);


        


        

        





};