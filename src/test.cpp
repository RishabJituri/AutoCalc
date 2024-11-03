#include "./variable.hpp"
#include <iostream>
int main(){
    forward_variable a(2.3,true);
    forward_variable b(2.3,false);
    forward_variable c = a*b + b;
    forward_variable d = a^b;
    std::cout << d.item;
    std::cout << "\n";
    std::cout << d.grad;
    
    
}