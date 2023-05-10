#include "mfem.hpp"
#include "../../linalg/dual.hpp"

template<class DType>
DType test_f(DType* inp){
    //DType rez={double(0.0),double(1.0)};
    DType rez={0.0};
    //DType
    rez=pow(inp[0],inp[1]);
    //DType rez=pow(5.0,inp[1]);
    return rez;
}


int main(int argc, char *argv[])
{

    typedef mfem::internal::dual<double,double> ADType;

    ADType inp[2];
    inp[0].value=2.0; inp[0].gradient=0.0;
    inp[1].value=1.4; inp[1].gradient=1.0;

    ADType rez;
    rez=test_f<ADType>(inp);

    std::cout<<"grad_1="<<rez.gradient<<std::endl;

}
