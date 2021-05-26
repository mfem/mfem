#include "admfem.hpp"

template<typename TDataType, typename TParamVector, typename TStateVector
         , int state_size, int param_size>
class DiffusionFunctional
{
public:
    TDataType operator() (TParamVector& vparam, TStateVector& uu)
    {
        MFEM_ASSERT(state_size==4,"ExampleFunctor state_size should be equal to 4!");
        MFEM_ASSERT(param_size==2,"ExampleFunctor param_size should be equal to 2!");
        auto kapa = vparam[0]; //diffusion coefficient
        auto load = vparam[1]; //volumetric influx
        TDataType rez = kapa*(uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2])/2.0 - load*uu[3];
        return rez;
    }

};

template<typename TDataType, typename TParamVector, typename TStateVector,
         int residual_size, int state_size, int param_size>
class DiffusionResidual
{
public:
    void operator ()(TParamVector& vparam, TStateVector& uu, TStateVector& rr)
    {
        MFEM_ASSERT(residual_size==4,"DiffusionResidual residual_size should be equal to 4!")
        MFEM_ASSERT(state_size==4,"ExampleFunctor state_size should be equal to 4!");
        MFEM_ASSERT(param_size==2,"ExampleFunctor param_size should be equal to 2!");
        auto kapa = vparam[0]; //diffusion coefficient
        auto load = vparam[1]; //volumetric influx

        rr[0] = kapa * uu[0];
        rr[1] = kapa * uu[1];
        rr[2] = kapa * uu[2];
        rr[3] = -load;
    }
};





int main(int argc, char *argv[])
{

    mfem::Vector param(2);
    param[0]=3.0; //diffusion coefficient
    param[1]=2.0; //volumetric influx

    mfem::Vector state(4);
    state[0]=1.0; // grad_x
    state[1]=2.0; // grad_y
    state[2]=3.0; // grad_z
    state[4]=4.0; // state value

    mfem::QFunctionAutoDiff<DiffusionFunctional,4,2> adf;
    mfem::QResidualAutoDiff<DiffusionResidual,4,4,2> rdf;

    mfem::Vector rr0(4);
    mfem::DenseMatrix hh0(4,4);

    mfem::Vector rr1(4);
    mfem::DenseMatrix hh1(4,4);

    adf.QResidual(param,state,rr0);
    adf.QGradResidual(param, state, hh0);
    // dump out the results
    std::cout<<"FunctionAutoDiff"<<std::endl;
    std::cout<< adf.QEval(param,state)<<std::endl;
    rr0.Print(std::cout);
    hh0.Print(std::cout);

    rdf.QResidual(param,state, rr1);
    rdf.QGradResidual(param, state, hh1);

    std::cout<<"ResidualAutoDiff"<<std::endl;
    rr1.Print(std::cout);
    hh1.Print(std::cout);

    //using lambda expression
    auto func = [](mfem::Vector& vparam, mfem::ad::ADVectorType& uu, mfem::ad::ADVectorType& vres) {
    //auto func = [](auto& vparam, auto& uu, auto& vres) { //c++14
        auto kappa = vparam[0]; //diffusion coefficient
        auto load = vparam[1]; //volumetric influx

        vres[0] = kappa * uu[0];
        vres[1] = kappa * uu[1];
        vres[2] = kappa * uu[2];
        vres[3] = -load;
    };

    mfem::FResidualAutoDiff<4,4,2> fdr(func);
    fdr.QGradResidual(param,state,hh1); //computes the gradient of func and stores the result in hh1
    std::cout<<"LambdaAutoDiff"<<std::endl;
    hh1.Print(std::cout);


    double kappa = param[0];
    double load =  param[1];
    //using lambda expression
    auto func01 = [&kappa,&load](mfem::Vector& vparam, mfem::ad::ADVectorType& uu, mfem::ad::ADVectorType& vres) {
    //auto func = [](auto& vparam, auto& uu, auto& vres) { //c++14

        vres[0] = kappa * uu[0];
        vres[1] = kappa * uu[1];
        vres[2] = kappa * uu[2];
        vres[3] = -load;
    };

    mfem::FResidualAutoDiff<4,4,2> fdr01(func01);
    fdr01.QGradResidual(param,state,hh1);
    std::cout<<"LambdaAutoDiff 01"<<std::endl;
    hh1.Print(std::cout);


}
