// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "admfem.hpp"
#include "mfem.hpp"

template<typename TDataType, typename TParamVector, typename TStateVector
         , int state_size, int param_size>
class DiffusionFunctional
{
public:
   TDataType operator() (TParamVector& vparam, TStateVector& uu)
   {
      MFEM_ASSERT(state_size==4,"ExampleFunctor state_size should be equal to 4!");
      MFEM_ASSERT(param_size==2,"ExampleFunctor param_size should be equal to 2!");
      auto kappa = vparam[0]; // diffusion coefficient
      auto load = vparam[1];  // volumetric influx
      TDataType rez = kappa*(uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2])/2.0 - load*uu[3];
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
      MFEM_ASSERT(residual_size==4,
                  "DiffusionResidual residual_size should be equal to 4!");
      MFEM_ASSERT(state_size==4,"ExampleFunctor state_size should be equal to 4!");
      MFEM_ASSERT(param_size==2,"ExampleFunctor param_size should be equal to 2!");
      auto kappa = vparam[0]; // diffusion coefficient
      auto load = vparam[1];  // volumetric influx

      rr[0] = kappa * uu[0];
      rr[1] = kappa * uu[1];
      rr[2] = kappa * uu[2];
      rr[3] = -load;
   }
};

int main(int argc, char *argv[])
{

#ifdef MFEM_USE_ADFORWARD
   std::cout<<"MFEM_USE_ADFORWARD == true"<<std::endl;
#else
   std::cout<<"MFEM_USE_ADFORWARD == false"<<std::endl;
#endif

#ifdef MFEM_USE_CALIPER
   cali::ConfigManager mgr;
#endif
   // Caliper instrumentation
   MFEM_PERF_FUNCTION;
#ifdef MFEM_USE_CALIPER
   const char* cali_config = "runtime-report";
   mgr.add(cali_config);
   mgr.start();
#endif
   mfem::Vector param(2);
   param[0]=3.0; // diffusion coefficient
   param[1]=2.0; // volumetric influx

   mfem::Vector state(4);
   state[0]=1.0; // grad_x
   state[1]=2.0; // grad_y
   state[2]=3.0; // grad_z
   state[3]=4.0; // state value

   mfem::QFunctionAutoDiff<DiffusionFunctional,4,2> adf;

   mfem::Vector rr0(4);
   mfem::DenseMatrix hh0(4,4);

   mfem::Vector rr1(4);
   mfem::DenseMatrix hh1(4,4);
   MFEM_PERF_BEGIN("Grad");
   adf.Grad(param,state,rr0);
   MFEM_PERF_END("Grad");
   MFEM_PERF_BEGIN("Hessian");
   adf.Hessian(param, state, hh0);
   MFEM_PERF_END("Hessian");
   // dump out the results
   std::cout<<"FunctionAutoDiff"<<std::endl;
   std::cout<< adf.Eval(param,state)<<std::endl;
   rr0.Print(std::cout);
   hh0.Print(std::cout);

   mfem::QVectorFuncAutoDiff<DiffusionResidual,4,4,2> rdf;
   MFEM_PERF_BEGIN("Jacobian");
   rdf.Jacobian(param, state, hh1);
   MFEM_PERF_END("Jacobian");

   std::cout<<"ResidualAutoDiff"<<std::endl;
   hh1.Print(std::cout);

   // using lambda expression
   auto func = [](mfem::Vector& vparam,
                  mfem::ad::ADVectorType& uu,
                  mfem::ad::ADVectorType& vres)
   {
      // auto func = [](auto& vparam, auto& uu, auto& vres) { //c++14
      auto kappa = vparam[0]; // diffusion coefficient
      auto load = vparam[1];  // volumetric influx

      vres[0] = kappa * uu[0];
      vres[1] = kappa * uu[1];
      vres[2] = kappa * uu[2];
      vres[3] = -load;
   };

   mfem::VectorFuncAutoDiff<4,4,2> fdr(func);
   MFEM_PERF_BEGIN("JacobianV");
   fdr.Jacobian(param,state,
                hh1); // computes the gradient of func and stores the result in hh1
   MFEM_PERF_END("JacobianV");
   std::cout<<"LambdaAutoDiff"<<std::endl;
   hh1.Print(std::cout);


   double kappa = param[0];
   double load =  param[1];
   // using lambda expression
   auto func01 = [&kappa,&load](mfem::Vector& vparam,
                                mfem::ad::ADVectorType& uu,
                                mfem::ad::ADVectorType& vres)
   {
      // auto func = [](auto& vparam, auto& uu, auto& vres) { //c++14

      vres[0] = kappa * uu[0];
      vres[1] = kappa * uu[1];
      vres[2] = kappa * uu[2];
      vres[3] = -load;
   };

   mfem::VectorFuncAutoDiff<4,4,2> fdr01(func01);
   MFEM_PERF_BEGIN("Jacobian1");
   fdr01.Jacobian(param,state,hh1);
   MFEM_PERF_END("Jacobian1");
   std::cout<<"LambdaAutoDiff 01"<<std::endl;
   hh1.Print(std::cout);

#ifdef MFEM_USE_CALIPER
   mgr.flush();
#endif
}
