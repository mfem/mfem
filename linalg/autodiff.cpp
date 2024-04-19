
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

#include "autodiff.hpp"

namespace mfem
{

template<int param_size, int vector_size, int state_size>
void ADVectorFunc<param_size,vector_size, state_size>
::Jacobian(mfem::Vector &vstate,mfem::DenseMatrix &jac)
{
   jac.SetSize(vector_size, state_size);
   jac = 0.0;
#ifdef MFEM_USE_CODIPACK
   ADVector ad_state(state_size);
   ADVector ad_result(vector_size);
#ifdef MFEM_USE_ADFORWARD
   for (int i=0; i<state_size; i++)
   {
      ad_state[i].setValue(vstate[i]);
      ad_state[i].setGradient(0.0);
   }
   for (int ii=0; ii<state_size; ii++)
   {
      ad_state[ii].setGradient(1.0);
      F(param,ad_state,ad_result);
      for (int jj=0; jj<vector_size; jj++)
      {
         jac(jj,ii)=ad_result[jj].getGradient();
      }
      ad_state[ii].setGradient(0.0);
   }
#else // use reverse mode
   for (int i=0; i<state_size; i++)
   {
      ad_state[i]=vstate[i];
   }

   ADFloat::Tape& tape = ADFloat::getTape();
   typename ADFloat::Tape::Position pos = tape.getPosition();

   tape.setActive();
   for (int ii=0; ii<state_size; ii++) { tape.registerInput(ad_state[ii]); }
   F(param,ad_state,ad_result);
   for (int ii=0; ii<vector_size; ii++) { tape.registerOutput(ad_result[ii]); }
   tape.setPassive();

   for (int jj=0; jj<vector_size; jj++)
   {
      ad_result[jj].setGradient(1.0);
      tape.evaluate();
      for (int ii=0; ii<state_size; ii++)
      {
         jac(jj,ii)=ad_state[ii].getGradient();
      }
      tape.clearAdjoints();
      ad_result[jj].setGradient(0.0);
   }
   tape.resetTo(pos);
#endif
#else
   MFEM_WARNING("not compiled with CoDiPack -- will return zero gradient");
#endif
}

template<int param_size, int vector_size, int state_size>
void ADVectorFunc<param_size,vector_size, state_size>
::Curl(mfem::Vector &vstate, mfem::Vector &curl)
{
   MFEM_ASSERT(state_size == 3, "!");
   MFEM_ASSERT(vector_size == 3, "!");
   mfem::DenseMatrix jac;
   Jacobian(vstate, jac);
   curl.SetSize(state_size);

   curl[0] = jac(2,1) - jac(1,2);
   curl[1] = jac(0,2) - jac(2,0);
   curl[2] = jac(1,0) - jac(0,1);
}

template<int param_size, int vector_size, int state_size>
real_t ADVectorFunc<param_size,vector_size, state_size>
::Divergence(mfem::Vector &vstate)
{
   mfem::DenseMatrix jac;
   Jacobian(vstate, jac);
   real_t div = 0.0;
   for (int ii=0; ii<state_size; ii++)
   {
      div += jac(ii,ii);
   }
   return div;
}

template<int param_size, int vector_size, int state_size>
void ADVectorFunc<param_size,vector_size, state_size>
::Gradient(mfem::Vector &vstate, mfem::Vector &grad)
{
   mfem::DenseMatrix jac;
   Jacobian(vstate, jac);
   grad.SetSize(state_size);
   jac.GetRow(0, grad);
}

template<int param_size, int vector_size, int state_size>
void ADVectorFunc<param_size,vector_size, state_size>
::Solution(mfem::Vector &vstate, mfem::Vector &sol)
{
   sol.SetSize(vector_size);
   sol = 0.0;
   {
      ADVector ad_state(state_size);
      ADVector ad_result(vector_size);
      for (int i=0; i<state_size; i++)
      {
         ad_state[i].setValue(vstate[i]);
         ad_state[i].setGradient(0.0);
      }

      F(param,ad_state,ad_result);
      for (int jj=0; jj<vector_size; jj++)
      {
         sol[jj] = ad_result[jj].getValue();
      }
   }
}

template<int param_size, int vector_size, int state_size>
real_t ADVectorFunc<param_size,vector_size, state_size>
::Solution(mfem::Vector &vstate)
{
   mfem::Vector sol;
   Solution(vstate, sol);
   return sol[0];
}

}

