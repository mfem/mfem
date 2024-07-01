
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

void ADVectorFunc::Jacobian(const Vector &vstate,DenseMatrix &jac)
{
   MFEM_ASSERT(vstate.Size() == jac.Width(),
               "state and jacobian size do not match");

   jac = 0.0;
#ifdef MFEM_USE_CODIPACK
   int vector_size = jac.Height();
   int state_size = jac.Width();
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

void ADVectorFunc::Curl(const Vector &vstate, Vector &curl)
{
   MFEM_ASSERT(vstate.Size() == 3, "state size expected to be 3");
   MFEM_ASSERT(curl.Size() == 3, "solution size expected to be 3");
   DenseMatrix jac(vstate.Size()); // Assume the jacobian is square
   Jacobian(vstate, jac);

   curl[0] = jac(2,1) - jac(1,2);
   curl[1] = jac(0,2) - jac(2,0);
   curl[2] = jac(1,0) - jac(0,1);
}

real_t ADVectorFunc::Divergence(const Vector &vstate)
{
   DenseMatrix jac(vstate.Size()); // Assume the jacobian is square
   Jacobian(vstate, jac);
   real_t div = 0.0;
   for (int ii = 0; ii<vstate.Size(); ii++)
   {
      div += jac(ii,ii);
   }
   return div;
}

void ADVectorFunc::Gradient(const Vector &vstate, Vector &grad)
{
   MFEM_ASSERT( grad.Size() == vstate.Size(), "grad and state size not equal");

   DenseMatrix jac(1, vstate.Size());
   Jacobian(vstate, jac);

   jac.GetRow(0, grad);
}

void ADVectorFunc::Solution(const Vector &vstate, Vector &sol)
{
   ADVector ad_state(vstate.Size());
   ADVector ad_result(sol.Size());
   for (int i=0; i<vstate.Size(); i++)
   {
#ifdef MFEM_USE_CODIPACK
      ad_state[i].setValue(vstate[i]);
      ad_state[i].setGradient(0.0);
#else
      ad_state[i] = vstate[i];
#endif
   }

   F(param,ad_state,ad_result);
   for (int jj=0; jj<sol.Size(); jj++)
   {
#ifdef MFEM_USE_CODIPACK
      sol[jj] = ad_result[jj].getValue();
#else
      sol[jj] = ad_result[jj];
#endif
   }
}

real_t ADVectorFunc::ScalarSolution(const Vector &vstate)
{
   Vector sol(1);
   Solution(vstate, sol);
   return sol[0];
}

void ADVectorTDFunc::Jacobian(const Vector &vstate, const real_t time,
                              DenseMatrix &jac)
{
   MFEM_ASSERT(vstate.Size() == jac.Width(),
               "state and jacobian size do not match");

   jac = 0.0;
#ifdef MFEM_USE_CODIPACK
   int vector_size = jac.Height();
   int state_size = jac.Width();
   ADVector ad_state(state_size);
   ADVector ad_result(vector_size);
   ADFloat ad_t(time);
#ifdef MFEM_USE_ADFORWARD
   ad_t.setGradient(0.0);
   for (int i=0; i<state_size; i++)
   {
      ad_state[i].setValue(vstate[i]);
      ad_state[i].setGradient(0.0);
   }

   for (int ii=0; ii<state_size; ii++)
   {
      ad_state[ii].setGradient(1.0);
      F(param,ad_state, ad_t, ad_result);
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
   F(param,ad_state,ad_t,ad_result);
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

void ADVectorTDFunc::Curl(const Vector &vstate, const real_t time, Vector &curl)
{
   MFEM_ASSERT(vstate.Size() == 3, "state size expected to be 3");
   MFEM_ASSERT(curl.Size() == 3, "solution size expected to be 3");
   DenseMatrix jac(vstate.Size()); // Assume the jacobian is square
   Jacobian(vstate, time, jac);

   curl[0] = jac(2,1) - jac(1,2);
   curl[1] = jac(0,2) - jac(2,0);
   curl[2] = jac(1,0) - jac(0,1);
}

real_t ADVectorTDFunc::Divergence(const Vector &vstate, const real_t time)
{
   DenseMatrix jac(vstate.Size()); // Assume the jacobian is square
   Jacobian(vstate, time, jac);
   real_t div = 0.0;
   for (int ii = 0; ii<vstate.Size(); ii++)
   {
      div += jac(ii,ii);
   }
   return div;
}

void ADVectorTDFunc::Gradient(const Vector &vstate, const real_t time,
                              Vector &grad)
{
   MFEM_ASSERT( grad.Size() == vstate.Size(), "grad and state size not equal");

   DenseMatrix jac(1, vstate.Size());
   Jacobian(vstate, time, jac);

   jac.GetRow(0, grad);
}

void ADVectorTDFunc::Rate(const Vector &vstate, const real_t time, Vector &rate)
{
#ifdef MFEM_USE_CODIPACK
   ADVector ad_state(vstate.Size());
   ADVector ad_result(rate.Size());
   ADFloat  ad_t(time);
#ifdef MFEM_USE_ADFORWARD
   ad_t.setGradient(1.0);
   for (int i=0; i<vstate.Size(); i++)
   {
      ad_state[i].setValue(vstate[i]);
      ad_state[i].setGradient(0.0);
   }

   F(param,ad_state,ad_t,ad_result);
   for (int jj=0; jj<rate.Size(); jj++)
   {
      rate[jj] = ad_result[jj].getGradient();
   }
#else
   for (int i=0; i<vstate.Size(); i++)
   {
      ad_state[i] = vstate[i];
   }

   ADFloat::Tape& tape = ADFloat::getTape();
   typename ADFloat::Tape::Position pos = tape.getPosition();

   tape.setActive();
   tape.registerInput(ad_t);
   F(param,ad_state,ad_t,ad_result);
   for (int ii=0; ii<rate.Size(); ii++) { tape.registerOutput(ad_result[ii]); }
   tape.setPassive();

   for (int jj=0; jj<rate.Size(); jj++)
   {
      ad_result[jj].setGradient(1.0);
      tape.evaluate();

      rate[jj] = ad_t.getGradient();
      tape.clearAdjoints();
      ad_result[jj].setGradient(0.0);
   }
   tape.resetTo(pos);
#endif
#else
   MFEM_WARNING("not compiled with CoDiPack -- will return zero rate");
   rate = 0.0;
#endif
}

real_t ADVectorTDFunc::ScalarRate(const Vector &vstate, const real_t time)
{
   Vector rv(1);
   Rate(vstate, time, rv);
   return rv[0];
}

void ADVectorTDFunc::Solution(const Vector &vstate, const real_t time,
                              Vector &sol)
{
   ADVector ad_state(vstate.Size());
   ADVector ad_result(sol.Size());
   ADFloat  ad_t(time);
   for (int i=0; i<vstate.Size(); i++)
   {
#ifdef MFEM_USE_CODIPACK
      ad_state[i].setValue(vstate[i]);
      ad_state[i].setGradient(0.0);
#else
      ad_state[i] = vstate[i];
#endif
   }

   F(param,ad_state,ad_t,ad_result);
   for (int jj=0; jj<sol.Size(); jj++)
   {
#ifdef MFEM_USE_CODIPACK
      sol[jj] = ad_result[jj].getValue();
#else
      sol[jj] = ad_result[jj];
#endif
   }
}

real_t ADVectorTDFunc::ScalarSolution(const Vector &vstate, const real_t time)
{
   Vector sol(1);
   Solution(vstate, time, sol);
   return sol[0];
}

}
