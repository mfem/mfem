// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

KernelsGridFunction::KernelsGridFunction(kFiniteElementSpace &f)
   : PArray(f.KernelsVLayout()),
     Array(f.KernelsVLayout(), sizeof(double)),
     Vector(f.KernelsVLayout()),
     fes(f),
     sequence(0) {push();pop();}

KernelsGridFunction::KernelsGridFunction(const KernelsGridFunction &v)
   : PArray(v),
     Array(v),
     Vector(v),
     fes(v.fes),
     sequence(v.sequence) {push();pop();}

KernelsGridFunction& KernelsGridFunction::operator = (double value)
{
   Fill(value);
   return *this;
}

KernelsGridFunction& KernelsGridFunction::operator = (const Vector &v)
{
   Assign<double>(v);
   return *this;
}

KernelsGridFunction& KernelsGridFunction::operator = (const KernelsGridFunction
                                                      &v)
{
   Assign<double>(v);
   return *this;
}

void KernelsGridFunction::GetTrueDofs(Vector &v)
{
   push();
   const mfem::Operator *R = fes.GetRestrictionOperator();
   if (!R)
   {
      v.MakeRef(*this);
   }
   else
   {
      v.Resize<double>(R->OutLayout(), NULL);
      mfem::Vector mfem_v(v);
      R->Mult(this->Wrap(), mfem_v);
   }
   pop();
}

void KernelsGridFunction::SetFromTrueDofs(Vector &v)
{
   push();
   const mfem::Operator *P = fes.GetProlongationOperator();
   if (!P)
   {
      MakeRef(v);
   }
   else
   {
      Resize<double>(P->OutLayout(), NULL);
      mfem::Vector mfem_this(*this);
      P->Mult(v.Wrap(), mfem_this);
   }
   pop();
}

mfem::FiniteElementSpace* KernelsGridFunction::GetFESpace()
{
   return fes.GetFESpace();
}

const mfem::FiniteElementSpace* KernelsGridFunction::GetFESpace() const
{
   return fes.GetFESpace();
}

void KernelsGridFunction::ToQuad(const IntegrationRule &ir, Vector &quadValues)
{
   push();
   const Engine &engine = KernelsLayout().KernelsEngine();
   kernels::device device = engine.GetDevice();

   const int elements = fes.GetNE();
   const int numQuad  = ir.GetNPoints();
   quadValues.Resize<double>(*(new Layout(engine, numQuad * elements)), NULL);
   assert(false);
   pop();
}

void KernelsGridFunction::Distribute(const Vector &v)
{
   push();
   if (fes.isDistributed())
   {
      mfem::Vector mfem_this(*this);
      fes.GetProlongationOperator()->Mult(v.Wrap(), mfem_this);
   }
   else
   {
      *this = v;
   }
   pop();
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
