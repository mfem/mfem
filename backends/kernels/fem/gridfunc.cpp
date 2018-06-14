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

//std::map<std::string, ::kernels::kernel> gridFunctionKernels;

/*::kernels::kernel GetGridFunctionKernel(::kernels::device device,
                                     FiniteElementSpace &fespace,
                                     const mfem::IntegrationRule &ir)
{
   const int numQuad = ir.GetNPoints();

   const FiniteElement &fe = *(fespace.GetFE(0));
   const int dim  = fe.GetDim();
   const int vdim = fespace.GetVDim();

   std::stringstream ss;
   ss << ::kernels::hash(device)
      << "FEColl : " << fespace.FEColl()->Name()
      << "Quad: "    << numQuad
      << "Dim: "     << dim
      << "VDim: "    << vdim;
   std::string hash = ss.str();

   // Kernel defines
   ::kernels::properties props;
   props["defines/NUM_VDIM"] = vdim;

   SetProperties(fespace, ir, props);

   ::kernels::kernel kernel = gridFunctionKernels[hash];
   if (!kernel.isInitialized())
   {
      const std::string &okl_path = fespace.KernelsEngine().GetOklPath();
      kernel = device.buildKernel(okl_path + "/gridfunc.okl",
                                  stringWithDim("GridFuncToQuad", dim),
                                  props);
   }
   return kernel;
   }*/

// KernelsGridFunction::KernelsGridFunction() :
//    Vector(),
//    ofespace(NULL),
//    sequence(0) {}

KernelsGridFunction::KernelsGridFunction(KernelsFiniteElementSpace &f)
   : PArray(f.KernelsVLayout()),
     Array(f.KernelsVLayout(), sizeof(double)),
     Vector(f.KernelsVLayout()),
     fes(f),
     sequence(0) {}

KernelsGridFunction::KernelsGridFunction(KernelsFiniteElementSpace &f,
                                   const KernelsVector *_v) :
   PArray(f.KernelsVLayout()),
   Array(f.KernelsVLayout(), sizeof(double)),
   Vector(f.KernelsVLayout()),
   fes(f),
   sequence(0),
   v(_v) {}


KernelsGridFunction::KernelsGridFunction(const KernelsGridFunction &v)
   : PArray(v),
     Array(v),
     Vector(v),
     fes(v.fes),
     sequence(v.sequence) {}

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

// KernelsGridFunction& KernelsGridFunction::operator = (const KernelsVectorRef &v)
// {
//    KernelsVector::operator = (v);
//    return *this;
// }

KernelsGridFunction& KernelsGridFunction::operator = (const KernelsGridFunction &v)
{
   Assign<double>(v);
   return *this;
}

// void KernelsGridFunction::SetGridFunction(mfem::GridFunction &gf)
// {
//    Vector v = *this;
//    gf.MakeRef(ofespace->GetFESpace(), v, 0);
//    // Make gf the owner of the data
//    v.Swap(gf);
// }

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

   //KernelsDofQuadMaps *maps = KernelsDofQuadMaps::Get(*ofespace, ir);

   const int elements = fes.GetNE();
   const int numQuad  = ir.GetNPoints();
   quadValues.Resize<double>(*(new Layout(engine, numQuad * elements)), NULL);

   //::kernels::kernel g2qKernel = GetGridFunctionKernel(device, *ofespace, ir);
   assert(false);/*
   g2qKernel(elements,
             maps.dofToQuad,
             ofespace->GetLocalToGlobalMap(),
             this->KernelsMem(),
             quadValues.KernelsMem());*/
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
