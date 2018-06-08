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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"

namespace mfem
{

namespace raja
{

//std::map<std::string, ::raja::kernel> gridFunctionKernels;

/*::raja::kernel GetGridFunctionKernel(::raja::device device,
                                     FiniteElementSpace &fespace,
                                     const mfem::IntegrationRule &ir)
{
   const int numQuad = ir.GetNPoints();

   const FiniteElement &fe = *(fespace.GetFE(0));
   const int dim  = fe.GetDim();
   const int vdim = fespace.GetVDim();

   std::stringstream ss;
   ss << ::raja::hash(device)
      << "FEColl : " << fespace.FEColl()->Name()
      << "Quad: "    << numQuad
      << "Dim: "     << dim
      << "VDim: "    << vdim;
   std::string hash = ss.str();

   // Kernel defines
   ::raja::properties props;
   props["defines/NUM_VDIM"] = vdim;

   SetProperties(fespace, ir, props);

   ::raja::kernel kernel = gridFunctionKernels[hash];
   if (!kernel.isInitialized())
   {
      const std::string &okl_path = fespace.RajaEngine().GetOklPath();
      kernel = device.buildKernel(okl_path + "/gridfunc.okl",
                                  stringWithDim("GridFuncToQuad", dim),
                                  props);
   }
   return kernel;
   }*/

// RajaGridFunction::RajaGridFunction() :
//    Vector(),
//    ofespace(NULL),
//    sequence(0) {}

RajaGridFunction::RajaGridFunction(FiniteElementSpace &f)
   : PArray(f.RajaVLayout()),
     Array(f.RajaVLayout(), sizeof(double)),
     Vector(f.RajaVLayout()),
     fes(f),
     sequence(0) {}

// RajaGridFunction::RajaGridFunction(RajaFiniteElementSpace *ofespace_,
//                                    RajaVectorRef ref) :
//    RajaVector(ref),
//    ofespace(ofespace_),
//    sequence(0) {}

RajaGridFunction::RajaGridFunction(const RajaGridFunction &v)
   : PArray(v),
     Array(v),
     Vector(v),
     fes(v.fes),
     sequence(v.sequence) {}

RajaGridFunction& RajaGridFunction::operator = (double value)
{
   Fill(value);
   return *this;
}

RajaGridFunction& RajaGridFunction::operator = (const Vector &v)
{
   Assign<double>(v);
   return *this;
}

// RajaGridFunction& RajaGridFunction::operator = (const RajaVectorRef &v)
// {
//    RajaVector::operator = (v);
//    return *this;
// }

RajaGridFunction& RajaGridFunction::operator = (const RajaGridFunction &v)
{
   Assign<double>(v);
   return *this;
}

// void RajaGridFunction::SetGridFunction(mfem::GridFunction &gf)
// {
//    Vector v = *this;
//    gf.MakeRef(ofespace->GetFESpace(), v, 0);
//    // Make gf the owner of the data
//    v.Swap(gf);
// }

void RajaGridFunction::GetTrueDofs(Vector &v)
{
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
}

void RajaGridFunction::SetFromTrueDofs(Vector &v)
{
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
}

mfem::FiniteElementSpace* RajaGridFunction::GetFESpace()
{
   return fes.GetFESpace();
}

const mfem::FiniteElementSpace* RajaGridFunction::GetFESpace() const
{
   return fes.GetFESpace();
}

void RajaGridFunction::ToQuad(const IntegrationRule &ir, Vector &quadValues)
{
   const Engine &engine = RajaLayout().RajaEngine();
   raja::device device = engine.GetDevice();

   //RajaDofQuadMaps *maps = RajaDofQuadMaps::Get(*ofespace, ir);

   const int elements = fes.GetNE();
   const int numQuad  = ir.GetNPoints();
   quadValues.Resize<double>(*(new Layout(engine, numQuad * elements)), NULL);

   //::raja::kernel g2qKernel = GetGridFunctionKernel(device, *ofespace, ir);
   assert(false);/*
   g2qKernel(elements,
             maps.dofToQuad,
             ofespace->GetLocalToGlobalMap(),
             this->RajaMem(),
             quadValues.RajaMem());*/
}

void RajaGridFunction::Distribute(const Vector &v)
{
   if (fes.isDistributed())
   {
      mfem::Vector mfem_this(*this);
      fes.GetProlongationOperator()->Mult(v.Wrap(), mfem_this);
   }
   else
   {
      *this = v;
   }
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
