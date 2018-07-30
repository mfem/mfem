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

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "gridfunc.hpp"
#include "bilininteg.hpp"
#include "../../fem/gridfunc.hpp"

namespace mfem
{

namespace occa
{

std::map<std::string, ::occa::kernel> gridFunctionKernels;

::occa::kernel GetGridFunctionKernel(::occa::device device,
                                     FiniteElementSpace &fespace,
                                     const mfem::IntegrationRule &ir)
{
   const int numQuad = ir.GetNPoints();

   const FiniteElement &fe = *(fespace.GetFE(0));
   const int dim  = fe.GetDim();
   const int vdim = fespace.GetVDim();

   std::stringstream ss;
   ss << ::occa::hash(device)
      << "FEColl : " << fespace.FEColl()->Name()
      << "Quad: "    << numQuad
      << "Dim: "     << dim
      << "VDim: "    << vdim;
   std::string hash = ss.str();

   // Kernel defines
   ::occa::properties props;
   props["defines/NUM_VDIM"] = vdim;

   SetProperties(fespace, ir, props);

   ::occa::kernel kernel = gridFunctionKernels[hash];
   if (!kernel.isInitialized())
   {
      const std::string &okl_path = fespace.OccaEngine().GetOklPath();
      kernel = device.buildKernel(okl_path + "gridfunc.okl",
                                  stringWithDim("GridFuncToQuad", dim),
                                  props);
   }
   return kernel;
}

// OccaGridFunction::OccaGridFunction() :
//    Vector(),
//    ofespace(NULL),
//    sequence(0) {}

OccaGridFunction::OccaGridFunction(FiniteElementSpace *ofespace_)
   : PArray(ofespace_->OccaVLayout()),
     Array(ofespace_->OccaVLayout(), sizeof(double)),
     Vector(ofespace_->OccaVLayout()),
     ofespace(ofespace_),
     sequence(0) {}

// OccaGridFunction::OccaGridFunction(OccaFiniteElementSpace *ofespace_,
//                                    OccaVectorRef ref) :
//    OccaVector(ref),
//    ofespace(ofespace_),
//    sequence(0) {}

OccaGridFunction::OccaGridFunction(const OccaGridFunction &v)
   : PArray(v),
     Array(v),
     Vector(v),
     ofespace(v.ofespace),
     sequence(v.sequence) {}

OccaGridFunction& OccaGridFunction::operator = (double value)
{
   Fill(value);
   return *this;
}

OccaGridFunction& OccaGridFunction::operator = (const Vector &v)
{
   Assign<double>(v);
   return *this;
}

// OccaGridFunction& OccaGridFunction::operator = (const OccaVectorRef &v)
// {
//    OccaVector::operator = (v);
//    return *this;
// }

OccaGridFunction& OccaGridFunction::operator = (const OccaGridFunction &v)
{
   Assign<double>(v);
   return *this;
}

// void OccaGridFunction::SetGridFunction(mfem::GridFunction &gf)
// {
//    Vector v = *this;
//    gf.MakeRef(ofespace->GetFESpace(), v, 0);
//    // Make gf the owner of the data
//    v.Swap(gf);
// }

void OccaGridFunction::GetTrueDofs(Vector &v)
{
   const mfem::Operator *R = ofespace->GetRestrictionOperator();
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

void OccaGridFunction::SetFromTrueDofs(Vector &v)
{
   const mfem::Operator *P = ofespace->GetProlongationOperator();
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

mfem::FiniteElementSpace* OccaGridFunction::GetFESpace()
{
   return ofespace->GetFESpace();
}

const mfem::FiniteElementSpace* OccaGridFunction::GetFESpace() const
{
   return ofespace->GetFESpace();
}

void OccaGridFunction::ToQuad(const IntegrationRule &ir, Vector &quadValues)
{
   const Engine &engine = OccaLayout().OccaEngine();
   ::occa::device device = engine.GetDevice();

   OccaDofQuadMaps &maps = OccaDofQuadMaps::Get(device, *ofespace, ir);

   const int elements = ofespace->GetNE();
   const int numQuad  = ir.GetNPoints();
   quadValues.Resize<double>(*(new Layout(engine, numQuad * elements)), NULL);

   ::occa::kernel g2qKernel = GetGridFunctionKernel(device, *ofespace, ir);
   g2qKernel(elements,
             maps.dofToQuad,
             ofespace->GetLocalToGlobalMap(),
             this->OccaMem(),
             quadValues.OccaMem());
}

void ToQuad(const IntegrationRule &ir, FiniteElementSpace &fespace, Vector &gf, Vector &quadValues)
{
   const Engine &engine = fespace.OccaEngine();
   ::occa::device device = engine.GetDevice();

   OccaDofQuadMaps &maps = OccaDofQuadMaps::Get(device, fespace, ir);

   const int elements = fespace.GetNE();
   const int numQuad  = ir.GetNPoints();
   quadValues.Resize<double>(*(new Layout(engine, numQuad * elements)), NULL);

   ::occa::kernel g2qKernel = GetGridFunctionKernel(device, fespace, ir);
   g2qKernel(elements,
             maps.dofToQuad,
             fespace.GetLocalToGlobalMap(),
             gf.OccaMem(),
             quadValues.OccaMem());
}


void OccaGridFunction::Distribute(const Vector &v)
{
   if (ofespace->isDistributed())
   {
      mfem::Vector mfem_this(*this);
      ofespace->GetProlongationOperator()->Mult(v.Wrap(), mfem_this);
   }
   else
   {
      *this = v;
   }
}

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
