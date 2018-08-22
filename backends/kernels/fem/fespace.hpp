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

#ifndef MFEM_BACKENDS_KERNELS_FESPACE_HPP
#define MFEM_BACKENDS_KERNELS_FESPACE_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

// *****************************************************************************
class kFiniteElementSpace : public mfem::PFiniteElementSpace
{
protected:
   Layout e_layout;
   int globalDofs, localDofs;
   int vdim;
   mfem::Ordering::Type ordering;
   kernels::array<int> offsets;
   kernels::array<int> indices,*reorderIndices;
   kernels::array<int> map;
   mfem::Operator *restrictionOp, *prolongationOp;
public:
   /// TODO: doxygen
   kFiniteElementSpace(const Engine&, mfem::FiniteElementSpace&);

   /// Virtual destructor
   virtual ~kFiniteElementSpace();

   /// TODO: doxygen
   const kernels::Engine &KernelsEngine() const
   { return *static_cast<const Engine *>(engine.Get()); }

   /// TODO: doxygen
   kernels::device GetDevice(int idx = 0) const
   { return KernelsEngine().GetDevice(idx); }

   mfem::Mesh* GetMesh() const { return fes->GetMesh(); }

   mfem::FiniteElementSpace* GetFESpace() const { return fes; }

#ifdef MFEM_USE_MPI
   mfem::ParFiniteElementSpace& GetParFESpace() const
   {
      return *static_cast<ParFiniteElementSpace*>(fes);
   }
#endif

   Layout &KernelsVLayout() const
   { return *fes->GetVLayout().As<Layout>(); }

   Layout &KernelsTrueVLayout() const
   { return *fes->GetTrueVLayout().As<Layout>(); }

   Layout &KernelsEVLayout() { return e_layout; }

#ifdef MFEM_USE_MPI
   bool isDistributed() const
   {
      return (KernelsEngine().GetComm() != MPI_COMM_NULL);
   }
#else
   bool isDistributed() const { return false; }
#endif

   bool hasTensorBasis() const
   {
      return dynamic_cast<const mfem::TensorBasisElement*>(fes->GetFE(0));
   }

   mfem::Ordering::Type GetOrdering() const { return ordering; }

   int GetGlobalDofs() const { return globalDofs; }
   int GetLocalDofs() const { return localDofs; }

   int GetDim() const { return fes->GetMesh()->Dimension(); }
   int GetVDim() const { return vdim; }

   int GetVSize() const { return globalDofs * vdim; }
   int GetTrueVSize() const { return fes->GetTrueVSize(); }
   int GetGlobalVSize() const { return globalDofs*vdim; /* FIXME: MPI */ }
   int GetGlobalTrueVSize() const { return fes->GetTrueVSize(); }

   int GetNE() const { return fes->GetNE(); }

   const mfem::FiniteElementCollection* FEColl() const
   { return fes->FEColl(); }

   const mfem::FiniteElement* GetFE(const int idx) const
   { return fes->GetFE(idx); }

   const mfem::Operator* GetRestrictionOperator() const { return restrictionOp; }
   const mfem::Operator* GetProlongationOperator() const { return prolongationOp; }
   
   virtual const mfem::Operator *GetInterpolationOperator(
      const mfem::QuadratureSpace &qspace) const
   { return NULL; /* FIXME */ }
   
   virtual const mfem::Operator *GetGradientOperator(
      const mfem::QuadratureSpace &qspace) const
   { return NULL; /* FIXME */ }

   const kernels::array<int> GetLocalToGlobalMap() const
   { return map; }

   void GlobalToLocal(const kernels::Vector &global, kernels::Vector &local) const;
   void LocalToGlobal(const kernels::Vector &local, kernels::Vector &global) const;

};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_FESPACE_HPP
