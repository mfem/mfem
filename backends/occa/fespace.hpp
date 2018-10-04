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

#ifndef MFEM_BACKENDS_OCCA_FE_SPACE_HPP
#define MFEM_BACKENDS_OCCA_FE_SPACE_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "engine.hpp"
#include "operator.hpp"
#include "../../fem/fem.hpp"

namespace mfem
{

namespace occa
{

/// TODO: doxygen
class FiniteElementSpace : public mfem::PFiniteElementSpace
{
protected:
   //
   // Inherited fields
   //
   // SharedPtr<const mfem::Engine> engine;
   // mfem::FiniteElementSpace *fes;

   SharedPtr<Layout> e_layout;

   ::occa::array<int> globalToLocalOffsets;
   ::occa::array<int> globalToLocalIndices;
   ::occa::array<int> localToGlobalMap;
   ::occa::kernel globalToLocalKernel, localToGlobalKernel;

   mfem::Ordering::Type ordering;

   int globalDofs, localDofs;
   int vdim;

   mutable Operator *prolongationOp, *restrictionOp;

   void SetupLocalGlobalMaps();
   void SetupOperators() const; // calls virtual methods of 'fes' !!!
   void SetupKernels();

public:
   /// TODO: doxygen
   FiniteElementSpace(const Engine &e, mfem::FiniteElementSpace &fespace);

   /// Virtual destructor
   virtual ~FiniteElementSpace();

   /// TODO: doxygen
   const Engine &OccaEngine() const { return engine->As<Engine>(); }

   /// TODO: doxygen
   ::occa::device GetDevice(int idx = 0) const
   { return OccaEngine().GetDevice(idx); }

   mfem::Mesh* GetMesh() const { return fes->GetMesh(); }

   Layout &OccaVLayout() const
   { return *fes->GetVLayout().As<Layout>(); }

   Layout &OccaTrueVLayout() const
   { return *fes->GetTrueVLayout().As<Layout>(); }

   Layout &OccaEVLayout() { return *e_layout; }

   bool hasTensorBasis() const
   { return dynamic_cast<const mfem::TensorBasisElement*>(fes->GetFE(0)); }

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

   const mfem::FiniteElementCollection *FEColl() const
   { return fes->FEColl(); }
   const mfem::FiniteElement *GetFE(const int idx) const
   { return fes->GetFE(idx); }

   virtual const mfem::Operator *GetProlongationOperator() const
   { return prolongationOp; }

   virtual const mfem::Operator *GetRestrictionOperator() const
   { return restrictionOp; }

   virtual const mfem::Operator *GetInterpolationOperator(
      const mfem::QuadratureSpace &qspace) const
   { return NULL; /* FIXME */ }

   virtual const mfem::Operator *GetGradientOperator(
      const mfem::QuadratureSpace &qspace) const
   { return NULL; /* FIXME */ }

   const ::occa::array<int> GetLocalToGlobalMap() const
   { return localToGlobalMap; }

   /// L-vector to E-vector
   void GlobalToLocal(const Vector &globalVec, Vector &localVec) const
   {
      globalToLocalKernel(globalDofs,
                          localDofs * fes->GetNE(),
                          globalToLocalOffsets,
                          globalToLocalIndices,
                          globalVec.OccaMem(), localVec.OccaMem());
   }

   /// E-vector to L-vector, transpose of GlobalToLocal
   void LocalToGlobal(const Vector &localVec, Vector &globalVec) const
   {
      localToGlobalKernel(globalDofs,
                          localDofs * fes->GetNE(),
                          globalToLocalOffsets,
                          globalToLocalIndices,
                          localVec.OccaMem(), globalVec.OccaMem());
   }
};


#ifdef MFEM_USE_MPI

/// OCCA version of mfem::ConformingProlongationOperator
class OccaConformingProlongation : public Operator
{
protected:
   // size(shr_buf)=size(shr_ltdof)
   // size(ext_buf)=size(ext_ldof)
   Array shr_ltdof, ext_ldof;
   mutable Array shr_buf, ext_buf;
   mutable char *host_shr_buf, *host_ext_buf;
   // Offsets into {shr,ext}_buf; size is num. neighbors, i.e.
   // gc.GetGroupTopology().GetNumNeighbors():
   int *shr_buf_offsets, *ext_buf_offsets;

   ::occa::memory ltdof_ldof; // shared with the restriction operator

   ::occa::memory unq_ltdof; // enumeration of the unique ltdofs in shr_ltdof
   ::occa::memory unq_shr_i, unq_shr_j;

   ::occa::kernel ExtractSubVector, SetSubVector, AddSubVector;

   MPI_Request *requests;

   const GroupCommunicator &gc;

   // Kernel: copy ltdofs from 'src' to 'shr_buf' - prepare for send.
   //         shr_buf[i] = src[shr_ltdof[i]]
   void BcastBeginCopy(const ::occa::memory &src, std::size_t item_size) const;
   // Kernel: copy ltdofs from 'src' to ldofs in 'dst'.
   //         dst[ltdof_ldof[i]] = src[i]
   void BcastLocalCopy(const ::occa::memory &src, ::occa::memory &dst,
                       std::size_t item_size) const;
   // Kernel: copy ext. dofs from 'ext_buf' to 'dst' - after recv.
   //         dst[ext_ldof[i]] = ext_buf[i]
   void BcastEndCopy(::occa::memory &dst, std::size_t item_size) const;

   // Kernel: copy ext. dofs from 'src' to 'ext_buf' - prepare for send.
   //         ext_buf[i] = src[ext_ldof[i]]
   void ReduceBeginCopy(const ::occa::memory &src, std::size_t item_size) const;
   // Kernel: copy owned ldofs from 'src' to ltdofs in 'dst'.
   //         dst[i] = src[ltdof_ldof[i]]
   void ReduceLocalCopy(const ::occa::memory &src, ::occa::memory &dst,
                       std::size_t item_size) const;
   // Kernel: assemble dofs from 'shr_buf' into to 'dst' - after recv.
   //         dst[shr_ltdof[i]] += shr_buf[i]
   void ReduceEndAssemble(::occa::memory &dst, std::size_t item_size) const;

public:
   OccaConformingProlongation(const FiniteElementSpace &ofes,
                              const mfem::ParFiniteElementSpace &pfes,
                              ::occa::memory ltdof_ldof_);

   virtual ~OccaConformingProlongation();

   // overrides
   virtual void Mult_(const Vector &x, Vector &y) const;
   virtual void MultTranspose_(const Vector &x, Vector &y) const;
};

#endif // MFEM_USE_MPI

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#endif // MFEM_BACKENDS_OCCA_FE_SPACE_HPP
