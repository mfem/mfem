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

#ifndef MFEM_BACKENDS_RAJA_PFESPACE_HPP
#define MFEM_BACKENDS_RAJA_PFESPACE_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{

   // ***************************************************************************
   // * RajaParFiniteElementSpace
   //  **************************************************************************
   class RajaParFiniteElementSpace : public mfem::PParFiniteElementSpace,
                                     public mfem::ParFiniteElementSpace {
   protected:
      Layout e_layout;
      int globalDofs, localDofs;
      raja::array<int> offsets;
      raja::array<int> indices, *reorderIndices;
      raja::array<int> map;
      mfem::Operator *restrictionOp, *prolongationOp;
   public:
      RajaParFiniteElementSpace(const Engine &e, mfem::ParFiniteElementSpace&);
      ~RajaParFiniteElementSpace();
      // ***********************************************************************
      const Engine &RajaEngine() const
      { return *static_cast<const Engine *>(engine.Get()); }
      
      raja::device GetDevice(int idx = 0) const
      { return RajaEngine().GetDevice(idx); }
 
      Layout &RajaVLayout() const
      { return *pfes->GetVLayout().As<Layout>(); }

      Layout &RajaTrueVLayout() const
      { return *pfes->GetTrueVLayout().As<Layout>(); }

      Layout &RajaEVLayout() { return e_layout; }

#ifdef MFEM_USE_MPI
      bool isDistributed() const { return (RajaEngine().GetComm() != MPI_COMM_NULL); }
#else
      bool isDistributed() const { return false; }
#endif
      // *************************************************************************
      bool hasTensorBasis() const;
      int GetLocalDofs() const { return localDofs; }
      const mfem::Operator* GetRestrictionOperator() { return restrictionOp; }
      const mfem::Operator* GetProlongationOperator() { return prolongationOp; }
      const raja::array<int>& GetLocalToGlobalMap() const { return map; }
      // *************************************************************************
      void GlobalToLocal(const raja::RajaVector&, raja::RajaVector&) const;
      void LocalToGlobal(const raja::RajaVector&, raja::RajaVector&) const;
   };
   
} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_PFESPACE_HPP
