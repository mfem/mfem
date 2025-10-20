// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "partial_qfunc.hpp"
#include "partial_qspace.hpp"

#include "../general/forall.hpp"

namespace mfem
{

PartialQuadratureFunction &
PartialQuadratureFunction::operator=(const QuadratureFunction &qf)
{
   MFEM_ASSERT(qf.GetVDim() == vdim, "Vector dimensions don't match");
   MFEM_ASSERT(qf.GetSpace()->GetSize() >= part_quad_space->GetSize(), 
               "QuadratureSpace sizes aren't of equivalent sizes");
   
   if (qf.GetSpace()->GetSize() == part_quad_space->GetSize())
   {
      Vector::operator=(qf);
      return *this;
   }
   else
   {
      // Very basic check to see if the two spaces are roughly equivalent...
      // We would need to do a much more thorough job if we wanted to be 100% certain
      MFEM_ASSERT(qf.GetSpace()->GetMesh()->GetNE() == part_quad_space->GetMesh()->GetNE(),
                  "QSpaces have meshes with different # of elements");
      MFEM_ASSERT(qf.GetSpace()->GetOrder() == part_quad_space->GetOrder(),
                  "QSpaces don't have the same integration order");
      
      // We now need to copy all of the relevant data over that we'll need
      auto l2g = part_quad_space->local2global.Read();
      auto loc_offsets = part_quad_space->offsets.Read();
      auto global_offsets = (part_quad_space->global_offsets.Size() > 1) ?
                            part_quad_space->global_offsets.Read() : loc_offsets;
      auto qf_data = qf.Read();
      auto loc_data = this->ReadWrite();

      auto NE = part_quad_space->GetNE();
      // For now this is fine. Later on we might want to leverage like RAJA views
      // and the IndexLayout to make things even more performant.
      // Additionally, we could look at using 2D kernels if need be but probably
      // overkill for now...
      const auto vdim_ = vdim;
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int ie)
      {
         const int global_idx = l2g[ie];
         const int global_offset_idx = global_offsets[global_idx];
         const int local_offset_idx = loc_offsets[ie];
         const int nqpts = loc_offsets[ie + 1] - local_offset_idx;
         const int npts = nqpts * vdim_;
         for (int jv = 0; jv < npts; jv++)
         {
            loc_data[local_offset_idx * vdim_ + jv] = 
               qf_data[global_offset_idx * vdim_ + jv];
         }
      });
   }
   return *this;
}

void PartialQuadratureFunction::FillQuadratureFunction(QuadratureFunction &qf,
                                                       const bool fill)
{
   if (qf.GetSpace()->GetSize() == part_quad_space->GetSize())
   {
      qf = *this;
   }
   else
   {
      // Very basic check to see if the two spaces are roughly equivalent...
      // We would need to do a much more thorough job if we wanted to be 100% certain
      MFEM_ASSERT(qf.GetVDim() == vdim, "Vector dimensions don't match");
      MFEM_ASSERT(qf.GetSpace()->GetMesh()->GetNE() == part_quad_space->GetMesh()->GetNE(),
                  "QSpaces have meshes with different # of elements");
      MFEM_ASSERT(qf.GetSpace()->GetOrder() == part_quad_space->GetOrder(),
                  "QSpaces don't have the same integration order");
      
      // We now need to copy all of the relevant data over that we'll need
      auto l2g = part_quad_space->local2global.Read();
      auto offsets = part_quad_space->offsets.Read();
      auto global_offsets = (part_quad_space->global_offsets.Size() > 1) ?
                            part_quad_space->global_offsets.Read() : offsets;
      auto qf_data = qf.ReadWrite();
      auto loc_data = this->Read();
      
      // First set all values to default
      if (fill)
      {
         qf = default_value;
      }
      
      auto NE = part_quad_space->GetNE();
      // Then copy our partial values to their proper places
      const auto vdim_ = vdim;
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int ie)
      {
         const int global_idx = l2g[ie];
         const int global_offset_idx = global_offsets[global_idx];
         const int local_offset_idx = offsets[ie];
         const int nqpts = offsets[ie + 1] - local_offset_idx;
         const int npts = nqpts * vdim_;
         for (int jv = 0; jv < npts; jv++)
         {
            qf_data[global_offset_idx * vdim_ + jv] = 
               loc_data[local_offset_idx * vdim_ + jv];
         }
      });
   }
}

} // namespace mfem