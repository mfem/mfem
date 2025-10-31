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

#ifndef MFEM_ESTIMATORS_HDG
#define MFEM_ESTIMATORS_HDG

#include "../estimators.hpp"

namespace mfem
{

class HDGErrorEstimator : public AnisotropicErrorEstimator
{
public:
   enum class Type
   {
      Residual,
      Energy,
   };

private:
   BilinearFormIntegrator &bfi;
   const GridFunction &sol_tr, &sol_p;
   Type type;

   long current_sequence{-1};
   Vector error_estimates;
   real_t total_error{};
   bool anisotropic{};
   Array<int> aniso_flags;

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = sol_tr.FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

public:
   HDGErrorEstimator(BilinearFormIntegrator &integ, const GridFunction &solr,
                     const GridFunction &solp, Type type_ = Type::Energy)
      : bfi(integ), sol_tr(solr), sol_p(solp), type(type_) { }

   /// Enable/disable anisotropic estimates.
   /** To enable this option, the BilinearFormIntegrator must support the
       'd_energy' parameter in its ComputeHDGFaceEnergy() method. */
   void SetAnisotropic(bool aniso = true) { anisotropic = aniso; }

   /// Return the total error from the last error estimate.
   real_t GetTotalError() const override { return total_error; }

   /// Get a Vector with all element errors.
   const Vector &GetLocalErrors() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /// Get an Array<int> with anisotropic flags for all mesh elements.
   /** Return an empty array when anisotropic estimates are not available or
       enabled. */
   const Array<int> &GetAnisotropicFlags()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return aniso_flags;
   }

   /// Reset the error estimator.
   void Reset() override { current_sequence = -1; }
};

} // namespace mfem

#endif // MFEM_ESTIMATORS_HDG
