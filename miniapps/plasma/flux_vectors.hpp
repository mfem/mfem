// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PLASMA_FLUX_VECTORS
#define MFEM_PLASMA_FLUX_VECTORS

#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

using common::RT_ParFESpace;

namespace plasma
{

class ParFluxVectors
{
private:
   bool df_active_;
   bool af_active_;
   bool m_rt_assm_;
   bool pa_;

   Coefficient       * dCoef_;
   MatrixCoefficient * DCoef_;
   VectorCoefficient * VCoef_;

   RT_ParFESpace fes_rt_;

   ParLinearForm DGradU_lf_;
   ParLinearForm VProdU_lf_;

   Array<int> ess_tdof_;
   ParBilinearForm m_rt_;
   OperatorPtr     M_rt_;
   Vector RHS_;
   Vector X_;

   ParGridFunction df_gf_;
   ParGridFunction af_gf_;

   // void Activate();

public:
   ParFluxVectors(ParMesh& pmesh, int order, bool pa);

   void Update();
   void Assemble();

   void SetDiffusionCoef(Coefficient &D);
   void SetDiffusionCoef(MatrixCoefficient &D);
   void SetAdvectionCoef(VectorCoefficient &V);

   void ComputeDiffusiveFlux(const ParGridFunction& u);
   void ComputeAdvectiveFlux(const ParGridFunction& u);

   // const ParGridFunction & GetDiffusiveFlux() const { return df_gf_; }
   // const ParGridFunction & GetAdvectiveFlux() const { return af_gf_; }

   ParGridFunction & GetDiffusiveFlux();
   ParGridFunction & GetAdvectiveFlux();

};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PLASMA_FLUX_VECTORS
