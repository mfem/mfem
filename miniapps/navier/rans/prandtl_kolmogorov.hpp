// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NAVIER_RANS_PRANDTL_KOLMOGOROV_HPP
#define MFEM_NAVIER_RANS_PRANDTL_KOLMOGOROV_HPP

#include "rans_model.hpp"

namespace mfem
{
namespace navier
{

class PrandtlKolmogorov : public RANSModel
{
public:
   PrandtlKolmogorov(ParFiniteElementSpace &kfes,
                     VectorCoefficient &vel_coeff,
                     Coefficient &kv_coeff,
                     Coefficient &f_coeff,
                     Coefficient &turb_length_scale,
                     Coefficient &k_bdrcoeff,
                     const double mu_calibration_const,
                     Array<int> ess_attr);

   void Mult(const Vector &x, Vector &y) const override;

   void SetTime(double t) override;

   void Apply(const Vector &k, Vector &y) const;

   void ApplyEssentialBC(const double t, Vector& y);

   static int PostProcessCallback(realtype t, N_Vector zpred, void *user_data);

   const ParFiniteElementSpace &kfes;
   const IntegrationRule *ir = nullptr; /// not owned
   IntegrationRules *gll_rules = nullptr;
   const Operator *Pk = nullptr;
   const Operator *Rk = nullptr;
   const Operator *Pu = nullptr;
   const Operator *Ru = nullptr;
   const QuadratureInterpolator* qi_vel = nullptr;
   const QuadratureInterpolator* qi_kv = nullptr;
   const QuadratureInterpolator* qi_tls = nullptr;
   const DofToQuad *maps = nullptr;
   const GeometricFactors *geom = nullptr;
   const int dim, ne = 0;
   const double mu_calibration_const;
   Vector kv_q, u_q, kv_e, u_e, f_q, tls_e, tls_q, b, k_bdr_values;
   mutable Vector k_l, k_e, y_l, y_e, z, z1, z2, x_ess;
   ParGridFunction k_gf;
   VectorCoefficient &vel_coeff;
   Coefficient &kv_coeff;
   Coefficient &f_coeff;
   Coefficient &k_bdrcoeff;
   Coefficient &tls_coeff;
   mutable Array<int> ess_attr, ess_tdof_list;

   ParBilinearForm *Mform = nullptr, *Kform = nullptr;
   ParLinearForm *bform = nullptr;
   CGSolver *M_inv = nullptr;
   OperatorHandle M, K;
};

} // namespace navier

} // namespace mfem

#endif