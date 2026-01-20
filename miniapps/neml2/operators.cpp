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

#include "operators.hpp"
#include <torch/torch.h>

namespace mfem
{

NEML2StressDivergenceIntegrator::NEML2StressDivergenceIntegrator(std::shared_ptr<neml2::Model> cmodel)
    : _constit_op(cmodel)
{
}

void NEML2StressDivergenceIntegrator::AssemblePA(const FiniteElementSpace &fe_space)
{
   StressDivergenceIntegrator<NonlinearFormIntegrator>::AssemblePA(fe_space);
   if (this->vdim < 2)
   {
      mfem_error("NEML2StressDivergenceIntegrator is only meant to be used in "
                 "multiple dimensions");
   }

   _ordering = fe_space.GetOrdering();
   auto *const mesh = fe_space.GetMesh();
   const auto *const nodes = mesh->GetNodes();

   _q_space_symr2 = std::make_unique<UniformParameterSpace>(*mesh,
                                                            *this->IntRule, 6);
   _strain = std::make_unique<ParameterFunction>(*_q_space_symr2);
   _stress = std::make_unique<ParameterFunction>(*_q_space_symr2);

   // Put strain and stress storage on device
   _strain->UseDevice(true);
   _stress->UseDevice(true);
}

template <int vdim>
void NEML2StressDivergenceIntegrator::ComputeStrainImpl(const Vector &x,
                                                        ParameterFunction &strain)
                                                                                const
{
   using future::tensor;

   constexpr int d = vdim;

   // Assuming all elements are the same
   const QuadratureInterpolator *E_To_Q_Map = this->fespace->GetQuadratureInterpolator(
                                                                                   *this->IntRule);
   E_To_Q_Map->SetOutputLayout(_ordering == Ordering::byNODES ? QVectorLayout::byNODES
                                                              : QVectorLayout::byVDIM);
   // interpolate physical derivatives to quadrature points.
   E_To_Q_Map->PhysDerivatives(x, *this->q_vec);

   const int numPoints = this->IntRule->GetNPoints();
   const int numEls = this->fespace->GetNE();
   const auto Q = Reshape(this->q_vec->Read(), numPoints, d, d, numEls);
   // device strain
   auto dStrain = Reshape(strain.Write(), 6, numPoints, numEls);
   mfem::forall_2D(numEls, numPoints, 1,
                   [=] MFEM_HOST_DEVICE(int e)
                   {
                      // for(int p = 0; p < numPoints, )
                      MFEM_FOREACH_THREAD(p, x, numPoints)
                      {
                         tensor<real_t, d, d> dudx;
                         // load grad(x) into dudx
                         for (int j = 0; j < d; j++)
                         {
                            for (int i = 0; i < d; i++)
                            {
                               dudx(i, j) = Q(p, i, j, e);
                            }
                         }
                         const auto epsilon = real_t(0.5) *
                                              (dudx + transpose(dudx));
                         // NEML2 uses Mandel notation for symmetric 2nd order tensors.
                         constexpr real_t sqrt2 = 1.4142135623730951_r;
                         dStrain(0, p, e) = epsilon(0, 0);
                         dStrain(1, p, e) = epsilon(1, 1);
                         dStrain(5, p, e) = sqrt2 * epsilon(0, 1);
                         if constexpr (d == 2) // NEML2 always expects 3D
                         {
                            dStrain(2, p, e) = 0;
                            dStrain(3, p, e) = 0;
                            dStrain(4, p, e) = 0;
                         }
                         else
                         {
                            dStrain(2, p, e) = epsilon(2, 2);
                            dStrain(3, p, e) = sqrt2 * epsilon(1, 2);
                            dStrain(4, p, e) = sqrt2 * epsilon(0, 2);
                         }
                      }
                   });
}

void NEML2StressDivergenceIntegrator::ComputeStrain(const Vector &X,
                                                    ParameterFunction &strain) const
{
   if (this->vdim == 2)
   {
      this->ComputeStrainImpl<2>(X, *_strain);
   }
   else if (this->vdim == 3)
   {
      this->ComputeStrainImpl<3>(X, *_strain);
   }
}

template <int vdim>
void NEML2StressDivergenceIntegrator::ComputeRImpl(const ParameterFunction &stress,
                                                   Vector &R) const
{
   using future::det;
   using future::inv;
   using future::make_tensor;
   using future::tensor;

   constexpr int d = vdim;

   const int numPoints = this->IntRule->GetNPoints();
   const int numEls = this->fespace->GetNE();
   auto Q = Reshape(this->q_vec->Write(), numPoints, d, d, numEls);
   // device stress
   const auto dStress = Reshape(stress.Read(), 6, numPoints, numEls);
   const auto J = Reshape(this->geom->J.Read(), numPoints, d, d, numEls);

   const real_t *ipWeights = this->IntRule->GetWeights().Read();
   mfem::forall_2D(numEls, numPoints, 1,
                   [=] MFEM_HOST_DEVICE(int e)
                   {
                      // for(int p = 0; p < numPoints, )
                      MFEM_FOREACH_THREAD(p, x, numPoints)
                      {
                         // clang-format off
                           const auto invJ = inv(make_tensor<d, d>([&](int i, int j)
                                                                  { return J(p, i, j, e); }));
                         // clang-format on
                         constexpr real_t sqrt2 = 1.4142135623730951_r;
                         tensor<real_t, d, d> stress_tensor;
                         stress_tensor(0, 0) = dStress(0, p, e);
                         stress_tensor(1, 1) = dStress(1, p, e);
                         stress_tensor(0, 1) = dStress(5, p, e) / sqrt2;
                         stress_tensor(1, 0) = stress_tensor(0, 1);
                         if constexpr (d == 3)
                         {
                            stress_tensor(2, 2) = dStress(2, p, e);
                            stress_tensor(1, 2) = dStress(3, p, e) / sqrt2;
                            stress_tensor(0, 2) = dStress(4, p, e) / sqrt2;
                            stress_tensor(2, 1) = stress_tensor(1, 2);
                            stress_tensor(2, 0) = stress_tensor(0, 2);
                         }
                         const auto JxW = ipWeights[p] / det(invJ);
                         const auto sigma_ref_weighted = stress_tensor *
                                                         transpose(invJ) * JxW;
                         for (int m = 0; m < d; ++m)
                         {
                            for (int q = 0; q < d; ++q)
                            {
                               Q(p, m, q, e) = sigma_ref_weighted(q, m);
                            }
                         }
                      }
                   });

   // Reduce quadrature function to an E-Vector
   const auto QRead = Reshape(this->q_vec->Read(), numPoints, d, d, numEls);
   const auto G = Reshape(this->maps->G.Read(), numPoints, d, this->ndofs);
   auto rDev = Reshape(R.ReadWrite(), this->ndofs, d, numEls);
   mfem::forall_2D(numEls, d, this->ndofs,
                   [=] MFEM_HOST_DEVICE(int e)
                   {
                      MFEM_FOREACH_THREAD(i, y, this->ndofs)
                      {
                         MFEM_FOREACH_THREAD(q, x, d)
                         {
                            real_t sum = 0.;
                            for (int m = 0; m < d; m++)
                            {
                               for (int p = 0; p < numPoints; p++)
                               {
                                  sum += QRead(p, m, q, e) * G(p, m, i);
                               }
                            }
                            rDev(i, q, e) += sum;
                         }
                      }
                   });
}

void NEML2StressDivergenceIntegrator::ComputeR(const ParameterFunction &stress,
                                               Vector &R) const
{
   if (this->vdim == 2)
   {
      this->ComputeRImpl<2>(stress, R);
   }
   else if (this->vdim == 3)
   {
      this->ComputeRImpl<3>(stress, R);
   }
}

void NEML2StressDivergenceIntegrator::AddMultPA(const Vector &X,
                                                Vector &R) const
{
   // displacement -> strain
   this->ComputeStrain(X, *_strain);

   // strain -> stress via NEML2
   _constit_op.Mult(*_strain, *_stress);

   // stress -> residuals
   this->ComputeR(*_stress, R);
}

void NEML2StressDivergenceIntegrator::AssembleGradPA(const Vector &X,
                                                     const FiniteElementSpace &fes)
{
   // Evaluate the tangent at the current state
   // displacement -> strain
   this->ComputeStrain(X, *_strain);
   if (!_tangent.has_value())
   {
      _tangent.emplace();
   }
   _constit_op.Tangent(*_strain, _tangent.value());
}

void NEML2StressDivergenceIntegrator::AddMultGradPA(const Vector &dX,
                                                    Vector &dR) const
{
   // dε = strain_op(dX)
   this->ComputeStrain(dX, *_strain);

   // dσ = C(ε) : dε
   _constit_op.ApplyTangent(_tangent.value(), *_strain, *_stress);

   // dR = stressdiv_op(dσ)
   this->ComputeR(*_stress, dR);
}

template void
NEML2StressDivergenceIntegrator::ComputeStrainImpl<2>(const Vector &x,
                                                      ParameterFunction &strain)
                                                                                const;
template void
NEML2StressDivergenceIntegrator::ComputeStrainImpl<3>(const Vector &x,
                                                      ParameterFunction &strain)
                                                                                const;
template void
NEML2StressDivergenceIntegrator::ComputeRImpl<2>(const ParameterFunction &stress,
                                                 Vector &R) const;
template void
NEML2StressDivergenceIntegrator::ComputeRImpl<3>(const ParameterFunction &stress,
                                                 Vector &R) const;
} // namespace mfem
