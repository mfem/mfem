// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "neo_hookean.hpp"
#include "linalg/tensor.hpp"

namespace mfem
{
namespace
{

const IntegrationRule &MaterialRule(const FiniteElementSpace &fes)
{
   const FiniteElement &fe = *fes.GetTypicalFE();
   MFEM_VERIFY((fes.GetMesh()->Dimension() == 2 ||
                fes.GetMesh()->Dimension() == 3) &&
               fes.GetVDim() == fes.GetMesh()->Dimension() &&
               !fes.IsVariableOrder() &&
               fes.GetMesh()->GetNumGeometries(fes.GetMesh()->Dimension()) <= 1,
               "Device Neo-Hookean evaluation requires a uniform 2D or 3D "
               "vector H1 space.");
   // Match HyperelasticNLFIntegrator's integration rule.
   return IntRules.Get(fe.GetGeomType(), 2*fe.GetOrder() + 3);
}

template<int dim>
void SetupGeometry(const int nd, const int nq, const int ne,
                   const DofToQuad &maps, const GeometricFactors &geometry,
                   const IntegrationRule &ir, Vector &gradients, Vector &weights)
{
   const auto J = Reshape(geometry.J.Read(), nq, dim, dim, ne);
   const auto G = Reshape(maps.G.Read(), nq, dim, nd);
   const real_t *W = ir.GetWeights().Read();
   auto PG = Reshape(gradients.Write(), nd, dim, nq, ne);
   auto weights_d = Reshape(weights.Write(), nq, ne);
   mfem::forall(nq*ne, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int q = idx % nq, e = idx / nq;
      const auto reference = future::make_tensor<dim, dim>(
      [&](int i, int j) { return J(q, i, j, e); });
      const auto inverse = future::inv(reference);
      weights_d(q, e) = W[q]*future::det(reference);
      for (int i = 0; i < nd; i++)
      {
         for (int c = 0; c < dim; c++)
         {
            real_t value = 0.0;
            for (int d = 0; d < dim; d++)
            {
               value += G(q, d, i)*inverse(d, c);
            }
            PG(i, c, q, e) = value;
         }
      }
   });
}

template<int dim>
void EvaluateMaterial(const int nd, const int nq, const int ne,
                      const Vector &state, const Vector &gradients,
                      const Vector &weights, const Vector &shear,
                      const Vector &bulk, Vector &material_data,
                      Vector &projected_gradients, Vector &energy_data,
                      Vector &determinants, Vector &element_gradient,
                      Vector &element_hessian, const bool compute_gradient,
                      const bool compute_hessian)
{
   const auto X = Reshape(state.Read(), nd, dim, ne);
   const auto PG = Reshape(gradients.Read(), nd, dim, nq, ne);
   const auto W = Reshape(weights.Read(), nq, ne);
   const auto MU = Reshape(shear.Read(), nq, ne);
   const auto K = Reshape(bulk.Read(), nq, ne);
   auto D = Reshape(material_data.Write(), 3, nq, ne);
   auto CG = Reshape(projected_gradients.Write(), nd, dim, 2, nq, ne);
   auto EQ = Reshape(energy_data.Write(), nq, ne);
   real_t *determinants_d = determinants.Write();
   const real_t inf = infinity();
   mfem::forall(nq*ne, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int q = idx % nq, e = idx / nq;
      const auto F = future::make_tensor<dim, dim>([&](int i, int j)
      {
         real_t value = 0.0;
         for (int k = 0; k < nd; k++) { value += X(k, i, e)*PG(k, j, q, e); }
         return value;
      });
      const real_t determinant = future::det(F);
      determinants_d[idx] = determinant > 0.0 && determinant < inf
                            ? determinant : -inf;
      const auto inverse = future::inv(F);
      real_t I1 = 0.0;
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++) { I1 += F(i, j)*F(i, j); }
      }
      const real_t a = MU(q, e)*pow(determinant, -2.0/dim);
      const real_t bc = a*I1/dim;
      // Coefficients and contractions match NeoHookeanModel::AssembleH.
      D(0, q, e) = W(q, e)*a;
      D(1, q, e) = W(q, e)*(bc - K(q, e)*determinant*(determinant - 1.0));
      D(2, q, e) = W(q, e)*(2.0*bc/dim +
                            K(q, e)*determinant*(2.0*determinant - 1.0));
      EQ(q, e) = 0.5*W(q, e)*(a*I1 - MU(q, e)*dim +
                              K(q, e)*(determinant - 1.0)*(determinant - 1.0));
      for (int i = 0; i < nd; i++)
      {
         for (int c = 0; c < dim; c++)
         {
            real_t C = 0.0, G = 0.0;
            for (int d = 0; d < dim; d++)
            {
               C += PG(i, d, q, e)*F(c, d);
               G += PG(i, d, q, e)*inverse(d, c);
            }
            CG(i, c, 0, q, e) = C;
            CG(i, c, 1, q, e) = G;
         }
      }
   });

   const auto data = Reshape(material_data.Read(), 3, nq, ne);
   const auto contractions = Reshape(projected_gradients.Read(), nd, dim, 2, nq,
                                     ne);
   if (compute_gradient)
   {
      auto EG = Reshape(element_gradient.Write(), nd, dim, ne);
      mfem::forall(nd*dim*ne, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int i = idx % nd, c = (idx / nd) % dim, e = idx / nd / dim;
         real_t value = 0.0;
         for (int q = 0; q < nq; q++)
         {
            value += data(0, q, e)*contractions(i, c, 0, q, e) -
                     data(1, q, e)*contractions(i, c, 1, q, e);
         }
         EG(i, c, e) = value;
      });
   }

   if (!compute_hessian) { return; }
   const int vd = nd*dim;
   auto H = Reshape(element_hessian.Write(), vd, vd, ne);
   const int threads = nd < 16 ? nd : 16;
   mfem::forall_2D(ne, threads, threads, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(i, x, nd)
      {
         MFEM_FOREACH_THREAD(k, y, nd)
         {
            for (int j = 0; j < dim; j++)
            {
               for (int l = 0; l < dim; l++)
               {
                  real_t value = 0.0;
                  for (int q = 0; q < nq; q++)
                  {
                     real_t dot = 0.0;
                     for (int d = 0; d < dim; d++)
                     {
                        dot += PG(i, d, q, e)*PG(k, d, q, e);
                     }
                     const real_t Cij = contractions(i, j, 0, q, e);
                     const real_t Ckl = contractions(k, l, 0, q, e);
                     const real_t Gij = contractions(i, j, 1, q, e);
                     const real_t Gkl = contractions(k, l, 1, q, e);
                     value += data(0, q, e)*((j == l ? dot : 0.0) -
                                             (2.0/dim)*(Cij*Gkl + Gij*Ckl)) +
                              data(1, q, e)*contractions(i, l, 1, q, e)*
                              contractions(k, j, 1, q, e) +
                              data(2, q, e)*Gij*Gkl;
                  }
                  H(k + l*nd, i + j*nd, e) = value;
               }
            }
         }
      }
   });
}

}

DeviceNeoHookean::DeviceNeoHookean(ParFiniteElementSpace &fes_,
                                   Coefficient &shear_, Coefficient &bulk_)
   : fes(fes_), quadrature(*fes.GetMesh(), MaterialRule(fes)),
     shear(shear_, quadrature, CoefficientStorage::FULL),
     bulk(bulk_, quadrature, CoefficientStorage::FULL),
     restriction(dynamic_cast<const ElementRestriction &>(
                    *fes.GetElementRestriction(GetEVectorOrdering(fes)))),
     maps(fes.GetTypicalFE()->GetDofToQuad(MaterialRule(fes),
                                           GetEVectorOrdering(fes) == ElementDofOrdering::NATIVE
                                           ? DofToQuad::FULL : DofToQuad::LEXICOGRAPHIC_FULL)),
     geometry(*fes.GetMesh()->GetGeometricFactors(MaterialRule(fes),
                                                  GeometricFactors::JACOBIANS)),
     local_hessian(fes.GetVSize())
{
   const int dim = fes.GetVDim(), nd = fes.GetTypicalFE()->GetDof();
   const int nq = MaterialRule(fes).GetNPoints(), ne = fes.GetNE();
   physical_gradients.SetSize(nd*dim*nq*ne);
   weights.SetSize(nq*ne);
   local_state.SetSize(fes.GetVSize());
   element_state.SetSize(restriction.Height());
   local_gradient.SetSize(fes.GetVSize());
   element_gradient.SetSize(restriction.Height());
   material_data.SetSize(3*nq*ne);
   projected_gradients.SetSize(nd*dim*2*nq*ne);
   energy_data.SetSize(nq*ne);
   determinants.SetSize(nq*ne);
   element_hessian.SetSize(nd*dim*nd*dim*ne);
   for (Vector *v :
        {
           &local_state, &element_state, &local_gradient,
           &element_gradient, &energy_data, &determinants
        })
   {
      v->UseDevice(true);
   }
   if (dim == 2)
   {
      SetupGeometry<2>(nd, nq, ne, maps, geometry, MaterialRule(fes),
                       physical_gradients, weights);
   }
   else
   {
      SetupGeometry<3>(nd, nq, ne, maps, geometry, MaterialRule(fes),
                       physical_gradients, weights);
   }
}

void DeviceNeoHookean::Evaluate(const Vector &coordinates,
                                const Array<int> &ess_tdofs, real_t &energy,
                                Vector *gradient, OperatorHandle *hessian)
{
   const int dim = fes.GetVDim(), nd = fes.GetTypicalFE()->GetDof();
   const int nq = MaterialRule(fes).GetNPoints(), ne = fes.GetNE();
   fes.GetProlongationMatrix()->Mult(coordinates, local_state);
   restriction.Mult(local_state, element_state);
   auto kernel = dim == 2 ? EvaluateMaterial<2> : EvaluateMaterial<3>;
   kernel(nd, nq, ne, element_state, physical_gradients, weights, shear, bulk,
          material_data, projected_gradients, energy_data, determinants,
          element_gradient, element_hessian, gradient != nullptr,
          hessian != nullptr);
   const real_t local_min = ne ? determinants.Min() : infinity();
   real_t global_min;
   MPI_Allreduce(&local_min, &global_min, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, fes.GetComm());
   MFEM_VERIFY(IsFinite(global_min) && global_min > 0.0,
               "Nonpositive deformation determinant in Neo-Hookean evaluation.");
   const real_t local_energy = energy_data.Sum();
   MPI_Allreduce(&local_energy, &energy, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, fes.GetComm());
   if (gradient)
   {
      restriction.MultTranspose(element_gradient, local_gradient);
      gradient->SetSize(fes.GetTrueVSize());
      gradient->UseDevice(true);
      fes.GetProlongationMatrix()->MultTranspose(local_gradient, *gradient);
      gradient->SetSubVector(ess_tdofs, 0.0);
   }

   if (!hessian) { return; }
   if (!matrix_initialized)
   {
      restriction.FillSparseMatrix(element_hessian, local_hessian);
      matrix_initialized = true;
   }
   else
   {
      restriction.FillJAndData(element_hessian, local_hessian);
   }
   hessian->Clear();
   ParBilinearForm assembler(&fes);
   assembler.ParallelRAP(local_hessian, *hessian);
   if (IsIdentityProlongation(fes.GetProlongationMatrix()))
   {
      hessian->Reset(new HypreParMatrix(*hessian->As<HypreParMatrix>()));
   }
   hessian->As<HypreParMatrix>()->EliminateBC(ess_tdofs, Operator::DIAG_ONE);
}

}
