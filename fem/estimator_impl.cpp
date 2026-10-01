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

#include "estimator_impl.hpp"
#include "complex_fem.hpp"

namespace mfem
{

real_t MaxwellResidualEstimator::EpsilonMin(ElementTransformation &trans,
                                             const IntegrationPoint &ip) const
{
   if (epsilon) { return epsilon->Eval(trans, ip); }

   DenseMatrix eps_tensor;
   epsilon_matrix->Eval(eps_tensor, trans, ip);
   MFEM_VERIFY(eps_tensor.Height() == 3 && eps_tensor.Width() == 3,
               "epsilon matrix must be 3 by 3.");
   const real_t symmetry_tol = 1e-12 * std::max(1.0, eps_tensor.MaxMaxNorm());
   for (int i = 0; i < 3; i++)
   {
      for (int j = i + 1; j < 3; j++)
      {
         MFEM_VERIFY(std::abs(eps_tensor(i, j) - eps_tensor(j, i)) <= symmetry_tol,
                     "epsilon matrix must be symmetric positive definite.");
      }
   }
   real_t eigenvalues[3], eigenvectors[9];
   eps_tensor.CalcEigenvalues(eigenvalues, eigenvectors);
   return std::min(eigenvalues[0], std::min(eigenvalues[1], eigenvalues[2]));
}

void MaxwellResidualEstimator::ComputeEstimates()
{
   FiniteElementSpace *fes = solution.FESpace();
   Mesh *mesh = fes->GetMesh();
   MFEM_VERIFY(mesh->Dimension() == 3 && mesh->SpaceDimension() == 3,
               "MaxwellResidualEstimator requires a three-dimensional mesh.");
   MFEM_VERIFY(source.FESpace()->GetMesh() == mesh,
               "solution and source must share a mesh.");

   // The residual fields must be discontinuous: their interface jumps are
   // part of the indicator. Vector-valued L2 spaces also provide elementwise
   // strong curl and divergence through GridFunction.
   L2_FECollection h_fec(std::max(0, order - 1), 3);
   L2_FECollection d_fec(order, 3);
   FiniteElementSpace h_fes(mesh, &h_fec, 3, Ordering::byVDIM);
   FiniteElementSpace d_fes(mesh, &d_fec, 3, Ordering::byVDIM);
   GridFunction h(&h_fes), d(&d_fes);
   CurlGridFunctionCoefficient curl_e(&solution);
   ScalarVectorProductCoefficient h_coef(mu_inv, curl_e);
   VectorGridFunctionCoefficient e(&solution);
   h.ProjectCoefficient(h_coef);
   if (epsilon)
   {
      ScalarVectorProductCoefficient d_coef(*epsilon, e);
      d.ProjectCoefficient(d_coef);
   }
   else
   {
      MatrixVectorProductCoefficient d_coef(*epsilon_matrix, e);
      d.ProjectCoefficient(d_coef);
   }

   const int ne = mesh->GetNE();
   error_estimates.SetSize(ne);
   error_estimates = 0.0;
   Vector f(3), curl_h(3), e_val(3), eps_e(3), h1(3), h2(3), d1(3), d2(3),
          normal(3);

   for (int el = 0; el < ne; el++)
   {
      ElementTransformation *tr = mesh->GetElementTransformation(el);
      const FiniteElement *fe = fes->GetFE(el);
      const int qorder = std::max(2 * fe->GetOrder() + 2, 2);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), qorder);
      const real_t h_el = mesh->GetElementSize(el, 0);
      const IntegrationPoint &center = Geometries.GetCenter(fe->GetGeomType());
      tr->SetIntPoint(&center);
      const real_t eps = EpsilonMin(*tr, center);
      const real_t mu = 1.0 / mu_inv.Eval(*tr, center);
      MFEM_VERIFY(eps > 0.0 && mu > 0.0,
                  "epsilon and mu_inv must be positive.");
      real_t curl_residual = 0.0, div_residual = 0.0;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         tr->SetIntPoint(&ip);
         source.GetVectorValue(*tr, ip, f);
         h.GetCurl(*tr, curl_h);
         solution.GetVectorValue(*tr, ip, e_val);
         if (epsilon)
         {
            f.Add(omega * omega * epsilon->Eval(*tr, ip), e_val);
         }
         else
         {
            DenseMatrix eps_tensor;
            epsilon_matrix->Eval(eps_tensor, *tr, ip);
            eps_tensor.Mult(e_val, eps_e);
            f.Add(omega * omega, eps_e);
         }
         f -= curl_h;
         curl_residual += (f * f) * ip.weight * tr->Weight();
         const real_t div = source.GetDivergence(*tr)
                            + omega * omega * d.GetDivergence(*tr);
         div_residual += div * div * ip.weight * tr->Weight();
      }
      error_estimates(el) = mu * pow(h_el / order, 2) * curl_residual
                            + pow(h_el / order, 2) * div_residual
                              / (omega * omega * eps);
   }

   // Each interior-face jump contributes to both adjacent element indicators.
   for (int face = 0; face < mesh->GetNumFaces(); face++)
   {
      FaceElementTransformations *tr = mesh->GetInteriorFaceTransformations(face);
      if (!tr) { continue; }
      const int e1 = tr->Elem1No, e2 = tr->Elem2No;
      const int qorder = std::max(2 * fes->GetFE(e1)->GetOrder() + 2, 2);
      const IntegrationRule &ir = IntRules.Get(tr->FaceGeom, qorder);
      real_t tangential_jump = 0.0, normal_jump = 0.0;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         tr->SetAllIntPoints(&ip);
         CalcOrtho(tr->Face->Jacobian(), normal);
         h.GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), h1);
         h.GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), h2);
         h1 -= h2;
         normal.cross3D(h1, f);
         tangential_jump += (f * f) * ip.weight * tr->Face->Weight();
         d.GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), d1);
         d.GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), d2);
         d1 -= d2;
         normal_jump += pow(normal * d1, 2) * ip.weight * tr->Face->Weight();
      }
      const real_t h1_el = mesh->GetElementSize(e1, 0);
      const real_t h2_el = mesh->GetElementSize(e2, 0);
      for (int el : {e1, e2})
      {
         ElementTransformation *et = mesh->GetElementTransformation(el);
         const IntegrationPoint &c = Geometries.GetCenter(fes->GetFE(el)->GetGeomType());
         et->SetIntPoint(&c);
         const real_t eps = EpsilonMin(*et, c);
         const real_t mu = 1.0 / mu_inv.Eval(*et, c);
         const real_t he = (el == e1) ? h1_el : h2_el;
         error_estimates(el) += mu * he / order * tangential_jump
                                + omega * omega * he / (order * eps) * normal_jump;
      }
   }
   for (int el = 0; el < ne; el++) { error_estimates(el) = sqrt(error_estimates(el)); }
   total_error = error_estimates.Norml2();
   current_sequence = mesh->GetSequence();
}

bool ComplexMaxwellResidualEstimator::MeshIsModified()
{
   const long sequence = solution.FESpace()->GetMesh()->GetSequence();
   MFEM_ASSERT(sequence >= current_sequence, "improper mesh update sequence");
   return sequence > current_sequence;
}

real_t ComplexMaxwellResidualEstimator::EpsilonMin(
   ElementTransformation &trans, const IntegrationPoint &ip) const
{
   DenseMatrix eps;
   if (epsilon_real_scalar) { return epsilon_real_scalar->Eval(trans, ip); }
   epsilon_real->Eval(eps, trans, ip);
   MFEM_VERIFY(eps.Height() == 3 && eps.Width() == 3,
               "epsilon_real must be a 3 by 3 matrix.");
   const real_t tol = 1e-12 * std::max(1.0, eps.MaxMaxNorm());
   for (int i = 0; i < 3; i++)
   {
      for (int j = i + 1; j < 3; j++)
      {
         MFEM_VERIFY(std::abs(eps(i, j) - eps(j, i)) <= tol,
                     "epsilon_real must be symmetric positive definite.");
      }
   }
   real_t values[3], vectors[9];
   eps.CalcEigenvalues(values, vectors);
   return std::min(values[0], std::min(values[1], values[2]));
}

void ComplexMaxwellResidualEstimator::ComputeEstimates()
{
   FiniteElementSpace *fes = solution.FESpace();
   Mesh *mesh = fes->GetMesh();
   MFEM_VERIFY(mesh->Dimension() == 3 && mesh->SpaceDimension() == 3,
               "ComplexMaxwellResidualEstimator requires a 3D mesh.");
   MFEM_VERIFY(source.FESpace()->GetMesh() == mesh,
               "solution and source must share a mesh.");

   L2_FECollection h_fec(std::max(0, order - 1), 3), d_fec(order, 3);
   FiniteElementSpace h_fes(mesh, &h_fec, 3, Ordering::byVDIM);
   FiniteElementSpace d_fes(mesh, &d_fec, 3, Ordering::byVDIM);
   ComplexGridFunction h(&h_fes), d(&d_fes), tmp(&d_fes);
   CurlGridFunctionCoefficient curl_er(&solution.real()), curl_ei(&solution.imag());
   ScalarVectorProductCoefficient hr_coef(mu_inv, curl_er), hi_coef(mu_inv, curl_ei);
   h.ProjectCoefficient(hr_coef, hi_coef);
   VectorGridFunctionCoefficient er(&solution.real()), ei(&solution.imag());
   if (epsilon_real_scalar)
   {
      ScalarVectorProductCoefficient epsr_er(*epsilon_real_scalar, er);
      ScalarVectorProductCoefficient epsi_ei(*epsilon_imag_scalar, ei);
      ScalarVectorProductCoefficient epsr_ei(*epsilon_real_scalar, ei);
      ScalarVectorProductCoefficient epsi_er(*epsilon_imag_scalar, er);
      d.ProjectCoefficient(epsr_er, epsr_ei);
      tmp.ProjectCoefficient(epsi_ei, epsi_er);
      d.real() -= tmp.real(); d.imag() += tmp.imag();
   }
   else
   {
      MatrixVectorProductCoefficient epsr_er(*epsilon_real, er), epsi_ei(*epsilon_imag, ei);
      MatrixVectorProductCoefficient epsr_ei(*epsilon_real, ei), epsi_er(*epsilon_imag, er);
      d.ProjectCoefficient(epsr_er, epsr_ei);
      tmp.ProjectCoefficient(epsi_ei, epsi_er);
      d.real() -= tmp.real(); d.imag() += tmp.imag();
   }

   const int ne = mesh->GetNE();
   error_estimates.SetSize(ne); error_estimates = 0.0;
   Vector fr(3), fi(3), cr(3), ci(3), er_val(3), ei_val(3), epsr(3), epsi(3);
   for (int el = 0; el < ne; el++)
   {
      ElementTransformation *tr = mesh->GetElementTransformation(el);
      const FiniteElement *fe = fes->GetFE(el);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                std::max(2 * fe->GetOrder() + 2, 2));
      const IntegrationPoint &center = Geometries.GetCenter(fe->GetGeomType());
      tr->SetIntPoint(&center);
      const real_t eps_min = EpsilonMin(*tr, center), mu = 1.0 / mu_inv.Eval(*tr, center);
      MFEM_VERIFY(eps_min > 0.0 && mu > 0.0,
                  "epsilon_real and mu_inv must be positive.");
      real_t curl_res = 0.0, div_res = 0.0;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q); tr->SetIntPoint(&ip);
         source.real().GetVectorValue(*tr, ip, fr); source.imag().GetVectorValue(*tr, ip, fi);
         solution.real().GetVectorValue(*tr, ip, er_val);
         solution.imag().GetVectorValue(*tr, ip, ei_val);
         if (epsilon_real_scalar)
         {
            epsr = er_val; epsr *= epsilon_real_scalar->Eval(*tr, ip);
            epsi = ei_val; epsi *= epsilon_imag_scalar->Eval(*tr, ip);
         }
         else
         {
            DenseMatrix ar, ai; epsilon_real->Eval(ar, *tr, ip); epsilon_imag->Eval(ai, *tr, ip);
            ar.Mult(er_val, epsr); ai.Mult(ei_val, epsi);
         }
         epsr -= epsi;
         fr.Add(omega * omega, epsr); h.real().GetCurl(*tr, cr); fr -= cr;
         if (epsilon_real_scalar)
         {
            epsr = ei_val; epsr *= epsilon_real_scalar->Eval(*tr, ip);
            epsi = er_val; epsi *= epsilon_imag_scalar->Eval(*tr, ip);
         }
         else
         {
            DenseMatrix ar, ai; epsilon_real->Eval(ar, *tr, ip); epsilon_imag->Eval(ai, *tr, ip);
            ar.Mult(ei_val, epsr); ai.Mult(er_val, epsi);
         }
         epsr += epsi;
         fi.Add(omega * omega, epsr); h.imag().GetCurl(*tr, ci); fi -= ci;
         curl_res += (fr * fr + fi * fi) * ip.weight * tr->Weight();
         const real_t div_r = source.real().GetDivergence(*tr) + omega * omega * d.real().GetDivergence(*tr);
         const real_t div_i = source.imag().GetDivergence(*tr) + omega * omega * d.imag().GetDivergence(*tr);
         div_res += (div_r * div_r + div_i * div_i) * ip.weight * tr->Weight();
      }
      const real_t he = mesh->GetElementSize(el);
      error_estimates(el) = mu * pow(he / order, 2) * curl_res
                            + pow(he / order, 2) * div_res / (omega * omega * eps_min);
   }
   // Face terms use the complex norm of the jumps of H and D=epsilon E.
   for (int face = 0; face < mesh->GetNumFaces(); face++)
   {
      FaceElementTransformations *tr = mesh->GetInteriorFaceTransformations(face);
      if (!tr) { continue; }
      const int e1 = tr->Elem1No, e2 = tr->Elem2No;
      const IntegrationRule &ir = IntRules.Get(tr->FaceGeom,
                                                std::max(2 * fes->GetFE(e1)->GetOrder() + 2, 2));
      real_t jt = 0.0, jn = 0.0; Vector a(3), b(3), n(3);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q); tr->SetAllIntPoints(&ip);
         CalcOrtho(tr->Face->Jacobian(), n);
         h.real().GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), a);
         h.real().GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), b); a -= b; n.cross3D(a, fr);
         h.imag().GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), a);
         h.imag().GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), b); a -= b; n.cross3D(a, fi);
         jt += (fr * fr + fi * fi) * ip.weight * tr->Face->Weight();
         d.real().GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), a);
         d.real().GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), b); a -= b;
         const real_t nr = n * a;
         d.imag().GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), a);
         d.imag().GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), b); a -= b;
         jn += (nr * nr + pow(n * a, 2)) * ip.weight * tr->Face->Weight();
      }
      for (int el : {e1, e2})
      {
         ElementTransformation *et = mesh->GetElementTransformation(el);
         const IntegrationPoint &c = Geometries.GetCenter(fes->GetFE(el)->GetGeomType()); et->SetIntPoint(&c);
         const real_t eps = EpsilonMin(*et, c), mu = 1.0 / mu_inv.Eval(*et, c), he = mesh->GetElementSize(el);
         error_estimates(el) += mu * he / order * jt + omega * omega * he / (order * eps) * jn;
      }
   }
   for (int el = 0; el < ne; el++) { error_estimates(el) = sqrt(error_estimates(el)); }
   total_error = error_estimates.Norml2(); current_sequence = mesh->GetSequence();
}


namespace
{
real_t MaxwellEpsilonMin(Coefficient *epsilon, MatrixCoefficient *epsilon_matrix,
                         ElementTransformation &trans, const IntegrationPoint &ip)
{
   if (epsilon) { return epsilon->Eval(trans, ip); }
   DenseMatrix eps;
   epsilon_matrix->Eval(eps, trans, ip);
   real_t values[3], vectors[9];
   eps.CalcEigenvalues(values, vectors);
   return std::min(values[0], std::min(values[1], values[2]));
}
}

MaxwellResidualDomainEstimator::MaxwellResidualDomainEstimator(
   GridFunction &solution_, GridFunction &source_, GridFunction &h_, GridFunction &d_,
   Coefficient &epsilon_, Coefficient &mu_inv_, real_t omega_, int order_)
   : solution(solution_), source(source_), h(h_), d(d_), epsilon(&epsilon_),
     epsilon_matrix(nullptr), mu_inv(mu_inv_), omega(omega_), order(order_) { }

MaxwellResidualDomainEstimator::MaxwellResidualDomainEstimator(
   GridFunction &solution_, GridFunction &source_, GridFunction &h_, GridFunction &d_,
   MatrixCoefficient &epsilon_, Coefficient &mu_inv_, real_t omega_, int order_)
   : solution(solution_), source(source_), h(h_), d(d_), epsilon(nullptr),
     epsilon_matrix(&epsilon_), mu_inv(mu_inv_), omega(omega_), order(order_) { }

real_t MaxwellResidualDomainEstimator::EpsilonMin(
   ElementTransformation &trans, const IntegrationPoint &ip) const
{ return MaxwellEpsilonMin(epsilon, epsilon_matrix, trans, ip); }

void BuildMaxwellResidualFields(GridFunction &solution, Coefficient &mu_inv,
                                Coefficient &epsilon, GridFunction &h,
                                GridFunction &d)
{
   CurlGridFunctionCoefficient curl_e(&solution);
   ScalarVectorProductCoefficient h_coef(mu_inv, curl_e);
   VectorGridFunctionCoefficient e(&solution);
   ScalarVectorProductCoefficient d_coef(epsilon, e);
   h.ProjectCoefficient(h_coef);
   d.ProjectCoefficient(d_coef);
}

void BuildMaxwellResidualFields(GridFunction &solution, Coefficient &mu_inv,
                                MatrixCoefficient &epsilon, GridFunction &h,
                                GridFunction &d)
{
   CurlGridFunctionCoefficient curl_e(&solution);
   ScalarVectorProductCoefficient h_coef(mu_inv, curl_e);
   VectorGridFunctionCoefficient e(&solution);
   MatrixVectorProductCoefficient d_coef(epsilon, e);
   h.ProjectCoefficient(h_coef);
   d.ProjectCoefficient(d_coef);
}

real_t MaxwellResidualDomainEstimator::GetElementError(const FiniteElement &el,
                                                        ElementTransformation &tr)
{
   MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive.");
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                             std::max(2 * el.GetOrder() + 2, 2));
   const IntegrationPoint &center = Geometries.GetCenter(el.GetGeomType());
   tr.SetIntPoint(&center);
   const real_t eps = EpsilonMin(tr, center), mu = 1.0 / mu_inv.Eval(tr, center);
   MFEM_VERIFY(eps > 0.0 && mu > 0.0, "epsilon and mu_inv must be positive.");
   Vector residual(3), curl_h(3), e(3), eps_e(3);
   real_t curl_term = 0.0, div_term = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetIntPoint(&ip);
      source.GetVectorValue(tr, ip, residual);
      solution.GetVectorValue(tr, ip, e);
      if (epsilon) { residual.Add(omega * omega * epsilon->Eval(tr, ip), e); }
      else
      {
         DenseMatrix eps_tensor; epsilon_matrix->Eval(eps_tensor, tr, ip);
         eps_tensor.Mult(e, eps_e); residual.Add(omega * omega, eps_e);
      }
      h.GetCurl(tr, curl_h); residual -= curl_h;
      curl_term += (residual * residual) * ip.weight * tr.Weight();
      const real_t div = source.GetDivergence(tr) + omega * omega * d.GetDivergence(tr);
      div_term += div * div * ip.weight * tr.Weight();
   }
   const real_t he = solution.FESpace()->GetMesh()->GetElementSize(tr.ElementNo);
   return mu * pow(he / order, 2) * curl_term
          + pow(he / order, 2) * div_term / (omega * omega * eps);
}

MaxwellResidualFaceEstimator::MaxwellResidualFaceEstimator(
   GridFunction &h_, GridFunction &d_, Coefficient &epsilon_, Coefficient &mu_inv_,
   real_t omega_, int order_)
   : h(h_), d(d_), epsilon(&epsilon_), epsilon_matrix(nullptr), mu_inv(mu_inv_),
     omega(omega_), order(order_) { }

MaxwellResidualFaceEstimator::MaxwellResidualFaceEstimator(
   GridFunction &h_, GridFunction &d_, MatrixCoefficient &epsilon_, Coefficient &mu_inv_,
   real_t omega_, int order_)
   : h(h_), d(d_), epsilon(nullptr), epsilon_matrix(&epsilon_), mu_inv(mu_inv_),
     omega(omega_), order(order_) { }

real_t MaxwellResidualFaceEstimator::EpsilonMin(
   ElementTransformation &trans, const IntegrationPoint &ip) const
{ return MaxwellEpsilonMin(epsilon, epsilon_matrix, trans, ip); }

void MaxwellResidualFaceEstimator::GetFaceError(
   const FiniteElement &el1, const FiniteElement &, FaceElementTransformations &tr,
   real_t &error1, real_t &error2)
{
   MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive.");
   const IntegrationRule &ir = IntRules.Get(tr.FaceGeom,
                                             std::max(2 * el1.GetOrder() + 2, 2));
   Vector h1(3), h2(3), d1(3), d2(3), normal(3), jump(3);
   real_t tangential = 0.0, normal_jump = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetAllIntPoints(&ip);
      CalcOrtho(tr.Face->Jacobian(), normal);
      h.GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), h1);
      h.GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), h2); h1 -= h2;
      normal.cross3D(h1, jump);
      tangential += (jump * jump) * ip.weight * tr.Face->Weight();
      d.GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), d1);
      d.GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), d2); d1 -= d2;
      normal_jump += pow(normal * d1, 2) * ip.weight * tr.Face->Weight();
   }
   auto contribution = [&](int el)
   {
      Mesh *mesh = h.FESpace()->GetMesh();
      ElementTransformation *et = NULL;
      real_t he = 0.0;
      if (el < mesh->GetNE())
      {
         et = mesh->GetElementTransformation(el);
         he = mesh->GetElementSize(el);
      }
#ifdef MFEM_USE_MPI
      else if (auto *pmesh = dynamic_cast<ParMesh*>(mesh))
      {
         const int nbr_el = el - mesh->GetNE();
         et = pmesh->GetFaceNbrElementTransformation(nbr_el);
         he = pmesh->GetFaceNbrElementSize(nbr_el);
      }
#endif
      else { MFEM_ABORT("invalid element number in face estimator"); }
      const IntegrationPoint &center = Geometries.GetCenter(h.FESpace()->GetFE(el)->GetGeomType());
      et->SetIntPoint(&center);
      const real_t eps = EpsilonMin(*et, center), mu = 1.0 / mu_inv.Eval(*et, center);
      return mu * he / order * tangential + omega * omega * he / (order * eps) * normal_jump;
   };
   error1 = contribution(tr.Elem1No);
   error2 = contribution(tr.Elem2No);
}

void MaxwellResidualFaceEstimator::ExchangeFaceNbrData()
{
#ifdef MFEM_USE_MPI
   if (auto *ph = dynamic_cast<ParGridFunction*>(&h))
   {
      ph->ExchangeFaceNbrData();
   }
   if (auto *pd = dynamic_cast<ParGridFunction*>(&d))
   {
      pd->ExchangeFaceNbrData();
   }
#endif
}

namespace
{
real_t ComplexEpsilonMin(Coefficient *er, MatrixCoefficient *mr,
                         ElementTransformation &tr, const IntegrationPoint &ip)
{
   if (er) { return er->Eval(tr, ip); }
   DenseMatrix a; mr->Eval(a, tr, ip);
   real_t values[3], vectors[9]; a.CalcEigenvalues(values, vectors);
   return std::min(values[0], std::min(values[1], values[2]));
}
}

ComplexMaxwellResidualDomainEstimator::ComplexMaxwellResidualDomainEstimator(
   ComplexGridFunction &s, ComplexGridFunction &j, ComplexGridFunction &h_,
   ComplexGridFunction &d_, MatrixCoefficient &er, MatrixCoefficient &ei,
   Coefficient &mu, real_t w, int p)
   : solution(s), source(j), h(h_), d(d_), epsilon_real(&er), epsilon_imag(&ei),
     epsilon_real_scalar(nullptr), epsilon_imag_scalar(nullptr), mu_inv(mu), omega(w), order(p) { }

ComplexMaxwellResidualDomainEstimator::ComplexMaxwellResidualDomainEstimator(
   ComplexGridFunction &s, ComplexGridFunction &j, ComplexGridFunction &h_,
   ComplexGridFunction &d_, Coefficient &er, Coefficient &ei, Coefficient &mu,
   real_t w, int p)
   : solution(s), source(j), h(h_), d(d_), epsilon_real(nullptr), epsilon_imag(nullptr),
     epsilon_real_scalar(&er), epsilon_imag_scalar(&ei), mu_inv(mu), omega(w), order(p) { }

real_t ComplexMaxwellResidualDomainEstimator::EpsilonMin(
   ElementTransformation &tr, const IntegrationPoint &ip) const
{ return ComplexEpsilonMin(epsilon_real_scalar, epsilon_real, tr, ip); }

void BuildComplexMaxwellResidualFields(ComplexGridFunction &solution,
                                       Coefficient &mu_inv, MatrixCoefficient &er,
                                       MatrixCoefficient &ei, ComplexGridFunction &h,
                                       ComplexGridFunction &d)
{
   CurlGridFunctionCoefficient cer(&solution.real()), cei(&solution.imag());
   ScalarVectorProductCoefficient hr(mu_inv, cer), hi(mu_inv, cei);
   h.ProjectCoefficient(hr, hi);
   VectorGridFunctionCoefficient vr(&solution.real()), vi(&solution.imag());
   MatrixVectorProductCoefficient er_r(er, vr), er_i(er, vi), ei_r(ei, vr), ei_i(ei, vi);
   d.ProjectCoefficient(er_r, er_i);
   ComplexGridFunction tmp(d.FESpace()); tmp.ProjectCoefficient(ei_i, ei_r);
   d.real() -= tmp.real(); d.imag() += tmp.imag();
}

void BuildComplexMaxwellResidualFields(ComplexGridFunction &solution,
                                       Coefficient &mu_inv, Coefficient &er,
                                       Coefficient &ei, ComplexGridFunction &h,
                                       ComplexGridFunction &d)
{
   CurlGridFunctionCoefficient cer(&solution.real()), cei(&solution.imag());
   ScalarVectorProductCoefficient hr(mu_inv, cer), hi(mu_inv, cei);
   h.ProjectCoefficient(hr, hi);
   VectorGridFunctionCoefficient vr(&solution.real()), vi(&solution.imag());
   ScalarVectorProductCoefficient er_r(er, vr), er_i(er, vi), ei_r(ei, vr), ei_i(ei, vi);
   d.ProjectCoefficient(er_r, er_i);
   ComplexGridFunction tmp(d.FESpace()); tmp.ProjectCoefficient(ei_i, ei_r);
   d.real() -= tmp.real(); d.imag() += tmp.imag();
}

real_t ComplexMaxwellResidualDomainEstimator::GetElementError(
   const FiniteElement &el, ElementTransformation &tr)
{
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), std::max(2 * el.GetOrder() + 2, 2));
   const IntegrationPoint &c = Geometries.GetCenter(el.GetGeomType()); tr.SetIntPoint(&c);
   const real_t eps = EpsilonMin(tr, c), mu = 1.0 / mu_inv.Eval(tr, c);
   Vector rr(3), ri(3), cr(3), ci(3), evr(3), evi(3), a(3), b(3);
   real_t curl_term = 0.0, div_term = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetIntPoint(&ip);
      source.real().GetVectorValue(tr, ip, rr); source.imag().GetVectorValue(tr, ip, ri);
      solution.real().GetVectorValue(tr, ip, evr); solution.imag().GetVectorValue(tr, ip, evi);
      if (epsilon_real_scalar)
      {
         a = evr; a *= epsilon_real_scalar->Eval(tr, ip); b = evi; b *= epsilon_imag_scalar->Eval(tr, ip);
      }
      else { DenseMatrix ar, ai; epsilon_real->Eval(ar, tr, ip); epsilon_imag->Eval(ai, tr, ip); ar.Mult(evr, a); ai.Mult(evi, b); }
      a -= b; rr.Add(omega * omega, a); h.real().GetCurl(tr, cr); rr -= cr;
      if (epsilon_real_scalar)
      {
         a = evi; a *= epsilon_real_scalar->Eval(tr, ip); b = evr; b *= epsilon_imag_scalar->Eval(tr, ip);
      }
      else { DenseMatrix ar, ai; epsilon_real->Eval(ar, tr, ip); epsilon_imag->Eval(ai, tr, ip); ar.Mult(evi, a); ai.Mult(evr, b); }
      a += b; ri.Add(omega * omega, a); h.imag().GetCurl(tr, ci); ri -= ci;
      curl_term += (rr * rr + ri * ri) * ip.weight * tr.Weight();
      const real_t dr = source.real().GetDivergence(tr) + omega * omega * d.real().GetDivergence(tr);
      const real_t di = source.imag().GetDivergence(tr) + omega * omega * d.imag().GetDivergence(tr);
      div_term += (dr * dr + di * di) * ip.weight * tr.Weight();
   }
   const real_t he = solution.FESpace()->GetMesh()->GetElementSize(tr.ElementNo);
   return mu * pow(he / order, 2) * curl_term + pow(he / order, 2) * div_term / (omega * omega * eps);
}

ComplexMaxwellResidualFaceEstimator::ComplexMaxwellResidualFaceEstimator(
   ComplexGridFunction &h_, ComplexGridFunction &d_, MatrixCoefficient &er,
   MatrixCoefficient &ei, Coefficient &mu, real_t w, int p)
   : h(h_), d(d_), epsilon_real(&er), epsilon_imag(&ei), epsilon_real_scalar(nullptr),
     epsilon_imag_scalar(nullptr), mu_inv(mu), omega(w), order(p) { }

ComplexMaxwellResidualFaceEstimator::ComplexMaxwellResidualFaceEstimator(
   ComplexGridFunction &h_, ComplexGridFunction &d_, Coefficient &er, Coefficient &ei,
   Coefficient &mu, real_t w, int p)
   : h(h_), d(d_), epsilon_real(nullptr), epsilon_imag(nullptr), epsilon_real_scalar(&er),
     epsilon_imag_scalar(&ei), mu_inv(mu), omega(w), order(p) { }

real_t ComplexMaxwellResidualFaceEstimator::EpsilonMin(
   ElementTransformation &tr, const IntegrationPoint &ip) const
{ return ComplexEpsilonMin(epsilon_real_scalar, epsilon_real, tr, ip); }

void ComplexMaxwellResidualFaceEstimator::GetFaceError(
   const FiniteElement &el1, const FiniteElement &, FaceElementTransformations &tr,
   real_t &error1, real_t &error2)
{
   const IntegrationRule &ir = IntRules.Get(tr.FaceGeom, std::max(2 * el1.GetOrder() + 2, 2));
   Vector a(3), b(3), n(3), jr(3), ji(3);
   real_t tangential = 0.0, normal_jump = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetAllIntPoints(&ip); CalcOrtho(tr.Face->Jacobian(), n);
      h.real().GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), a);
      h.real().GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), b); a -= b; n.cross3D(a, jr);
      h.imag().GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), a);
      h.imag().GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), b); a -= b; n.cross3D(a, ji);
      tangential += (jr * jr + ji * ji) * ip.weight * tr.Face->Weight();
      d.real().GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), a);
      d.real().GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), b); a -= b;
      const real_t nr = n * a;
      d.imag().GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), a);
      d.imag().GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), b); a -= b;
      normal_jump += (nr * nr + pow(n * a, 2)) * ip.weight * tr.Face->Weight();
   }
   auto contribution = [&](int el)
   {
      Mesh *mesh = h.FESpace()->GetMesh();
      ElementTransformation *et = NULL;
      real_t he = 0.0;
      if (el < mesh->GetNE())
      {
         et = mesh->GetElementTransformation(el);
         he = mesh->GetElementSize(el);
      }
#ifdef MFEM_USE_MPI
      else if (auto *pmesh = dynamic_cast<ParMesh*>(mesh))
      {
         const int nbr_el = el - mesh->GetNE();
         et = pmesh->GetFaceNbrElementTransformation(nbr_el);
         he = pmesh->GetFaceNbrElementSize(nbr_el);
      }
#endif
      else { MFEM_ABORT("invalid element number in face estimator"); }
      const IntegrationPoint &c = Geometries.GetCenter(h.FESpace()->GetFE(el)->GetGeomType()); et->SetIntPoint(&c);
      const real_t eps = EpsilonMin(*et, c), mu = 1.0 / mu_inv.Eval(*et, c);
      return mu * he / order * tangential + omega * omega * he / (order * eps) * normal_jump;
   };
   error1 = contribution(tr.Elem1No); error2 = contribution(tr.Elem2No);
}

void ComplexMaxwellResidualFaceEstimator::ExchangeFaceNbrData()
{
#ifdef MFEM_USE_MPI
   if (auto *ph_real = dynamic_cast<ParGridFunction*>(&h.real()))
   {
      ph_real->ExchangeFaceNbrData();
   }
   if (auto *ph_imag = dynamic_cast<ParGridFunction*>(&h.imag()))
   {
      ph_imag->ExchangeFaceNbrData();
   }
   if (auto *pd_real = dynamic_cast<ParGridFunction*>(&d.real()))
   {
      pd_real->ExchangeFaceNbrData();
   }
   if (auto *pd_imag = dynamic_cast<ParGridFunction*>(&d.imag()))
   {
      pd_imag->ExchangeFaceNbrData();
   }
#endif
}

namespace
{
real_t ComplexTangentialTraceError(ComplexGridFunction &field,
                                   VectorCoefficient *data_real,
                                   VectorCoefficient *data_imag,
                                   const FiniteElement &el,
                                   FaceElementTransformations &tr)
{
   const IntegrationRule &ir = IntRules.Get(tr.FaceGeom,
                                             std::max(2 * el.GetOrder() + 2, 2));
   Vector value_r(3), value_i(3), trace_r(3), trace_i(3), prescribed(3), normal(3);
   real_t error = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      tr.SetAllIntPoints(&ip);
      CalcOrtho(tr.Face->Jacobian(), normal);
      normal /= normal.Norml2();
      const IntegrationPoint &eip = tr.GetElement1IntPoint();
      field.real().GetVectorValue(*tr.Elem1, eip, value_r);
      field.imag().GetVectorValue(*tr.Elem1, eip, value_i);
      value_r.cross3D(normal, trace_r);
      value_i.cross3D(normal, trace_i);
      if (data_real)
      {
         data_real->Eval(prescribed, *tr.Elem1, eip);
         trace_r -= prescribed;
      }
      if (data_imag)
      {
         data_imag->Eval(prescribed, *tr.Elem1, eip);
         trace_i -= prescribed;
      }
      error += (trace_r * trace_r + trace_i * trace_i) * ip.weight * tr.Face->Weight();
   }
   return error;
}
}

real_t ComplexMaxwellDirichletBCErrorEstimator::GetFaceError(
   const FiniteElement &el, FaceElementTransformations &tr)
{
   return ComplexTangentialTraceError(electric, data_real, data_imag, el, tr);
}

real_t ComplexMaxwellNeumannBCErrorEstimator::GetFaceError(
   const FiniteElement &el, FaceElementTransformations &tr)
{
   return ComplexTangentialTraceError(magnetic_flux, data_real, data_imag, el, tr);
}

} // namespace mfem
