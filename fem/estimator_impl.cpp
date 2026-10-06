// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and j_src code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "estimator_impl.hpp"
#include "complex_fem.hpp"

namespace mfem
{

MaxwellResidualEstimatorBase::FieldLayout
MaxwellResidualEstimatorBase::GetFieldLayout(const GridFunction &field)
{
   return GetFieldLayout(*field.FESpace(), field.VectorDim());
}

MaxwellResidualEstimatorBase::FieldLayout
MaxwellResidualEstimatorBase::GetFieldLayout(const FiniteElementSpace &fes,
                                             const int vector_dim)
{
   const Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.Dimension();
   MFEM_VERIFY((dim == 1 || dim == 2 || dim == 3) && mesh.SpaceDimension() == dim,
               "Maxwell residual estimators require a one-, two-, or "
               "three-dimensional full-dimensional mesh.");
   return {dim, vector_dim, fes.GetCurlDim()};
}

real_t FaceJumpEstimatorBase::SmallestEigenvalue(const DenseMatrix &a,
                                                 const int vector_dim,
                                                 const char *name)
{
   MFEM_VERIFY(a.Height() == vector_dim && a.Width() == vector_dim,
               name << " must have dimensions matching the mesh.");
   const real_t tol = 1e-12 * std::max(1.0, a.MaxMaxNorm());
   for (int i = 0; i < vector_dim; i++)
   {
      for (int j = i + 1; j < vector_dim; j++)
      {
         MFEM_VERIFY(std::abs(a(i, j) - a(j, i)) <= tol,
                     name << " must be symmetric positive definite.");
      }
   }
   Vector values(vector_dim), vectors(vector_dim * vector_dim);
   a.CalcEigenvalues(values.GetData(), vectors.GetData());
   return values.Min();
}

real_t FaceJumpEstimatorBase::CoefficientMinimum(
   Coefficient *scalar, MatrixCoefficient *matrix, ElementTransformation &tr,
   const IntegrationPoint &ip, const int vector_dim, const char *name)
{
   if (scalar) { return scalar->Eval(tr, ip); }
   DenseMatrix value;
   matrix->Eval(value, tr, ip);
   return SmallestEigenvalue(value, vector_dim, name);
}

void MaxwellResidualEstimatorBase::GetCurl(const GridFunction &field,
                                           ElementTransformation &tr,
                                           const FieldLayout &layout,
                                           Vector &curl)
{
   if (layout.mesh_dim == 2 && layout.vector_dim == 3)
   {
      DenseMatrix grad;
      field.GetVectorGradient(tr, grad);
      MFEM_VERIFY(grad.Height() == 3 && grad.Width() == 2,
                  "The R2D magnetic reconstruction must have three components.");
      curl.SetSize(3);
      curl(0) = grad(2, 1);
      curl(1) = -grad(2, 0);
      curl(2) = grad(1, 0) - grad(0, 1);
      return;
   }
   if (layout.mesh_dim == 1 && layout.vector_dim == 3)
   {
      DenseMatrix grad;
      field.GetVectorGradient(tr, grad);
      MFEM_VERIFY(grad.Height() == 3 && grad.Width() == 1,
                  "The R1D magnetic reconstruction must have three components.");
      curl.SetSize(3);
      curl(0) = 0.0;
      curl(1) = -grad(2, 0);
      curl(2) = grad(1, 0);
      return;
   }
   if (layout.mesh_dim == 3)
   {
      field.GetCurl(tr, curl);
      return;
   }
   Vector grad;
   field.GetGradient(tr, grad);
   MFEM_VERIFY(grad.Size() == 2,
               "The 2D magnetic reconstruction must be scalar-valued.");
   curl.SetSize(2);
   curl(0) = grad(1);
   curl(1) = -grad(0);
}

void FaceJumpEstimatorBase::GetFaceNormal(FaceElementTransformations &tr,
                                          const int vector_dim,
                                          Vector &normal)
{
   const DenseMatrix &jacobian = tr.Face->Jacobian();
   if (jacobian.Width() == 0)
   {
      // A 1D interior face is a point. Its orientation has no effect on the
      // squared residual jump, so use the positive embedded x normal.
      normal.SetSize(vector_dim);
      normal = 0.0;
      normal(0) = 1.0;
      return;
   }
   Vector face_normal(jacobian.Height());
   CalcOrtho(jacobian, face_normal);
   MFEM_VERIFY(vector_dim == face_normal.Size() || vector_dim == 3,
               "incompatible field and face-normal dimensions");
   normal.SetSize(vector_dim);
   normal = 0.0;
   for (int i = 0; i < face_normal.Size(); i++) { normal(i) = face_normal(i); }
}

real_t FaceJumpEstimatorBase::TangentialComponentSquared(
   const Vector &normal, const Vector &value)
{
   MFEM_VERIFY(normal.Size() == value.Size(),
               "normal and vector field dimensions must agree.");
   if (normal.Size() == 1) { return 0.0; }
   if (normal.Size() == 2)
   {
      const real_t cross = normal(0) * value(1) - normal(1) * value(0);
      return cross * cross;
   }
   Vector cross;
   normal.cross3D(value, cross);
   return cross * cross;
}

real_t MaxwellResidualEstimatorBase::TangentialJump(
   const GridFunction &field, FaceElementTransformations &tr,
   const FieldLayout &layout, const Vector &normal, Vector &first,
   Vector &second, Vector &cross)
{
   if (layout.mesh_dim == 2 && layout.vector_dim == 1)
   {
      const real_t jump = field.GetValue(*tr.Elem1, tr.GetElement1IntPoint()) -
                          field.GetValue(*tr.Elem2, tr.GetElement2IntPoint());
      return jump * jump;
   }
   field.GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), first);
   field.GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), second);
   first -= second;
   if (normal.Size() == 3)
   {
      normal.cross3D(first, cross);
      return cross * cross;
   }
   return TangentialComponentSquared(normal, first);
}

namespace
{
real_t GlobalMaxwellEstimatorError(const FiniteElementSpace *fes,
                                   const real_t local_error)
{
#ifdef MFEM_USE_MPI
   if (auto *pfes = dynamic_cast<const ParFiniteElementSpace*>(fes))
   {
      real_t global_error_sq = 0.0;
      const real_t local_error_sq = local_error * local_error;
      MPI_Allreduce(&local_error_sq, &global_error_sq, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, pfes->GetComm());
      return sqrt(global_error_sq);
   }
#endif
   return local_error;
}
}

real_t MaxwellResidualEstimator::GetTotalError() const
{
   return GlobalMaxwellEstimatorError(e.FESpace(), total_error);
}

real_t ComplexMaxwellResidualEstimator::GetTotalError() const
{
   return GlobalMaxwellEstimatorError(e.FESpace(), total_error);
}

MaxwellResidualFields::MaxwellResidualFields(GridFunction &e_,
                                             Coefficient &epsilon_,
                                             Coefficient &mu_inv_, int order_)
   : e(e_), epsilon(&epsilon_), epsilon_matrix(nullptr),
     mu_inv(mu_inv_), order(order_) { }

MaxwellResidualFields::MaxwellResidualFields(GridFunction &e_,
                                             MatrixCoefficient &epsilon_,
                                             Coefficient &mu_inv_, int order_)
   : e(e_), epsilon(nullptr), epsilon_matrix(&epsilon_),
     mu_inv(mu_inv_), order(order_) { }

MaxwellResidualFields::~MaxwellResidualFields() = default;

void MaxwellResidualFields::BuildSpaces()
{
   Mesh *mesh = e.FESpace()->GetMesh();
   h.reset(); d.reset();
   h_fes.reset(); d_fes.reset();
   h_fec.reset(); d_fec.reset();
   const FieldLayout layout = GetFieldLayout(e);
   const int dim = layout.mesh_dim;
   const int hdim = e.FESpace()->GetCurlDim();
   const int ddim = e.VectorDim();
   h_fec = std::make_unique<L2_FECollection>(std::max(0, order - 1), dim);
   d_fec = std::make_unique<L2_FECollection>(order, dim);
#ifdef MFEM_USE_MPI
   if (auto *pfes = dynamic_cast<ParFiniteElementSpace*>(e.FESpace()))
   {
      ParMesh *pmesh = pfes->GetParMesh();
      h_fes = std::make_unique<ParFiniteElementSpace>(pmesh, h_fec.get(),
                                                      hdim,
                                                      Ordering::byVDIM);
      d_fes = std::make_unique<ParFiniteElementSpace>(pmesh, d_fec.get(), ddim,
                                                      Ordering::byVDIM);
      h = std::make_unique<ParGridFunction>(
             static_cast<ParFiniteElementSpace*>(h_fes.get()));
      d = std::make_unique<ParGridFunction>(
             static_cast<ParFiniteElementSpace*>(d_fes.get()));
   }
   else
#endif
   {
      h_fes = std::make_unique<FiniteElementSpace>(mesh, h_fec.get(),
                                                   hdim,
                                                   Ordering::byVDIM);
      d_fes = std::make_unique<FiniteElementSpace>(mesh, d_fec.get(), ddim,
                                                   Ordering::byVDIM);
      h = std::make_unique<GridFunction>(h_fes.get());
      d = std::make_unique<GridFunction>(d_fes.get());
   }
   mesh_sequence = mesh->GetSequence();
}

void MaxwellResidualFields::Update()
{
   MFEM_VERIFY(order > 0, "order must be positive.");
   if (!h || mesh_sequence != e.FESpace()->GetMesh()->GetSequence())
   {
      BuildSpaces();
   }
   BuildCurlFlux(e, mu_inv, *h);
   if (epsilon) { BuildElectricDisplacement(e, *epsilon, *d); }
   else { BuildElectricDisplacement(e, *epsilon_matrix, *d); }
}

ComplexMaxwellResidualFields::ComplexMaxwellResidualFields(
   ComplexGridFunction &e_, MatrixCoefficient &epsilon_real_,
   MatrixCoefficient &epsilon_imag_, Coefficient &mu_inv_, int order_)
   : e(e_), epsilon_real(&epsilon_real_),
     epsilon_imag(&epsilon_imag_),
     epsilon_real_scalar(nullptr), epsilon_imag_scalar(nullptr), mu_inv(mu_inv_),
     order(order_) { }

ComplexMaxwellResidualFields::ComplexMaxwellResidualFields(
   ComplexGridFunction &e_, Coefficient &epsilon_real_,
   Coefficient &epsilon_imag_, Coefficient &mu_inv_, int order_)
   : e(e_), epsilon_real(nullptr), epsilon_imag(nullptr),
     epsilon_real_scalar(&epsilon_real_), epsilon_imag_scalar(&epsilon_imag_),
     mu_inv(mu_inv_), order(order_) { }

ComplexMaxwellResidualFields::~ComplexMaxwellResidualFields() = default;

void ComplexMaxwellResidualFields::BuildSpaces()
{
   FiniteElementSpace *efes = e.FESpace();
   Mesh *mesh = efes->GetMesh();
   h.reset(); d.reset();
   h_fes.reset(); d_fes.reset();
   h_fec.reset(); d_fec.reset();
   const FieldLayout layout = GetFieldLayout(*efes, e.VectorDim());
   const int dim = layout.mesh_dim;
   const int hdim = efes->GetCurlDim();
   const int ddim = e.VectorDim();
   h_fec = std::make_unique<L2_FECollection>(std::max(0, order - 1), dim);
   d_fec = std::make_unique<L2_FECollection>(order, dim);
#ifdef MFEM_USE_MPI
   if (auto *pfes = dynamic_cast<ParFiniteElementSpace*>(e.FESpace()))
   {
      ParMesh *pmesh = pfes->GetParMesh();
      h_fes = std::make_unique<ParFiniteElementSpace>(pmesh, h_fec.get(),
                                                      hdim,
                                                      Ordering::byVDIM);
      d_fes = std::make_unique<ParFiniteElementSpace>(pmesh, d_fec.get(), ddim,
                                                      Ordering::byVDIM);
      h = std::make_unique<ParComplexGridFunction>(
             static_cast<ParFiniteElementSpace *>(h_fes.get()));
      d = std::make_unique<ParComplexGridFunction>(
             static_cast<ParFiniteElementSpace *>(d_fes.get()));
   }
   else
#endif
   {
      h_fes = std::make_unique<FiniteElementSpace>(mesh, h_fec.get(),
                                                   hdim,
                                                   Ordering::byVDIM);
      d_fes = std::make_unique<FiniteElementSpace>(mesh, d_fec.get(), ddim,
                                                   Ordering::byVDIM);
      h = std::make_unique<ComplexGridFunction>(h_fes.get());
      d = std::make_unique<ComplexGridFunction>(d_fes.get());
   }
   mesh_sequence = mesh->GetSequence();
}

void ComplexMaxwellResidualFields::Update()
{
   MFEM_VERIFY(order > 0, "order must be positive.");
   if (!h || mesh_sequence != e.FESpace()->GetMesh()->GetSequence())
   {
      BuildSpaces();
   }
   if (epsilon_real_scalar)
   {
      BuildComplexCurlFlux(e, mu_inv, *h);
      BuildComplexElectricDisplacement(e, *epsilon_real_scalar,
                                       *epsilon_imag_scalar, *d);
   }
   else
   {
      BuildComplexCurlFlux(e, mu_inv, *h);
      BuildComplexElectricDisplacement(e, *epsilon_real, *epsilon_imag,
                                       *d);
   }
}

real_t MaxwellResidualEstimator::EpsilonMin(ElementTransformation &trans,
                                            const IntegrationPoint &ip) const
{
   if (epsilon) { return epsilon->Eval(trans, ip); }

   return CoefficientMinimum(epsilon, epsilon_matrix, trans, ip,
                             e.VectorDim(), "epsilon matrix");
}

void MaxwellResidualEstimator::ComputeEstimates()
{
   FiniteElementSpace *fes = e.FESpace();
   Mesh *mesh = fes->GetMesh();
   const FieldLayout layout = GetFieldLayout(e);
   const int dim = layout.mesh_dim;
   MFEM_VERIFY(j_src.FESpace()->GetMesh() == mesh,
               "e and j_src must share a mesh.");

   // The residual fields must be discontinuous: their interface jumps are
   // part of the indicator. Vector-valued L2 spaces also provide elementwise
   // strong curl and divergence through GridFunction.
   L2_FECollection h_fec(std::max(0, order - 1), dim);
   L2_FECollection d_fec(order, dim);
   const int hdim = fes->GetCurlDim();
   const int ddim = e.VectorDim();
   FiniteElementSpace h_fes(mesh, &h_fec, hdim, Ordering::byVDIM);
   FiniteElementSpace d_fes(mesh, &d_fec, ddim, Ordering::byVDIM);
   GridFunction h(&h_fes), d(&d_fes);
   const FieldLayout h_layout = GetFieldLayout(h);
   CurlGridFunctionCoefficient curl_e(&e);
   ScalarVectorProductCoefficient h_coef(mu_inv, curl_e);
   VectorGridFunctionCoefficient e_coef(&e);
   h.ProjectCoefficient(h_coef);
   if (epsilon)
   {
      ScalarVectorProductCoefficient d_coef(*epsilon, e_coef);
      d.ProjectCoefficient(d_coef);
   }
   else
   {
      MatrixVectorProductCoefficient d_coef(*epsilon_matrix, e_coef);
      d.ProjectCoefficient(d_coef);
   }

   const int ne = mesh->GetNE();
   error_estimates.SetSize(ne);
   error_estimates = 0.0;
   const int vector_dim = e.VectorDim();
   Vector f(vector_dim), curl_h(vector_dim), e_val(vector_dim), eps_e(vector_dim),
          h1(e.FESpace()->GetCurlDim()), h2(h1.Size()), d1(vector_dim),
          d2(vector_dim), normal(vector_dim), cross(vector_dim);

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
         j_src.GetVectorValue(*tr, ip, f);
         GetCurl(h, *tr, h_layout, curl_h);
         e.GetVectorValue(*tr, ip, e_val);
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
         const real_t div = j_src.GetDivergence(*tr)
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
         GetFaceNormal(*tr, vector_dim, normal);
         tangential_jump += TangentialJump(h, *tr, h_layout, normal, h1, h2, cross)
                            * ip.weight * tr->Face->Weight();
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
   const long sequence = e.FESpace()->GetMesh()->GetSequence();
   MFEM_ASSERT(sequence >= current_sequence, "improper mesh update sequence");
   return sequence > current_sequence;
}

real_t ComplexMaxwellResidualEstimator::EpsilonMin(
   ElementTransformation &trans, const IntegrationPoint &ip) const
{
   return CoefficientMinimum(epsilon_real_scalar, epsilon_real, trans, ip,
                             e.real().VectorDim(), "epsilon_real");
}

void ComplexMaxwellResidualEstimator::ComputeEstimates()
{
   FiniteElementSpace *fes = e.FESpace();
   Mesh *mesh = fes->GetMesh();
   MFEM_VERIFY(j_src.FESpace()->GetMesh() == mesh,
               "e and j_src must share a mesh.");

   std::unique_ptr<ComplexMaxwellResidualFields> fields;
   if (epsilon_real_scalar)
   {
      fields = std::make_unique<ComplexMaxwellResidualFields>(
                  e, *epsilon_real_scalar, *epsilon_imag_scalar, mu_inv, order);
   }
   else
   {
      fields = std::make_unique<ComplexMaxwellResidualFields>(
                  e, *epsilon_real, *epsilon_imag, mu_inv, order);
   }
   fields->Update();
   ComplexGridFunction &h = fields->CurlFlux();
   ComplexGridFunction &d = fields->D();
   const FieldLayout h_layout = GetFieldLayout(*h.FESpace(),
                                               h.FESpace()->GetVDim());

   const int ne = mesh->GetNE();
   error_estimates.SetSize(ne); error_estimates = 0.0;
   const int vector_dim = e.VectorDim();
   Vector fr(vector_dim), fi(vector_dim), cr(vector_dim), ci(vector_dim),
          er_val(vector_dim), ei_val(vector_dim), epsr(vector_dim), epsi(vector_dim),
          a(e.FESpace()->GetCurlDim()), b(a.Size()), n(vector_dim),
          cross(vector_dim);
   for (int el = 0; el < ne; el++)
   {
      ElementTransformation *tr = mesh->GetElementTransformation(el);
      const FiniteElement *fe = fes->GetFE(el);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               std::max(2 * fe->GetOrder() + 2, 2));
      const IntegrationPoint &center = Geometries.GetCenter(fe->GetGeomType());
      tr->SetIntPoint(&center);
      const real_t eps_min = EpsilonMin(*tr, center), mu = 1.0 / mu_inv.Eval(*tr,
                                                                             center);
      MFEM_VERIFY(eps_min > 0.0 && mu > 0.0,
                  "epsilon_real and mu_inv must be positive.");
      real_t curl_res = 0.0, div_res = 0.0;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q); tr->SetIntPoint(&ip);
         j_src.real().GetVectorValue(*tr, ip, fr);
         j_src.imag().GetVectorValue(*tr, ip, fi);
         e.real().GetVectorValue(*tr, ip, er_val);
         e.imag().GetVectorValue(*tr, ip, ei_val);
         if (epsilon_real_scalar)
         {
            epsr = er_val; epsr *= epsilon_real_scalar->Eval(*tr, ip);
            epsi = ei_val; epsi *= epsilon_imag_scalar->Eval(*tr, ip);
         }
         else
         {
            DenseMatrix ar, ai; epsilon_real->Eval(ar, *tr, ip);
            epsilon_imag->Eval(ai, *tr, ip);
            ar.Mult(er_val, epsr); ai.Mult(ei_val, epsi);
         }
         epsr -= epsi;
         fr.Add(omega * omega, epsr);
         GetCurl(h.real(), *tr, h_layout, cr); fr -= cr;
         if (epsilon_real_scalar)
         {
            epsr = ei_val; epsr *= epsilon_real_scalar->Eval(*tr, ip);
            epsi = er_val; epsi *= epsilon_imag_scalar->Eval(*tr, ip);
         }
         else
         {
            DenseMatrix ar, ai; epsilon_real->Eval(ar, *tr, ip);
            epsilon_imag->Eval(ai, *tr, ip);
            ar.Mult(ei_val, epsr); ai.Mult(er_val, epsi);
         }
         epsr += epsi;
         fi.Add(omega * omega, epsr);
         GetCurl(h.imag(), *tr, h_layout, ci); fi -= ci;
         curl_res += (fr * fr + fi * fi) * ip.weight * tr->Weight();
         const real_t div_r = j_src.real().GetDivergence(*tr) + omega * omega *
                              d.real().GetDivergence(*tr);
         const real_t div_i = j_src.imag().GetDivergence(*tr) + omega * omega *
                              d.imag().GetDivergence(*tr);
         div_res += (div_r * div_r + div_i * div_i) * ip.weight * tr->Weight();
      }
      const real_t he = mesh->GetElementSize(el);
      error_estimates(el) = mu * pow(he / order, 2) * curl_res
                            + pow(he / order, 2) * div_res / (omega * omega * eps_min);
   }
   // Face terms use the complex norm of the jumps of H and D=epsilon E.
   // A shared face contributes only to the locally owned element; its other
   // element belongs to the neighboring rank.
   auto add_face = [&](FaceElementTransformations *tr, const bool local_second)
   {
      const int e1 = tr->Elem1No, e2 = tr->Elem2No;
      const IntegrationRule &ir = IntRules.Get(tr->FaceGeom,
                                               std::max(2 * fes->GetFE(e1)->GetOrder() + 2, 2));
      real_t jt = 0.0, jn = 0.0;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q); tr->SetAllIntPoints(&ip);
         GetFaceNormal(*tr, vector_dim, n);
         jt += (TangentialJump(h.real(), *tr, h_layout, n, a, b, cross) +
                TangentialJump(h.imag(), *tr, h_layout, n, a, b, cross)) *
               ip.weight * tr->Face->Weight();
         d.real().GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), a);
         d.real().GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), b); a -= b;
         const real_t nr = n * a;
         d.imag().GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), a);
         d.imag().GetVectorValue(*tr->Elem2, tr->GetElement2IntPoint(), b); a -= b;
         jn += (nr * nr + pow(n * a, 2)) * ip.weight * tr->Face->Weight();
      }
      const int count = local_second ? 2 : 1;
      for (int side = 0; side < count; side++)
      {
         const int el = side ? e2 : e1;
         ElementTransformation *et = mesh->GetElementTransformation(el);
         const IntegrationPoint &c = Geometries.GetCenter(fes->GetFE(el)->GetGeomType());
         et->SetIntPoint(&c);
         const real_t eps = EpsilonMin(*et, c), mu = 1.0 / mu_inv.Eval(*et, c),
                      he = mesh->GetElementSize(el);
         error_estimates(el) += mu * he / order * jt + omega * omega * he /
                                (order * eps) * jn;
      }
   };
   for (int face = 0; face < mesh->GetNumFaces(); face++)
   {
      if (FaceElementTransformations *tr = mesh->GetInteriorFaceTransformations(face))
      {
         add_face(tr, true);
      }
   }
#ifdef MFEM_USE_MPI
   if (auto *pmesh = dynamic_cast<ParMesh *>(mesh))
   {
      auto *h_real = dynamic_cast<ParGridFunction *>(&h.real());
      auto *h_imag = dynamic_cast<ParGridFunction *>(&h.imag());
      auto *d_real = dynamic_cast<ParGridFunction *>(&d.real());
      auto *d_imag = dynamic_cast<ParGridFunction *>(&d.imag());
      MFEM_VERIFY(h_real && h_imag && d_real && d_imag,
                  "parallel Maxwell reconstructions must be ParGridFunctions.");
      h_real->ExchangeFaceNbrData(); h_imag->ExchangeFaceNbrData();
      d_real->ExchangeFaceNbrData(); d_imag->ExchangeFaceNbrData();
      pmesh->ExchangeFaceNbrData();
      for (int sf = 0; sf < pmesh->GetNSharedFaces(); sf++)
      {
         add_face(pmesh->GetSharedFaceTransformations(sf, true), false);
      }
   }
#endif
   for (int el = 0; el < ne; el++) { error_estimates(el) = sqrt(error_estimates(el)); }
   total_error = error_estimates.Norml2(); current_sequence = mesh->GetSequence();
}

void MaxwellResidualDomainEstimatorBase::Prepare(ErrorEstimatorContext &context)
{ if (fields) { context.Ensure(*fields); } }

MaxwellResidualCurlDomainEstimator::MaxwellResidualCurlDomainEstimator(
   GridFunction &e_, GridFunction &j_src_, GridFunction &h_,
   Coefficient &epsilon_, Coefficient &mu_inv_, real_t omega_, int order_)
   : MaxwellResidualDomainEstimatorBase(e_, j_src_), h(&h_),
     epsilon(&epsilon_), epsilon_matrix(nullptr), mu_inv(mu_inv_), omega(omega_),
     order(order_) { }

MaxwellResidualCurlDomainEstimator::MaxwellResidualCurlDomainEstimator(
   GridFunction &e_, GridFunction &j_src_, GridFunction &h_,
   MatrixCoefficient &epsilon_, Coefficient &mu_inv_, real_t omega_, int order_)
   : MaxwellResidualDomainEstimatorBase(e_, j_src_), h(&h_), epsilon(nullptr),
     epsilon_matrix(&epsilon_), mu_inv(mu_inv_), omega(omega_), order(order_) { }

MaxwellResidualCurlDomainEstimator::MaxwellResidualCurlDomainEstimator(
   GridFunction &e_, GridFunction &j_src_,
   std::shared_ptr<MaxwellResidualFields> fields_, Coefficient &epsilon_,
   Coefficient &mu_inv_, real_t omega_, int order_)
   : MaxwellResidualDomainEstimatorBase(e_, j_src_, std::move(fields_)),
     h(nullptr), epsilon(&epsilon_), epsilon_matrix(nullptr),
     mu_inv(mu_inv_), omega(omega_), order(order_)
{ MFEM_VERIFY(fields, "Maxwell residual fields must be provided."); }

MaxwellResidualCurlDomainEstimator::MaxwellResidualCurlDomainEstimator(
   GridFunction &e_, GridFunction &j_src_,
   std::shared_ptr<MaxwellResidualFields> fields_, MatrixCoefficient &epsilon_,
   Coefficient &mu_inv_, real_t omega_, int order_)
   : MaxwellResidualDomainEstimatorBase(e_, j_src_, std::move(fields_)),
     h(nullptr), epsilon(nullptr), epsilon_matrix(&epsilon_),
     mu_inv(mu_inv_), omega(omega_), order(order_)
{ MFEM_VERIFY(fields, "Maxwell residual fields must be provided."); }

void MaxwellResidualEstimatorBase::BuildCurlFlux(
   GridFunction &e, Coefficient &mu_inv, GridFunction &h)
{
   CurlGridFunctionCoefficient curl_e(&e);
   ScalarVectorProductCoefficient h_coef(mu_inv, curl_e);
   h.ProjectCoefficient(h_coef);
}

void MaxwellResidualEstimatorBase::BuildElectricDisplacement(
   GridFunction &e, Coefficient &epsilon, GridFunction &d)
{
   VectorGridFunctionCoefficient e_coef(&e);
   ScalarVectorProductCoefficient d_coef(epsilon, e_coef);
   d.ProjectCoefficient(d_coef);
}

void MaxwellResidualEstimatorBase::BuildElectricDisplacement(
   GridFunction &e, MatrixCoefficient &epsilon, GridFunction &d)
{
   VectorGridFunctionCoefficient e_coef(&e);
   MatrixVectorProductCoefficient d_coef(epsilon, e_coef);
   d.ProjectCoefficient(d_coef);
}

real_t MaxwellResidualCurlDomainEstimator::GetElementError(
   ElementTransformation &tr)
{
   const FiniteElement &el = *e.FESpace()->GetFE(tr.ElementNo);
   MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive.");
   GridFunction &h_field = fields ? fields->CurlFlux() : *h;
   const FieldLayout h_layout = GetFieldLayout(h_field);
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                            std::max(2 * el.GetOrder() + 2, 2));
   const IntegrationPoint &center = Geometries.GetCenter(el.GetGeomType());
   tr.SetIntPoint(&center);
   const real_t mu = 1.0 / mu_inv.Eval(tr, center);
   MFEM_VERIFY(mu > 0.0, "mu_inv must be positive.");
   const int vector_dim = e.VectorDim();
   Vector residual(vector_dim), curl_h(vector_dim), e_value(vector_dim),
          eps_e(vector_dim);
   real_t curl_term = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetIntPoint(&ip);
      j_src.GetVectorValue(tr, ip, residual);
      e.GetVectorValue(tr, ip, e_value);
      if (epsilon)
      {
         residual.Add(omega * omega * epsilon->Eval(tr, ip), e_value);
      }
      else
      {
         DenseMatrix eps_tensor; epsilon_matrix->Eval(eps_tensor, tr, ip);
         eps_tensor.Mult(e_value, eps_e); residual.Add(omega * omega, eps_e);
      }
      GetCurl(h_field, tr, h_layout, curl_h); residual -= curl_h;
      curl_term += (residual * residual) * ip.weight * tr.Weight();
   }
   const real_t he = e.FESpace()->GetMesh()->GetElementSize(tr.ElementNo, 0);
   return mu * pow(he / order, 2) * curl_term;
}

real_t MaxwellResidualDivergenceDomainEstimator::GetElementError(
   ElementTransformation &tr)
{
   const FiniteElement &el = *e.FESpace()->GetFE(tr.ElementNo);
   MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive.");
   GridFunction &d_field = fields ? fields->D() : *d;
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                            std::max(2 * el.GetOrder() + 2, 2));
   const IntegrationPoint &center = Geometries.GetCenter(el.GetGeomType());
   tr.SetIntPoint(&center);
   const real_t eps = EpsilonMin(tr, center);
   MFEM_VERIFY(eps > 0.0, "epsilon must be positive.");
   real_t div_term = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetIntPoint(&ip);
      const real_t div = j_src.GetDivergence(tr) + omega * omega *
                         d_field.GetDivergence(tr);
      div_term += div * div * ip.weight * tr.Weight();
   }
   const real_t he = e.FESpace()->GetMesh()->GetElementSize(tr.ElementNo, 0);
   return pow(he / order, 2) * div_term / (omega * omega * eps);
}

MaxwellResidualDivergenceDomainEstimator::MaxwellResidualDivergenceDomainEstimator(
   GridFunction &e_, GridFunction &j_src_, GridFunction &d_,
   Coefficient &epsilon_, real_t omega_, int order_)
   : MaxwellResidualDomainEstimatorBase(e_, j_src_), d(&d_),
     epsilon(&epsilon_), epsilon_matrix(nullptr), omega(omega_), order(order_) { }

MaxwellResidualDivergenceDomainEstimator::MaxwellResidualDivergenceDomainEstimator(
   GridFunction &e_, GridFunction &j_src_, GridFunction &d_,
   MatrixCoefficient &epsilon_, real_t omega_, int order_)
   : MaxwellResidualDomainEstimatorBase(e_, j_src_), d(&d_),
     epsilon(nullptr), epsilon_matrix(&epsilon_), omega(omega_), order(order_) { }

MaxwellResidualDivergenceDomainEstimator::MaxwellResidualDivergenceDomainEstimator(
   GridFunction &e_, GridFunction &j_src_,
   std::shared_ptr<MaxwellResidualFields> fields_, Coefficient &epsilon_,
   real_t omega_, int order_)
   : MaxwellResidualDomainEstimatorBase(e_, j_src_, std::move(fields_)),
     d(nullptr), epsilon(&epsilon_), epsilon_matrix(nullptr), omega(omega_),
     order(order_)
{ MFEM_VERIFY(fields, "Maxwell residual fields must be provided."); }

MaxwellResidualDivergenceDomainEstimator::MaxwellResidualDivergenceDomainEstimator(
   GridFunction &e_, GridFunction &j_src_,
   std::shared_ptr<MaxwellResidualFields> fields_, MatrixCoefficient &epsilon_,
   real_t omega_, int order_)
   : MaxwellResidualDomainEstimatorBase(e_, j_src_, std::move(fields_)),
     d(nullptr), epsilon(nullptr), epsilon_matrix(&epsilon_), omega(omega_),
     order(order_)
{ MFEM_VERIFY(fields, "Maxwell residual fields must be provided."); }

real_t MaxwellResidualDivergenceDomainEstimator::EpsilonMin(
   ElementTransformation &trans, const IntegrationPoint &ip) const
{
   return CoefficientMinimum(epsilon, epsilon_matrix, trans, ip, e.VectorDim(),
                             "epsilon matrix");
}

void MaxwellResidualFaceEstimatorBase::Prepare(ErrorEstimatorContext &context)
{ if (fields) { context.Ensure(*fields); } }

MaxwellResidualTangentialFaceEstimator::MaxwellResidualTangentialFaceEstimator(
   GridFunction &h_, Coefficient &mu_inv_, int order_)
   : MaxwellResidualFaceEstimatorBase(), h(&h_), mu_inv(mu_inv_), order(order_) { }

MaxwellResidualTangentialFaceEstimator::MaxwellResidualTangentialFaceEstimator(
   std::shared_ptr<MaxwellResidualFields> fields_, Coefficient &mu_inv_,
   int order_)
   : MaxwellResidualFaceEstimatorBase(std::move(fields_)), h(nullptr),
     mu_inv(mu_inv_), order(order_)
{ MFEM_VERIFY(fields, "Maxwell residual fields must be provided."); }

void MaxwellResidualTangentialFaceEstimator::GetFaceError(
   FaceElementTransformations &tr,
   real_t &error1, real_t &error2)
{
   MFEM_VERIFY(order > 0, "order must be positive.");
   GridFunction &h_field = fields ? fields->CurlFlux() : *h;
   const FiniteElement &el1 = *h_field.FESpace()->GetFE(tr.Elem1No);
   const IntegrationRule &ir = IntRules.Get(tr.FaceGeom,
                                            std::max(2 * el1.GetOrder() + 2, 2));
   const FieldLayout layout = GetFieldLayout(h_field);
   const int vector_dim = GetFieldLayout(h_field).mesh_dim == 2 &&
                          h_field.VectorDim() == 1 ? 2 : h_field.VectorDim();
   Vector h1(h_field.VectorDim()), h2(h1.Size()), normal(vector_dim),
          jump(vector_dim);
   real_t tangential = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetAllIntPoints(&ip);
      GetFaceNormal(tr, vector_dim, normal);
      tangential += TangentialJump(h_field, tr, layout, normal, h1, h2, jump) *
                    ip.weight * tr.Face->Weight();
   }
   auto contribution = [&](int el)
   {
      Mesh *mesh = h_field.FESpace()->GetMesh();
      ElementTransformation *et = NULL;
      real_t he = 0.0;
      if (el < mesh->GetNE())
      {
         et = mesh->GetElementTransformation(el);
         he = mesh->GetElementSize(el, 0);
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
      const IntegrationPoint &center = Geometries.GetCenter(h_field.FESpace()->GetFE(
                                                               el)->GetGeomType());
      et->SetIntPoint(&center);
      const real_t mu = 1.0 / mu_inv.Eval(*et, center);
      return mu * he / order * tangential;
   };
   error1 = contribution(tr.Elem1No);
   error2 = contribution(tr.Elem2No);
}

MaxwellResidualNormalFaceEstimator::MaxwellResidualNormalFaceEstimator(
   GridFunction &d_, Coefficient &epsilon_, real_t omega_, int order_)
   : MaxwellResidualFaceEstimatorBase(), d(&d_), epsilon(&epsilon_),
     epsilon_matrix(nullptr), omega(omega_), order(order_) { }

MaxwellResidualNormalFaceEstimator::MaxwellResidualNormalFaceEstimator(
   GridFunction &d_, MatrixCoefficient &epsilon_, real_t omega_, int order_)
   : MaxwellResidualFaceEstimatorBase(), d(&d_), epsilon(nullptr),
     epsilon_matrix(&epsilon_), omega(omega_), order(order_) { }

MaxwellResidualNormalFaceEstimator::MaxwellResidualNormalFaceEstimator(
   std::shared_ptr<MaxwellResidualFields> fields_, Coefficient &epsilon_,
   real_t omega_, int order_)
   : MaxwellResidualFaceEstimatorBase(std::move(fields_)), d(nullptr),
     epsilon(&epsilon_), epsilon_matrix(nullptr), omega(omega_), order(order_)
{ MFEM_VERIFY(fields, "Maxwell residual fields must be provided."); }

MaxwellResidualNormalFaceEstimator::MaxwellResidualNormalFaceEstimator(
   std::shared_ptr<MaxwellResidualFields> fields_, MatrixCoefficient &epsilon_,
   real_t omega_, int order_)
   : MaxwellResidualFaceEstimatorBase(std::move(fields_)), d(nullptr),
     epsilon(nullptr), epsilon_matrix(&epsilon_), omega(omega_), order(order_)
{ MFEM_VERIFY(fields, "Maxwell residual fields must be provided."); }

real_t MaxwellResidualNormalFaceEstimator::EpsilonMin(
   ElementTransformation &trans, const IntegrationPoint &ip) const
{
   const GridFunction &d_field = fields ? fields->D() : *d;
   return CoefficientMinimum(epsilon, epsilon_matrix, trans, ip,
                             d_field.VectorDim(), "epsilon matrix");
}

void MaxwellResidualNormalFaceEstimator::GetFaceError(FaceElementTransformations
                                                      &tr,
                                                      real_t &error1, real_t &error2)
{
   MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive.");
   GridFunction &d_field = fields ? fields->D() : *d;
   const FiniteElement &el1 = *d_field.FESpace()->GetFE(tr.Elem1No);
   const IntegrationRule &ir = IntRules.Get(tr.FaceGeom,
                                            std::max(2 * el1.GetOrder() + 2, 2));
   const int vector_dim = d_field.VectorDim();
   Vector d1(vector_dim), d2(vector_dim), normal(vector_dim);
   real_t normal_jump = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetAllIntPoints(&ip);
      GetFaceNormal(tr, vector_dim, normal);
      d_field.GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), d1);
      d_field.GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), d2); d1 -= d2;
      normal_jump += pow(normal * d1, 2) * ip.weight * tr.Face->Weight();
   }
   auto contribution = [&](int el)
   {
      Mesh *mesh = d_field.FESpace()->GetMesh();
      ElementTransformation *et = NULL;
      real_t he = 0.0;
      if (el < mesh->GetNE())
      {
         et = mesh->GetElementTransformation(el);
         he = mesh->GetElementSize(el, 0);
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
      const IntegrationPoint &center = Geometries.GetCenter(
                                          d_field.FESpace()->GetFE(el)->GetGeomType());
      et->SetIntPoint(&center);
      const real_t eps = EpsilonMin(*et, center);
      return omega * omega * he * normal_jump / (order * eps);
   };
   error1 = contribution(tr.Elem1No);
   error2 = contribution(tr.Elem2No);
}

void MaxwellResidualTangentialFaceEstimator::ExchangeFaceNbrData()
{
#ifdef MFEM_USE_MPI
   GridFunction &h_field = fields ? fields->CurlFlux() : *h;
   if (auto *ph = dynamic_cast<ParGridFunction*>(&h_field))
   {
      ph->ExchangeFaceNbrData();
   }
#endif
}

void MaxwellResidualNormalFaceEstimator::ExchangeFaceNbrData()
{
#ifdef MFEM_USE_MPI
   GridFunction &d_field = fields ? fields->D() : *d;
   if (auto *pd = dynamic_cast<ParGridFunction*>(&d_field))
   {
      pd->ExchangeFaceNbrData();
   }
#endif
}

WeightedFaceJumpErrorEstimatorBase::WeightedFaceJumpErrorEstimatorBase(
   GridFunction &x_, Coefficient *a_, MatrixCoefficient *a_matrix_,
   real_t alpha_, FaceJumpScaling scaling_)
   : x(x_), a(a_), a_matrix(a_matrix_), alpha(alpha_), scaling(scaling_)
{
   MFEM_VERIFY(alpha >= 0.0, "alpha must be non-negative.");
   if (a_matrix)
   {
      MFEM_VERIFY(a_matrix->GetHeight() == x.VectorDim() &&
                  a_matrix->GetWidth() == x.VectorDim(),
                  "a matrix dimensions must match the field vector dimension.");
   }
}

real_t WeightedFaceJumpErrorEstimatorBase::FaceScale(const int element) const
{
   if (scaling == FaceJumpScaling::NONE) { return alpha; }

   Mesh *mesh = x.FESpace()->GetMesh();
   ElementTransformation *tr = nullptr;
   real_t h = 0.0;
   if (element < mesh->GetNE())
   {
      tr = mesh->GetElementTransformation(element);
      h = mesh->GetElementSize(element, 0);
   }
#ifdef MFEM_USE_MPI
   else if (auto *pmesh = dynamic_cast<ParMesh *>(mesh))
   {
      const int face_nbr_element = element - mesh->GetNE();
      tr = pmesh->GetFaceNbrElementTransformation(face_nbr_element);
      h = pmesh->GetFaceNbrElementSize(face_nbr_element);
   }
#endif
   else { MFEM_ABORT("invalid element number in face-jump estimator"); }

   const int p = std::max(1, x.FESpace()->GetFE(element)->GetOrder());
   real_t scale = alpha * h / p;
   if (scaling == FaceJumpScaling::H_OVER_P_OVER_COEFFICIENT)
   {
      const IntegrationPoint &center = Geometries.GetCenter(
                                          x.FESpace()->GetFE(element)->GetGeomType());
      tr->SetIntPoint(&center);
      const real_t a_min = a || a_matrix ?
                           CoefficientMinimum(a, a_matrix, *tr, center,
                                              x.VectorDim(), "a matrix") : 1.0;
      MFEM_VERIFY(a_min > 0.0,
                  "coefficient-aware face-jump scaling requires a positive "
                  "scalar coefficient or SPD matrix coefficient.");
      scale /= a_min;
   }
   return scale;
}

void WeightedFaceJumpErrorEstimatorBase::ExchangeFaceNbrData()
{
#ifdef MFEM_USE_MPI
   if (auto *px = dynamic_cast<ParGridFunction *>(&x))
   {
      px->ExchangeFaceNbrData();
   }
#endif
}

NedelecNormalJumpErrorEstimator::NedelecNormalJumpErrorEstimator(
   GridFunction &x_, real_t alpha_, FaceJumpScaling scaling_)
   : WeightedFaceJumpErrorEstimatorBase(x_, nullptr, nullptr, alpha_, scaling_) { }

NedelecNormalJumpErrorEstimator::NedelecNormalJumpErrorEstimator(
   GridFunction &x_, Coefficient &a_, real_t alpha_, FaceJumpScaling scaling_)
   : WeightedFaceJumpErrorEstimatorBase(x_, &a_, nullptr, alpha_, scaling_) { }

NedelecNormalJumpErrorEstimator::NedelecNormalJumpErrorEstimator(
   GridFunction &x_, MatrixCoefficient &a_, real_t alpha_,
   FaceJumpScaling scaling_)
   : WeightedFaceJumpErrorEstimatorBase(x_, nullptr, &a_, alpha_, scaling_) { }

void NedelecNormalJumpErrorEstimator::GetFaceError(
   FaceElementTransformations &tr, real_t &error1, real_t &error2)
{
   const FiniteElement &el1 = *x.FESpace()->GetFE(tr.Elem1No);
   const int map_type = el1.GetMapType();
   MFEM_VERIFY(map_type == FiniteElement::H_CURL ||
               map_type == FiniteElement::H_CURL_R2D ||
               map_type == FiniteElement::H_CURL_R1D,
               "NedelecNormalJumpErrorEstimator requires an H(curl) field.");
   const IntegrationRule &ir = IntRules.Get(tr.FaceGeom,
                                            std::max(2 * el1.GetOrder() + 2, 2));
   const int vector_dim = x.VectorDim();
   Vector x1(vector_dim), x2(vector_dim), ax1(vector_dim), ax2(vector_dim),
          normal(vector_dim);
   DenseMatrix a1, a2;
   real_t jump_error = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      tr.SetAllIntPoints(&ip);
      GetFaceNormal(tr, vector_dim, normal);
      x.GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), x1);
      x.GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), x2);
      if (a)
      {
         ax1 = x1;
         ax1 *= a->Eval(*tr.Elem1, tr.GetElement1IntPoint());
         ax2 = x2;
         ax2 *= a->Eval(*tr.Elem2, tr.GetElement2IntPoint());
      }
      else if (a_matrix)
      {
         a_matrix->Eval(a1, *tr.Elem1, tr.GetElement1IntPoint());
         a1.Mult(x1, ax1);
         a_matrix->Eval(a2, *tr.Elem2, tr.GetElement2IntPoint());
         a2.Mult(x2, ax2);
      }
      else
      {
         ax1 = x1;
         ax2 = x2;
      }
      ax1 -= ax2;
      jump_error += pow(normal * ax1, 2) * ip.weight * tr.Face->Weight();
   }
   error1 = FaceScale(tr.Elem1No) * jump_error;
   error2 = FaceScale(tr.Elem2No) * jump_error;
}

RTTangentialJumpErrorEstimator::RTTangentialJumpErrorEstimator(
   GridFunction &x_, real_t alpha_, FaceJumpScaling scaling_)
   : WeightedFaceJumpErrorEstimatorBase(x_, nullptr, nullptr, alpha_, scaling_) { }

RTTangentialJumpErrorEstimator::RTTangentialJumpErrorEstimator(
   GridFunction &x_, Coefficient &a_, real_t alpha_, FaceJumpScaling scaling_)
   : WeightedFaceJumpErrorEstimatorBase(x_, &a_, nullptr, alpha_, scaling_) { }

RTTangentialJumpErrorEstimator::RTTangentialJumpErrorEstimator(
   GridFunction &x_, MatrixCoefficient &a_, real_t alpha_,
   FaceJumpScaling scaling_)
   : WeightedFaceJumpErrorEstimatorBase(x_, nullptr, &a_, alpha_, scaling_) { }

void RTTangentialJumpErrorEstimator::GetFaceError(
   FaceElementTransformations &tr, real_t &error1, real_t &error2)
{
   const FiniteElement &el1 = *x.FESpace()->GetFE(tr.Elem1No);
   const int map_type = el1.GetMapType();
   MFEM_VERIFY(map_type == FiniteElement::H_DIV ||
               map_type == FiniteElement::H_DIV_R2D ||
               map_type == FiniteElement::H_DIV_R1D,
               "RTTangentialJumpErrorEstimator requires an H(div) field.");
   const IntegrationRule &ir = IntRules.Get(tr.FaceGeom,
                                            std::max(2 * el1.GetOrder() + 2, 2));
   const int vector_dim = x.VectorDim();
   Vector x1(vector_dim), x2(vector_dim), ax1(vector_dim), ax2(vector_dim),
          normal(vector_dim);
   DenseMatrix a1, a2;
   real_t jump_error = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      tr.SetAllIntPoints(&ip);
      GetFaceNormal(tr, vector_dim, normal);
      x.GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), x1);
      x.GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), x2);
      if (a)
      {
         ax1 = x1;
         ax1 *= a->Eval(*tr.Elem1, tr.GetElement1IntPoint());
         ax2 = x2;
         ax2 *= a->Eval(*tr.Elem2, tr.GetElement2IntPoint());
      }
      else if (a_matrix)
      {
         a_matrix->Eval(a1, *tr.Elem1, tr.GetElement1IntPoint());
         a1.Mult(x1, ax1);
         a_matrix->Eval(a2, *tr.Elem2, tr.GetElement2IntPoint());
         a2.Mult(x2, ax2);
      }
      else
      {
         ax1 = x1;
         ax2 = x2;
      }
      ax1 -= ax2;
      jump_error += TangentialComponentSquared(normal, ax1) * ip.weight *
                    tr.Face->Weight();
   }
   error1 = FaceScale(tr.Elem1No) * jump_error;
   error2 = FaceScale(tr.Elem2No) * jump_error;
}

ComplexMaxwellResidualDomainEstimatorBase::
ComplexMaxwellResidualDomainEstimatorBase(
   ComplexGridFunction &s, ComplexGridFunction &j,
   ComplexGridFunction &h_, ComplexGridFunction &d_,
   MatrixCoefficient &er, MatrixCoefficient &ei,
   Coefficient &mu, real_t w, int p)
   : e(s), j_src(j), h(&h_), d(&d_), epsilon_real(&er), epsilon_imag(&ei),
     epsilon_real_scalar(nullptr), epsilon_imag_scalar(nullptr), mu_inv(mu),
     omega(w), order(p) { }

ComplexMaxwellResidualDomainEstimatorBase::
ComplexMaxwellResidualDomainEstimatorBase(
   ComplexGridFunction &s, ComplexGridFunction &j,
   ComplexGridFunction &h_, ComplexGridFunction &d_,
   Coefficient &er, Coefficient &ei, Coefficient &mu,
   real_t w, int p)
   : e(s), j_src(j), h(&h_), d(&d_), epsilon_real(nullptr),
     epsilon_imag(nullptr),
     epsilon_real_scalar(&er), epsilon_imag_scalar(&ei), mu_inv(mu), omega(w),
     order(p) { }

ComplexMaxwellResidualDomainEstimatorBase::
ComplexMaxwellResidualDomainEstimatorBase(
   ComplexGridFunction &s, ComplexGridFunction &j,
   std::shared_ptr<ComplexMaxwellResidualFields> fields_,
   MatrixCoefficient &er, MatrixCoefficient &ei, Coefficient &mu, real_t w, int p)
   : e(s), j_src(j), h(nullptr), d(nullptr), fields(std::move(fields_)),
     epsilon_real(&er), epsilon_imag(&ei), epsilon_real_scalar(nullptr),
     epsilon_imag_scalar(nullptr), mu_inv(mu), omega(w), order(p)
{ MFEM_VERIFY(fields, "Complex Maxwell residual fields must be provided."); }

ComplexMaxwellResidualDomainEstimatorBase::
ComplexMaxwellResidualDomainEstimatorBase(
   ComplexGridFunction &s, ComplexGridFunction &j,
   std::shared_ptr<ComplexMaxwellResidualFields> fields_, Coefficient &er,
   Coefficient &ei, Coefficient &mu, real_t w, int p)
   : e(s), j_src(j), h(nullptr), d(nullptr), fields(std::move(fields_)),
     epsilon_real(nullptr), epsilon_imag(nullptr), epsilon_real_scalar(&er),
     epsilon_imag_scalar(&ei), mu_inv(mu), omega(w), order(p)
{ MFEM_VERIFY(fields, "Complex Maxwell residual fields must be provided."); }

void ComplexMaxwellResidualDomainEstimatorBase::Prepare(ErrorEstimatorContext
                                                        &context)
{ if (fields) { context.Ensure(*fields); } }

real_t ComplexMaxwellResidualDomainEstimatorBase::EpsilonMin(
   ElementTransformation &tr, const IntegrationPoint &ip) const
{
   return CoefficientMinimum(epsilon_real_scalar, epsilon_real, tr, ip,
                             e.real().VectorDim(), "epsilon_real");
}

void MaxwellResidualEstimatorBase::BuildComplexCurlFlux(
   ComplexGridFunction &e, Coefficient &mu_inv, ComplexGridFunction &h)
{
   CurlGridFunctionCoefficient cer(&e.real()), cei(&e.imag());
   ScalarVectorProductCoefficient hr(mu_inv, cer), hi(mu_inv, cei);
#ifdef MFEM_USE_MPI
   if (auto *ph = dynamic_cast<ParComplexGridFunction *>(&h))
   {
      ph->ProjectCoefficient(hr, hi);
      return;
   }
#endif
   auto *sh = dynamic_cast<ComplexGridFunction *>(&h);
   MFEM_VERIFY(sh, "complex Maxwell residual fields have an invalid type.");
   sh->ProjectCoefficient(hr, hi);
}

void MaxwellResidualEstimatorBase::BuildComplexElectricDisplacement(
   ComplexGridFunction &e, MatrixCoefficient &er, MatrixCoefficient &ei,
   ComplexGridFunction &d)
{
#ifdef MFEM_USE_MPI
   ParComplexGridFunction *pd = dynamic_cast<ParComplexGridFunction *>(&d);
   if (pd) { pd->Sync(); }
#endif
   VectorGridFunctionCoefficient vr(&e.real()), vi(&e.imag());
   MatrixVectorProductCoefficient er_r(er, vr), er_i(er, vi), ei_r(ei, vr),
                                  ei_i(ei, vi);
   d.real().ProjectCoefficient(er_r);
   d.imag().ProjectCoefficient(er_i);
   std::unique_ptr<GridFunction> tmp;
#ifdef MFEM_USE_MPI
   if (auto *pfes = dynamic_cast<ParFiniteElementSpace *>(d.FESpace()))
   {
      tmp = std::make_unique<ParGridFunction>(pfes);
   }
   else
#endif
   {
      tmp = std::make_unique<GridFunction>(d.FESpace());
   }
   tmp->ProjectCoefficient(ei_i); d.real() -= *tmp;
   tmp->ProjectCoefficient(ei_r); d.imag() += *tmp;
#ifdef MFEM_USE_MPI
   if (pd) { pd->SyncAlias(); }
#endif
}

void MaxwellResidualEstimatorBase::BuildComplexElectricDisplacement(
   ComplexGridFunction &e, Coefficient &er, Coefficient &ei,
   ComplexGridFunction &d)
{
#ifdef MFEM_USE_MPI
   ParComplexGridFunction *pd = dynamic_cast<ParComplexGridFunction *>(&d);
   if (pd) { pd->Sync(); }
#endif
   VectorGridFunctionCoefficient vr(&e.real()), vi(&e.imag());
   ScalarVectorProductCoefficient er_r(er, vr), er_i(er, vi), ei_r(ei, vr),
                                  ei_i(ei, vi);
   d.real().ProjectCoefficient(er_r);
   d.imag().ProjectCoefficient(er_i);
   std::unique_ptr<GridFunction> tmp;
#ifdef MFEM_USE_MPI
   if (auto *pfes = dynamic_cast<ParFiniteElementSpace *>(d.FESpace()))
   {
      tmp = std::make_unique<ParGridFunction>(pfes);
   }
   else
#endif
   {
      tmp = std::make_unique<GridFunction>(d.FESpace());
   }
   tmp->ProjectCoefficient(ei_i); d.real() -= *tmp;
   tmp->ProjectCoefficient(ei_r); d.imag() += *tmp;
#ifdef MFEM_USE_MPI
   if (pd) { pd->SyncAlias(); }
#endif
}

real_t ComplexMaxwellResidualCurlDomainEstimator::GetElementError(
   ElementTransformation &tr)
{
   const FiniteElement &el = *e.FESpace()->GetFE(tr.ElementNo);
   ComplexGridFunction &h_field = fields ? fields->CurlFlux() : *h;
   const FieldLayout h_layout = GetFieldLayout(h_field.real());
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                            std::max(2 * el.GetOrder() + 2, 2));
   const IntegrationPoint &c = Geometries.GetCenter(el.GetGeomType());
   tr.SetIntPoint(&c);
   const real_t mu = 1.0 / mu_inv.Eval(tr, c);
   MFEM_VERIFY(mu > 0.0, "mu_inv must be positive.");
   const int vector_dim = e.real().VectorDim();
   Vector rr(vector_dim), ri(vector_dim), cr(vector_dim), ci(vector_dim),
          evr(vector_dim), evi(vector_dim), a(vector_dim), b(vector_dim);
   real_t curl_term = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetIntPoint(&ip);
      j_src.real().GetVectorValue(tr, ip, rr);
      j_src.imag().GetVectorValue(tr, ip, ri);
      e.real().GetVectorValue(tr, ip, evr);
      e.imag().GetVectorValue(tr, ip, evi);
      if (epsilon_real_scalar)
      {
         a = evr; a *= epsilon_real_scalar->Eval(tr, ip); b = evi;
         b *= epsilon_imag_scalar->Eval(tr, ip);
      }
      else { DenseMatrix ar, ai; epsilon_real->Eval(ar, tr, ip); epsilon_imag->Eval(ai, tr, ip); ar.Mult(evr, a); ai.Mult(evi, b); }
      a -= b; rr.Add(omega * omega, a);
      GetCurl(h_field.real(), tr, h_layout, cr); rr -= cr;
      if (epsilon_real_scalar)
      {
         a = evi; a *= epsilon_real_scalar->Eval(tr, ip); b = evr;
         b *= epsilon_imag_scalar->Eval(tr, ip);
      }
      else { DenseMatrix ar, ai; epsilon_real->Eval(ar, tr, ip); epsilon_imag->Eval(ai, tr, ip); ar.Mult(evi, a); ai.Mult(evr, b); }
      a += b; ri.Add(omega * omega, a);
      GetCurl(h_field.imag(), tr, h_layout, ci); ri -= ci;
      curl_term += (rr * rr + ri * ri) * ip.weight * tr.Weight();
   }
   const real_t he = e.FESpace()->GetMesh()->GetElementSize(tr.ElementNo);
   return mu * pow(he / order, 2) * curl_term;
}

real_t ComplexMaxwellResidualDivergenceDomainEstimator::GetElementError(
   ElementTransformation &tr)
{
   const FiniteElement &el = *e.FESpace()->GetFE(tr.ElementNo);
   MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive.");
   ComplexGridFunction &d_field = fields ? fields->D() : *d;
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                            std::max(2 * el.GetOrder() + 2, 2));
   const IntegrationPoint &center = Geometries.GetCenter(el.GetGeomType());
   tr.SetIntPoint(&center);
   const real_t eps = EpsilonMin(tr, center);
   MFEM_VERIFY(eps > 0.0, "epsilon_real must be positive.");
   real_t div_term = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetIntPoint(&ip);
      const real_t dr = j_src.real().GetDivergence(tr) + omega * omega *
                        d_field.real().GetDivergence(tr);
      const real_t di = j_src.imag().GetDivergence(tr) + omega * omega *
                        d_field.imag().GetDivergence(tr);
      div_term += (dr * dr + di * di) * ip.weight * tr.Weight();
   }
   const real_t he = e.FESpace()->GetMesh()->GetElementSize(tr.ElementNo);
   return pow(he / order, 2) * div_term / (omega * omega * eps);
}

void ComplexMaxwellResidualFaceEstimatorBase::Prepare(
   ErrorEstimatorContext &context)
{ if (fields) { context.Ensure(*fields); } }

void ComplexMaxwellResidualFaceEstimatorBase::ExchangeFieldFaceNbrData(
   ComplexGridFunction &field)
{
#ifdef MFEM_USE_MPI
   if (auto *real = dynamic_cast<ParGridFunction *>(&field.real()))
   {
      real->ExchangeFaceNbrData();
   }
   if (auto *imag = dynamic_cast<ParGridFunction *>(&field.imag()))
   {
      imag->ExchangeFaceNbrData();
   }
#endif
}

ComplexMaxwellResidualTangentialFaceEstimator::
ComplexMaxwellResidualTangentialFaceEstimator(ComplexGridFunction &h_,
                                              Coefficient &mu_inv_, int order_)
   : ComplexMaxwellResidualFaceEstimatorBase(&h_, nullptr), mu_inv(mu_inv_),
     order(order_) { }

ComplexMaxwellResidualTangentialFaceEstimator::
ComplexMaxwellResidualTangentialFaceEstimator(
   std::shared_ptr<ComplexMaxwellResidualFields> fields_, Coefficient &mu_inv_,
   int order_)
   : ComplexMaxwellResidualFaceEstimatorBase(std::move(fields_)), mu_inv(mu_inv_),
     order(order_) { }

void ComplexMaxwellResidualTangentialFaceEstimator::GetFaceError(
   FaceElementTransformations &tr,
   real_t &error1, real_t &error2)
{
   MFEM_VERIFY(order > 0, "order must be positive.");
   ComplexGridFunction &h_field = CurlFlux();
   const FiniteElement &el1 = *h_field.FESpace()->GetFE(tr.Elem1No);
   const IntegrationRule &ir = IntRules.Get(tr.FaceGeom,
                                            std::max(2 * el1.GetOrder() + 2, 2));
   const FieldLayout layout = GetFieldLayout(h_field.real());
   const int vector_dim = layout.mesh_dim == 2 &&
                          h_field.real().VectorDim() == 1 ? 2 :
                          h_field.real().VectorDim();
   Vector a(h_field.real().VectorDim()), b(a.Size()), n(vector_dim),
          jr(vector_dim);
   real_t tangential = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetAllIntPoints(&ip);
      GetFaceNormal(tr, vector_dim, n);
      tangential += (TangentialJump(h_field.real(), tr, layout, n, a, b, jr) +
                     TangentialJump(h_field.imag(), tr, layout, n, a, b, jr)) *
                    ip.weight * tr.Face->Weight();
   }
   auto contribution = [&](int el)
   {
      Mesh *mesh = h_field.FESpace()->GetMesh();
      ElementTransformation *et = NULL;
      real_t he = 0.0;
      if (el < mesh->GetNE())
      {
         et = mesh->GetElementTransformation(el);
         he = mesh->GetElementSize(el, 0);
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
      const IntegrationPoint &c = Geometries.GetCenter(h_field.FESpace()->GetFE(
                                                          el)->GetGeomType()); et->SetIntPoint(&c);
      const real_t mu = 1.0 / mu_inv.Eval(*et, c);
      return mu * he / order * tangential;
   };
   error1 = contribution(tr.Elem1No); error2 = contribution(tr.Elem2No);
}

void ComplexMaxwellResidualTangentialFaceEstimator::ExchangeFaceNbrData()
{ ExchangeFieldFaceNbrData(CurlFlux()); }

ComplexMaxwellResidualNormalFaceEstimator::
ComplexMaxwellResidualNormalFaceEstimator(ComplexGridFunction &d_,
                                          MatrixCoefficient &epsilon_real_,
                                          real_t omega_, int order_)
   : ComplexMaxwellResidualFaceEstimatorBase(nullptr, &d_),
     epsilon_real(&epsilon_real_), epsilon_real_scalar(nullptr), omega(omega_),
     order(order_) { }

ComplexMaxwellResidualNormalFaceEstimator::
ComplexMaxwellResidualNormalFaceEstimator(ComplexGridFunction &d_,
                                          Coefficient &epsilon_real_,
                                          real_t omega_, int order_)
   : ComplexMaxwellResidualFaceEstimatorBase(nullptr, &d_),
     epsilon_real(nullptr), epsilon_real_scalar(&epsilon_real_), omega(omega_),
     order(order_) { }

ComplexMaxwellResidualNormalFaceEstimator::
ComplexMaxwellResidualNormalFaceEstimator(
   std::shared_ptr<ComplexMaxwellResidualFields> fields_,
   MatrixCoefficient &epsilon_real_, real_t omega_, int order_)
   : ComplexMaxwellResidualFaceEstimatorBase(std::move(fields_)),
     epsilon_real(&epsilon_real_), epsilon_real_scalar(nullptr), omega(omega_),
     order(order_) { }

ComplexMaxwellResidualNormalFaceEstimator::
ComplexMaxwellResidualNormalFaceEstimator(
   std::shared_ptr<ComplexMaxwellResidualFields> fields_,
   Coefficient &epsilon_real_, real_t omega_, int order_)
   : ComplexMaxwellResidualFaceEstimatorBase(std::move(fields_)),
     epsilon_real(nullptr), epsilon_real_scalar(&epsilon_real_), omega(omega_),
     order(order_) { }

real_t ComplexMaxwellResidualNormalFaceEstimator::EpsilonMin(
   ElementTransformation &tr, const IntegrationPoint &ip) const
{
   return CoefficientMinimum(epsilon_real_scalar, epsilon_real, tr, ip,
                             D().real().VectorDim(), "epsilon_real");
}

void ComplexMaxwellResidualNormalFaceEstimator::GetFaceError(
   FaceElementTransformations &tr,
   real_t &error1, real_t &error2)
{
   MFEM_VERIFY(omega > 0.0 && order > 0, "omega and order must be positive.");
   ComplexGridFunction &d_field = D();
   const FiniteElement &el1 = *d_field.FESpace()->GetFE(tr.Elem1No);
   const IntegrationRule &ir = IntRules.Get(tr.FaceGeom,
                                            std::max(2 * el1.GetOrder() + 2, 2));
   const int vector_dim = d_field.real().VectorDim();
   Vector d1(vector_dim), d2(vector_dim), normal(vector_dim);
   real_t normal_jump = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q); tr.SetAllIntPoints(&ip);
      GetFaceNormal(tr, vector_dim, normal);
      d_field.real().GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), d1);
      d_field.real().GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), d2);
      d1 -= d2;
      const real_t real_jump = normal * d1;
      d_field.imag().GetVectorValue(*tr.Elem1, tr.GetElement1IntPoint(), d1);
      d_field.imag().GetVectorValue(*tr.Elem2, tr.GetElement2IntPoint(), d2);
      d1 -= d2;
      const real_t imag_jump = normal * d1;
      normal_jump += (real_jump * real_jump + imag_jump * imag_jump) * ip.weight *
                     tr.Face->Weight();
   }
   auto contribution = [&](int el)
   {
      Mesh *mesh = d_field.FESpace()->GetMesh();
      ElementTransformation *et = NULL;
      real_t he = 0.0;
      if (el < mesh->GetNE())
      {
         et = mesh->GetElementTransformation(el);
         he = mesh->GetElementSize(el, 0);
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
      const IntegrationPoint &center = Geometries.GetCenter(
                                          d_field.FESpace()->GetFE(el)->GetGeomType());
      et->SetIntPoint(&center);
      const real_t eps = EpsilonMin(*et, center);
      return omega * omega * he * normal_jump / (order * eps);
   };
   error1 = contribution(tr.Elem1No);
   error2 = contribution(tr.Elem2No);
}

void ComplexMaxwellResidualNormalFaceEstimator::ExchangeFaceNbrData()
{ ExchangeFieldFaceNbrData(D()); }

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
      error += (trace_r * trace_r + trace_i * trace_i) * ip.weight *
               tr.Face->Weight();
   }
   return error;
}
}

real_t ComplexMaxwellDirichletBCErrorEstimator::GetFaceError(
   FaceElementTransformations &tr)
{
   const FiniteElement &el = *electric.FESpace()->GetFE(tr.Elem1No);
   return ComplexTangentialTraceError(electric, data_real, data_imag, el, tr);
}

real_t ComplexMaxwellNeumannBCErrorEstimator::GetFaceError(
   FaceElementTransformations &tr)
{
   const FiniteElement &el = *magnetic_flux.FESpace()->GetFE(tr.Elem1No);
   return ComplexTangentialTraceError(magnetic_flux, data_real, data_imag, el, tr);
}

void AddMaxwellResidualEstimators(GeneralErrorEstimator &estimator,
                                  GridFunction &e, GridFunction &j_src,
                                  Coefficient &epsilon, Coefficient &mu_inv,
                                  real_t omega, int order)
{
   auto fields = std::make_shared<MaxwellResidualFields>(e, epsilon,
                                                         mu_inv, order);
   estimator.AddDomainEstimator(new MaxwellResidualCurlDomainEstimator(
                                   e, j_src, fields, epsilon, mu_inv,
                                   omega, order));
   estimator.AddDomainEstimator(new MaxwellResidualDivergenceDomainEstimator(
                                   e, j_src, fields, epsilon, omega, order));
   estimator.AddInteriorFaceEstimator(new MaxwellResidualTangentialFaceEstimator(
                                         fields, mu_inv, order));
   estimator.AddInteriorFaceEstimator(new MaxwellResidualNormalFaceEstimator(
                                         fields, epsilon, omega, order));
}

void AddMaxwellResidualEstimators(GeneralErrorEstimator &estimator,
                                  GridFunction &e, GridFunction &j_src,
                                  MatrixCoefficient &epsilon, Coefficient &mu_inv,
                                  real_t omega, int order)
{
   auto fields = std::make_shared<MaxwellResidualFields>(e, epsilon,
                                                         mu_inv, order);
   estimator.AddDomainEstimator(new MaxwellResidualCurlDomainEstimator(
                                   e, j_src, fields, epsilon, mu_inv,
                                   omega, order));
   estimator.AddDomainEstimator(new MaxwellResidualDivergenceDomainEstimator(
                                   e, j_src, fields, epsilon, omega, order));
   estimator.AddInteriorFaceEstimator(new MaxwellResidualTangentialFaceEstimator(
                                         fields, mu_inv, order));
   estimator.AddInteriorFaceEstimator(new MaxwellResidualNormalFaceEstimator(
                                         fields, epsilon, omega, order));
}

void AddComplexMaxwellResidualEstimators(
   GeneralErrorEstimator &estimator, ComplexGridFunction &e,
   ComplexGridFunction &j_src, MatrixCoefficient &epsilon_real,
   MatrixCoefficient &epsilon_imag, Coefficient &mu_inv, real_t omega, int order)
{
   auto fields = std::make_shared<ComplexMaxwellResidualFields>(
                    e, epsilon_real, epsilon_imag, mu_inv, order);
   estimator.AddDomainEstimator(new ComplexMaxwellResidualCurlDomainEstimator(
                                   e, j_src, fields, epsilon_real,
                                   epsilon_imag, mu_inv, omega, order));
   estimator.AddDomainEstimator(
      new ComplexMaxwellResidualDivergenceDomainEstimator(
         e, j_src, fields, epsilon_real, epsilon_imag, mu_inv, omega, order));
   estimator.AddInteriorFaceEstimator(
      new ComplexMaxwellResidualTangentialFaceEstimator(
         fields, mu_inv, order));
   estimator.AddInteriorFaceEstimator(
      new ComplexMaxwellResidualNormalFaceEstimator(
         fields, epsilon_real, omega, order));
}

void AddComplexMaxwellResidualEstimators(
   GeneralErrorEstimator &estimator, ComplexGridFunction &e,
   ComplexGridFunction &j_src, Coefficient &epsilon_real,
   Coefficient &epsilon_imag, Coefficient &mu_inv, real_t omega, int order)
{
   auto fields = std::make_shared<ComplexMaxwellResidualFields>(
                    e, epsilon_real, epsilon_imag, mu_inv, order);
   estimator.AddDomainEstimator(new ComplexMaxwellResidualCurlDomainEstimator(
                                   e, j_src, fields, epsilon_real,
                                   epsilon_imag, mu_inv, omega, order));
   estimator.AddDomainEstimator(
      new ComplexMaxwellResidualDivergenceDomainEstimator(
         e, j_src, fields, epsilon_real, epsilon_imag, mu_inv, omega, order));
   estimator.AddInteriorFaceEstimator(
      new ComplexMaxwellResidualTangentialFaceEstimator(
         fields, mu_inv, order));
   estimator.AddInteriorFaceEstimator(
      new ComplexMaxwellResidualNormalFaceEstimator(
         fields, epsilon_real, omega, order));
}

} // namespace mfem
