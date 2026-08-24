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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace bilininteg_hdg
{

typedef NonlinearFormIntegrator NLFI;

/// A mesh with one interior face and a boundary, plus the element and trace
/// spaces the HDG integrators are assembled between.
struct Faces
{
   Mesh mesh;
   L2_FECollection el_coll;
   DG_Interface_FECollection tr_coll;
   FiniteElementSpace fes_el, fes_tr;

   Faces(int dim, int order, Element::Type type)
      : mesh((dim == 2)
             ? Mesh::MakeCartesian2D(2, 2, type, false, 1.0, 1.0)
             : Mesh::MakeCartesian3D(2, 2, 2, type, 1.0, 1.0, 1.0)),
        el_coll(order, dim),
        tr_coll(order, dim),
        fes_el(&mesh, &el_coll),
        fes_tr(&mesh, &tr_coll)
   { }

   /// The first interior face, and the first boundary face.
   int InteriorFace() const
   {
      for (int f = 0; f < mesh.GetNumFaces(); f++)
      {
         if (mesh.FaceIsInterior(f)) { return f; }
      }
      return -1;
   }
};

/// Deterministic, non-symmetric test data -- a constant would satisfy far too
/// many identities by accident.
void FillVarying(Vector &v, real_t shift)
{
   for (int i = 0; i < v.Size(); i++)
   {
      v(i) = std::sin(1.7 * i + shift) + 0.5 * std::cos(0.3 * i);
   }
}

/// Apply the blocks of a one-sided HDG face matrix to (elfun, trfun) exactly
/// as the HDGFaceType mask prescribes. This is the contract that the
/// integrators' own AssembleHDGFaceVector must satisfy; it is written out here
/// rather than delegated so that the test does not depend on the very code
/// path it is checking.
void ApplyBlocks(const DenseMatrix &elmat, int type, int ndof_el,
                 int ndof_tr, const Vector &elfun, const Vector &trfun,
                 Vector &out)
{
   int n = 0;
   if (type & (NLFI::ELEM | NLFI::TRACE))   { n += ndof_el; }
   if (type & (NLFI::CONSTR | NLFI::FACE))  { n += ndof_tr; }

   out.SetSize(n);
   out = 0.0;

   int ioff = 0;
   if (type & NLFI::ELEM)
   {
      for (int i = 0; i < ndof_el; i++)
         for (int j = 0; j < ndof_el; j++)
         {
            out(ioff + i) += elmat(i, j) * elfun(j);
         }
   }
   if (type & NLFI::TRACE)
   {
      for (int i = 0; i < ndof_el; i++)
         for (int j = 0; j < ndof_tr; j++)
         {
            out(ioff + i) += elmat(i, ndof_el + j) * trfun(j);
         }
   }
   if (type & (NLFI::ELEM | NLFI::TRACE)) { ioff += ndof_el; }

   if (type & NLFI::CONSTR)
   {
      for (int i = 0; i < ndof_tr; i++)
         for (int j = 0; j < ndof_el; j++)
         {
            out(ioff + i) += elmat(ndof_el + i, j) * elfun(j);
         }
   }
   if (type & NLFI::FACE)
   {
      for (int i = 0; i < ndof_tr; i++)
         for (int j = 0; j < ndof_tr; j++)
         {
            out(ioff + i) += elmat(ndof_el + i, ndof_el + j) * trfun(j);
         }
   }
}

/// Check the integrator's specialized face-vector routine against its own
/// face matrix, over every combination of the HDGFaceType mask.
void CheckVectorAgainstMatrix(BilinearFormIntegrator &integ,
                              const FiniteElement &tr_fe,
                              const FiniteElement &el_fe,
                              FaceElementTransformations &Tr,
                              int side)
{
   const int ndof_el = el_fe.GetDof();
   const int ndof_tr = tr_fe.GetDof();

   Vector elfun(ndof_el), trfun(ndof_tr);
   FillVarying(elfun, 0.0);
   FillVarying(trfun, 1.1);

   DenseMatrix elmat;
   integ.AssembleHDGFaceMatrix(side, tr_fe, el_fe, Tr, elmat);
   REQUIRE(elmat.Height() == ndof_el + ndof_tr);
   REQUIRE(elmat.Width() == ndof_el + ndof_tr);

   const int bits[4] = {NLFI::ELEM, NLFI::TRACE, NLFI::CONSTR, NLFI::FACE};

   for (int mask = 1; mask < 16; mask++)
   {
      int type = side & 1;
      for (int b = 0; b < 4; b++)
      {
         if (mask & (1 << b)) { type |= bits[b]; }
      }

      Vector expected;
      ApplyBlocks(elmat, type, ndof_el, ndof_tr, elfun, trfun, expected);

      Vector actual;
      integ.AssembleHDGFaceVector(type, tr_fe, el_fe, Tr, trfun, elfun, actual);

      CAPTURE(side, mask, type);
      REQUIRE(actual.Size() == expected.Size());
      for (int i = 0; i < expected.Size(); i++)
      {
         CAPTURE(i, actual(i), expected(i));
         REQUIRE(actual(i) == MFEM_Approx(expected(i), 1e-11, 1e-10));
      }
   }
}

} // namespace bilininteg_hdg

TEST_CASE("HDG face integrators: vector action matches the face matrix",
          "[HDGIntegrator]")
{
   using namespace bilininteg_hdg;

   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   const Element::Type type = (dim == 2) ? Element::QUADRILATERAL
                              : Element::HEXAHEDRON;

   CAPTURE(dim, order);

   Faces fx(dim, order, type);
   const int f = fx.InteriorFace();
   REQUIRE(f >= 0);

   FaceElementTransformations *Tr = fx.mesh.GetFaceElementTransformations(f);
   REQUIRE(Tr->Elem2No >= 0);

   const FiniteElement &tr_fe = *fx.fes_tr.GetFaceElement(f);

   Vector vel(dim);
   vel = 0.0;
   vel(0) = 1.3;
   if (dim > 1) { vel(1) = -0.7; }
   VectorConstantCoefficient vcoeff(vel);
   ConstantCoefficient q(2.5);

   DenseMatrix Q(dim);
   Q = 0.0;
   for (int d = 0; d < dim; d++) { Q(d, d) = 1.0 + d; }
   Q(0, 1) = Q(1, 0) = 0.3;
   MatrixConstantCoefficient mq(Q);

   for (int side = 0; side < 2; side++)
   {
      const int elno = (side == 0) ? Tr->Elem1No : Tr->Elem2No;
      const FiniteElement &el_fe = *fx.fes_el.GetFE(elno);

      SECTION("diffusion, scalar coefficient")
      {
         HDGDiffusionIntegrator integ(q);
         CheckVectorAgainstMatrix(integ, tr_fe, el_fe, *Tr, side);
      }
      SECTION("diffusion, matrix coefficient")
      {
         HDGDiffusionIntegrator integ(mq);
         CheckVectorAgainstMatrix(integ, tr_fe, el_fe, *Tr, side);
      }
      SECTION("diffusion with an advective direction")
      {
         HDGDiffusionIntegrator integ(vcoeff, q);
         CheckVectorAgainstMatrix(integ, tr_fe, el_fe, *Tr, side);
      }
      SECTION("convection, centered")
      {
         HDGConvectionCenteredIntegrator integ(vcoeff);
         CheckVectorAgainstMatrix(integ, tr_fe, el_fe, *Tr, side);
      }
      SECTION("convection, upwinded")
      {
         HDGConvectionUpwindedIntegrator integ(vcoeff);
         CheckVectorAgainstMatrix(integ, tr_fe, el_fe, *Tr, side);
      }
   }
}

TEST_CASE("HDG face integrators are consistent on the trace",
          "[HDGIntegrator]")
{
   using namespace bilininteg_hdg;

   // The stabilization only ever sees the jump between an element value and
   // the trace, so a state in which both are the same constant must produce no
   // trace-equation residual. This is the discrete conservation statement of
   // the numerical trace, and it holds for the convective integrators too even
   // though their element rows do not vanish (those carry the consistent
   // convective flux).
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   const Element::Type type = (dim == 2) ? Element::QUADRILATERAL
                              : Element::HEXAHEDRON;

   CAPTURE(dim, order);

   Faces fx(dim, order, type);
   const int f = fx.InteriorFace();
   FaceElementTransformations *Tr = fx.mesh.GetFaceElementTransformations(f);

   const FiniteElement &tr_fe = *fx.fes_tr.GetFaceElement(f);
   const FiniteElement &el1 = *fx.fes_el.GetFE(Tr->Elem1No);
   const FiniteElement &el2 = *fx.fes_el.GetFE(Tr->Elem2No);

   Vector vel(dim);
   vel = 0.0;
   vel(0) = 1.3;
   if (dim > 1) { vel(1) = -0.7; }
   VectorConstantCoefficient vcoeff(vel);
   ConstantCoefficient q(2.5);

   // Nodal bases, so the coefficient vector of the constant 1 is all ones.
   const int n1 = el1.GetDof(), n2 = el2.GetDof(), nt = tr_fe.GetDof();
   Vector ones(n1 + n2 + nt);
   ones = 1.0;

   auto TraceRowsVanish = [&](BilinearFormIntegrator & integ,
                              const char *what)
   {
      DenseMatrix elmat;
      integ.AssembleHDGFaceMatrix(tr_fe, el1, el2, *Tr, elmat);
      REQUIRE(elmat.Height() == n1 + n2 + nt);

      Vector res(elmat.Height());
      elmat.Mult(ones, res);

      for (int i = 0; i < nt; i++)
      {
         INFO(what << ": trace row " << i << " gives " << res(n1 + n2 + i));
         REQUIRE(std::abs(res(n1 + n2 + i)) < 1e-11);
      }
      return res;
   };

   SECTION("diffusion")
   {
      HDGDiffusionIntegrator integ(q);
      Vector res = TraceRowsVanish(integ, "diffusion");

      // With no advective direction the diffusion stabilization is pure
      // penalty, so the element rows vanish on constants as well.
      for (int i = 0; i < n1 + n2; i++)
      {
         INFO("element row " << i << " gives " << res(i));
         REQUIRE(std::abs(res(i)) < 1e-11);
      }

      // The face matrix is symmetric only up to the sign convention that
      // DarcyForm applies to the second block row (see DarcyForm's bsym): the
      // element block and the face block are each symmetric, while the
      // constraint block is the *negative* transpose of the trace-flux block.
      // Asserting plain symmetry here would be wrong, and asserting nothing
      // would miss a transposed block.
      DenseMatrix elmat;
      integ.AssembleHDGFaceMatrix(tr_fe, el1, el2, *Tr, elmat);
      const int nel = n1 + n2;

      for (int i = 0; i < nel; i++)
         for (int j = 0; j < i; j++)
         {
            CAPTURE(i, j);
            REQUIRE(elmat(i, j) == MFEM_Approx(elmat(j, i), 1e-11, 1e-10));
         }

      for (int i = 0; i < nt; i++)
         for (int j = 0; j < i; j++)
         {
            CAPTURE(i, j);
            REQUIRE(elmat(nel + i, nel + j) ==
                    MFEM_Approx(elmat(nel + j, nel + i), 1e-11, 1e-10));
         }

      for (int i = 0; i < nt; i++)
         for (int j = 0; j < nel; j++)
         {
            CAPTURE(i, j);
            REQUIRE(elmat(nel + i, j) ==
                    MFEM_Approx(-elmat(j, nel + i), 1e-11, 1e-10));
         }
   }

   SECTION("convection, centered")
   {
      HDGConvectionCenteredIntegrator integ(vcoeff);
      TraceRowsVanish(integ, "centered convection");
   }

   SECTION("convection, upwinded")
   {
      HDGConvectionUpwindedIntegrator integ(vcoeff);
      TraceRowsVanish(integ, "upwinded convection");
   }
}

TEST_CASE("HDGDiffusionIntegrator face energy", "[HDGIntegrator]")
{
   using namespace bilininteg_hdg;

   // The energy the HDG error estimator is built on: ~ (p - lambda)^T tau
   // (p - lambda). It must be non-negative, and it must vanish exactly when
   // the element value and the trace agree.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   const Element::Type type = (dim == 2) ? Element::QUADRILATERAL
                              : Element::HEXAHEDRON;

   CAPTURE(dim, order);

   Faces fx(dim, order, type);
   const int f = fx.InteriorFace();
   FaceElementTransformations *Tr = fx.mesh.GetFaceElementTransformations(f);

   const FiniteElement &tr_fe = *fx.fes_tr.GetFaceElement(f);
   const FiniteElement &el_fe = *fx.fes_el.GetFE(Tr->Elem1No);

   ConstantCoefficient q(2.5);
   HDGDiffusionIntegrator integ(q);

   const int ndof_el = el_fe.GetDof(), ndof_tr = tr_fe.GetDof();

   SECTION("vanishes when the element value matches the trace")
   {
      Vector elfun(ndof_el), trfun(ndof_tr);
      elfun = 1.0;
      trfun = 1.0;
      const real_t e = integ.ComputeHDGFaceEnergy(0, tr_fe, el_fe, *Tr,
                                                  trfun, elfun);
      REQUIRE(std::abs(e) < 1e-12);
   }

   SECTION("is non-negative and grows with the jump")
   {
      Vector elfun(ndof_el), trfun(ndof_tr);
      elfun = 1.0;
      trfun = 0.0;
      const real_t e1 = integ.ComputeHDGFaceEnergy(0, tr_fe, el_fe, *Tr,
                                                   trfun, elfun);
      REQUIRE(e1 > 0.0);

      elfun = 2.0;
      const real_t e2 = integ.ComputeHDGFaceEnergy(0, tr_fe, el_fe, *Tr,
                                                   trfun, elfun);
      REQUIRE(e2 > e1);
      // Quadratic in the jump.
      REQUIRE(e2 == MFEM_Approx(4.0 * e1, 1e-10, 1e-9));
   }

   SECTION("the directional split sums to the total")
   {
      Vector elfun(ndof_el), trfun(ndof_tr);
      FillVarying(elfun, 0.4);
      FillVarying(trfun, 2.2);

      Vector d_energy;
      const real_t e = integ.ComputeHDGFaceEnergy(0, tr_fe, el_fe, *Tr,
                                                  trfun, elfun, &d_energy);
      REQUIRE(d_energy.Size() == dim);

      real_t sum = 0.0;
      for (int d = 0; d < dim; d++) { sum += d_energy(d); }
      INFO("total " << e << " vs sum of directional parts " << sum);
      REQUIRE(sum == MFEM_Approx(e, 1e-10, 1e-9));
   }
}

namespace bilininteg_hdg
{

/// Constant, and deliberately not the identity: it scales the built-in value.
/// A user object that is ignored by an assembly path is the failure this
/// guards against.
class ScaledStabilization : public HDGStabilization
{
   real_t f;
public:
   ScaledStabilization(real_t f_) : f(f_) { }
   real_t Eval(real_t s_diff, real_t, real_t, real_t,
               ElementTransformation &) const override
   { return f * s_diff; }
};

/// s = s_diff + c (u^2 + uhat^2) / 2, which depends on the state and on its
/// trace separately, so both partials are needed and neither can be inferred
/// from the other.
class QuadraticStabilization : public HDGStabilization
{
   real_t c;
public:
   QuadraticStabilization(real_t c_) : c(c_) { }
   bool IsConstant() const override { return false; }
   real_t Eval(real_t s_diff, real_t, real_t u, real_t uhat,
               ElementTransformation &) const override
   { return s_diff + 0.5 * c * (u * u + uhat * uhat); }
   void EvalGrad(real_t, real_t, real_t u, real_t uhat,
                 ElementTransformation &, real_t &d1s, real_t &d2s) const override
   { d1s = c * u; d2s = c * uhat; }
};

/// The control: claims a dependence but reports no derivatives.
class LyingStabilization : public QuadraticStabilization
{
public:
   LyingStabilization(real_t c_) : QuadraticStabilization(c_) { }
   void EvalGrad(real_t, real_t, real_t, real_t,
                 ElementTransformation &, real_t &d1s, real_t &d2s) const override
   { d1s = 0.; d2s = 0.; }
};

} // namespace bilininteg_hdg

TEST_CASE("A user stabilization reaches the bilinear assembly",
          "[HDGIntegrator][Stabilization]")
{
   using namespace bilininteg_hdg;

   // A constant stabilization must be honoured by the face matrix, not
   // silently replaced by the built-in expression. Scaling it by a known
   // factor makes an ignored object impossible to miss.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1);
   const Element::Type type = (dim == 2) ? Element::QUADRILATERAL
                              : Element::HEXAHEDRON;
   CAPTURE(dim, order);

   Faces fx(dim, order, type);
   const int f = fx.InteriorFace();
   FaceElementTransformations *Tr = fx.mesh.GetFaceElementTransformations(f);
   const FiniteElement &tr_fe = *fx.fes_tr.GetFaceElement(f);
   const FiniteElement &el_fe = *fx.fes_el.GetFE(Tr->Elem1No);

   ConstantCoefficient q(2.5);

   DenseMatrix plain, doubled, unit;
   {
      HDGDiffusionIntegrator integ(q);
      integ.AssembleHDGFaceMatrix(0, tr_fe, el_fe, *Tr, plain);
   }
   {
      HDGDiffusionIntegrator integ(q);
      ScaledStabilization s(1.0);
      integ.SetStabilization(s);
      integ.AssembleHDGFaceMatrix(0, tr_fe, el_fe, *Tr, unit);
   }
   {
      HDGDiffusionIntegrator integ(q);
      ScaledStabilization s(2.0);
      integ.SetStabilization(s);
      integ.AssembleHDGFaceMatrix(0, tr_fe, el_fe, *Tr, doubled);
   }

   REQUIRE(unit.Height() == plain.Height());
   for (int i = 0; i < plain.Height(); i++)
      for (int j = 0; j < plain.Width(); j++)
      {
         CAPTURE(i, j);
         // Identity to round-off rather than bitwise: with an object set the
         // weight is divided out and multiplied back.
         REQUIRE(unit(i, j) == MFEM_Approx(plain(i, j), 1e-12, 1e-12));
         REQUIRE(doubled(i, j) == MFEM_Approx(2.0 * plain(i, j), 1e-12, 1e-11));
      }
}

TEST_CASE("A state-dependent stabilization has a matching gradient",
          "[HDGIntegrator][Stabilization]")
{
   using namespace bilininteg_hdg;

   // Nguyen, Peraire and Cockburn Eq. (5) makes s a function of the potential
   // and of its trace, and Eq. (15) linearises it with both partials. Omit
   // them and Newton slows down without the answer changing, so the only
   // check that finds it is this one.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   const int side = GENERATE(0, 1);
   const Element::Type type = (dim == 2) ? Element::QUADRILATERAL
                              : Element::HEXAHEDRON;
   CAPTURE(dim, order, side);

   Faces fx(dim, order, type);
   const int f = fx.InteriorFace();
   FaceElementTransformations *Tr = fx.mesh.GetFaceElementTransformations(f);
   const FiniteElement &tr_fe = *fx.fes_tr.GetFaceElement(f);
   const int elno = (side == 0) ? Tr->Elem1No : Tr->Elem2No;
   const FiniteElement &el_fe = *fx.fes_el.GetFE(elno);

   const int ne = el_fe.GetDof(), nt = tr_fe.GetDof();

   ConstantCoefficient q(2.5);
   HDGDiffusionIntegrator integ(q);
   QuadraticStabilization stab(0.7);
   integ.SetStabilization(stab);

   const int mask = NLFI::ELEM | NLFI::TRACE | NLFI::CONSTR | NLFI::FACE;
   const int itype = mask | (side & 1);

   Vector elfun(ne), trfun(nt);
   FillVarying(elfun, 0.0);
   FillVarying(trfun, 1.1);

   DenseMatrix J;
   integ.AssembleHDGFaceGrad(itype, tr_fe, el_fe, *Tr, trfun, elfun, J);
   REQUIRE(J.Height() == ne + nt);
   REQUIRE(J.Width() == ne + nt);

   Vector dy(ne + nt);
   FillVarying(dy, 2.4);

   Vector Jdy(ne + nt);
   J.Mult(dy, Jdy);

   const real_t h = std::cbrt(std::numeric_limits<real_t>::epsilon());

   auto residual = [&](real_t eps, Vector & r)
   {
      Vector e(elfun), t(trfun);
      for (int i = 0; i < ne; i++) { e(i) += eps * dy(i); }
      for (int i = 0; i < nt; i++) { t(i) += eps * dy(ne + i); }
      integ.AssembleHDGFaceVector(itype, tr_fe, el_fe, *Tr, t, e, r);
   };

   Vector rp, rm;
   residual(h, rp);
   residual(-h, rm);

   Vector fd(rp);
   fd -= rm;
   fd /= (2.0 * h);

   Vector diff(Jdy);
   diff -= fd;
   INFO("||J dy - fd||_inf = " << diff.Normlinf()
        << " against ||fd||_inf = " << fd.Normlinf());
   REQUIRE(diff.Normlinf() < 1e-6 * std::max(fd.Normlinf(), real_t(1.0)));
}

TEST_CASE("A stabilization that hides its derivatives is caught",
          "[HDGIntegrator][Stabilization]")
{
   using namespace bilininteg_hdg;

   Faces fx(2, 1, Element::QUADRILATERAL);
   const int f = fx.InteriorFace();
   FaceElementTransformations *Tr = fx.mesh.GetFaceElementTransformations(f);
   const FiniteElement &tr_fe = *fx.fes_tr.GetFaceElement(f);
   const FiniteElement &el_fe = *fx.fes_el.GetFE(Tr->Elem1No);
   const int ne = el_fe.GetDof(), nt = tr_fe.GetDof();

   ConstantCoefficient q(2.5);
   HDGDiffusionIntegrator integ(q);
   LyingStabilization stab(0.7);
   integ.SetStabilization(stab);

   const int itype = NLFI::ELEM | NLFI::TRACE | NLFI::CONSTR | NLFI::FACE;

   Vector elfun(ne), trfun(nt), dy(ne + nt);
   FillVarying(elfun, 0.0);
   FillVarying(trfun, 1.1);
   FillVarying(dy, 2.4);

   DenseMatrix J;
   integ.AssembleHDGFaceGrad(itype, tr_fe, el_fe, *Tr, trfun, elfun, J);
   Vector Jdy(ne + nt);
   J.Mult(dy, Jdy);

   const real_t h = std::cbrt(std::numeric_limits<real_t>::epsilon());
   Vector rp, rm, e(elfun), t(trfun);
   for (int i = 0; i < ne; i++) { e(i) += h * dy(i); }
   for (int i = 0; i < nt; i++) { t(i) += h * dy(ne + i); }
   integ.AssembleHDGFaceVector(itype, tr_fe, el_fe, *Tr, t, e, rp);
   e = elfun; t = trfun;
   for (int i = 0; i < ne; i++) { e(i) -= h * dy(i); }
   for (int i = 0; i < nt; i++) { t(i) -= h * dy(ne + i); }
   integ.AssembleHDGFaceVector(itype, tr_fe, el_fe, *Tr, t, e, rm);

   Vector fd(rp);
   fd -= rm;
   fd /= (2.0 * h);
   Vector diff(Jdy);
   diff -= fd;

   INFO("dropping both partials shifts J dy by " << diff.Normlinf());
   REQUIRE(diff.Normlinf() > 1e-4 * std::max(fd.Normlinf(), real_t(1.0)));
}
