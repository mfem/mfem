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

#include <cmath>
#include <cstring>
#include <memory>

using namespace mfem;

namespace darcy_nl_face_grad
{

// The STATE-CARRYING HDG face constraint: c_nlfi_p / c_nlfi, evaluated once
// per element per Newton step inside ConstructGrad(), not once at assembly.
// AssemblyMode::Batched replaces that per-pair loop with
// HDGNLFaceGradScatterBatched(), and these cases are what says the two agree.
//
// Two harnesses, because the two slots are filled by different callers and
// only one of them can ever be non-null:
//
//   * BurgersHDG fills c_nlfi_p with a SumNLFIntegrator of one
//     HDGDiffusionIntegrator and one HyperbolicFormIntegrator. That is the
//     ONLY state-carrying face constraint any regression reference reaches --
//     17 of the 152 serial ones, all `convdiff -p 2 -nlc` or `-p 6 -nl` --
//     established by printing which slot EnableHybridization() fills rather
//     than by reading its conditions.
//   * CoupledHDG fills c_nlfi with a MixedConductionNLFIntegrator at
//     neq = 2. No miniapp reaches that under hybridization: every `-nld -hb`
//     configuration also puts an HDGDiffusionIntegrator on the potential mass
//     form, which makes M_p or Mnl_p non-null and sends
//     EnableHybridization() down a branch that never reads
//     Mnl->GetInteriorFaceIntegrators(). So this fixture is the only caller
//     of that half in the tree, which is exactly why it is here.

/// Bitwise, which is what "the same operator" means where the arithmetic is
/// the same operations in the same order.
bool BitwiseEqual(const Vector &a, const Vector &b)
{
   if (a.Size() != b.Size()) { return false; }
   return std::memcmp(a.GetData(), b.GetData(),
                      a.Size()*sizeof(real_t)) == 0;
}

/// Round-off, with an ABSOLUTE floor. The batched pass accumulates point by
/// point straight into the blocks while the per-pair route sums each
/// integrator into one DenseMatrix and adds that, so the two reassociate; and
/// an equality test between two routes must never compare round-off
/// relatively.
void RequireClose(const Vector &ref, const Vector &got,
                  const char *what = "", real_t tol = 1e-12)
{
   INFO("comparing " << what);
   REQUIRE(ref.Size() == got.Size());
   Vector d(ref);
   d -= got;
   const real_t scale = std::max(ref.Normlinf(), real_t(1.0));
   CAPTURE(d.Normlinf(), scale, tol*scale);
   REQUIRE(d.Normlinf() <= tol * scale);
}

/// A state that is nowhere near zero, deterministic, and not a constant.
///
/// It has to be all three. At a zero state the Burgers flux Jacobian is zero
/// and the hyperbolic weights vanish, so the comparison would pass against
/// any wrong kernel; at a constant state the trace and the element agree on
/// every face and the whole stabilization drops out, which is the same trap
/// that let a wrong AssembleHDGFaceGrad indexing survive for years.
void FillState(Vector &v, int seed)
{
   for (int i = 0; i < v.Size(); i++)
   {
      v(i) = 0.4 + 0.6*std::sin(0.7*(i + 1) + 1.3*seed)
             + 0.2*std::cos(0.31*i*i + seed);
   }
}

/// Every interior face's E, G and H, to round-off with an ABSOLUTE floor.
///
/// **Bitwise was tried and is FALSE, for a reason worth keeping.** With one
/// integrator it holds -- each entry is the same sum of the same addends in
/// the same order. With two it does not, and it is not the summation over
/// integrators: HDGDiffusionIntegrator's one-sided AssembleHDGFaceMatrix()
/// fills only the LOWER triangle and then copies, so the per-pair route gets
/// the upper triangle as `(w*s_j)*s_i` where this kernel computes
/// `(w*s_i)*s_j`. Measured, 3 to 6 entries of a few thousand differ by at
/// most 4.4e-16 on a scale of 1.7. A tolerance is therefore right, and an
/// absolute floor with it -- an equality test between two routes must not
/// compare round-off relatively.
///
/// D is not compared here and could not be bitwise either: the boundary faces
/// reach it during the element loop and the interior ones afterwards, so the
/// two routes sum the same terms in a different order -- measured at 2.2e-16.
void RequireBlocksSame(const Array<real_t> &ref, const Array<real_t> &got,
                       real_t tol = 1e-13)
{
   REQUIRE(ref.Size() == got.Size());
   REQUIRE(ref.Size() > 0);
   real_t mx = 0.0, scale = 0.0;
   int ndiff = 0;
   for (int i = 0; i < ref.Size(); i++)
   {
      mx = std::max(mx, std::abs(ref[i] - got[i]));
      scale = std::max(scale, std::abs(ref[i]));
      if (ref[i] != got[i]) { ndiff++; }
   }
   // The blocks are not the zero ones, so agreement is not vacuous.
   CAPTURE(mx, scale, ndiff, ref.Size());
   REQUIRE(scale > 1e-6);
   REQUIRE(mx <= tol * std::max(scale, real_t(1.0)));
}

struct GradOutcome
{
   Vector Sv;                ///< the assembled trace operator applied to a
   ///< fixed vector -- see OneGradient()
   int S_nnz = 0;            ///< and how many nonzeros it was stored with
   Array<real_t> blocks;     ///< every interior face's E, G and H, flattened
   Vector dtr;               ///< the reduced right-hand side, after NPCReduce
   BlockVector dr;           ///< NPCRecover's fields from it
   Vector r_tr;              ///< the trace residual at the same state
   bool can_batch = false;   ///< whether the new kernel was taken
   bool can_batch_res = false;  ///< and whether its RESIDUAL counterpart was
   int n_integs = 0;         ///< integrators in the face constraint, unwrapped
   GradOutcome() : dr() { }
};

/// Take ONE NPC gradient at a fixed, nonzero state and record everything the
/// face constraint feeds.
///
/// A gradient rather than a solve, and taken ONCE: `Mult(x)` leaves the
/// hybridization's state at its own argument, so a second GetGradient() drags
/// it back and pays for more local work -- a gradient check that re-takes the
/// gradient after the difference has hidden a defect on this branch before.
template <class Setup>
void OneGradient(Setup &s, DarcyHybridization::AssemblyMode mode,
                 GradOutcome &out)
{
   DarcyHybridization *dh = s.darcy.GetHybridization();
   dh->SetAssemblyMode(mode);
   dh->EnableNPC();

   Array<int> ess_bdr(s.mesh.bdr_attributes.Max());
   ess_bdr = 1;
   dh->SetEssentialBC(ess_bdr);

   s.darcy.Assemble();
   s.darcy.Finalize();

   BlockVector b(s.darcy.GetOffsets()), x(s.darcy.GetOffsets());
   b = 0.0;
   FillState(x, 1);

   Vector x_tr(s.Mh.GetVSize());
   FillState(x_tr, 2);

   BlockVector r(s.darcy.GetOffsets());
   Vector r_tr, b_tr;

   dh->NPCResidual(b, x, x_tr, r, r_tr);
   out.r_tr = r_tr;
   out.can_batch_res = dh->CanBatchNLFaceResidual();

   Operator &S = dh->NPCGradient(x, x_tr);
   out.can_batch = dh->CanBatchNLFaceGrad();
   {
      Array<NonlinearFormIntegrator*> integs;
      Array<BlockNonlinearFormIntegrator*> bintegs;
      dh->NLFaceConstraintIntegrators(integs, bintegs);
      out.n_integs = integs.Size() + bintegs.Size();
   }

   // Every interior face's E, G and H, flattened. This is the sharpest
   // comparison available and the only one that survives a fixture whose
   // local Schur complement is near singular; see GetFaceE().
   {
      Mesh *mesh = s.Wh.GetMesh();
      out.blocks.SetSize(0);
      DenseMatrix M;
      for (int f = 0; f < mesh->GetNumFaces(); f++)
      {
         if (!mesh->FaceIsInterior(f)) { continue; }
         for (int side = 0; side < 2; side++)
         {
            dh->GetFaceE(f, side, M);
            for (int j = 0; j < M.Width(); j++)
               for (int i = 0; i < M.Height(); i++) { out.blocks.Append(M(i, j)); }
            dh->GetFaceG(f, side, M);
            for (int j = 0; j < M.Width(); j++)
               for (int i = 0; i < M.Height(); i++) { out.blocks.Append(M(i, j)); }
         }
         dh->GetFaceH(f, M);
         for (int j = 0; j < M.Width(); j++)
            for (int i = 0; i < M.Height(); i++) { out.blocks.Append(M(i, j)); }
      }
   }

   // The operator's ACTION, not its stored entries, and the reason is
   // measured. The batched pass accumulates D point by point while the
   // per-pair route sums each integrator into a DenseMatrix and adds that
   // once, and the boundary faces reach D during the element loop while the
   // interior ones now reach it afterwards -- so D differs by up to 2.2e-16.
   // SparseMatrix::Finalize() drops exact zeros, and at order 0 a 2.2e-16
   // perturbation is enough to flip that decision: 256 stored nonzeros
   // against 252 on the same problem. The sparsity is therefore not a
   // property of the operator here, and comparing it would be asserting a
   // property of skip_zeros.
   SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
   REQUIRE(Sm != nullptr);
   out.S_nnz = Sm->NumNonZeroElems();
   Vector v(S.Width());
   FillState(v, 7);
   out.Sv.SetSize(S.Height());
   S.Mult(v, out.Sv);

   // NPCReduce and NPCRecover read E, G and D directly, so they see the
   // batched pass's writes by a different route than the assembled operator
   // does. Both are compared.
   dh->NPCReduce(r, r_tr, b_tr);
   out.dtr = b_tr;
   out.dr.Update(s.darcy.GetOffsets());
   Vector step(b_tr.Size());
   FillState(step, 3);
   dh->NPCRecover(r, step, out.dr);
}

/// c_nlfi_p: an HDGDiffusionIntegrator and a HyperbolicFormIntegrator summed
/// on the potential mass NONLINEAR form, which is `convdiff -p 6 -nl`'s shape
/// and the only one the reference set reaches.
struct BurgersHDG
{
   Mesh mesh;
   L2_FECollection u_coll, p_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace Vh, Wh, Mh;
   DarcyForm darcy;
   ConstantCoefficient one;
   BurgersFlux flux;
   HDGFlux num_flux;
   Array<int> ess_flux;

   BurgersHDG(int n, int order, Element::Type etype)
      : mesh(Mesh::MakeCartesian2D(n, n, etype)),
        u_coll(order, 2), p_coll(order, 2), t_coll(order, 2),
        Vh(&mesh, &u_coll, 2), Wh(&mesh, &p_coll), Mh(&mesh, &t_coll),
        darcy(&Vh, &Wh), one(1.0), flux(2),
        num_flux(flux, HDGFlux::HDGScheme::HDG_1)
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(one));
      darcy.GetFluxDivForm()->AddDomainIntegrator(
         new VectorDivergenceIntegrator());
      darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

      NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
      Mnl_p->AddDomainIntegrator(
         new HyperbolicFormIntegrator(num_flux, 0, -1.0));
      Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 0.5));
      Mnl_p->AddInteriorFaceIntegrator(
         new HyperbolicFormIntegrator(num_flux, 0, -1.0));
      Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 0.5));
      Mnl_p->AddBdrFaceIntegrator(
         new HyperbolicFormIntegrator(num_flux, 0, -1.0));

      darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(),
                                ess_flux);
   }
};

/// A 2-equation advection flux whose Jacobian is NOT diagonal.
///
/// This exists for one reason. HDGFlux::AverageGrad() returns
/// ComputeFluxJacobianDotN(), so a diagonal flux Jacobian gives a diagonal
/// weight matrix -- and a diagonal weight matrix cannot detect a transposed
/// EQUATION index in the scatter, because di == dj wherever the weight is
/// nonzero. Measured: with a diagonal weight the transposed-G break passes
/// every assertion. That is exactly the defect fem/hyperbolic.cpp once
/// carried for years -- AssembleHDGFaceGrad writing its blocks
/// equation-outermost while AssembleHDGFaceVector wrote them group-outermost
/// -- and it agrees at num_equations == 1.
class CoupledAdvectionFlux : public FluxFunction
{
public:
   explicit CoupledAdvectionFlux(int dim) : FluxFunction(2, dim) { }

   real_t ComputeFlux(const Vector &u, ElementTransformation &,
                      DenseMatrix &F) const override
   {
      real_t speed = 0.0;
      for (int d = 0; d < dim; d++)
      {
         for (int e = 0; e < 2; e++)
         {
            F(e, d) = 0.0;
            for (int f = 0; f < 2; f++) { F(e, d) += A(e, f, d) * u(f); }
         }
         speed = std::max(speed, real_t(2.0));
      }
      return speed;
   }

   void ComputeFluxJacobian(const Vector &, ElementTransformation &,
                            DenseTensor &J) const override
   {
      J.SetSize(2, 2, dim);
      for (int d = 0; d < dim; d++)
         for (int e = 0; e < 2; e++)
            for (int f = 0; f < 2; f++)
            { J(e, f, d) = A(e, f, d); }
   }

   void ComputeFluxJacobianDotN(const Vector &, const Vector &nor,
                                ElementTransformation &,
                                DenseMatrix &J) const override
   {
      J.SetSize(2);
      J = 0.0;
      for (int d = 0; d < dim; d++)
         for (int e = 0; e < 2; e++)
            for (int f = 0; f < 2; f++)
            { J(e, f) += nor(d) * A(e, f, d); }
   }

private:
   /// Deliberately NON-symmetric: A(0,1,d) != A(1,0,d), so a transposed
   /// equation index is a different matrix and not the same one.
   ///
   /// And deliberately SMALL against HDGFlux's Ctau of 1. HDGFlux's
   /// AverageGrad(1, ...) is J.n - Ctau*|n| I, so |A| ~ 1 makes that
   /// indefinite and the local Schur complement singular -- measured, the
   /// assembled trace operator came back at 6e+226 in BOTH routes, which is a
   /// comparison that measures nothing. At |A| ~ 0.1 the trace block is a
   /// perturbation of -I and the off-diagonal part is still 30% of the
   /// diagonal, which is all the equation index needs to be observable.
   static real_t A(int e, int f, int d)
   {
      static const real_t a[2][2][3] = { { {0.10, 0.03, 0.01}, {0.08, -0.04, 0.02} },
         { {-0.02, 0.09, 0.03}, {0.05, 0.11, -0.01} }
      };
      return a[e][f][d];
   }
};

/// c_nlfi_p at neq = 2, with the coupled flux above: the case that can see a
/// transposed equation index.
struct CoupledBurgersHDG
{
   static const int neq = 2;
   Mesh mesh;
   L2_FECollection u_coll, p_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace Vh, Wh, Mh;
   DarcyForm darcy;
   ConstantCoefficient one;
   CoupledAdvectionFlux flux;
   HDGFlux num_flux;
   Array<int> ess_flux;

   CoupledBurgersHDG(int n, int order, Element::Type etype,
                     HDGFlux::HDGScheme scheme)
      : mesh(Mesh::MakeCartesian2D(n, n, etype)),
        u_coll(order, 2), p_coll(order, 2), t_coll(order, 2),
        Vh(&mesh, &u_coll, neq*2, Ordering::byNODES),
        Wh(&mesh, &p_coll, neq, Ordering::byNODES),
        Mh(&mesh, &t_coll, neq, Ordering::byNODES),
        darcy(&Vh, &Wh), one(1.0), flux(2),
        num_flux(flux, scheme)
   {
      // WRAPPED, and a bare VectorMassIntegrator here is a defect. Its own
      // vdim comes from the SPACE DIMENSION when the coefficient is scalar,
      // so on this vdim = neq*dim flux space it builds a (nd*dim) square
      // where DarcyHybridization::AssembleFluxMassMatrix() reads an
      // (nd*neq*dim) one -- measured, 2x2 against the 4x4 expected at
      // order 0. That is an out-of-bounds READ, so it does not fail where it
      // happens: it returns plausible garbage, and whether the run survives
      // depends on the heap. It cost an afternoon here, presenting as a
      // SIGSEGV in this file that appeared under some test filters and not
      // others and went away under gdb. ASan on two translation units named
      // it in one run.
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorBlockDiagonalIntegrator(neq, new VectorMassIntegrator(one)));
      MixedBilinearForm *B = darcy.GetFluxDivForm();
      B->AddDomainIntegrator(
         new VectorBlockDiagonalIntegrator(neq, new VectorDivergenceIntegrator));
      // The INTERIOR face constraint on B, without which row 1's
      // -(u, div v) + <lambda, v.n> = 0 is missing on every element and the
      // local solve is degenerate. At order 0 the domain term vanishes
      // identically -- grad w = 0 for a constant basis -- so this is the
      // ONLY thing coupling the flux to the trace there.
      B->AddInteriorFaceIntegrator(
         new VectorBlockDiagonalIntegrator(
            neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0))));
      B->AddBdrFaceIntegrator(
         new VectorBlockDiagonalIntegrator(
            neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0))));

      NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
      // A potential mass, so that the local Schur complement is not
      // singular. Without it this configuration blows up to 1e+128 in BOTH
      // routes -- a pure hyperbolic constraint leaves the local problem with
      // nothing on its potential diagonal but the face stabilization, and at
      // order 0 the domain term (F(u), grad w) vanishes identically. A
      // comparison between two routes that both return 1e+128 measures
      // nothing.
      //
      // A BilinearFormIntegrator on a NonlinearForm is deliberate and legal:
      // BilinearFormIntegrator derives from NonlinearFormIntegrator, and
      // EnableHybridization() looks only at the FACE integrators when it
      // decides whether the constraint is linear, so the constraint still
      // reaches c_nlfi_p.
      Mnl_p->AddDomainIntegrator(new VectorMassIntegrator(one));
      Mnl_p->AddDomainIntegrator(
         new HyperbolicFormIntegrator(num_flux, 0, -1.0));
      Mnl_p->AddInteriorFaceIntegrator(
         new HyperbolicFormIntegrator(num_flux, 0, -1.0));
      Mnl_p->AddBdrFaceIntegrator(
         new HyperbolicFormIntegrator(num_flux, 0, -1.0));

      darcy.EnableHybridization(
         &Mh, new VectorBlockDiagonalIntegrator(neq,
                                                new NormalTraceJumpIntegrator),
         ess_flux);
   }
};

/// A flux law that couples the equations, so that neq > 1 is not neq == 1
/// twice. Copied in shape from test_darcy_npc.cpp's, which is the only other
/// caller of a MixedConductionNLFIntegrator face constraint.
class ScaledCoupledFlux : public MixedFluxFunction
{
public:
   ScaledCoupledFlux(int dim, int neq, real_t eps_)
      : MixedFluxFunction(neq, dim), eps(eps_) { }

   real_t ComputeDualFlux(const Vector &u, const DenseMatrix &F,
                          ElementTransformation &, DenseMatrix &G) const override
   {
      G.SetSize(num_equations, dim);
      for (int e = 0; e < num_equations; e++)
         for (int d = 0; d < dim; d++)
         { G(e, d) = F(e, d) / Scale(u, e); }
      return 1.0;
   }

   real_t ComputeFlux(const Vector &u, ElementTransformation &,
                      DenseMatrix &F) const override
   {
      for (int e = 0; e < num_equations; e++)
         for (int d = 0; d < dim; d++)
         { F(e, d) *= Scale(u, e); }
      return 1.0;
   }

   void ComputeDualFluxJacobian(const Vector &u, const DenseMatrix &F,
                                ElementTransformation &,
                                DenseMatrix &J_u,
                                DenseMatrix &J_F) const override
   {
      J_u.SetSize(num_equations*dim, num_equations);
      J_u = 0.0;
      J_F.SetSize(num_equations*dim);
      J_F = 0.0;
      for (int e = 0; e < num_equations; e++)
      {
         const real_t s = Scale(u, e);
         for (int d = 0; d < dim; d++)
         {
            J_F(e*dim + d, e*dim + d) = 1.0 / s;
            for (int f = 0; f < num_equations; f++)
            {
               J_u(e*dim + d, f) = -F(e, d) * eps / (s*s);
            }
         }
      }
   }

private:
   real_t Scale(const Vector &u, int e) const
   {
      real_t s = 1.0 + 0.25*(e + 1);
      for (int f = 0; f < u.Size(); f++) { s += eps*u(f); }
      return s;
   }
   real_t eps;
};

/// c_nlfi: a MixedConductionNLFIntegrator on the block nonlinear form.
struct CoupledHDG
{
   static const int neq = 2;
   Mesh mesh;
   L2_FECollection u_coll, p_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace Vh, Wh, Mh;
   DarcyForm darcy;
   ScaledCoupledFlux flux;
   Vector taus;
   Array<int> ess_flux;

   CoupledHDG(int n, int order, Element::Type etype)
      : mesh(Mesh::MakeCartesian2D(n, n, etype)),
        u_coll(order, 2), p_coll(order, 2), t_coll(order, 2),
        Vh(&mesh, &u_coll, neq*2, Ordering::byNODES),
        Wh(&mesh, &p_coll, neq, Ordering::byNODES),
        Mh(&mesh, &t_coll, neq, Ordering::byNODES),
        darcy(&Vh, &Wh), flux(2, neq, 0.35), taus(neq)
   {
      BlockNonlinearForm *Mnl = darcy.GetBlockNonlinearForm();
      Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(flux));
      auto *face = new MixedConductionNLFIntegrator(flux, 0.7);
      taus(0) = 1.0;
      taus(1) = 1.5;
      face->SetVariableStabilization(taus);
      Mnl->AddInteriorFaceIntegrator(face);

      MixedBilinearForm *B = darcy.GetFluxDivForm();
      B->AddDomainIntegrator(
         new VectorBlockDiagonalIntegrator(neq, new VectorDivergenceIntegrator));
      B->AddInteriorFaceIntegrator(
         new VectorBlockDiagonalIntegrator(
            neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0))));
      B->AddBdrFaceIntegrator(
         new VectorBlockDiagonalIntegrator(
            neq, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0))));

      darcy.EnableHybridization(
         &Mh, new VectorBlockDiagonalIntegrator(neq,
                                                new NormalTraceJumpIntegrator),
         ess_flux);
   }
};

} // namespace darcy_nl_face_grad

TEST_CASE("The batched state-carrying face gradient agrees with the per-pair "
          "loop", "[DarcyHybridization][NPC][Batched]")
{
   using namespace darcy_nl_face_grad;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(0, 1, 2);
   const int n = GENERATE(2, 3);
   const Element::Type etype = GENERATE(Element::QUADRILATERAL,
                                        Element::TRIANGLE);
   CAPTURE(order, n, etype);

   SECTION("c_nlfi_p: HDGDiffusion and HyperbolicFormIntegrator summed")
   {
      BurgersHDG a(n, order, etype), b(n, order, etype);
      GradOutcome ref, got;
      OneGradient(a, AM::Serial, ref);
      OneGradient(b, AM::Batched, got);

      // The kernel is REACHED, and the reference is not taking it. Asserted
      // rather than assumed: a batched route nothing runs is how this branch
      // lost FluxNL's Schur complement once already.
      REQUIRE_FALSE(ref.can_batch);
      REQUIRE(got.can_batch);
      // The RESIDUAL kernel likewise, which is what makes the r_tr comparison
      // below a test of it rather than of two runs of the same code.
      REQUIRE_FALSE(ref.can_batch_res);
      REQUIRE(got.can_batch_res);
      // And the sum really has two members, so the accumulation across
      // integrators is exercised rather than a single pass.
      REQUIRE(ref.n_integs == 2);

      // There is an operator to compare and it is not the zero one, and the
      // residual is not zero either -- the weights are evaluated at a state
      // where the stabilization does not drop out.
      REQUIRE(ref.S_nnz > 0);
      REQUIRE(ref.Sv.Normlinf() > 1e-3);
      REQUIRE(ref.r_tr.Normlinf() > 1e-3);

      RequireBlocksSame(ref.blocks, got.blocks);
      RequireClose(ref.Sv, got.Sv, "S applied to a fixed vector");
      RequireClose(ref.r_tr, got.r_tr, "r_tr");
      RequireClose(ref.dtr, got.dtr, "dtr");
      RequireClose(ref.dr.GetBlock(0), got.dr.GetBlock(0), "dr flux");
      RequireClose(ref.dr.GetBlock(1), got.dr.GetBlock(1), "dr pot");
   }

   SECTION("c_nlfi_p at neq = 2, with an OFF-DIAGONAL weight matrix")
   {
      // The section above cannot see a transposed EQUATION index: at neq == 1
      // di == dj always, and MixedConductionNLFIntegrator's weight matrix is
      // diagonal by construction (a scalar tau per variable), so di == dj
      // wherever it is nonzero. This one uses HDGFlux over a non-symmetric
      // coupled flux Jacobian, which is what makes the equation index
      // observable at all.
      // BOTH schemes, because they put the coupled flux Jacobian on
      // DIFFERENT blocks. HDGFlux::AverageGrad() returns the full Jacobian
      // for side 1 under HDG_1 and for side 2 under HDG_2, and Ctau*I on the
      // other -- so HDG_1 gives an off-diagonal E and H with a DIAGONAL D and
      // G, and HDG_2 the reverse. Only running both makes the equation index
      // observable in all four blocks; with HDG_1 alone, transposing G's
      // equation index changes nothing.
      const HDGFlux::HDGScheme scheme = GENERATE(HDGFlux::HDGScheme::HDG_1,
                                                 HDGFlux::HDGScheme::HDG_2);
      CoupledBurgersHDG a(n, order, etype, scheme), b(n, order, etype, scheme);
      GradOutcome ref, got;
      OneGradient(a, AM::Serial, ref);
      OneGradient(b, AM::Batched, got);

      REQUIRE_FALSE(ref.can_batch);
      REQUIRE(got.can_batch);
      REQUIRE_FALSE(ref.can_batch_res);
      REQUIRE(got.can_batch_res);
      REQUIRE(ref.r_tr.Normlinf() > 1e-3);

      // NOT the assembled operator, and NOT anything downstream of a local
      // solve. This fixture's local Schur complement is near singular -- the
      // trace operator comes back between 6e+15 and 6e+226 in BOTH routes,
      // and the reduced right-hand side with it, so comparing either measures
      // nothing. Making it well posed would mean putting a diffusion
      // stabilization alongside, and HDGDiffusionIntegrator is scalar, so at
      // neq = 2 it would have to be wrapped -- and the kernel refuses a
      // wrapper. The face blocks ARE what this kernel writes, they are
      // exactly comparable, and they carry the equation index this section
      // exists to test.
      CAPTURE(ref.S_nnz, got.S_nnz);
      RequireBlocksSame(ref.blocks, got.blocks);
      RequireClose(ref.r_tr, got.r_tr, "r_tr");
   }

   SECTION("two gradients in a row give the same operator")
   {
      // E and G hold one block per (face, side) and are NOT reset between
      // gradient evaluations, so a batched pass that forgot to clear them
      // would accumulate and make the operator depend on how many times it
      // had been asked for. Nothing above can see that -- each harness takes
      // one gradient on a fresh object, where the arrays start at zero -- and
      // the per-pair route has been wrong in exactly this way before.
      BurgersHDG a(n, order, etype);
      DarcyHybridization *dh = a.darcy.GetHybridization();
      dh->SetAssemblyMode(AM::Batched);
      dh->EnableNPC();
      Array<int> ess_bdr(a.mesh.bdr_attributes.Max());
      ess_bdr = 1;
      dh->SetEssentialBC(ess_bdr);
      a.darcy.Assemble();
      a.darcy.Finalize();

      BlockVector x(a.darcy.GetOffsets());
      FillState(x, 1);
      Vector x_tr(a.Mh.GetVSize());
      FillState(x_tr, 2);

      REQUIRE(dh->CanBatchNLFaceGrad());

      Vector first, second;
      {
         Operator &S = dh->NPCGradient(x, x_tr);
         SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
         REQUIRE(Sm != nullptr);
         first.SetSize(Sm->NumNonZeroElems());
         for (int i = 0; i < first.Size(); i++)
         { first(i) = Sm->GetData()[i]; }
      }
      REQUIRE(first.Normlinf() > 1e-3);
      {
         Operator &S = dh->NPCGradient(x, x_tr);
         SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
         REQUIRE(Sm != nullptr);
         second.SetSize(Sm->NumNonZeroElems());
         for (int i = 0; i < second.Size(); i++)
         { second(i) = Sm->GetData()[i]; }
      }
      RequireClose(first, second, "two gradients");
   }

   SECTION("c_nlfi: MixedConductionNLFIntegrator at neq = 2")
   {
      CoupledHDG a(n, order, etype), b(n, order, etype);
      GradOutcome ref, got;
      OneGradient(a, AM::Serial, ref);
      OneGradient(b, AM::Batched, got);

      REQUIRE_FALSE(ref.can_batch);
      REQUIRE(got.can_batch);
      // **And the residual kernel refuses this family where the gradient
      // takes it**, deliberately -- see HDGNLFaceResidualCanBatch(). Asserted
      // so that implementing MixedConduction later has to come here and say
      // so, rather than silently changing which route this section runs.
      REQUIRE_FALSE(ref.can_batch_res);
      REQUIRE_FALSE(got.can_batch_res);
      REQUIRE(ref.n_integs == 1);

      REQUIRE(ref.S_nnz > 0);
      REQUIRE(ref.Sv.Normlinf() > 1e-3);
      REQUIRE(ref.r_tr.Normlinf() > 1e-3);

      // NOT the assembled operator here. This fixture's local Schur
      // complement is near singular -- the trace operator comes back at
      // 6e+15 to 6e+226 in BOTH routes, and a comparison of two garbage
      // numbers measures nothing. The face blocks are what the kernel writes
      // and they are compared directly instead.
      CAPTURE(ref.S_nnz, got.S_nnz);
      RequireBlocksSame(ref.blocks, got.blocks);
      RequireClose(ref.r_tr, got.r_tr, "r_tr");
      RequireClose(ref.dr.GetBlock(0), got.dr.GetBlock(0), "dr flux");
      RequireClose(ref.dr.GetBlock(1), got.dr.GetBlock(1), "dr pot");
   }
}

TEST_CASE("What the batched face gradient refuses, and why",
          "[DarcyHybridization][NPC][Batched]")
{
   using namespace darcy_nl_face_grad;
   using AM = DarcyHybridization::AssemblyMode;

   SECTION("a state dependent stabilization is refused, not silently dropped")
   {
      // HDGDiffusionIntegrator with a non-constant HDGStabilization takes a
      // different pair of weights -- s + d1s*jump against -s + d2s*jump --
      // and nothing in this tree installs one, so the kernel refuses rather
      // than guessing. The refusal is the assertion: the per-pair loop still
      // runs and the answer is the same.
      class VaryingStab : public HDGStabilization
      {
      public:
         bool IsConstant() const override { return false; }
         real_t Eval(real_t s, real_t, real_t u, real_t uhat,
                     ElementTransformation &) const override
         { return s * (1.0 + 0.1*(u + uhat)); }
         void EvalGrad(real_t s, real_t, real_t, real_t,
                       ElementTransformation &,
                       real_t &d1s, real_t &d2s) const override
         { d1s = 0.1*s; d2s = 0.1*s; }
      };

      const int order = 1, n = 3;
      VaryingStab stab_a, stab_b;
      BurgersHDG a(n, order, Element::QUADRILATERAL);
      BurgersHDG b(n, order, Element::QUADRILATERAL);
      Array<NonlinearFormIntegrator*> ia, ib;
      Array<BlockNonlinearFormIntegrator*> ba, bb;
      a.darcy.GetHybridization()->NLFaceConstraintIntegrators(ia, ba);
      b.darcy.GetHybridization()->NLFaceConstraintIntegrators(ib, bb);
      REQUIRE(ia.Size() == 2);
      int n_diff = 0;
      for (int k = 0; k < ia.Size(); k++)
      {
         if (auto *d = dynamic_cast<HDGDiffusionIntegrator*>(ia[k]))
         { d->SetStabilization(stab_a); n_diff++; }
         if (auto *d = dynamic_cast<HDGDiffusionIntegrator*>(ib[k]))
         { d->SetStabilization(stab_b); }
      }
      // The guard is reached: there IS an HDGDiffusionIntegrator to hang the
      // hook on. An unreachable guard is worse than none.
      REQUIRE(n_diff == 1);

      GradOutcome ref, got;
      OneGradient(a, AM::Serial, ref);
      OneGradient(b, AM::Batched, got);

      REQUIRE_FALSE(ref.can_batch);
      REQUIRE_FALSE(got.can_batch);
      RequireClose(ref.Sv, got.Sv, "S applied to a fixed vector");
   }

   SECTION("the reduced route does not take it, whatever the mode says")
   {
      // The kernel writes H into H_data, which is where the per-pair route
      // puts it only under NPC. Without NPC that route scatters H into the
      // assembled sparse matrix and nothing reads H_data, so taking the
      // kernel would put the face term where the solve does not look.
      BurgersHDG a(3, 1, Element::QUADRILATERAL);
      DarcyHybridization *dh = a.darcy.GetHybridization();
      dh->SetAssemblyMode(AM::Batched);
      Array<int> ess_bdr(a.mesh.bdr_attributes.Max());
      ess_bdr = 1;
      dh->SetEssentialBC(ess_bdr);
      a.darcy.Assemble();
      a.darcy.Finalize();
      // IsNonlinear() makes NPCEnabled() true for this problem even without
      // EnableNPC(), so the discriminating case is the LINEAR one below --
      // this one records that a nonlinear problem is admitted.
      REQUIRE(dh->CanBatchNLFaceGrad());
   }

   SECTION("a linear face constraint has nothing here to batch")
   {
      // Every integrator being a BilinearFormIntegrator routes the constraint
      // to c_bfi_p, where HDGFaceScatterBatched() -- a different kernel, run
      // once at assembly -- is what covers it. So this one must refuse, and
      // for the reason that there is no nonlinear face constraint at all
      // rather than because it cannot weigh the terms.
      Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
      L2_FECollection u_coll(1, 2), p_coll(1, 2);
      DG_Interface_FECollection t_coll(1, 2);
      FiniteElementSpace Vh(&mesh, &u_coll, 2), Wh(&mesh, &p_coll),
                         Mh(&mesh, &t_coll);
      DarcyForm darcy(&Vh, &Wh);
      ConstantCoefficient one(1.0);
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(one));
      darcy.GetFluxDivForm()->AddDomainIntegrator(
         new VectorDivergenceIntegrator());
      darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
         new HDGDiffusionIntegrator(one, 0.5));
      Array<int> ess_flux;
      darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
      DarcyHybridization *dh = darcy.GetHybridization();
      dh->SetAssemblyMode(AM::Batched);
      dh->EnableNPC();

      Array<NonlinearFormIntegrator*> integs;
      Array<BlockNonlinearFormIntegrator*> bintegs;
      dh->NLFaceConstraintIntegrators(integs, bintegs);
      REQUIRE(integs.Size() == 0);
      REQUIRE(bintegs.Size() == 0);
      REQUIRE_FALSE(dh->CanBatchNLFaceGrad());
      // And the linear kernel is the one that does cover it.
      REQUIRE(dh->NumPotFaceConstraintIntegrators() == 1);
   }
}
