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

namespace darcy_batched_bdrflux
{

/** @brief A flux-mass boundary face integrator: @a s <C q, v>_F over the
    adjacent element's own shape functions, with C a constant and deliberately
    NON-SYMMETRIC coupling of the vdim components.

    It discretises nothing, and it has to be written here because there is
    nothing in the library to use instead. **No BilinearFormIntegrator in MFEM
    returns the one-sided element block of a vector flux space on a boundary
    face**: BoundaryMassIntegrator is a scalar MassIntegrator, so on an L2
    space with vdim = dim it produces a block vdim times too small in each
    direction and DarcyForm::AssembleFluxMassBdrFaces()'s own MFEM_VERIFY
    rejects it, while an H(div) space has no scalar shape for it to evaluate
    at all. VectorMassIntegrator and VectorFEMassIntegrator have no
    AssembleFaceMatrix(). That is why the batched pass under test batches the
    SCATTER and not the quadrature -- there is no integrator family to
    dispatch on.

    C non-symmetric is what makes a transposed read visible: the block is
    C (x) (face mass), the face mass is symmetric, so with C symmetric the
    whole block would be too and reading M[j + a*i] would test nothing. */
class BdrFluxMassIntegrator : public BilinearFormIntegrator
{
   const real_t s;
   const int vd;

public:
   BdrFluxMassIntegrator(real_t s_, int vd_) : s(s_), vd(vd_) { }

   static real_t C(int d, int e)
   {
      return 1.0 + 0.5 * d - 0.25 * e + ((d > e) ? 0.75 : 0.0);
   }

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override
   {
      const int dof = el1.GetDof();
      elmat.SetSize(dof * vd);
      elmat = 0.0;

      Vector shape(dof);
      const IntegrationRule &ir =
         IntRules.Get(Trans.GetGeometryType(), 2 * el1.GetOrder() + 2);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Trans.SetAllIntPoints(&ip);
         el1.CalcShape(Trans.GetElement1IntPoint(), shape);

         const real_t w = s * ip.weight * Trans.Weight();
         for (int d = 0; d < vd; d++)
         {
            for (int e = 0; e < vd; e++)
            {
               const real_t wc = w * C(d, e);
               for (int i = 0; i < dof; i++)
                  for (int j = 0; j < dof; j++)
                  {
                     elmat(d * dof + i, e * dof + j) += wc * shape(i) * shape(j);
                  }
            }
         }
      }
   }
};

/// How the mesh is built. Periodic is here because a periodic mesh keeps the
/// boundary ELEMENTS whose faces the identification turned interior, and both
/// routes have to drop them.
enum class MeshKind { Quad, Tri, PeriodicX };

const char *Name(MeshKind k)
{
   switch (k)
   {
      case MeshKind::Quad:      return "quads";
      case MeshKind::Tri:       return "triangles";
      case MeshKind::PeriodicX: return "quads, periodic in x";
   }
   return "?";
}

Mesh MakeMesh(MeshKind kind, int n)
{
   if (kind == MeshKind::Tri)
   {
      return Mesh::MakeCartesian2D(n, n, Element::TRIANGLE, false, 0.8, 1.2);
   }
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   if (kind != MeshKind::PeriodicX) { return mesh; }

   Vector tx(2);
   tx(0) = 0.8;
   tx(1) = 0.0;
   std::vector<Vector> trans;
   trans.push_back(tx);
   return Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping(trans));
}

/// Which boundary attributes each flux-mass boundary integrator is given.
enum class Markers
{
   All,        ///< no marker at all: every attribute
   Split,      ///< integrator 0 on {1,3}, integrator 1 on {2,4}
   Periodic,   ///< {2, 4} only -- exactly the attributes MakePeriodic identifies
   Physical    ///< {1, 3} only -- the attributes a periodic-in-x mesh keeps
};

struct Cfg
{
   int order = 1;
   int n = 3;
   MeshKind kind = MeshKind::Quad;
   int nbdr = 1;               ///< how many flux-mass boundary integrators
   real_t scale = 0.7;         ///< the scale each of them carries
   Markers markers = Markers::All;
   bool ess_flux = false;      ///< a hand-made essential flux dof list
   DarcyHybridization::AssemblyMode am =
      DarcyHybridization::AssemblyMode::Serial;
};

struct Out
{
   Array<int> I, J;
   Vector data;                ///< the assembled NPC trace gradient
   Vector B;                   ///< the reduced right-hand side
   bool bdr_taken = false;     ///< CanBatchFluxMassBdrFaces()
   bool mass_taken = false;    ///< CanBatchFluxMass(), the element pass
};

/** @a keep owns the marker arrays. BilinearForm::AddBdrFaceIntegrator()
    stores the POINTER, not a copy, so a marker built on the stack of this
    function dangles the moment it returns -- measured as a SIGSEGV inside
    Assemble(), on the first case that used a marker at all. */
void AddBdrIntegrators(Mesh &mesh, BilinearForm *M_u, const Cfg &cfg, int vd,
                       std::vector<std::unique_ptr<Array<int>>> &keep)
{
   const int nattr = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
   for (int k = 0; k < cfg.nbdr; k++)
   {
      auto *bfi = new BdrFluxMassIntegrator(cfg.scale, vd);
      if (cfg.markers == Markers::All) { M_u->AddBdrFaceIntegrator(bfi); continue; }

      keep.emplace_back(new Array<int>(nattr));
      Array<int> &m = *keep.back();
      m = 0;
      auto set = [&](int a) { if (a >= 1 && a <= nattr) { m[a-1] = 1; } };
      switch (cfg.markers)
      {
         case Markers::Split:
            if (k % 2 == 0) { set(1); set(3); }
            else { set(2); set(4); }
            break;
         case Markers::Periodic: set(2); set(4); break;
         case Markers::Physical: set(1); set(3); break;
         default: break;
      }
      M_u->AddBdrFaceIntegrator(bfi, m);
   }
}

/** Build the hybridized problem and assemble the NPC trace gradient, entry
    for entry.

    The GRADIENT rather than a solution, for the reason
    test_darcy_batched_face.cpp gives: an iterative trace solve at a finite
    tolerance can absorb a difference the assembled operator cannot. It is the
    right instrument for Af, which is what the boundary flux mass writes and
    what the Schur complement inverts. It is the WRONG instrument for Ae --
    see ReducedRHS() below. */
void Gradient(const Cfg &cfg, Out &out)
{
   const int dim = 2;
   Mesh mesh = MakeMesh(cfg.kind, cfg.n);

   L2_FECollection u_coll(cfg.order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(cfg.order, dim);
   DG_Interface_FECollection t_coll(cfg.order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);
   FunctionCoefficient kappa([](const Vector &X)
   { return 1.0 + 0.5 * X(0) * X(1); });

   std::vector<std::unique_ptr<Array<int>>> keep;
   BilinearForm *M_u = darcy.GetFluxMassForm();
   M_u->AddDomainIntegrator(new VectorMassIntegrator(kappa));
   AddBdrIntegrators(mesh, M_u, cfg, dim, keep);

   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   BilinearForm *M_p = darcy.GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   if (cfg.ess_flux)
   {
      // A HAND-MADE list: the flux space is L2, so it has no boundary dofs
      // and GetEssentialTrueDofs() returns nothing whatever the marker says.
      for (int i = 0; i < Vh.GetVSize(); i += 7) { ess_flux.Append(i); }
   }
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(cfg.am);
   dh->EnableNPC();
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      dh->SetEssentialBC(ess_bdr);
   }

   darcy.Assemble();
   darcy.Finalize();

   out.bdr_taken = dh->CanBatchFluxMassBdrFaces(M_u);
   out.mass_taken = dh->CanBatchFluxMass(M_u);

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   Vector x_tr(Mh.GetVSize());
   x_tr = 0.0;

   Operator &S = dh->NPCGradient(x, x_tr);
   SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
   REQUIRE(Sm != nullptr);

   const int nrows = Sm->Height(), nnz = Sm->NumNonZeroElems();
   out.I.SetSize(nrows + 1);
   std::copy(Sm->GetI(), Sm->GetI() + nrows + 1, out.I.begin());
   out.J.SetSize(nnz);
   std::copy(Sm->GetJ(), Sm->GetJ() + nnz, out.J.begin());
   out.data.SetSize(nnz);
   std::copy(Sm->GetData(), Sm->GetData() + nnz, out.data.GetData());
}

/** Reduce a LINEAR problem with nonzero essential flux values and hand back
    the reduced right-hand side.

    Not NPC, and that is the point. Ae_data -- the essential COLUMNS of each
    element's flux mass, which the boundary pass writes alongside Af -- is read
    only by the elimination `bu -= A_e u_e` in EliminateVDofsInRHS(), on the
    reduced route. NPCGradient() never touches it, so Gradient() above cannot
    see the Ae half of the kernel however many essential dofs are present. */
void ReducedRHS(const Cfg &cfg, Out &out)
{
   const int dim = 2;
   Mesh mesh = MakeMesh(cfg.kind, cfg.n);

   L2_FECollection u_coll(cfg.order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(cfg.order, dim);
   DG_Interface_FECollection t_coll(cfg.order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);
   FunctionCoefficient kappa([](const Vector &X)
   { return 1.0 + 0.5 * X(0) * X(1); });
   FunctionCoefficient src([](const Vector &X)
   { return std::sin(M_PI * X(0)) * std::sin(M_PI * X(1)); });

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));

   std::vector<std::unique_ptr<Array<int>>> keep;
   BilinearForm *M_u = darcy.GetFluxMassForm();
   M_u->AddDomainIntegrator(new VectorMassIntegrator(kappa));
   AddBdrIntegrators(mesh, M_u, cfg, dim, keep);

   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   BilinearForm *M_p = darcy.GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kappa, 1.0));
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kappa, 1.0));

   Array<int> ess_flux;
   for (int i = 0; i < Vh.GetVSize(); i += 7) { ess_flux.Append(i); }
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(cfg.am);
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   dh->SetEssentialBC(ess_bdr);

   darcy.Assemble();
   out.bdr_taken = dh->CanBatchFluxMassBdrFaces(M_u);
   out.mass_taken = dh->CanBatchFluxMass(M_u);

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   // NONZERO on the essential flux dofs, which is what makes Ae do anything:
   // the elimination forms bu -= A_e u_e and u_e is exactly this.
   for (int i = 0; i < ess_flux.Size(); i++)
   {
      x.GetBlock(0)(ess_flux[i]) = 0.3 + 0.1 * (i % 5);
   }
   x.SyncFromBlocks();

   OperatorHandle R;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux, x, R, X, B, true);
   out.B.SetSize(B.Size());
   out.B = B;
   out.B.HostRead();
}

/** @brief s * I on the adjacent element's block, for the H(div) harness below.

    An RT element has no scalar shape, so BdrFluxMassIntegrator's face mass
    cannot be written for it -- CalcShape() aborts on a vector-valued element.
    This is the shape test_darcy_hybridization.cpp's own flux-mass boundary
    test uses, and for the same reason. It is symmetric, so it cannot see a
    transposed read; what it is here for is the ROUTE, not the arithmetic. */
class ScaledIdentityFace : public BilinearFormIntegrator
{
   const real_t s;

public:
   ScaledIdentityFace(real_t s_) : s(s_) { }

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &,
                           FaceElementTransformations &,
                           DenseMatrix &elmat) override
   {
      const int n = el1.GetDof();
      elmat.SetSize(n);
      elmat = 0.0;
      for (int i = 0; i < n; i++) { elmat(i, i) = s; }
   }
};

/** The hybridized MIXED method -- an H(div) flux and an L2 potential -- with
    two boundary face integrators on the flux mass, reduced to the trace
    system. Hands back the assembled trace operator entry for entry.

    The configuration matters more than the numbers. VectorFEMassIntegrator is
    not a family the batched ELEMENT mass implements, so the element pass falls
    back to the host loop and the boundary pass is then the FIRST and only
    kernel in the flux mass group -- the opposite order from every other case
    in this file, and the one where the offsets are host-valid on entry rather
    than device-valid. And it is an H(div) space on the REDUCED route, which
    every other batched face kernel in DarcyHybridization refuses outright. */
void MixedTraceOperator(int order, int n,
                        DarcyHybridization::AssemblyMode am, Out &out)
{
   const int dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false, 1., 1.);
   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll), Wh(&mesh, &p_coll), Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);
   FunctionCoefficient src([](const Vector &X)
   { return std::exp(X(0)) * std::sin(X(1)); });

   BilinearForm *M_u = darcy.GetFluxMassForm();
   M_u->AddDomainIntegrator(new VectorFEMassIntegrator(one));
   M_u->AddBdrFaceIntegrator(new ScaledIdentityFace(0.7));
   M_u->AddBdrFaceIntegrator(new ScaledIdentityFace(0.4));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));

   Array<int> ess;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(am);
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   dh->SetEssentialBC(all);

   darcy.Assemble();
   out.bdr_taken = dh->CanBatchFluxMassBdrFaces(M_u);
   out.mass_taken = dh->CanBatchFluxMass(M_u);
   darcy.Finalize();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   OperatorHandle R;
   Vector X, B;
   darcy.FormLinearSystem(ess, x, R, X, B, true);
   SparseMatrix *Sm = R.As<SparseMatrix>();
   REQUIRE(Sm != nullptr);
   const int nrows = Sm->Height(), nnz = Sm->NumNonZeroElems();
   out.I.SetSize(nrows + 1);
   std::copy(Sm->GetI(), Sm->GetI() + nrows + 1, out.I.begin());
   out.J.SetSize(nnz);
   std::copy(Sm->GetJ(), Sm->GetJ() + nnz, out.J.begin());
   out.data.SetSize(nnz);
   std::copy(Sm->GetData(), Sm->GetData() + nnz, out.data.GetData());
   out.data.HostRead();
}

/// The two routes agree entry for entry, with a floored relative bound.
void RequireSameMatrix(const Out &ref, const Out &got, real_t rtol)
{
   REQUIRE(got.I.Size() == ref.I.Size());
   REQUIRE(got.J.Size() == ref.J.Size());
   for (int i = 0; i < ref.I.Size(); i++) { REQUIRE(got.I[i] == ref.I[i]); }
   for (int i = 0; i < ref.J.Size(); i++) { REQUIRE(got.J[i] == ref.J[i]); }
   Vector d(ref.data);
   d -= got.data;
   const real_t scale = ref.data.Normlinf();
   CAPTURE(d.Normlinf(), scale);
   REQUIRE(d.Normlinf() <= rtol * scale);
}

} // namespace darcy_batched_bdrflux

/**
 * @brief The batched boundary flux-mass pass assembles the per-face loop's
 * blocks, bit for bit.
 *
 * DarcyForm::AssembleFluxMassBdrFaces() was the one assembly loop on the
 * hybridized path with no kernel; AssemblyMode::Batched now replaces its
 * scatter with one mfem::forall, leaving the integrator evaluation on the host
 * because there is no integrator family here to dispatch on -- see
 * BdrFluxMassIntegrator above, which had to be written for this test because
 * the library has nothing that fits.
 *
 * BIT FOR BIT, and that is a design property rather than luck. The work is
 * grouped by ELEMENT and each element is one thread, so a corner element's two
 * boundary faces are summed by the same thread in the loop's own order and
 * every entry of Af sees the same sequence of additions. One thread per
 * contribution would have needed AtomicAdd and would have been round-off at
 * best.
 *
 * The bound here is nevertheless 1e-13 relative and not zero, because what
 * this compares is two whole ASSEMBLIES: switching AssemblyMode also switches
 * the element mass, the divergence and the interior/boundary face kernels, all
 * of which are round-off. Measured with ONLY the boundary pass switched -- an
 * environment gate inside CanBatchFluxMassBdrFaces(), used as a probe and not
 * committed -- the difference is 0.0 exactly, on eight configurations spanning
 * order 0 to 2, quadrilaterals and triangles, one and two integrators, and
 * split markers; and on four reduced-route configurations for the Ae half. The
 * worst Serial-against-Batched difference over the 18 combinations below is
 * 8.3e-13 against a matrix norm of 58, i.e. 1.4e-14 relative, on quads at
 * order 2 -- so 1e-13 leaves a factor of seven, and it still discriminates by
 * a wide margin: dropping the Ae branch of the kernel, or transposing its read
 * of the packed block, fails these cases at 1e-01 and worse.
 */
TEST_CASE("The batched boundary flux mass assembles the per-face blocks",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_bdrflux;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(0, 1, 2);
   const MeshKind kind = GENERATE(MeshKind::Quad, MeshKind::Tri);
   // One integrator; two, which is what makes the pass accumulate against
   // itself as well as against the element block; and two with disjoint
   // attribute markers, which is the structural difference from an interior
   // face pass -- two boundary integrators need not apply to the same faces.
   const int which = GENERATE(0, 1, 2);
   CAPTURE(order, Name(kind), which);

   Cfg cfg;
   cfg.order = order;
   cfg.kind = kind;
   cfg.nbdr = (which == 0) ? 1 : 2;
   cfg.markers = (which == 2) ? Markers::Split : Markers::All;

   Out ref, got;
   cfg.am = AM::Serial;
   Gradient(cfg, ref);
   cfg.am = AM::Batched;
   Gradient(cfg, got);

   // The kernel was actually TAKEN, and the reference route did not take it.
   // Two fallbacks would agree perfectly and test nothing -- which is how
   // AssemblyMode::Batched's face kernel stayed dead code for three commits.
   REQUIRE_FALSE(ref.bdr_taken);
   REQUIRE(got.bdr_taken);

   REQUIRE(ref.data.Size() > 0);
   REQUIRE(ref.data.Normlinf() > 1e-6);
   RequireSameMatrix(ref, got, 1e-13);

   // And the boundary term is not a no-op, or the agreement above is vacuous.
   Cfg none = cfg;
   none.nbdr = 0;
   none.am = AM::Batched;
   Out plain;
   Gradient(none, plain);
   REQUIRE_FALSE(plain.bdr_taken);
   Vector d(ref.data);
   d -= plain.data;
   CAPTURE(d.Normlinf(), ref.data.Normlinf());
   REQUIRE(d.Normlinf() > 1e-3 * ref.data.Normlinf());
}

/**
 * @brief An arithmetically inert boundary flux-mass integrator changes
 * nothing, which is what tells an assigning kernel from an accumulating one.
 *
 * The scale here is ZERO, so every block the pass writes is exactly zero and
 * adding it must leave the operator bit-identical to the one assembled with no
 * boundary integrator at all. A kernel that ASSIGNED would instead replace the
 * whole flux mass block of every element touching the boundary with zero, and
 * the answer would be nonsense -- silently, and only on those elements.
 *
 * That is not a hypothetical failure mode. It is the recorded history of
 * DarcyHybridization::AssembleFluxMassMatrix(), which assigned and is called a
 * second time for the element owning a boundary face; hybridized then
 * disagreed with monolithic by 5-6%. The inert-knob form is the sharp one,
 * because the discrete problem is unchanged by construction, so whichever
 * route fails to return the no-integrator answer is the wrong one.
 *
 * The gate is asserted TRUE with the inert integrator installed: an integrator
 * whose value is zero is still work to be visited, and if the pass were
 * refused this case would be comparing two identical fallbacks.
 */
TEST_CASE("An inert boundary flux mass integrator changes nothing, batched",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_bdrflux;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(0, 1, 2);
   const int nbdr = GENERATE(1, 2);
   CAPTURE(order, nbdr);

   Cfg zero;
   zero.order = order;
   zero.nbdr = nbdr;
   zero.scale = 0.0;
   zero.am = AM::Batched;

   Cfg none = zero;
   none.nbdr = 0;

   Out with_inert, without;
   Gradient(zero, with_inert);
   Gradient(none, without);

   REQUIRE(with_inert.bdr_taken);
   REQUIRE_FALSE(without.bdr_taken);
   REQUIRE(without.data.Size() > 0);
   REQUIRE(without.data.Normlinf() > 1e-6);

   // BITWISE. Adding exactly zero, in the same order, is the identity.
   RequireSameMatrix(without, with_inert, 0.0);
}

/**
 * @brief The batched boundary flux-mass pass fills Ae, for the right-hand side
 * elimination.
 *
 * Ae_data holds each element's essential flux COLUMNS taken with every row,
 * and the only reader is `bu -= A_e u_e` in EliminateVDofsInRHS(). So a
 * boundary contribution to Ae shows up in the reduced right-hand side and
 * nowhere else -- not in the operator, and not under NPC at all, which never
 * calls that path. The gradient comparison above therefore cannot see it, and
 * the same trap has already been paid for once on the element pass, where a
 * comparison written believing it covered Ae kept passing with the Ae branch
 * disabled.
 */
TEST_CASE("The batched boundary flux mass fills Ae for the RHS elimination",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_bdrflux;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(1, 2);
   const int nbdr = GENERATE(1, 2);
   CAPTURE(order, nbdr);

   Cfg cfg;
   cfg.order = order;
   cfg.nbdr = nbdr;
   cfg.ess_flux = true;

   Out ref, got;
   cfg.am = AM::Serial;
   ReducedRHS(cfg, ref);
   cfg.am = AM::Batched;
   ReducedRHS(cfg, got);

   REQUIRE_FALSE(ref.bdr_taken);
   REQUIRE(got.bdr_taken);

   // The elimination did something: with a zero essential state the two
   // routes would agree whatever Ae held.
   REQUIRE(ref.B.Size() > 0);
   REQUIRE(ref.B.Normlinf() > 1e-6);

   REQUIRE(got.B.Size() == ref.B.Size());
   Vector d(ref.B);
   d -= got.B;
   CAPTURE(d.Normlinf(), ref.B.Normlinf());
   REQUIRE(d.Normlinf() <= 1e-13 * ref.B.Normlinf());
}

/**
 * @brief A periodic mesh's leftover boundary elements are dropped by the
 * batched pass, as the loop drops them.
 *
 * Mesh::MakePeriodic identifies the two ends, so the faces there become
 * INTERIOR -- and the boundary elements that sat on them remain, with
 * GetBdrElementFaceIndex() still naming the now-interior face.
 * Mesh::GetBdrFaceTransformations() returning null is how every boundary loop
 * in fem/darcy drops them, and the batched work list is built from that same
 * test rather than restating it.
 *
 * Two halves, and the first is the sharper. With the integrator marked on
 * exactly the identified attributes there is NO admitted work, so the gate
 * must report false and the operator must be bit-identical to the one with no
 * boundary integrator at all -- a pass that walked the boundary elements
 * blindly would add a term on the wrong (interior) faces' elements. With it
 * marked on the attributes the mesh keeps, the gate must report true and the
 * two routes must agree, so a mesh that mixes leftover and genuine boundary
 * elements is covered as well.
 */
TEST_CASE("The batched boundary flux mass drops a periodic mesh's leftovers",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_bdrflux;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(0, 1);
   CAPTURE(order);

   Cfg base;
   base.order = order;
   base.kind = MeshKind::PeriodicX;
   base.n = 4;

   Cfg none = base;
   none.nbdr = 0;
   none.am = AM::Batched;
   Out without;
   Gradient(none, without);
   REQUIRE(without.data.Size() > 0);
   REQUIRE(without.data.Normlinf() > 1e-6);

   // On the identified attributes only: nothing to do, and the gate says so.
   Cfg dead = base;
   dead.markers = Markers::Periodic;
   dead.am = AM::Batched;
   Out on_leftovers;
   Gradient(dead, on_leftovers);
   REQUIRE_FALSE(on_leftovers.bdr_taken);
   RequireSameMatrix(without, on_leftovers, 0.0);

   // On the attributes the periodic mesh keeps: real work, and the two routes
   // must agree on a mesh where only some boundary elements are admitted.
   Cfg live = base;
   live.markers = Markers::Physical;
   Out ref, got;
   live.am = AM::Serial;
   Gradient(live, ref);
   live.am = AM::Batched;
   Gradient(live, got);
   REQUIRE_FALSE(ref.bdr_taken);
   REQUIRE(got.bdr_taken);
   RequireSameMatrix(ref, got, 1e-13);

   Vector d(ref.data);
   d -= without.data;
   CAPTURE(d.Normlinf(), ref.data.Normlinf());
   REQUIRE(d.Normlinf() > 1e-3 * ref.data.Normlinf());
}

/**
 * @brief The batched boundary flux mass on an H(div) flux, where it is the
 * ONLY kernel in the flux mass group.
 *
 * VectorFEMassIntegrator is not a family HDGElementMassBatched() implements,
 * so with an RT flux space the element pass falls back to the host loop and
 * this pass runs first rather than second. That is the opposite memory
 * situation from every other case here -- the offset arrays are host-valid on
 * entry, not device-valid -- and it is worth pinning both ways round, since
 * the three HostRead() calls the other order needs must also be harmless in
 * this one.
 *
 * It is also the REDUCED route on an H(div) space, which no other batched face
 * kernel in DarcyHybridization will take: the two potential face kernels
 * require NPC because they write H into H_data, and this one writes only Af
 * and Ae. So this case is the whole of the evidence that the pass is not an
 * NPC-only facility.
 *
 * BITWISE, and here it can be: the element mass, the divergence and the
 * potential mass all take the same host route in both modes, so the boundary
 * pass is the only thing that differs and it is bit-for-bit by construction.
 * That makes this the sharpest comparison in the file -- and it says the
 * grouping is right, not merely close.
 */
TEST_CASE("The batched boundary flux mass on an H(div) flux, reduced route",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_bdrflux;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(0, 1, 2);
   const int n = GENERATE(3, 4);
   CAPTURE(order, n);

   Out ref, got;
   MixedTraceOperator(order, n, AM::Serial, ref);
   MixedTraceOperator(order, n, AM::Batched, got);

   // The boundary pass was taken and the ELEMENT pass was not, in either mode.
   REQUIRE_FALSE(ref.bdr_taken);
   REQUIRE(got.bdr_taken);
   REQUIRE_FALSE(ref.mass_taken);
   REQUIRE_FALSE(got.mass_taken);

   REQUIRE(ref.data.Size() > 0);
   REQUIRE(ref.data.Normlinf() > 1e-6);
   RequireSameMatrix(ref, got, 0.0);
}
