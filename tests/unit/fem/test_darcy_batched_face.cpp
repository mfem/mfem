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

namespace darcy_batched_face
{

/// Which face terms go on the potential mass form.
enum class FaceTerm
{
   Diffusion,             ///< HDGDiffusionIntegrator, scalar coefficient
   DiffusionVel,          ///< ... with a velocity, so alpha and beta bite
   DiffusionMat,          ///< ... with a matrix coefficient
   Centered,              ///< HDGConvectionCenteredIntegrator alone
   Upwinded,              ///< HDGConvectionUpwindedIntegrator alone
   DiffusionCentered,     ///< a SumIntegrator of two
   DiffusionUpwinded      ///< a SumIntegrator of two
};

const char *Name(FaceTerm t)
{
   switch (t)
   {
      case FaceTerm::Diffusion:         return "diffusion";
      case FaceTerm::DiffusionVel:      return "diffusion+velocity";
      case FaceTerm::DiffusionMat:      return "diffusion+matrix";
      case FaceTerm::Centered:          return "centred convection";
      case FaceTerm::Upwinded:          return "upwinded convection";
      case FaceTerm::DiffusionCentered: return "diffusion + centred";
      case FaceTerm::DiffusionUpwinded: return "diffusion + upwinded";
   }
   return "?";
}

/// How many integrators @a t puts on the constraint.
int NumIntegs(FaceTerm t)
{
   return (t == FaceTerm::DiffusionCentered ||
           t == FaceTerm::DiffusionUpwinded) ? 2 : 1;
}

/// Which mass terms go on the two element blocks.
enum class MassTerm
{
   Plain,        ///< VectorMassIntegrator(), no coefficient
   Scalar,       ///< ... with a Coefficient
   Diagonal,     ///< ... with a VectorCoefficient
   Matrix,       ///< ... with a MatrixCoefficient, the coupled block
   PotScalar     ///< a MassIntegrator on the potential block as well
};

const char *Name(MassTerm t)
{
   switch (t)
   {
      case MassTerm::Plain:     return "flux mass, no coefficient";
      case MassTerm::Scalar:    return "flux mass, scalar";
      case MassTerm::Diagonal:  return "flux mass, diagonal";
      case MassTerm::Matrix:    return "flux mass, matrix";
      case MassTerm::PotScalar: return "flux + potential mass";
   }
   return "?";
}

struct Outcome
{
   Array<int> I, J;
   Vector data;
   bool taken = false;
   bool bdr_taken = false;
   bool flux_mass_taken = false;
   bool pot_mass_taken = false;
   bool div_taken = false;
   int nintegs = 0;
};

/** Assemble the NPC trace gradient once, in the given assembly mode, and hand
    back the matrix entry for entry.

    The GRADIENT and not a solution, deliberately. A solution goes through an
    iterative trace solve whose tolerance is looser than the difference a
    mis-assembled face block makes, so it can absorb one; the assembled
    operator cannot. And it is the NPC gradient because that is the route on
    which the batched face kernel is reachable at all -- it writes H into
    H_data, where only NPC reads it. */
void AssembleGradient(Mesh &mesh, int order, FaceTerm term,
                      DarcyHybridization::AssemblyMode am, Outcome &out,
                      MassTerm mass = MassTerm::Plain,
                      bool ess_flux_dofs = false,
                      int neq = 1, bool repeat_integ = false)
{
   const int dim = mesh.Dimension();

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   // vdim = neq on the potential and the trace, neq*dim on the flux, and
   // Ordering::byNODES -- the layout every system caller in the tree uses
   // (navierstokes, examples/hdg/ex17 and ex21, test_darcy_system).
   FiniteElementSpace Vh(&mesh, &u_coll, neq * dim, Ordering::byNODES);
   FiniteElementSpace Wh(&mesh, &p_coll, neq, Ordering::byNODES);
   FiniteElementSpace Mh(&mesh, &t_coll, neq, Ordering::byNODES);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);
   // Deliberately not axis-aligned and not constant along the face, so that
   // sign(u.n) changes from face to face and the upwinded form's crossing of
   // the two sides is actually exercised.
   VectorFunctionCoefficient vel(dim, [](const Vector &X, Vector &v)
   {
      v(0) = 1.0 + 0.5 * std::sin(M_PI * X(1));
      v(1) = -0.7 + 0.3 * std::cos(M_PI * X(0));
   });
   MatrixFunctionCoefficient mat(dim, [](const Vector &X, DenseMatrix &m)
   {
      m.SetSize(2);
      m(0, 0) = 2.0 + X(0);
      m(1, 1) = 1.0 + X(1);
      m(0, 1) = m(1, 0) = 0.25;
   });
   FunctionCoefficient kappa([](const Vector &X)
   {
      return 1.0 + 0.5 * X(0) * X(1);
   });

   VectorFunctionCoefficient dcoeff(dim, [](const Vector &X, Vector &v)
   {
      v(0) = 1.5 + 0.4 * X(0);
      v(1) = 0.8 + 0.3 * X(1);
   });

   // PER-EQUATION coefficients, and they are the whole reason a vdim > 1 case
   // can discriminate an equation index. With one coefficient shared across
   // the blocks, permuting the equations leaves the operator unchanged and the
   // comparison below would pass on a scatter that wrote equation 1's block
   // where equation 0's belongs.
   std::vector<std::unique_ptr<Coefficient>> kap_e(neq);
   std::vector<std::unique_ptr<VectorCoefficient>> vel_e(neq);
   std::vector<std::unique_ptr<MatrixCoefficient>> mat_e(neq);
   for (int e = 0; e < neq; e++)
   {
      const real_t s = 1.0 + 0.75 * e;
      kap_e[e].reset(new FunctionCoefficient([s](const Vector &X)
      {
         return s * (1.0 + 0.5 * X(0) * X(1));
      }));
      vel_e[e].reset(new VectorFunctionCoefficient(
                        dim, [s](const Vector &X, Vector &v)
      {
         v(0) = s * (1.0 + 0.5 * std::sin(M_PI * X(1)));
         v(1) = -0.7 * s + 0.3 * std::cos(M_PI * X(0));
      }));
      mat_e[e].reset(new MatrixFunctionCoefficient(
                        dim, [s](const Vector &X, DenseMatrix &m)
      {
         m.SetSize(2);
         m(0, 0) = s * (2.0 + X(0));
         m(1, 1) = 1.0 + s * X(1);
         m(0, 1) = m(1, 0) = 0.25 * s;
      }));
   }

   /// One scalar face integrator of family @a t for equation @a e.
   auto mk = [&](FaceTerm t, int e) -> BilinearFormIntegrator *
   {
      Coefficient &kq = (neq == 1) ? kappa : *kap_e[e];
      VectorCoefficient &vq = (neq == 1) ? (VectorCoefficient&)vel : *vel_e[e];
      MatrixCoefficient &mq = (neq == 1) ? (MatrixCoefficient&)mat : *mat_e[e];
      switch (t)
      {
         case FaceTerm::Diffusion:    return new HDGDiffusionIntegrator(kq, 1.0);
         case FaceTerm::DiffusionVel: return new HDGDiffusionIntegrator(vq, kq, 1.0);
         case FaceTerm::DiffusionMat: return new HDGDiffusionIntegrator(mq, 1.0);
         case FaceTerm::Centered:
            return new HDGConvectionCenteredIntegrator(vq, 1.0);
         default:
            return new HDGConvectionUpwindedIntegrator(vq, 1.0, 0.5);
      }
   };

   /// Register family @a t on @a form, wrapped for the vector dimension.
   auto add = [&](BilinearForm *form, bool interior, FaceTerm t)
   {
      BilinearFormIntegrator *bfi;
      if (neq == 1)
      {
         bfi = mk(t, 0);
      }
      else if (repeat_integ)
      {
         // examples/hdg/ex17 and ex21's shape: ONE integrator replicated, so
         // VectorBlockDiagonalIntegrator holds one where it has neq blocks.
         // GetIntegrator(i) is out of bounds for i > 0 there, which is why
         // HDGFaceBlocksOf() asks GetNumIntegrators() first.
         bfi = new VectorBlockDiagonalIntegrator(neq, mk(t, 0));
      }
      else
      {
         std::vector<BilinearFormIntegrator*> blks(neq);
         for (int e = 0; e < neq; e++) { blks[e] = mk(t, e); }
         bfi = new VectorBlockDiagonalIntegrator(blks);
      }
      if (interior) { form->AddInteriorFaceIntegrator(bfi); }
      else { form->AddBdrFaceIntegrator(bfi); }
   };

   /// A vdim-shaped wrapper for a non-HDG integrator, or the integrator.
   auto wrap = [&](BilinearFormIntegrator *bfi) -> BilinearFormIntegrator *
   {
      return (neq == 1) ? bfi
      : new VectorBlockDiagonalIntegrator(neq, bfi);
   };

   switch (mass)
   {
      case MassTerm::Plain:
         darcy.GetFluxMassForm()->AddDomainIntegrator(
            wrap(new VectorMassIntegrator()));
         break;
      case MassTerm::Diagonal:
         darcy.GetFluxMassForm()->AddDomainIntegrator(
            wrap(new VectorMassIntegrator(dcoeff)));
         break;
      case MassTerm::Matrix:
         darcy.GetFluxMassForm()->AddDomainIntegrator(
            wrap(new VectorMassIntegrator(mat)));
         break;
      default:
         darcy.GetFluxMassForm()->AddDomainIntegrator(
            wrap(new VectorMassIntegrator(kappa)));
         break;
   }
   if (mass == MassTerm::PotScalar)
   {
      darcy.GetPotentialMassForm()->AddDomainIntegrator(
         wrap(new MassIntegrator(kappa)));
   }
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      wrap(new VectorDivergenceIntegrator()));
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      wrap(new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0))));

   BilinearForm *M_p = darcy.GetPotentialMassForm();
   switch (term)
   {
      case FaceTerm::Diffusion:
      case FaceTerm::DiffusionVel:
      case FaceTerm::DiffusionMat:
      case FaceTerm::Centered:
      case FaceTerm::Upwinded:
         add(M_p, true, term);
         break;
      case FaceTerm::DiffusionCentered:
         add(M_p, true, FaceTerm::Diffusion);
         add(M_p, true, FaceTerm::Centered);
         break;
      case FaceTerm::DiffusionUpwinded:
         add(M_p, true, FaceTerm::Diffusion);
         add(M_p, true, FaceTerm::Upwinded);
         break;
   }
   // The SAME terms on the boundary, which has its own kernel and its own
   // weights -- one-sided throughout, and the upwinded form's trace weight
   // does not reduce to the sum of the two sides there. Registering a
   // diffusion term on every case's boundary would have left the convection
   // boundary weights untested, which is how a real defect survived.
   switch (term)
   {
      case FaceTerm::Centered:
      case FaceTerm::Upwinded:
         add(M_p, false, term);
         break;
      case FaceTerm::DiffusionCentered:
         add(M_p, false, FaceTerm::Diffusion);
         add(M_p, false, FaceTerm::Centered);
         break;
      case FaceTerm::DiffusionUpwinded:
         add(M_p, false, FaceTerm::Diffusion);
         add(M_p, false, FaceTerm::Upwinded);
         break;
      default:
         add(M_p, false, FaceTerm::Diffusion);
         break;
   }

   Array<int> ess_flux;
   if (ess_flux_dofs)
   {
      // A HAND-MADE essential flux list, because the natural route cannot
      // produce one here: the flux space is L2, so it has no boundary dofs
      // and GetEssentialTrueDofs() returns nothing whatever the marker says.
      // Without this the Ae branch of the batched flux scatter -- an
      // essential COLUMN, taken with every row of the element -- is never
      // reached by any case in the suite.
      for (int i = 0; i < Vh.GetVSize(); i += 7) { ess_flux.Append(i); }
   }
   darcy.EnableHybridization(&Mh, wrap(new NormalTraceJumpIntegrator()),
                             ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(am);
   // A LINEAR problem with NPC asked for explicitly. NPCEnabled() is what
   // sends H to H_data, and a nonlinear potential mass would take the
   // constraint to c_nlfi_p instead, where there is no batched kernel at all.
   dh->EnableNPC();
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   dh->SetEssentialBC(ess_bdr);

   darcy.Assemble();
   darcy.Finalize();

   out.nintegs = dh->NumPotFaceConstraintIntegrators();
   out.taken = dh->CanBatchPotFaceAssembly();
   out.bdr_taken = dh->CanBatchPotBdrFaceAssembly();
   out.flux_mass_taken = dh->CanBatchFluxMass(darcy.GetFluxMassForm());
   out.pot_mass_taken = dh->CanBatchPotMass(darcy.GetPotentialMassForm());
   out.div_taken = dh->CanBatchDiv(darcy.GetFluxDivForm());

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

} // namespace darcy_batched_face

/**
 * @brief The batched face kernel assembles the per-face loop's operator, for
 * every face term it claims to cover.
 *
 * AssemblyMode::Batched replaces the per-face host loop over the potential
 * mass constraint with one kernel that scatters into E, G, H and D. It used to
 * cover a lone pure-diffusion term; it now covers a diffusion term with a
 * velocity or a matrix coefficient, both HDGConvection*Integrators, and a
 * SumIntegrator of them.
 *
 * The three families weigh a quadrature point differently and it is not a
 * detail: diffusion puts one weight on D, E and G per side; the centred form
 * puts E on a different weight from D and G; and the upwinded form CROSSES
 * them, side 1's E carrying side 2's weight. A kernel that got that wrong
 * would still produce a plausible operator, so this compares the assembled
 * matrix entry for entry rather than a solution -- a trace solve at a finite
 * tolerance can absorb an error the operator cannot hide.
 *
 * That the two modes agree is not enough on its own: this also requires the
 * kernel to have been TAKEN. It silently falls back, and for most problems it
 * does, so a comparison of two fallbacks would pass while testing nothing.
 * That is not hypothetical -- the mode was unreachable for every caller in the
 * tree for three commits, because DarcyForm wraps the constraint in a
 * SumIntegrator and the gate looked for a bare integrator.
 */
TEST_CASE("The batched HDG face kernel assembles the per-face operator",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_face;
   using AM = DarcyHybridization::AssemblyMode;

   const FaceTerm term = GENERATE(FaceTerm::Diffusion,
                                  FaceTerm::DiffusionVel,
                                  FaceTerm::DiffusionMat,
                                  FaceTerm::Centered,
                                  FaceTerm::Upwinded,
                                  FaceTerm::DiffusionCentered,
                                  FaceTerm::DiffusionUpwinded);
   const int order = GENERATE(0, 1, 2);
   const int n = GENERATE(2, 4);
   CAPTURE(Name(term), order, n);

   Mesh mesh_a = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   Mesh mesh_b = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);

   Outcome ref, got;
   AssembleGradient(mesh_a, order, term, AM::Serial, ref);
   AssembleGradient(mesh_b, order, term, AM::Batched, got);

   // The kernel was actually taken, and on the number of integrators the
   // case means to exercise.
   REQUIRE_FALSE(ref.taken);
   REQUIRE(got.taken);
   // The BOUNDARY kernel too. Asserting only the interior one would have let
   // the boundary route fall back and pass while testing nothing -- and the
   // boundary kernel is where the defect was.
   REQUIRE_FALSE(ref.bdr_taken);
   REQUIRE(got.bdr_taken);
   REQUIRE(ref.nintegs == NumIntegs(term));
   REQUIRE(got.nintegs == NumIntegs(term));

   // There is an operator to compare, and it is not the zero one.
   REQUIRE(ref.data.Size() > 0);
   REQUIRE(ref.data.Normlinf() > 1e-6);

   // The sparsity, which a wrong scatter would move even where the values
   // happened to agree.
   REQUIRE(got.I.Size() == ref.I.Size());
   REQUIRE(got.J.Size() == ref.J.Size());
   for (int i = 0; i < ref.I.Size(); i++) { REQUIRE(got.I[i] == ref.I[i]); }
   for (int i = 0; i < ref.J.Size(); i++) { REQUIRE(got.J[i] == ref.J[i]); }

   // The entries. NOT bitwise, and the reason is worth stating: the per-face
   // route sums every integrator into one element matrix and adds that to D
   // once, while the kernel accumulates point by point and integrator by
   // integrator. Same arithmetic, different association, so the difference is
   // round-off relative to the block -- and the tolerance is relative to the
   // matrix norm rather than to each entry, since H has entries of every size.
   //
   // The bound is MEASURED and not guessed: run with a tolerance of zero, the
   // worst of the 42 combinations here is 2.2e-15 against a matrix norm of
   // 3.18, i.e. 7.0e-16 relative. 1e-14 leaves an order of headroom and is
   // still far inside anything a wrong weight would produce: removing the
   // upwinded form's E crossing -- side 1's E carries side 2's weight -- fails
   // 12 assertions over 10 of the 42 combinations, worst 9.9e-02 relative.
   Vector d(ref.data);
   d -= got.data;
   const real_t scale = ref.data.Normlinf();
   CAPTURE(d.Normlinf(), scale);
   REQUIRE(d.Normlinf() <= 1e-14 * scale);
}

/**
 * @brief The same, on a SYSTEM: a face constraint of vector dimension > 1.
 *
 * `HDGFaceSpacesCanBatch()` used to refuse `vdim != 1` outright, and the
 * device-offload plan filed that as one refusal. It is two, and the measurement
 * says so: relaxing the vdim test alone leaves the kernel refused, because at
 * vdim > 1 the only shape any caller installs is a
 * VectorBlockDiagonalIntegrator and `HDGBatchKindOf()` reported that
 * Unsupported. Both had to go, and the second is what HDGFaceBlocksOf() is.
 *
 * WHAT IS BEING TESTED IS THE INDEXING, not the arithmetic: a pass writes one
 * equation's diagonal sub-block, at row/column offset e*ND inside a block of
 * leading dimension ND*neq, and the per-quadrature-point weights are the
 * scalar kernel's untouched. Two things make the comparison able to fail:
 *
 *  - PER-EQUATION coefficients, so permuting the equations changes the
 *    operator. With one coefficient shared, a scatter that put equation 1's
 *    block where equation 0's belongs would agree to the last bit.
 *  - order >= 1 TOGETHER WITH neq >= 2. At order 0 an element has one dof per
 *    equation and a face one trace dof per equation, so `e*ND + i` and
 *    `i*neq + e` -- field-outermost against interleaved -- are the same index,
 *    and that wrong layout passes. That is the same blind spot the batched
 *    divergence kernel has, and it is measured below rather than argued.
 *
 * MEASURED, by writing the wrong layout on purpose -- five breakages, each
 * gated on an environment variable inside the kernel so one TU rebuilds, of
 * the 60 combinations here:
 *
 *  - interleaving the element and trace index (`i*neq + e` for `e*ND + i`) in
 *    the INTERIOR kernel: 40 fail, and the 20 that pass carry `order := 0`
 *    exactly, none carry order 1 or 2. That is the predicted blind spot
 *    arriving on the nose.
 *  - the same in the BOUNDARY kernel: 40 fail. So the boundary pass is
 *    reached and checked, which is not free -- the boundary kernel is where
 *    the scalar case's defect was.
 *  - reversing the trace equation index (`(neq-1-e)*TRD`), interior: 60 fail;
 *    boundary: 60 fail here, and the [DebugDevice] companion case PASSES it,
 *    because that one compares the recovered SOLUTION and this test's
 *    boundary trace dofs are all essential, so the damage lands entirely in
 *    eliminated rows. Comparing the operator rather than a solve is what
 *    catches it.
 *  - dropping the equation offsets, every pass writing block 0: the first
 *    combinations fail on sparsity and the third aborts, the local solve
 *    having gone singular.
 *  - permuting the WEIGHTS between blocks while writing at the right offsets
 *    (equation e's block filled from equation neq-1-e's integrator): 30 fail,
 *    and they are exactly the 30 with `repeated := false`. The other 30 share
 *    one coefficient across the blocks, so the permutation really is a no-op
 *    there. This is the one breakage the VALUE comparison catches rather than
 *    the sparsity one -- the other four move a nonzero and are caught by
 *    `got.J.Size() == ref.J.Size()` before any value is looked at.
 */
TEST_CASE("The batched HDG face kernel assembles a vdim > 1 face constraint",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_face;
   using AM = DarcyHybridization::AssemblyMode;

   const FaceTerm term = GENERATE(FaceTerm::Diffusion,
                                  FaceTerm::DiffusionVel,
                                  FaceTerm::DiffusionMat,
                                  FaceTerm::Upwinded,
                                  FaceTerm::DiffusionUpwinded);
   // order 0 is kept as a control, not as coverage: it is the one order at
   // which this case CANNOT discriminate, and the write-up above says so.
   const int order = GENERATE(0, 1, 2);
   const int neq = GENERATE(2, 3);
   // The single-repeated wrapper is examples/hdg/ex17's shape and reaches a
   // different branch of HDGFaceBlocksOf(); it shares one coefficient across
   // the blocks, so it cannot discriminate an equation swap and is not asked
   // to -- what it establishes is that GetNumIntegrators() < GetNumBlocks()
   // is handled rather than read out of bounds.
   const bool repeated = GENERATE(false, true);
   CAPTURE(Name(term), order, neq, repeated);

   const int n = 2;
   Mesh mesh_a = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   Mesh mesh_b = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);

   Outcome ref, got;
   AssembleGradient(mesh_a, order, term, AM::Serial, ref, MassTerm::Plain,
                    false, neq, repeated);
   AssembleGradient(mesh_b, order, term, AM::Batched, got, MassTerm::Plain,
                    false, neq, repeated);

   REQUIRE_FALSE(ref.taken);
   REQUIRE(got.taken);
   REQUIRE_FALSE(ref.bdr_taken);
   REQUIRE(got.bdr_taken);
   REQUIRE(ref.nintegs == NumIntegs(term));
   REQUIRE(got.nintegs == NumIntegs(term));

   REQUIRE(ref.data.Size() > 0);
   REQUIRE(ref.data.Normlinf() > 1e-6);

   REQUIRE(got.I.Size() == ref.I.Size());
   REQUIRE(got.J.Size() == ref.J.Size());
   for (int i = 0; i < ref.I.Size(); i++) { REQUIRE(got.I[i] == ref.I[i]); }
   for (int i = 0; i < ref.J.Size(); i++) { REQUIRE(got.J[i] == ref.J[i]); }

   // Round-off relative to the matrix norm, for the reason the scalar case
   // gives: the per-face route sums the integrators into one element matrix
   // and the kernel accumulates pass by pass, so the association differs.
   // Worst over these 60 combinations with the tolerance set to zero:
   // 1.78e-15 against a norm of 4.14, i.e. 4.3e-16 relative -- so 1e-14 leaves
   // a factor of 23, the same headroom the scalar case measured for itself.
   Vector d(ref.data);
   d -= got.data;
   const real_t scale = ref.data.Normlinf();
   CAPTURE(d.Normlinf(), scale);
   REQUIRE(d.Normlinf() <= 1e-14 * scale);
}

/**
 * @brief The batched element-mass kernel assembles the per-element loop's
 * blocks, for every coefficient shape VectorMassIntegrator carries.
 *
 * AssemblyMode::Batched replaces DarcyForm::Assemble()'s two element loops --
 * `ComputeElementMatrix` then `AssembleFluxMassMatrix` / `AssemblePotMassMatrix`
 * per element -- with one kernel per form. The flux one is the interesting
 * half: its block is masked, a free column going to Af in its own compacted
 * indexing and an essential one going to Ae with every row of the element, and
 * the coefficient can be scalar, diagonal or a fully coupled matrix.
 *
 * The matrix case is the one that would be easy to get wrong and easy to miss:
 * every other shape leaves the vdim blocks uncoupled, so an off-diagonal block
 * written to the wrong place, or not at all, changes nothing until a
 * MatrixCoefficient appears.
 *
 * This route is deliberately NOT MFEM's AssemblyLevel::ELEMENT, and the test
 * would not exist if it could be: EABilinearFormExtension has no notion of
 * vdim -- it sizes ea_data as ne*ndof*ndof with a scalar ndof -- so the flux
 * space, which is L2 with vdim = dim, cannot go through it at all; and for a
 * DG space it folds the form's FACE terms into the element matrices, which is
 * exactly the work DarcyForm routes into the constraint blocks itself.
 */
TEST_CASE("The batched HDG element mass assembles the per-element blocks",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_face;
   using AM = DarcyHybridization::AssemblyMode;

   const MassTerm mass = GENERATE(MassTerm::Plain, MassTerm::Scalar,
                                  MassTerm::Diagonal, MassTerm::Matrix,
                                  MassTerm::PotScalar);
   const int order = GENERATE(0, 1, 2);
   const int n = GENERATE(2, 4);
   // TRIANGLES as well as quads, and that is the point rather than extra
   // coverage: MFEM's own element assembly is tensor-product only -- both its
   // paths refuse a simplex, one in GetDofToQuad and one in AssembleEA_ --
   // so a simplex is precisely where an upstream EA route could never serve
   // this and these kernels have to. See doc/UPSTREAM-SPLIT.md.
   const Element::Type geom = GENERATE(Element::QUADRILATERAL,
                                       Element::TRIANGLE);
   CAPTURE(Name(mass), order, n, (int)geom);

   Mesh mesh_a = Mesh::MakeCartesian2D(n, n, geom);
   Mesh mesh_b = Mesh::MakeCartesian2D(n, n, geom);

   Outcome ref, got;
   AssembleGradient(mesh_a, order, FaceTerm::Diffusion, AM::Serial, ref, mass);
   AssembleGradient(mesh_b, order, FaceTerm::Diffusion, AM::Batched, got, mass);

   // The flux kernel was taken, and the potential one exactly when there is a
   // potential mass domain integrator to take. Without this the comparison
   // could be two fallbacks agreeing perfectly.
   REQUIRE_FALSE(ref.flux_mass_taken);
   REQUIRE(got.flux_mass_taken);
   REQUIRE_FALSE(ref.pot_mass_taken);
   REQUIRE(got.pot_mass_taken == (mass == MassTerm::PotScalar));
   // The DIVERGENCE block too. It is a MixedBilinearForm, and MFEM has no
   // mixed element assembly at all -- SetAssemblyLevel aborts on ELEMENT --
   // so this kernel has no upstream counterpart to fall back to on any
   // element shape.
   REQUIRE_FALSE(ref.div_taken);
   REQUIRE(got.div_taken);

   REQUIRE(ref.data.Size() > 0);
   REQUIRE(ref.data.Normlinf() > 1e-6);

   REQUIRE(got.I.Size() == ref.I.Size());
   REQUIRE(got.J.Size() == ref.J.Size());
   for (int i = 0; i < ref.I.Size(); i++) { REQUIRE(got.I[i] == ref.I[i]); }
   for (int i = 0; i < ref.J.Size(); i++) { REQUIRE(got.J[i] == ref.J[i]); }

   // Round-off for the reason the face case gives: the kernel accumulates
   // point by point where the per-element route adds one matrix.
   Vector d(ref.data);
   d -= got.data;
   const real_t scale = ref.data.Normlinf();
   CAPTURE(d.Normlinf(), scale);
   REQUIRE(d.Normlinf() <= 1e-14 * scale);
}

/** Reduce a LINEAR problem with essential flux values, and hand back the
    reduced right-hand side.

    Not NPC, and that is the point: Ae_data -- the essential COLUMNS of each
    element's flux mass, which the batched scatter writes alongside Af -- is
    read only by EliminateVDofsInRHS()/EliminateTrueDofsInRHS(), on the
    reduced route. NPCGradient() never touches it, so the gradient comparison
    above cannot see it however many essential dofs are present. Measured:
    disabling the Ae branch of the kernel outright leaves that case passing. */
void ReduceWithEssentialFlux(Mesh &mesh, int order,
                             DarcyHybridization::AssemblyMode am,
                             Vector &B_out, bool &taken)
{
   const int dim = mesh.Dimension();
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);
   FunctionCoefficient kappa([](const Vector &X)
   { return 1.0 + 0.5 * X(0) * X(1); });
   FunctionCoefficient src([](const Vector &X)
   { return std::sin(M_PI*X(0))*std::sin(M_PI*X(1)); });

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxMassForm()->AddDomainIntegrator(
      new VectorMassIntegrator(kappa));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
   BilinearForm *M_p = darcy.GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kappa, 1.0));
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kappa, 1.0));

   Array<int> ess_flux;
   for (int i = 0; i < Vh.GetVSize(); i += 7) { ess_flux.Append(i); }
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(am);
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   dh->SetEssentialBC(ess_bdr);

   darcy.Assemble();
   taken = dh->CanBatchFluxMass(darcy.GetFluxMassForm());

   BlockVector x(darcy.GetOffsets());
   x = 0.0;
   // NONZERO on the essential flux dofs, which is what makes Ae do anything:
   // the elimination forms bu -= A_e u_e, and u_e is exactly this.
   for (int i = 0; i < ess_flux.Size(); i++)
   {
      x.GetBlock(0)(ess_flux[i]) = 0.3 + 0.1 * (i % 5);
   }
   x.SyncFromBlocks();

   OperatorHandle R;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux, x, R, X, B, true);
   B_out.SetSize(B.Size());
   B_out = B;
   B_out.HostRead();
}

/**
 * @brief The batched flux mass scatter, with essential flux dofs present.
 *
 * AssembleFluxMassMatrix() splits an element's block by a mask: a free column
 * goes to Af in its own compacted indexing, an ESSENTIAL one goes to Ae with
 * every row of the element. This covers the FIRST -- bypassing the free-dof
 * map fails all eight combinations, so the mask is doing work -- and the
 * second is covered by the reduced-route case below, because Ae never enters
 * the gradient this compares.
 *
 * Every other case in this file reaches neither: the flux space is L2, so
 * GetEssentialTrueDofs() returns nothing and no hat dof is ever marked
 * essential by the natural route.
 *
 * So the list here is made by hand. That is not artificial: EnableHybridization
 * takes whatever true-dof list the caller passes, and the RT pathway that does
 * produce one naturally is refused by the batched mass kernel for a different
 * reason (it carries a VectorFEMassIntegrator, which this kernel does not
 * implement). Building the configuration is the only way to know the branch
 * runs at all.
 *
 * The essential dofs also make the per-element Af blocks different sizes,
 * which is what CanBatchLocalFactor() refuses and this kernel does not have
 * to: the scatter reads each element's own count.
 */
TEST_CASE("The batched HDG flux mass scatter handles essential flux dofs",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_face;
   using AM = DarcyHybridization::AssemblyMode;

   const MassTerm mass = GENERATE(MassTerm::Scalar, MassTerm::Matrix);
   const int order = GENERATE(1, 2);
   const int n = GENERATE(2, 4);
   CAPTURE(Name(mass), order, n);

   Mesh mesh_a = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   Mesh mesh_b = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);

   Outcome ref, got;
   AssembleGradient(mesh_a, order, FaceTerm::Diffusion, AM::Serial, ref, mass,
                    true);
   AssembleGradient(mesh_b, order, FaceTerm::Diffusion, AM::Batched, got, mass,
                    true);

   REQUIRE_FALSE(ref.flux_mass_taken);
   REQUIRE(got.flux_mass_taken);

   REQUIRE(ref.data.Size() > 0);
   REQUIRE(ref.data.Normlinf() > 1e-6);
   REQUIRE(got.I.Size() == ref.I.Size());
   for (int i = 0; i < ref.I.Size(); i++) { REQUIRE(got.I[i] == ref.I[i]); }
   for (int i = 0; i < ref.J.Size(); i++) { REQUIRE(got.J[i] == ref.J[i]); }

   Vector d(ref.data);
   d -= got.data;
   const real_t scale = ref.data.Normlinf();
   CAPTURE(d.Normlinf(), scale);
   REQUIRE(d.Normlinf() <= 1e-14 * scale);
}

/**
 * @brief The batched flux mass scatter's Ae half: the essential columns.
 *
 * Ae_data holds each element's essential flux COLUMNS, taken with every row,
 * and the only thing that reads it is the right-hand side elimination
 * `bu -= A_e u_e` in EliminateVDofsInRHS(). So it shows up in the reduced
 * right-hand side and nowhere else -- not in the operator, and not under NPC
 * at all, which never calls that path.
 *
 * That is why this case exists separately, and it was not foreseen: the
 * gradient comparison above was written believing it covered Ae, and disabling
 * the Ae branch of the kernel left it passing. Checking that the guard fires
 * is what said so.
 */
TEST_CASE("The batched HDG flux mass scatter fills Ae for the RHS elimination",
          "[DarcyHybridization][BatchedLinAlg]")
{
   using namespace darcy_batched_face;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(1, 2);
   const int n = GENERATE(2, 4);
   CAPTURE(order, n);

   Mesh mesh_a = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   Mesh mesh_b = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);

   Vector Bref, Bbat;
   bool tref = true, tbat = false;
   ReduceWithEssentialFlux(mesh_a, order, AM::Serial, Bref, tref);
   ReduceWithEssentialFlux(mesh_b, order, AM::Batched, Bbat, tbat);

   REQUIRE_FALSE(tref);
   REQUIRE(tbat);

   // The elimination did something: with a zero essential state the two
   // routes would agree whatever Ae held.
   REQUIRE(Bref.Size() > 0);
   REQUIRE(Bref.Normlinf() > 1e-6);

   REQUIRE(Bbat.Size() == Bref.Size());
   Vector d(Bref);
   d -= Bbat;
   CAPTURE(d.Normlinf(), Bref.Normlinf());
   REQUIRE(d.Normlinf() <= 1e-13 * Bref.Normlinf());
}
