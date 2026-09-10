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

#include <algorithm>
#include <map>
#include <utility>

using namespace mfem;

namespace darcy_batched_traceh
{

/** The assembled trace operator, and the reason it is captured as a MAP
    rather than as three arrays compared elementwise.

    TraceAssemblyMode::Batched lays each row's columns out one neighbour face
    at a time, which is not the order the serial route leaves them in -- that
    order is the reverse of the order entries were first inserted, and
    AddSubMatrix(skip_zeros) declines to insert an element's exact zeros, so
    it is a function of the VALUES. Measured, not assumed: with element 0
    contributing an exact zero where element 1 does not, the two columns come
    back transposed in the row. So the two matrices are the same operator with
    the same pattern and the same values, in a different storage order, and
    the comparison that says so is a comparison of (row, col) -> value. */
struct Assembled
{
   std::map<std::pair<int,int>, real_t> entries;
   int height = 0;
   int nnz = 0;
   bool can_batch = false;
   Vector q, p, tr;

   void Capture(const SparseMatrix &H)
   {
      height = H.Height();
      nnz = H.NumNonZeroElems();
      const int *I = H.HostReadI();
      const int *J = H.HostReadJ();
      const real_t *V = H.HostReadData();
      for (int r = 0; r < height; r++)
      {
         for (int k = I[r]; k < I[r+1]; k++)
         {
            // A repeated column in one row would silently collapse here, so
            // it is checked rather than assumed away.
            const auto key = std::make_pair(r, J[k]);
            REQUIRE(entries.find(key) == entries.end());
            entries[key] = V[k];
         }
      }
   }
};

/** @brief Every entry of @a ref reproduced by @a got to the BIT, and every
    entry @a got has beyond @a ref exactly zero.

    Bitwise and not to a tolerance, and that is the right criterion rather
    than a strict one: an entry of H is a sum of at most two element
    contributions -- its row's face and its column's face must share an
    element, and a face has at most two -- and a two-term IEEE sum does not
    depend on the order of the terms. So the batched route adds the same two
    doubles the loop does, and any difference at all is a defect.

    The one-sided slack is for the pattern. The serial route's pattern is what
    AddSubMatrix(skip_zeros=1) inserted, which is measurably the structural
    pattern on every mesh tried -- every structurally present entry is nonzero
    in at least one of its elements -- but nothing guarantees that, and where
    it fails the batched route carries an extra entry that is exactly 0.0 and
    changes no product. */
void RequireSameOperator(const Assembled &ref, const Assembled &got)
{
   REQUIRE(got.height == ref.height);
   int extra = 0;
   for (const auto &kv : got.entries)
   {
      const auto it = ref.entries.find(kv.first);
      if (it == ref.entries.end())
      {
         CAPTURE(kv.first.first, kv.first.second, kv.second);
         REQUIRE(kv.second == 0.0);
         extra++;
         continue;
      }
      if (!(kv.second == it->second))
      {
         CAPTURE(kv.first.first, kv.first.second, it->second, kv.second);
         REQUIRE(kv.second == it->second);
      }
   }
   // Nothing the serial route has may be MISSING, which is the half a
   // one-sided comparison would let through -- a scatter that dropped every
   // second contribution would otherwise pass on values alone.
   for (const auto &kv : ref.entries)
   {
      if (got.entries.find(kv.first) == got.entries.end())
      {
         CAPTURE(kv.first.first, kv.first.second, kv.second);
         REQUIRE(false);
      }
   }
   CAPTURE(extra, ref.nnz, got.nnz);
}

/// A nonlinearity on the potential mass, for the NPC section.
class SquareSource : public NonlinearFormIntegrator
{
public:
   explicit SquareSource(real_t c_) : c(c_) { }

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               2*el.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, shape);
         const real_t u = shape * elfun;
         elvect.Add(ip.weight * Tr.Weight() * c * u * u, shape);
      }
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elmat.SetSize(dof);
      elmat = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               2*el.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, shape);
         const real_t u = shape * elfun;
         AddMult_a_VVt(ip.weight * Tr.Weight() * 2.0 * c * u, shape, elmat);
      }
   }

private:
   real_t c;
   Vector shape;
};

/// Which trace space to build, since one of them must be refused.
enum class Trace { DG, H1 };

/** Assemble the hybridized trace operator once, in the given mode, and solve.

    @a c == 0 gives the linear problem, which reaches ComputeH() from
    Finalize(); a nonzero @a c makes it semilinear, which is NPC and reaches
    ComputeH(Gradient) from GetGradient() once per Newton step. The two are
    different callers of the routine under test and neither substitutes for
    the other -- the linear one assembles once into a matrix Finalize() owns,
    the NPC one reassembles into a matrix GetGradient() has just reset. */
Assembled Run(Mesh &mesh, int order,
              DarcyHybridization::TraceAssemblyMode tmode,
              DarcyHybridization::LocalFactorMode lmode,
              real_t c = 0.0, Trace trace = Trace::DG)
{
   const int dim = mesh.Dimension();

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   std::unique_ptr<FiniteElementCollection> t_coll;
   if (trace == Trace::DG)
   { t_coll.reset(new DG_Interface_FECollection(order, dim)); }
   else
   { t_coll.reset(new H1_Trace_FECollection(std::max(order, 1), dim)); }

   FiniteElementSpace Vh(&mesh, &u_coll, dim);
   FiniteElementSpace Wh(&mesh, &p_coll);
   FiniteElementSpace Mh(&mesh, t_coll.get());

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);
   FunctionCoefficient src([](const Vector &X)
   {
      return std::sin(M_PI*X(0))*std::sin(M_PI*X(1));
   });

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   if (c == 0.0)
   {
      BilinearForm *M_p = darcy.GetPotentialMassForm();
      M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
      M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   }
   else
   {
      NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
      Mnl_p->AddDomainIntegrator(new SquareSource(c));
      Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
      Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   }

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);

   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetTraceAssemblyMode(tmode);
   dh->SetLocalFactorMode(lmode);
   if (c != 0.0)
   {
      dh->SetLocalNLSolver(DarcyHybridization::LSsolveType::Newton, 1000,
                           1e-14, 1e-30);
   }
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   dh->SetEssentialBC(ess_bdr);

   darcy.Assemble();

   Assembled out;
   out.can_batch = dh->CanBatchTraceAssembly();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorHandle R;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux, x, R, X, B, true);

   GSSmoother prec;
   GMRESSolver lin;
   lin.SetKDim(200);
   lin.SetMaxIter(2000);
   lin.SetRelTol(1e-14);
   lin.SetAbsTol(0.0);
   lin.SetPreconditioner(prec);
   lin.SetPrintLevel(-1);

   if (c == 0.0)
   {
      if (SparseMatrix *H = dynamic_cast<SparseMatrix*>(R.Ptr()))
      { out.Capture(*H); }
      lin.SetOperator(*R.Ptr());
      lin.Mult(B, X);
   }
   else
   {
      NewtonSolver newton;
      newton.SetSolver(lin);
      newton.SetOperator(*R.Ptr());
      newton.SetRelTol(1e-12);
      newton.SetAbsTol(1e-14);
      newton.SetMaxIter(30);
      newton.SetPrintLevel(-1);
      newton.Mult(B, X);
      // The gradient at the CONVERGED state, which is a matrix the NPC route
      // has just assembled through GetGradient() rather than Finalize().
      if (SparseMatrix *G =
             dynamic_cast<SparseMatrix*>(&R->GetGradient(X)))
      { out.Capture(*G); }
   }

   out.tr = X;
   darcy.RecoverFEMSolution(X, x);
   out.q = x.GetBlock(0);
   out.p = x.GetBlock(1);
   return out;
}

/// The solved fields, which reassociate and so want a tolerance.
void RequireCloseFields(const Assembled &ref, const Assembled &got)
{
   REQUIRE(got.q.Size() == ref.q.Size());
   REQUIRE(got.p.Size() == ref.p.Size());
   Vector d(ref.q);
   d -= got.q;
   REQUIRE(d.Normlinf() == MFEM_Approx(0.0, 1e-11, 1e-11));
   d = ref.p;
   d -= got.p;
   REQUIRE(d.Normlinf() == MFEM_Approx(0.0, 1e-11, 1e-11));
}

} // namespace darcy_batched_traceh

using namespace darcy_batched_traceh;

TEST_CASE("The batched trace assembly builds the serial one's matrix",
          "[DarcyHybridization][TraceAssembly][NPC]")
{
   const int order = GENERATE(0, 1, 2, 3);
   const auto elem = GENERATE(Element::QUADRILATERAL, Element::TRIANGLE);
   CAPTURE(order, (int)elem);

   Mesh mesh = Mesh::MakeCartesian2D(3, 3, elem);

   // NPC, and not the linear problem, because NPC is where the mode is
   // REACHABLE -- see the refusal case below, which is the other half of this
   // one. An earlier version of this case ran the linear problem, asserted
   // CanBatchTraceAssembly(), and passed against a deliberately broken
   // kernel; the predicate was answering a different question from the one
   // ComputeH() asks. Both now ask the same one.
   const auto ref = Run(mesh, order,
                        DarcyHybridization::TraceAssemblyMode::Serial,
                        DarcyHybridization::LocalFactorMode::Serial, 0.5);
   const auto got = Run(mesh, order,
                        DarcyHybridization::TraceAssemblyMode::Batched,
                        DarcyHybridization::LocalFactorMode::Serial, 0.5);

   // The lever must be live. A silent fallback would make every assertion
   // below a comparison of the serial route with itself, which is the shape
   // of a passing test that tests nothing -- and this file has already been
   // that shape once.
   REQUIRE_FALSE(ref.can_batch);
   REQUIRE(got.can_batch);
   REQUIRE(ref.nnz > 0);

   RequireSameOperator(ref, got);
   RequireCloseFields(ref, got);
}

TEST_CASE("The batched trace assembly runs beside the batched face pairs",
          "[DarcyHybridization][TraceAssembly][BatchedLinAlg]")
{
   const int order = GENERATE(1, 2);
   CAPTURE(order);

   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);

   // LocalFactorMode::Batched is what leaves the element blocks DEVICE-valid,
   // written by ComputeElementsHBatched() rather than by the host face-pair
   // loop -- so this pairing is the one where the trace assembly consumes a
   // buffer it did not have to read back, which is the whole point of it.
   const auto ref = Run(mesh, order,
                        DarcyHybridization::TraceAssemblyMode::Serial,
                        DarcyHybridization::LocalFactorMode::Batched, 0.5);
   const auto got = Run(mesh, order,
                        DarcyHybridization::TraceAssemblyMode::Batched,
                        DarcyHybridization::LocalFactorMode::Batched, 0.5);

   REQUIRE(got.can_batch);
   REQUIRE(ref.nnz > 0);

   RequireSameOperator(ref, got);
   RequireCloseFields(ref, got);
}

TEST_CASE("The batched trace assembly is refused on the reduced route",
          "[DarcyHybridization][TraceAssembly]")
{
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);

   // The scope limit, asserted rather than left as a remark. Without NPC the
   // face constraint writes its per-face diagonal blocks straight into the
   // sparse H during Assemble(), so ComputeH() inherits a linked-list matrix
   // instead of building a CSR and the mode cannot apply. The connectivity is
   // perfectly fine here -- it is the destination that is taken -- which is
   // why the predicate has to ask both questions.
   const auto got = Run(mesh, 2,
                        DarcyHybridization::TraceAssemblyMode::Batched,
                        DarcyHybridization::LocalFactorMode::Serial);
   REQUIRE_FALSE(got.can_batch);

   // And the fallback is the serial route exactly, so asking for the mode on
   // a problem that cannot take it costs nothing.
   const auto ref = Run(mesh, 2,
                        DarcyHybridization::TraceAssemblyMode::Serial,
                        DarcyHybridization::LocalFactorMode::Serial);
   REQUIRE(ref.nnz > 0);
   REQUIRE(got.nnz == ref.nnz);
   RequireSameOperator(ref, got);
}

TEST_CASE("The batched trace assembly refuses a shared trace dof",
          "[DarcyHybridization][TraceAssembly]")
{
   Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);

   // An H1 trace shares its vertex dofs between the faces meeting there, so
   // one thread no longer owns a whole row of H and the row-ownership the
   // whole mode rests on is gone. The refusal is the assertion; that it still
   // SOLVES is the other half, since a refusal that aborted would be a
   // regression for every EDG caller.
   const auto got = Run(mesh, 2,
                        DarcyHybridization::TraceAssemblyMode::Batched,
                        DarcyHybridization::LocalFactorMode::Serial,
                        0.0, Trace::H1);
   REQUIRE_FALSE(got.can_batch);

   const auto ref = Run(mesh, 2,
                        DarcyHybridization::TraceAssemblyMode::Serial,
                        DarcyHybridization::LocalFactorMode::Serial,
                        0.0, Trace::H1);
   REQUIRE(ref.nnz > 0);
   REQUIRE(got.nnz == ref.nnz);
   RequireSameOperator(ref, got);
}
