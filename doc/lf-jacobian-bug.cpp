// Demonstrator for the second finding of doc/LINEARISE-FIRST-RESIDUAL-BUG.md:
// under NLOrdering::LineariseThenCondense, GetGradient() was not the
// derivative of Mult(). Self-contained, MFEM only -- the earlier version of
// this file needed ../meq to get the essential trace dof list, and did not:
// DarcyHybridization::GetEssentialTrueDofs() returns exactly that list.
//
//   g++ -std=c++17 -O2 -I<mfem> -I<suitesparse> doc/lf-jacobian-bug.cpp \
//       <mfem>/libmfem.a $(MFEM_EXT_LIBS) -o lf_jac && ./lf_jac
//
// Four things are measured, and the first three are the ones that say the
// disagreement was a real Jacobian error rather than a differencing artefact:
//
//   * it did not move with the step h, across four decades;
//   * it grew with the strength of the nonlinearity;
//   * it fell with |x|, at FIXED nonlinearity, which is what identified the
//     cause -- it was proportional to the local Newton step the substitution
//     took at the linearisation point, and that step is proportional to the
//     trace here because the right-hand side is zero;
//   * and it was there only at a COLD linearisation, the first one, which
//     retained the caller's initial guess. One relinearisation and it fell to
//     round-off. That is why a mild problem was unaffected, why a stiff one
//     lost its first Newton step, and why a line search made it worse.
//
// CondenseThenLinearise is the control throughout and must trace the textbook
// round-off curve of an exact Jacobian, rising as h falls. Its local solve is
// tightened below, because its default rtol of 1e-6 would otherwise put the
// control at 1e-6 and hide everything smaller.
#include "mfem.hpp"
#include <cstdio>
#include <cmath>

using namespace mfem;

namespace
{
// (c u^2, w) on the potential block: a semilinear source, which is the path a
// Grad-Shafranov solver drives and the path that had no coverage. The existing
// unit tests drove the BLOCK nonlinear form, a nonlinear flux law, instead.
class SquareSource : public NonlinearFormIntegrator
{
public:
   explicit SquareSource(real_t c_) : c(c_) { }

   void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
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

struct Harness
{
   Mesh mesh;
   L2_FECollection u_coll, p_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace Vh, Wh, Mh;
   DarcyForm darcy;
   ConstantCoefficient one;
   Array<int> ess_flux;
   OperatorHandle R;
   Vector X, B;
   BlockVector sol;

   Harness(int n, int order, real_t c,
           DarcyHybridization::NLOrdering ordering)
      : mesh(Mesh::MakeCartesian2D(n, n, Element::TRIANGLE)),
        u_coll(order, 2, BasisType::GaussLobatto), p_coll(order, 2),
        t_coll(order, 2),
        Vh(&mesh, &u_coll, 2), Wh(&mesh, &p_coll), Mh(&mesh, &t_coll),
        darcy(&Vh, &Wh), one(1.0)
   {
      darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
      darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorDivergenceIntegrator());
      darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

      NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
      Mnl_p->AddDomainIntegrator(new SquareSource(c));
      Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
      Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

      // A genuinely well-posed Dirichlet problem. Without this the essential
      // trace dof list is empty and everything below measures nothing.
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
      darcy.GetHybridization()->SetNonlinearOrdering(ordering);
      darcy.GetHybridization()->SetLocalNLSolver(
         DarcyHybridization::LSsolveType::Newton, 1000, 1e-14, 1e-30);
      darcy.GetHybridization()->SetEssentialBC(ess_bdr);
      darcy.Assemble();

      sol.Update(darcy.GetOffsets());
      sol = 0.0;
      darcy.FormLinearSystem(ess_flux, sol, R, X, B, true);
   }

   Operator &op() { return *R.Ptr(); }
   const Array<int> &ess() const
   { return darcy.GetHybridization()->GetEssentialTrueDofs(); }
};

// |J v - (R(x+hv) - R(x-hv))/2h| / |(R(x+hv) - R(x-hv))/2h|, with the
// essential trace rows left out: the residual is masked there and the Jacobian
// carries a unit row, so including them makes the comparison meaningless.
real_t JacobianError(DarcyHybridization::NLOrdering ordering, real_t c,
                     real_t h, int n, int order, int *n_ess = nullptr,
                     real_t xscale = 0.05, int warm = 0)
{
   Harness H(n, order, c, ordering);
   Operator &op = H.op();
   const int m = op.Height();

   Array<int> ess_marker(m);
   ess_marker = 0;
   for (int i = 0; i < H.ess().Size(); i++) { ess_marker[H.ess()[i]] = 1; }
   if (n_ess) { *n_ess = H.ess().Size(); }

   Vector x(m), v(m);
   x.Randomize(3);
   x *= xscale;
   v.Randomize(7);
   for (int i = 0; i < m; i++)
   {
      if (ess_marker[i]) { x(i) = 0.0; v(i) = 0.0; }
   }
   v *= 1.0/v.Norml2();

   // Newton's own order, and the only order in which the question is well
   // posed: the residual, then the gradient at the same trace. The
   // linearisation then sits at x and Mult() never moves it, so both
   // difference evaluations see the linearisation the gradient belongs to.
   // (A line search is this same pairing, several trials deep.)
   Vector r0(m);
   op.Mult(x, r0);
   op.GetGradient(x);

   // A cold linearisation retains the caller's initial guess, which is as far
   // from the local solution as the run ever gets. A Newton iteration past its
   // first step is warm. Warming it here means relinearising at traces that
   // converge onto x; each advances the retained fields, and the last lands on
   // x exactly.
   for (int w = 0; w < warm; w++)
   {
      Vector xw(x);
      xw *= 1.0 + 1e-13*(w + 1);
      op.GetGradient(xw);
      op.GetGradient(x);
   }

   Vector xp(x), xm(x), rp(m), rm(m), Jv(m);
   xp.Add(h, v);
   xm.Add(-h, v);
   op.Mult(xp, rp);
   op.Mult(xm, rm);
   Vector fd(rp);
   fd -= rm;
   fd *= 1.0/(2.0*h);

   op.GetGradient(x).Mult(v, Jv);   // idempotent at the retained trace

   real_t num = 0.0, den = 0.0;
   for (int i = 0; i < m; i++)
   {
      if (ess_marker[i]) { continue; }
      const real_t d = Jv(i) - fd(i);
      num += d*d;
      den += fd(i)*fd(i);
   }
   return std::sqrt(num)/std::max(real_t(1e-300), std::sqrt(den));
}

void Row(const char *label, real_t a, real_t b)
{
   std::printf("    %-10s | %-24.3e | %-24.3e\n", label, a, b);
   std::fflush(stdout);
}

void Head(const char *what, const char *col)
{
   std::printf("\n  %s\n", what);
   std::printf("    %-10s | %-24s | %-24s\n", col,
               "condense-then-linearise", "LINEARISE-then-condense");
}
}

int main()
{
   using NL = DarcyHybridization::NLOrdering;
   const int n = 8, order = 1;
   char lab[32];

   int n_ess = 0;
   JacobianError(NL::CondenseThenLinearise, 1.0, 1e-5, n, order, &n_ess);
   std::printf("\n  F = c p^2 on the potential mass form, k = 1, 8x8 triangles\n");
   std::printf("  essential trace dofs found: %d   (zero would mean the "
               "problem is not\n  the Dirichlet problem it is supposed to be, "
               "and nothing below means anything)\n", n_ess);

   Head("independence of the step, at c = 100", "h");
   for (real_t h : {1e-4, 1e-5, 1e-6, 1e-7})
   {
      std::snprintf(lab, sizeof(lab), "%.0e", h);
      Row(lab, JacobianError(NL::CondenseThenLinearise, 100.0, h, n, order),
          JacobianError(NL::LineariseThenCondense, 100.0, h, n, order));
   }

   Head("growth with the nonlinearity, at h = 1e-5", "c");
   for (real_t c : {1.0, 10.0, 100.0, 1000.0})
   {
      std::snprintf(lab, sizeof(lab), "%g", c);
      Row(lab, JacobianError(NL::CondenseThenLinearise, c, 1e-5, n, order),
          JacobianError(NL::LineariseThenCondense, c, 1e-5, n, order));
   }

   Head("scaling of the trace at FIXED c = 100 -- the attribution", "|x| ~");
   for (real_t s : {5e-2, 5e-3, 5e-4, 5e-5})
   {
      std::snprintf(lab, sizeof(lab), "%.0e", s);
      Row(lab, JacobianError(NL::CondenseThenLinearise, 100.0, 1e-5, n, order,
                             nullptr, s),
          JacobianError(NL::LineariseThenCondense, 100.0, 1e-5, n, order,
                        nullptr, s));
   }

   Head("cold against warm, at c = 100 -- where the defect lived", "warm-ups");
   for (int w : {0, 1, 2, 4})
   {
      std::snprintf(lab, sizeof(lab), "%d", w);
      Row(lab, JacobianError(NL::CondenseThenLinearise, 100.0, 1e-5, n, order,
                             nullptr, 0.05, w),
          JacobianError(NL::LineariseThenCondense, 100.0, 1e-5, n, order,
                        nullptr, 0.05, w));
   }

   std::printf("\n");
   return 0;
}
