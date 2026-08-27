// Minimal demonstrator: under NLOrdering::LineariseThenCondense, with the
// nonlinearity on the POTENTIAL MASS form (Mnl_p), the reduced operator's
// residual is not a function of the trace.
//
//     R.Mult(x, r1);  R.GetGradient(x);  R.Mult(x, r2);   =>   r1 != r2
//
// x never changes, so r1 and r2 must be equal. Under CondenseThenLinearise they
// are, bit for bit. The existing unit test drives the BLOCK nonlinear form
// (GetBlockNonlinearForm / MixedConductionNLFIntegrator) and passes; this drives
// GetPotentialMassNonlinearForm instead, which is a different path.
//
//   g++ -std=c++17 -O2 -I<mfem>/include lf_bug.cpp <mfem>/lib/libmfem.a <tpls>
#include "mfem.hpp"
#include <cstdio>
#include <cmath>

using namespace mfem;

namespace
{
// (s(u), w) on the potential block, with s(u) = c u^2 -- any u-dependent source
// will do; c scales how nonlinear it is.
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

// One assembly, then r1 = R(x), refresh the linearisation at the same x,
// r2 = R(x). Returns |r2 - r1| / |r1|.
real_t Inconsistency(DarcyHybridization::NLOrdering ordering, real_t c, int n,
                     int order, bool verbose)
{
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::TRIANGLE);

   L2_FECollection u_coll(order, mesh.Dimension(), BasisType::GaussLobatto);
   L2_FECollection p_coll(order, mesh.Dimension());
   DG_Interface_FECollection t_coll(order, mesh.Dimension());
   FiniteElementSpace Vh(&mesh, &u_coll, mesh.Dimension());
   FiniteElementSpace Wh(&mesh, &p_coll);
   FiniteElementSpace Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0);

   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   darcy.GetFluxDivForm()->AddDomainIntegrator(new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   // THE POINT: the nonlinearity goes on the potential MASS form, not on the
   // block nonlinear form. The whole potential block must live there, so the
   // HDG stabilisation goes with it.
   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new SquareSource(c));
   Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   darcy.GetHybridization()->SetNonlinearOrdering(ordering);
   darcy.GetHybridization()->SetEssentialBC(ess_bdr);
   darcy.Assemble();

   Array<int> offs(3);
   offs[0] = 0; offs[1] = Vh.GetVSize(); offs[2] = offs[1] + Wh.GetVSize();
   BlockVector sol(offs), rhs(offs);
   sol = 0.0; rhs = 0.0;

   OperatorHandle R;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux, sol, rhs, R, X, B, true);

   Operator &op = *R.Ptr();
   Vector x(op.Height()), r1(op.Height()), r2(op.Height()), r3(op.Height());
   x.Randomize(1);
   x *= 0.1;

   op.Mult(x, r1);
   op.GetGradient(x);            // same x
   op.Mult(x, r2);
   op.GetGradient(x);            // same x again
   op.Mult(x, r3);

   Vector d(r2); d -= r1;
   Vector d2(r3); d2 -= r2;
   const real_t rel = d.Norml2() / std::max(real_t(1e-300), r1.Norml2());

   if (verbose)
   {
      std::printf("    |r1| %.4e  |r2| %.4e  |r2-r1|/|r1| %.3e   "
                  "|r3-r2|/|r2| %.3e\n",
                  r1.Norml2(), r2.Norml2(), rel,
                  d2.Norml2() / std::max(real_t(1e-300), r2.Norml2()));
   }
   return rel;
}
}

int main()
{
   using NL = DarcyHybridization::NLOrdering;
   std::printf("\nR.Mult(x,r1); R.GetGradient(x); R.Mult(x,r2)  -- x never changes,\n"
               "so r1 and r2 must agree. Nonlinearity on Mnl_p, k=1, 8x8.\n\n");

   for (real_t c : {1.0, 1.0e2, 1.0e4, 1.0e5})
   {
      std::printf("  c = %-6g condense-then-linearise:\n", c);
      Inconsistency(NL::CondenseThenLinearise, c, 8, 1, true);
      std::printf("  c = %-6g LINEARISE-then-condense:\n", c);
      Inconsistency(NL::LineariseThenCondense, c, 8, 1, true);
      std::printf("\n");
   }
   return 0;
}
