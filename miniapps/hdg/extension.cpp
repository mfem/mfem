//                    MFEM HDG Miniapp: extension from subdomains
//
// Compile with: make extension
//
// Sample runs:  extension -o 1 -n 8 -r 3
//               extension -o 2 -n 8 -r 3 -path ls
//               extension -o 3 -n 8 -r 3 -tau 1
//               extension -o 1 -n 16 -r 2 -no-ext
//               extension -p 2 -o 1 -n 8 -r 3
//               extension -p 3 -o 1 -n 16 -r 2
//               extension -o 2 -n 8 -r 3 -no-ctl -no-rec
//
// Description:  Solves the Darcy problem
//
//                   C u + grad p = 0,   -div u = g   in Omega,   p = f on Gamma
//
//               on a *polyhedral subdomain* D_h of a domain Omega whose
//               boundary Gamma the mesh does not follow, by the extension
//               technique of Cockburn and Solano. The Dirichlet datum given on
//               Gamma is transferred to the computational boundary Gamma_h
//               along a family of paths, by the line integral
//
//                   phi_h(x) = f(a(x)) + int_sigma C E_h(u_h) . m ds,
//
//               where E_h is the extension of the discrete flux out of the
//               element owning the face. The curved boundary is thereby
//               reduced to the evaluation of line integrals, and the design
//               order of the method is retained even though the distance from
//               Gamma_h to Gamma is only of order h -- which is the whole
//               point, earlier techniques having needed order h^(k+1).
//
//               Three of the reference's experiments, chosen with -p:
//
//                 1  Omega is a disc immersed in a triangulated square, and
//                    the whole of the computational boundary is transferred
//                    (section 3.2, Table 4).
//                 2  Omega is the square less a circular obstacle: potential
//                    flow around a disc. The domain is not convex, and its
//                    outer boundary is fitted by the mesh while the
//                    obstacle's is not, so the two boundary treatments run
//                    side by side (section 3.3, Table 5).
//                 3  the same with a Joukowsky airfoil replacing the disc.
//                    Its tail is a curved reentrant corner, and it has no
//                    closest-point map in closed form, which is what makes it
//                    the reference's most difficult case (section 3.4,
//                    Table 6).
//
//               The miniapp solves on a sequence of background meshes and
//               prints the history of convergence, because the claim being
//               tested is a claim about rates. With -rec, the default, it
//               also postprocesses onto spaces one order higher and reports
//               the errors of the total flux u_t and of the pair (u*, p*).
//               The potential gains a full order there -- k+2 -- with the
//               datum transferred as well as without it, which is the last
//               of the reference's claims about the method. With -ctl it also solves, on
//               the same D_h, the problem whose Dirichlet datum is read on
//               Gamma_h itself -- the boundary-fitted problem the extension is
//               trying to match -- and reports the ratio of the errors.
//
//               Reference: B. Cockburn and M. Solano, Solving Dirichlet
//               boundary-value problems on curved domains by extensions from
//               subdomains, SIAM J. Sci. Comput. 34 (2012) A497-A519.

#include "mfem.hpp"

#include <memory>
#include <iostream>
#include <iomanip>
#include <complex>

using namespace std;
using namespace mfem;

namespace
{

// Which experiment of the reference. See the header comment.
int problem = 1;

// Problem 1: Omega is a disc, immersed in the unit square.
real_t disc_R = 0.45;

// Problems 2 and 3: Omega is the unit square less an obstacle -- a disc, or
// the Joukowsky image of one -- so that the domain is not convex and its outer
// boundary *is* fitted by the mesh while the obstacle's is not.
real_t obst_R = 0.125;                                   ///< problem 2
real_t foil_R = 0.107, foil_s1 = 0.01, foil_s2 = 0.01;   ///< problem 3
real_t foil_lambda = -1.0;   ///< negative takes the reference's R - |s|

/// How many times to refine the lattice that decides whether an element is
/// inside Omega. The vertex test is exact only where Omega is convex, and for
/// problems 2 and 3 it is not: a triangle can have every vertex outside the
/// obstacle and still clip it, which puts a piece of D_h outside Omega and
/// makes nonsense of everything downstream.
int extra_refine = 0;

const real_t cx = 0.5, cy = 0.5;

real_t FoilLambda()
{
   return (foil_lambda >= 0.0)
          ? foil_lambda
          : foil_R - sqrt(foil_s1 * foil_s1 + foil_s2 * foil_s2);
}

/// The Joukowsky map J(z) = z + lambda^2 / z.
complex<real_t> Joukowsky(complex<real_t> z, real_t lambda)
{
   return z + lambda * lambda / z;
}

/// The branch of its inverse that is injective on |z| >= lambda.
/** The two roots of z^2 - w z + lambda^2 = 0 multiply to lambda^2, so one lies
    on each side of |z| = lambda and the wanted one is the larger. It is built
    with the square root signed so that the two terms add: taking the smaller
    root from a difference and inverting it would lose most of the digits for
    a point far from the obstacle, which is most of the mesh. */
complex<real_t> JoukowskyInverse(complex<real_t> w, real_t lambda)
{
   complex<real_t> r = sqrt(w * w - 4.0 * lambda * lambda);
   if (w.real() * r.real() + w.imag() * r.imag() < 0.0) { r = -r; }
   const complex<real_t> z = 0.5 * (w + r);
   return (abs(z) > 0.0) ? z : complex<real_t>(lambda, 0.0);
}

/// The level set: negative inside Omega, zero on Gamma, positive outside.
real_t LevelSet(const Vector &x)
{
   const real_t X = x(0) - cx, Y = x(1) - cy;
   switch (problem)
   {
      case 1:
         return sqrt(X * X + Y * Y) - disc_R;
      case 2:
         // Outside the obstacle is inside Omega, so the sign is the other way.
         return obst_R - sqrt(X * X + Y * Y);
      case 3:
      {
         const complex<real_t> z =
            JoukowskyInverse(complex<real_t>(X, Y), FoilLambda());
         return foil_R - abs(z - complex<real_t>(foil_s1, foil_s2));
      }
   }
   MFEM_ABORT("unknown problem " << problem);
   return 0.0;
}

/// The measure of Omega, which the regions swept by the paths must make up
/// together with that of D_h.
real_t OmegaMeasure()
{
   switch (problem)
   {
      case 1: return M_PI * disc_R * disc_R;
      case 2: return 1.0 - M_PI * obst_R * obst_R;
      case 3:
      {
         // By the shoelace formula on the image of the circle; the Joukowsky
         // image has no area formula as tidy as the map.
         const real_t lambda = FoilLambda();
         const int N = 400000;
         const complex<real_t> s(foil_s1, foil_s2);
         real_t A = 0.0;
         complex<real_t> prev = Joukowsky(s + foil_R, lambda);
         const complex<real_t> first = prev;
         for (int i = 1; i <= N; i++)
         {
            const real_t th = 2.0 * M_PI * i / N;
            const complex<real_t> w = (i == N) ? first
                                      : Joukowsky(s + foil_R * exp(complex<real_t>(0.0, th)),
                                                  lambda);
            A += prev.real() * w.imag() - w.real() * prev.imag();
            prev = w;
         }
         return 1.0 - 0.5 * fabs(A);
      }
   }
   MFEM_ABORT("unknown problem " << problem);
   return 0.0;
}

// The manufactured solutions, with C = I so that u = -grad p.

real_t pExact(const Vector &x)
{
   const real_t X = x(0) - cx, Y = x(1) - cy;
   switch (problem)
   {
      case 1:
         return sin(x(0)) * sin(x(1));
      case 2:
         // Potential flow past a cylinder: harmonic away from the centre, and
         // its normal flux vanishes on the circle of radius obst_R.
         return -X * (1.0 + obst_R * obst_R / (X * X + Y * Y));
      case 3:
         return sin(3.0 * M_PI * x(0)) * sin(3.0 * M_PI * x(1));
   }
   MFEM_ABORT("unknown problem " << problem);
   return 0.0;
}

void uExact(const Vector &x, Vector &u)
{
   const real_t X = x(0) - cx, Y = x(1) - cy;
   switch (problem)
   {
      case 1:
         u(0) = -cos(x(0)) * sin(x(1));
         u(1) = -sin(x(0)) * cos(x(1));
         return;
      case 2:
      {
         const real_t r2 = X * X + Y * Y, R2 = obst_R * obst_R;
         u(0) = 1.0 + R2 * (Y * Y - X * X) / (r2 * r2);
         u(1) = -2.0 * R2 * X * Y / (r2 * r2);
         return;
      }
      case 3:
      {
         const real_t k = 3.0 * M_PI;
         u(0) = -k * cos(k * x(0)) * sin(k * x(1));
         u(1) = -k * sin(k * x(0)) * cos(k * x(1));
         return;
      }
   }
   MFEM_ABORT("unknown problem " << problem);
}

/// The source of the potential equation, g = -div u = laplacian p.
real_t gExact(const Vector &x)
{
   switch (problem)
   {
      case 1: return -2.0 * pExact(x);
      case 2: return 0.0;
      case 3: return -18.0 * M_PI * M_PI * pExact(x);
   }
   MFEM_ABORT("unknown problem " << problem);
   return 0.0;
}

/// The datum as the flux equation takes it: negated, as elsewhere on this
/// branch. See HDGExtensionIntegrator on why that fixes the sign of the
/// extension term.
real_t pNatural(const Vector &x) { return -pExact(x); }

struct Result
{
   real_t err_u{}, err_p{};        ///< L2 errors on D_h
   real_t err_us{}, err_ps{};      ///< and of the postprocessed pair
   real_t err_ut{};                ///< and of the total flux
   real_t ext_u{}, ext_p{};        ///< and on the complement, normalized
   real_t area_c{};                ///< the measure of D_h^c, as swept
   real_t area_err{};              ///< how far that is from |Omega| - |D_h|
   real_t dist{};        ///< the largest distance from Gamma_h to Gamma
   int    widened{};     ///< vertices whose admissible fan had to be widened
   int    dofs{};        ///< size of the hybridized system
   int    elements{};
   bool   converged{};
};

/// Solve on D_h. With @a extend the Dirichlet datum is transferred from Gamma;
/// otherwise it is read on Gamma_h itself, which is the boundary-fitted
/// problem on the same subdomain and so is the thing to match.
enum class PathFamily { ClosestPoint, LevelSet, VertexCone };

Result Solve(int n, int order, real_t tau, real_t offset,
             PathFamily path_family, bool extend, int line_order,
             bool postprocess = true, bool visualization = false)
{
   const int dim = 2;

   Mesh background = Mesh::MakeCartesian2D(n, n, Element::TRIANGLE);

   Array<int> marker;
   const int inside = MarkLevelSetSubdomain(background, LevelSet, offset,
                                            marker);
   MFEM_VERIFY(inside > 0, "the subdomain is empty; the mesh is too coarse");
   for (int i = 0; i < background.GetNE(); i++)
   {
      background.SetAttribute(i, marker[i] ? 1 : 2);
   }
   background.SetAttributes();

   Array<int> domain_attr(1);
   domain_attr[0] = 1;
   auto D_h = make_unique<SubMesh>(
                 SubMesh::CreateFromDomain(background, domain_attr));

   // SubMesh gives the boundary it had to generate one new attribute, and
   // leaves the boundary inherited from the parent with the attributes it
   // already had. The generated part is Gamma_h, where the datum has to be
   // transferred; the inherited part -- the sides of the square, for the two
   // problems with an obstacle -- is a piece of Gamma that the mesh does fit,
   // and there the datum is simply read.
   const int parent_bdr = background.bdr_attributes.Max();
   const int gamma_h = D_h->bdr_attributes.Max();
   MFEM_VERIFY(gamma_h == parent_bdr + 1,
               "D_h has no boundary of its own: the level set selected either "
               "everything or nothing");

   Array<int> bdr_gamma_h(gamma_h), bdr_fitted(gamma_h);
   bdr_gamma_h = 0;
   bdr_fitted = 1;
   bdr_gamma_h[gamma_h - 1] = 1;
   bdr_fitted[gamma_h - 1] = 0;
   const bool any_fitted = (problem != 1);

   Result res;

   // The paths. Where the obstacle is a circle its closest-point map is a
   // closed form and is the family to use, being the only one of the two that
   // tiles the region beyond Gamma_h. The airfoil has no such map, so it is
   // left with the level-set family, which marches along the outward normal
   // and bisects.
   Vector centre(2);
   centre(0) = cx; centre(1) = cy;
   unique_ptr<TransferPath> path;
   int widened = 0;
   if (path_family == PathFamily::LevelSet)
   {
      path = make_unique<LevelSetPath>(LevelSet, 4.0 / n);
   }
   else if (path_family == PathFamily::VertexCone)
   {
      auto vcp = make_unique<VertexConePath>(*D_h, gamma_h, LevelSet, 4.0 / n);
      widened = vcp->NumWidened();
      path = std::move(vcp);
   }
   else
   {
      MFEM_VERIFY(problem != 3, "the airfoil has no closest-point map");
      path = make_unique<ClosestPointPath>(
                ClosestPointPath::Sphere(centre,
                                         (problem == 1) ? disc_R : obst_R));
   }
   res.widened = widened;

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(D_h.get(), &u_coll, dim);
   FiniteElementSpace fes_p(D_h.get(), &p_coll);

   ConstantCoefficient C(1.0);          // the inverse diffusion tensor
   VectorFunctionCoefficient zero(dim, [](const Vector &, Vector &f) { f = 0.0; });
   FunctionCoefficient gcoeff(gExact);
   FunctionCoefficient pcoeff(pExact);
   VectorFunctionCoefficient ucoeff(dim, uExact);

   FunctionCoefficient datum_here(pNatural);
   PathTraceCoefficient datum_there(*path, pNatural);
   Coefficient &datum = extend ? static_cast<Coefficient &>(datum_there)
                        : static_cast<Coefficient &>(datum_here);

   DarcyForm darcy(&fes_u, &fes_p);

   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(C));
   if (extend)
   {
      // The solution-dependent half of the transferred datum. It is local to
      // the element owning the face, so it goes into that element's own flux
      // mass block and the hybridization never sees it.
      darcy.GetFluxMassForm()->AddBdrFaceIntegrator(
         new HDGExtensionIntegrator(*path, C, +1.0, line_order), bdr_gamma_h);
   }

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   // HDGDiffusionIntegrator's parameter is tau*h/kappa, not tau. The meshes
   // are uniform on the unit square, so h = 1/n and passing tau/n holds the
   // stabilization fixed under refinement, which is the scaling the reference
   // uses. Holding the parameter itself fixed is a different method: it makes
   // tau grow like 1/h and costs the flux an order.
   darcy.GetPotentialMassForm()->AddInteriorFaceIntegrator(
      new HDGDiffusionIntegrator(C, tau / n));

   LinearForm *fform = darcy.GetFluxRHS();
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(zero));
   fform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(datum),
                               bdr_gamma_h);
   if (any_fitted)
   {
      fform->AddBdrFaceIntegrator(
         new VectorBoundaryFluxLFIntegrator(datum_here), bdr_fitted);
   }
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));

   Array<int> ess_flux_tdofs;
   DG_Interface_FECollection trace_coll(order, dim);
   FiniteElementSpace fes_t(D_h.get(), &trace_coll);
   darcy.EnableHybridization(&fes_t, new NormalTraceJumpIntegrator(),
                             ess_flux_tdofs);

   darcy.Assemble();

   BlockVector x(darcy.GetOffsets());
   x = 0.0;

   OperatorPtr A;
   Vector X, RHS;
   darcy.FormLinearSystem(ess_flux_tdofs, x, A, X, RHS, true);

   res.dofs = X.Size();
   res.elements = D_h->GetNE();

   GSSmoother prec;
   GMRESSolver solver;
   solver.SetKDim(500);
   solver.SetMaxIter(5000);
   solver.SetRelTol(0.0);
   solver.SetAbsTol(1e-12);
   solver.SetPreconditioner(prec);
   solver.SetOperator(*A);
   solver.Mult(RHS, X);
   res.converged = solver.GetConverged();

   darcy.RecoverFEMSolution(X, x);

   GridFunction u_h, p_h;
   u_h.MakeRef(&fes_u, x.GetBlock(0), 0);
   p_h.MakeRef(&fes_p, x.GetBlock(1), 0);
   res.err_u = u_h.ComputeL2Error(ucoeff);
   res.err_p = p_h.ComputeL2Error(pcoeff);

   // The local postprocessing, on spaces one order higher. Its local problem
   // is the same one the element was solved with, and on a face of Gamma_h
   // that includes the extension term -- which is the half of the transferred
   // datum that depends on the flux, and which the lift onto the enriched
   // space used to drop. The reference's claim for it is k+2 in the
   // potential; the reconstructed flux is not superconvergent and is not
   // claimed to be.
   GridFunction ut, u_s, p_s, tr_s;
   if (postprocess)
   {
      darcy.Reconstruct(x, X, ut, u_s, p_s, tr_s);
      res.err_ut = ut.ComputeL2Error(ucoeff);
      res.err_us = u_s.ComputeL2Error(ucoeff);
      res.err_ps = p_s.ComputeL2Error(pcoeff);
   }

   // How far the computational boundary stands from the true one, which is the
   // parameter the whole construction is about.
   Vector xp, xbar;
   for (int be = 0; be < D_h->GetNBE(); be++)
   {
      if (D_h->GetBdrAttribute(be) != gamma_h) { continue; }
      FaceElementTransformations *FTr = D_h->GetBdrFaceTransformations(be);
      if (!FTr) { continue; }
      const IntegrationRule &ir = IntRules.Get(FTr->GetGeometryType(), 4);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         path->Endpoint(*FTr, ir.IntPoint(q), xbar);
         FTr->Transform(ir.IntPoint(q), xp);
         xbar -= xp;
         res.dist = max(res.dist, xbar.Norml2());
      }
   }

   // The approximation on D_h^c: the flux is the extension of the element's
   // own polynomial, and the potential is the lifting evaluated there rather
   // than only on Gamma_h. This is the half of the method's claim that is
   // about the whole of Omega.
   {
      const int iro = 2 * order + 8;
      const IntegrationRule &fir = IntRules.Get(Geometry::SEGMENT, iro);
      const IntegrationRule &lir = IntRules.Get(Geometry::SEGMENT, iro);

      IsoparametricTransformation el_tr;
      ElementExtension ext;
      Vector ue(dim), ye(dim), v;
      real_t e2u = 0.0, e2p = 0.0;

      for (int be = 0; be < D_h->GetNBE(); be++)
      {
         if (D_h->GetBdrAttribute(be) != gamma_h) { continue; }
         FaceElementTransformations *FTr = D_h->GetBdrFaceTransformations(be);
         if (!FTr) { continue; }

         const int el = FTr->Elem1No;
         D_h->GetElementTransformation(el, &el_tr);
         ext.SetElement(el_tr);

         auto Cu = [&](const Vector &yy, Vector &vv)
         {
            IntegrationPoint eip;
            MFEM_VERIFY(ext.TransformBack(yy, eip), "no convergence");
            u_h.GetVectorValue(el, eip, vv);
         };

         ExtensionRegionQuadrature(
            *FTr, *path, fir, lir, [&](const ExtensionPoint &pt)
         {
            uExact(pt.y, ue);
            Cu(pt.y, v);
            real_t du = 0.0;
            for (int d = 0; d < dim; d++)
            {
               const real_t s = ue(d) - v(d);
               du += s * s;
            }
            // The lifting from the point itself, not from Gamma_h.
            const real_t lift =
               pExact(pt.xbar) + PathIntegral(Cu, pt.y, pt.xbar, lir);
            const real_t dp = pExact(pt.y) - lift;

            e2u += pt.weight * du;
            e2p += pt.weight * dp * dp;
            res.area_c += pt.weight;
         });
      }

      real_t vol_D = 0.0;
      for (int i = 0; i < D_h->GetNE(); i++) { vol_D += D_h->GetElementVolume(i); }
      const real_t vol_c = OmegaMeasure() - vol_D;
      res.area_err = (res.area_c - vol_c) / vol_c;   // signed: which way it fails

      // Normalized by the measure of the region, as the reference does: it
      // shrinks like h, and a raw L2 norm over it would carry half an order
      // that has nothing to do with the approximation.
      const real_t s = (res.area_c > 0.0) ? sqrt(res.area_c) : 1.0;
      res.ext_u = sqrt(e2u) / s;
      res.ext_p = sqrt(e2p) / s;
   }

   if (visualization)
   {
      // The subdomain and the solution on it. The region between Gamma_h and
      // Gamma is not meshed and so is not drawn; what is drawn stops where the
      // extension takes over.
      char vishost[] = "localhost";
      const int visport = 19916;
      socketstream sol_sock(vishost, visport);
      if (sol_sock.is_open())
      {
         sol_sock.precision(8);
         sol_sock << "solution\n" << *D_h << p_h
                  << "window_title 'potential on D_h'\n" << flush;
      }
      else
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
      }
   }

   return res;
}

} // namespace

int main(int argc, char *argv[])
{
   int order = 1;
   int n = 8;
   int refinements = 3;
   int line_order = -1;
   real_t tau = 1.0;
   real_t offset = 0.0;
   const char *path_type = "cp";
   bool extend = true;
   bool control = true;
   bool postprocess = true;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of the flux, potential and trace.");
   args.AddOption(&n, "-n", "--num-divisions",
                  "Divisions per side of the coarsest background mesh.");
   args.AddOption(&refinements, "-r", "--refinements",
                  "Halvings of the background mesh size after the first.");
   args.AddOption(&tau, "-tau", "--stabilization",
                  "The HDG stabilization, held fixed under refinement.");
   args.AddOption(&offset, "-d", "--offset",
                  "Select against phi <= -offset, pushing Gamma_h further "
                  "from Gamma. Zero takes every element that fits.");
   args.AddOption(&problem, "-p", "--problem",
                  "1: a disc (reference section 3.2, Table 4). "
                  "2: potential flow around a disc, so that Omega is not "
                  "convex and its outer boundary is fitted (section 3.3, "
                  "Table 5). "
                  "3: the same with a Joukowsky airfoil replacing the disc, "
                  "whose tail is a curved reentrant corner and which has no "
                  "closest-point map (section 3.4, Table 6).");
   args.AddOption(&disc_R, "-R", "--radius",
                  "Radius of the disc of problem 1.");
   args.AddOption(&obst_R, "-Ro", "--obstacle-radius",
                  "Radius of the circular obstacle of problem 2.");
   args.AddOption(&foil_s1, "-s1", "--foil-centre-x",
                  "Real part of the centre of that circle.");
   args.AddOption(&foil_s2, "-s2", "--foil-centre-y",
                  "Imaginary part of the centre of that circle.");
   args.AddOption(&foil_R, "-Rf", "--foil-radius",
                  "Radius of the circle the Joukowsky map carries to the "
                  "airfoil, for problem 3.");
   args.AddOption(&extra_refine, "-er", "--extra-refine",
                  "Refinements of the lattice tested when selecting D_h. Zero "
                  "tests the vertices only, which is exact where Omega is "
                  "convex and is not for problems 2 and 3.");
   args.AddOption(&foil_lambda, "-lam", "--foil-lambda",
                  "The Joukowsky parameter; negative takes the reference's "
                  "R - |s|, which makes the circle internally tangent to "
                  "|z| = lambda and gives the airfoil its thin curled tail.");
   args.AddOption(&path_type, "-path", "--path-family",
                  "'cp' for the closest-point map, 'ls' for a normal ray "
                  "bisected on the level set, 'vc' for the general family "
                  "with a direction searched at each vertex and interpolated "
                  "along the faces.");
   args.AddOption(&line_order, "-lo", "--line-order",
                  "Order of the quadrature along the paths; negative takes "
                  "twice the element order plus two.");
   args.AddOption(&extend, "-ext", "--extend", "-no-ext", "--no-extend",
                  "Transfer the datum from Gamma. Without it the datum is "
                  "read on Gamma_h itself, which is a different problem.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Send the potential on the finest D_h to GLVis.");
   args.AddOption(&control, "-ctl", "--control", "-no-ctl", "--no-control",
                  "Also solve with the datum read on Gamma_h and report the "
                  "ratio of the errors.");
   args.AddOption(&postprocess, "-rec", "--reconstruct", "-no-rec",
                  "--no-reconstruct",
                  "Postprocess flux and potential onto spaces one order "
                  "higher and report their errors and the total flux's.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   MFEM_VERIFY(problem >= 1 && problem <= 3, "unknown problem " << problem);

   PathFamily family;
   if (string(path_type) == "cp")      { family = PathFamily::ClosestPoint; }
   else if (string(path_type) == "ls") { family = PathFamily::LevelSet; }
   else if (string(path_type) == "vc") { family = PathFamily::VertexCone; }
   else { MFEM_ABORT("unknown path family '" << path_type << "'"); }

   if (problem == 3 && family == PathFamily::ClosestPoint)
   {
      cout << "\nthe airfoil has no closest-point map in closed form; "
           "using the vertex-cone paths\n";
      family = PathFamily::VertexCone;
   }

   cout << "\n"
        << "The errors on D_h are plain L2 norms; those on its complement are\n"
        << "divided by the square root of its measure, which shrinks like h.\n\n"
        << "   n    elem     dofs      dist       ||u-u_h||   rate"
        << "     ||p-p_h||   rate      ext_u      rate      ext_p      rate";
   if (postprocess)
   {
      cout << "     ||u-u_t||   rate     ||u-u*||    rate     ||p-p*||    rate";
   }
   if (control && extend) { cout << "    ratio_u  ratio_p"; }
   cout << "\n";

   auto column = [](real_t err, real_t prev)
   {
      cout << "   " << scientific << setprecision(4) << err << "  " << fixed
           << setw(5) << setprecision(2) << (prev > 0.0 ? log2(prev / err) : 0.0);
   };

   real_t prev_u = 0.0, prev_p = 0.0, prev_xu = 0.0, prev_xp = 0.0;
   real_t prev_ut = 0.0, prev_us = 0.0, prev_ps = 0.0;
   real_t worst_area = 0.0;
   int nn = n;
   for (int r = 0; r <= refinements; r++)
   {
      const Result e = Solve(nn, order, tau, offset, family, extend,
                             line_order, postprocess,
                             visualization && r == refinements);

      cout << setw(4) << nn << setw(8) << e.elements << setw(9) << e.dofs
           << "  " << scientific << setprecision(2) << e.dist;
      column(e.err_u, prev_u);
      column(e.err_p, prev_p);
      column(e.ext_u, prev_xu);
      column(e.ext_p, prev_xp);
      if (postprocess)
      {
         column(e.err_ut, prev_ut);
         column(e.err_us, prev_us);
         column(e.err_ps, prev_ps);
      }

      if (control && extend)
      {
         const Result c = Solve(nn, order, tau, offset, family, false,
                                line_order, false);
         cout << "  " << setw(7) << setprecision(3) << e.err_u / c.err_u
              << "  " << setw(7) << e.err_p / c.err_p;
         if (!c.converged) { cout << "  [control did not converge]"; }
      }
      if (!e.converged) { cout << "  [NOT CONVERGED]"; }
      cout << "\n";

      if (e.widened > 0)
      {
         cout << "  [" << e.widened << " vertices needed a widened fan]";
      }
      if (fabs(e.area_err) > fabs(worst_area)) { worst_area = e.area_err; }
      prev_u = e.err_u;
      prev_p = e.err_p;
      prev_xu = e.ext_u;
      prev_xp = e.ext_p;
      prev_ut = e.err_ut;
      prev_us = e.err_us;
      prev_ps = e.err_ps;
      nn *= 2;
   }

   // The regions swept by the paths must tile the complement exactly, or the
   // two columns above are integrals over the wrong set. The closest-point map
   // tiles; a family following each face's own normal does not, adjacent faces
   // disagreeing on the path through the vertex they share.
   cout << "\nlargest relative error in the measure of D_h^c (signed): "
        << scientific << setprecision(2) << worst_area << "\n";
   if (fabs(worst_area) > 1e-6)
   {
      cout << "the extension regions do not tile the complement, so the two "
           "columns above\nare not integrals over it\n";
   }
   cout << endl;

   return 0;
}
