#include "../mfem.hpp"
#include <fstream>
#include <iostream>
#include "../miniapps/meshing/mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

#define ring_radius 0.2
// use 0.2 ring for dist_fun because square_disc has the
// hole with this radius
double dist_fun(const Vector &x)
{
   double dx = x(0) - 0.5,
          dy = x(1) - 0.5,
          rv = dx*dx + dy*dy;
   rv = rv > 0 ? pow(rv, 0.5) : 0;
   double dist0 = rv - ring_radius; // +ve is the domain
   return dist0;
}

void dist_vec_c(const Vector &x, Vector &p)
{
   double dist0 = dist_fun(x);
   double theta = std::atan2(x(1)-0.5, x(0)-0.5);
   p(0) = -dist0*std::cos(theta);
   p(1) = -dist0*std::sin(theta);
}
#undef ring_radius

void optimize_mesh_with_distfun(Mesh &mesh, GridFunction &x, GridFunction &dist)
{
   // Mesh -> mesh
   // x -> nodal gridfunction
   // dist -> distance function
   int quad_order = 8;

   int solver_type       = 0;
   int solver_iter       = 200;
   double solver_rtol    = 1e-10;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = false;
   int verbosity_level   = 0;
   bool fdscheme         = false;
   int adapt_eval        = 0;
   bool exactaction      = false;
   bool visualization    = false;


   const double eps = 0.001;
   const double aspr_ratio = 20;
   const double size_ratio = 40.0;

   int mesh_poly_deg = x.FESpace()->GetFE(0)->GetOrder();
   FiniteElementSpace *fespace = x.FESpace();
   TMOP_QualityMetric *metric = new TMOP_Metric_007;
   x.SetTrueVector();
   x.SetFromTrueVector();

   // 5. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   GridFunction x0(fespace);
   x0 = x;

   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   DiscreteAdaptTC *tcd = NULL;
   GridFunction disc = dist, d_x = dist, d_y = dist;
   GridFunction size = dist, aspr = dist;
   size *= 0.;
   aspr *= 0.;

   for (int i = 0; i < disc.Size(); i++)
   {
      if (disc(i) <= 0.075) { disc(i) = 0; }
      else { disc(i) = 1; }
   }

   target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
   tcd = new DiscreteAdaptTC(target_t);
#ifdef MFEM_USE_GSLIB
   tcd->SetAdaptivityEvaluator(new InterpolatorFP);
#else
   MFEM_ABORT("MFEM is not built with GSLIB.");
#endif

   DiffuseField(disc, 1);
   //Get  partials with respect to x and y of the grid function
   disc.GetDerivative(1,0,d_x);
   disc.GetDerivative(1,1,d_y);

   //Compute the squared magnitude of the gradient
   for (int i = 0; i < size.Size(); i++)
   {
      size(i) = std::pow(d_x(i),2)+std::pow(d_y(i),2);
   }
   const double max = size.Max();

   for (int i = 0; i < d_x.Size(); i++)
   {
      d_x(i) = std::abs(d_x(i));
      d_y(i) = std::abs(d_y(i));
   }

   for (int i = 0; i < size.Size(); i++)
   {
      size(i) = (size(i)/max);
      aspr(i) = (d_x(i)+eps)/(d_y(i)+eps);
      if (aspr(i) > aspr_ratio) {aspr(i) = aspr_ratio;}
      if (aspr(i) < 1.0/aspr_ratio) {aspr(i) = 1.0/aspr_ratio;}
   }
   Vector vals;
   const int NE = mesh.GetNE();
   double volume = 0.0, volume_ind = 0.0;

   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *Tr = mesh.GetElementTransformation(i);
      const IntegrationRule &ir =
         IntRules.Get(mesh.GetElementBaseGeometry(i), Tr->OrderJ());
      size.GetValues(i, ir, vals);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         Tr->SetIntPoint(&ip);
         volume     += ip.weight * Tr->Weight();
         volume_ind += vals(j) * ip.weight * Tr->Weight();
      }
   }

   const double avg_zone_size = volume / NE;

   const double small_avg_ratio = (volume_ind + (volume - volume_ind) /
                                   size_ratio) /
                                  volume;

   const double small_zone_size = small_avg_ratio * avg_zone_size;
   const double big_zone_size   = size_ratio * small_zone_size * 2;

   for (int i = 0; i < size.Size(); i++)
   {
      const double val = size(i);
      const double a = (big_zone_size - small_zone_size) / small_zone_size;
      size(i) = big_zone_size / (1.0+a*val);
   }

   DiffuseField(size, 2);

   for (int i = 0; i < aspr.Size(); i++)
   {
      double aval = aspr(i);
      if (aval < 1.) { aspr(i) = 1. - (1./aval - 1.); }
      aspr(i) = aspr(i)-1.;
   }

   DiffuseField(aspr, 1);

   //move aspr back to 1
   for (int i = 0; i < aspr.Size(); i++)
   {
      aspr(i) = aspr(i)+1.;
      double aval = aspr(i);
      if (aval < 1.) { aval = 1. + (1. - aval); aspr(i) = 1./aval; }
   }

   tcd->SetSerialDiscreteTargetSize(size);
   //tcd->SetSerialDiscreteTargetAspectRatio(aspr);
   target_c = tcd;

   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh.Print(sock);
      disc.Save(sock);
      sock.send();
      sock << "window_title 'Indicator'\n"
           << "window_geometry "
           << 000 << " " << 0 << " " << 300 << " " << 300 << "\n"
           << "keys jRmclA" << endl;
   }

   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh.Print(sock);
      size.Save(sock);
      sock.send();
      sock << "window_title 'SIZE'\n"
           << "window_geometry "
           << 300 << " " << 0 << " " << 300 << " " << 300 << "\n"
           << "keys jRmclA" << endl;
   }
   //MFEM_ABORT(" ");

   target_c->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ =
      new TMOP_Integrator(metric, target_c);

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = &IntRulesLo;
   he_nlf_integ->SetIntegrationRules(*irules, quad_order);
   const int dim = mesh.Dimension();
   if (dim == 2)
   {
      cout << "Triangle quadrature points: "
           << irules->Get(Geometry::TRIANGLE, quad_order).GetNPoints()
           << "\nQuadrilateral quadrature points: "
           << irules->Get(Geometry::SQUARE, quad_order).GetNPoints() << endl;
   }
   if (dim == 3)
   {
      cout << "Tetrahedron quadrature points: "
           << irules->Get(Geometry::TETRAHEDRON, quad_order).GetNPoints()
           << "\nHexahedron quadrature points: "
           << irules->Get(Geometry::CUBE, quad_order).GetNPoints()
           << "\nPrism quadrature points: "
           << irules->Get(Geometry::PRISM, quad_order).GetNPoints() << endl;
   }


   NonlinearForm a(fespace);

   a.AddDomainIntegrator(he_nlf_integ);
   const double init_energy = a.GetGridFunctionEnergy(x)/mesh.GetNE();

   // fix_bnd
   if (true)
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }

   Solver *S = NULL;
   const double linsol_rtol = 1e-12;

   MINRESSolver *minres = new MINRESSolver;
   minres->SetMaxIter(max_lin_iter);
   minres->SetRelTol(linsol_rtol);
   minres->SetAbsTol(0.0);
   minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
   S = minres;

   const IntegrationRule &ir =
      irules->Get(fespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(ir, solver_type);

   // Provide all integration rules in case of a mixed mesh.
   solver.SetIntegrationRules(*irules, quad_order);
   if (solver_type == 0)
   {
      // Specify linear solver when we use a Newton-based solver.
      solver.SetPreconditioner(*S);
   }
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(0.0);
   solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   solver.SetOperator(a);
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();
   if (solver.GetConverged() == false)
   {
      cout << "Nonlinear solver: rtol = " << solver_rtol << " not achieved.\n";
   }

   delete S;
   delete target_c;
   delete metric;

}
