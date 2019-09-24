#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

// Metric values are visualized by creating an L2 finite element functions and
// computing the metric values at the nodes.
void vis_metric(int order, TMOP_QualityMetric &qm, const TargetConstructor &tc,
                Mesh &mesh, char *title, int position)
{
   L2_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 1);
   GridFunction metric(&fes);
   InterpolateTMOP_QualityMetric(qm, tc, mesh, metric);
   osockstream sock(19916, "localhost");
   sock << "solution\n";
   mesh.Print(sock);
   metric.Save(sock);
   sock.send();
   sock << "window_title '"<< title << "'\n"
        << "window_geometry "
        << position << " " << 0 << " " << 600 << " " << 600 << "\n"
        << "keys jRmclA" << endl;
}

class RelaxedNewtonSolver : public NewtonSolver
{
private:
   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;
   FiniteElementSpace *fes;
   mutable GridFunction x_gf;

public:
   RelaxedNewtonSolver(const IntegrationRule &irule, FiniteElementSpace *f)
      : ir(irule), fes(f) { }

   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const;
};

double RelaxedNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &b) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   MFEM_VERIFY(nlf != NULL, "invalid Operator subclass");
   const bool have_b = (b.Size() == Height());

   const int NE = fes->GetMesh()->GetNE(), dim = fes->GetFE(0)->GetDim(),
             dof = fes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   Vector x_out(x.Size());
   bool x_out_ok = false;
   const double energy_in = nlf->GetEnergy(x);
   double scale = 1.0, energy_out;
   double norm0 = Norm(r);
   x_gf.MakeTRef(fes, x_out, 0);

   // Decreases the scaling of the update until the new mesh is valid.
   for (int i = 0; i < 12; i++)
   {
      add(x, -scale, c, x_out);
      x_gf.SetFromTrueVector();

      energy_out = nlf->GetGridFunctionEnergy(x_gf);
      if (energy_out > 1.2*energy_in || std::isnan(energy_out) != 0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Increasing energy." << endl; }
         scale *= 0.5; continue;
      }

      int jac_ok = 1;
      for (int i = 0; i < NE; i++)
      {
         fes->GetElementVDofs(i, xdofs);
         x_gf.GetSubVector(xdofs, posV);
         for (int j = 0; j < nsp; j++)
         {
            fes->GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
            MultAtB(pos, dshape, Jpr);
            if (Jpr.Det() <= 0.0) { jac_ok = 0; goto break2; }
         }
      }
   break2:
      if (jac_ok == 0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Neg det(J) found." << endl; }
         scale *= 0.5; continue;
      }

      oper->Mult(x_out, r);
      if (have_b) { r -= b; }
      double norm = Norm(r);

      if (norm > 1.2*norm0)
      {
         if (print_level >= 0)
         { cout << "Scale = " << scale << " Norm increased." << endl; }
         scale *= 0.5; continue;
      }
      else { x_out_ok = true; break; }
   }

   if (print_level >= 0)
   {
      cout << "Energy decrease: "
           << (energy_in - energy_out) / energy_in * 100.0
           << "% with " << scale << " scaling." << endl;
   }

   if (x_out_ok == false) { scale = 0.0; }

   return scale;
}

// Allows negative Jacobians. Used in untangling metrics.
class DescentNewtonSolver : public NewtonSolver
{
private:
   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;
   FiniteElementSpace *fes;
   mutable GridFunction x_gf;

public:
   DescentNewtonSolver(const IntegrationRule &irule, FiniteElementSpace *f)
      : ir(irule), fes(f) { }

   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const;
};

double DescentNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &b) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   MFEM_VERIFY(nlf != NULL, "invalid Operator subclass");

   const int NE = fes->GetMesh()->GetNE(), dim = fes->GetFE(0)->GetDim(),
             dof = fes->GetFE(0)->GetDof(), nsp = ir.GetNPoints();
   Array<int> xdofs(dof * dim);
   DenseMatrix Jpr(dim), dshape(dof, dim), pos(dof, dim);
   Vector posV(pos.Data(), dof * dim);

   x_gf.MakeTRef(fes, x.GetData());
   x_gf.SetFromTrueVector();

   double min_detJ = infinity();
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, xdofs);
      x_gf.GetSubVector(xdofs, posV);
      for (int j = 0; j < nsp; j++)
      {
         fes->GetFE(i)->CalcDShape(ir.IntPoint(j), dshape);
         MultAtB(pos, dshape, Jpr);
         min_detJ = min(min_detJ, Jpr.Det());
      }
   }
   cout << "Minimum det(J) = " << min_detJ << endl;

   Vector x_out(x.Size());
   bool x_out_ok = false;
   const double energy_in = nlf->GetGridFunctionEnergy(x_gf);
   double scale = 1.0, energy_out;

   for (int i = 0; i < 7; i++)
   {
      add(x, -scale, c, x_out);

      energy_out = nlf->GetEnergy(x_out);
      if (energy_out > energy_in || std::isnan(energy_out) != 0)
      {
         scale *= 0.5;
      }
      else { x_out_ok = true; break; }
   }

   cout << "Energy decrease: " << (energy_in - energy_out) / energy_in * 100.0
        << "% with " << scale << " scaling." << endl;

   if (x_out_ok == false) { return 0.0; }

   return scale;
}

// Additional IntegrationRules that can be used with the --quad-type option.
IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);


int main (int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "../../data/beam-quad.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 2;
   double lim_const      = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int newton_iter       = 10;
   double newton_rtol    = 1e-12;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool visualization    = false;
   int verbosity_level   = 0;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "1  : |T|^2                          -- 2D shape\n\t");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "2: Ideal shape, equal size\n\t");
   args.AddOption(&lim_const, "-lc", "--limit-const", "Limiting constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&newton_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver: 0 - l1-Jacobi, 1 - CG, 2 - MINRES.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   cout << "Mesh curvature: ";
   if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // 3. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec = new H1_FECollection(mesh_poly_deg, dim); }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);

   // 4. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(fespace);

   // 5. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 6. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   GridFunction *x = mesh->GetNodes();

   // 7. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in pfespace. Note: this is partition-dependent.
   //
   //    In addition, compute average mesh size and total volume.
   Vector h0(fespace->GetNDofs());
   h0 = infinity();
   double volume = 0.0;
   Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const double hi = mesh->GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      volume += mesh->GetElementVolume(i);
   }
   const double small_phys_size = pow(volume, 1.0 / dim) / 100.0;

   // 8. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(fespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < fespace->GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fespace->DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fespace->GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      fespace->GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   *x -= rdm;
   // Set the perturbation of all nodes from the true nodes.
   x->SetTrueVector();
   x->SetFromTrueVector();

   // 9. Save the starting (prior to the optimization) mesh to a file. This
   //    output can be viewed later using GLVis: "glvis -m perturbed.mesh".
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 10. Store the starting (prior to the optimization) positions.
   GridFunction x0(fespace);
   x0 = *x;

   int p = x->Size()/2;
   GridFunction Q(fespace);
   Q = *x;

   GridFunction P(fespace);

   for (int i = 0; i < p; i++)
   {
      P(i) = 0.5*(std::pow(Q(i),2)+std::pow(Q(i+p),2));
   }

   GridFunction dP_x(fespace);
   GridFunction dP_y(fespace);
   GridFunction dP_xx(fespace);
   GridFunction dP_xy(fespace);
   GridFunction dP_yy(fespace);

   P.GetDerivative(1,0,dP_x);
   P.GetDerivative(1,1,dP_y);
   dP_x.GetDerivative(1,0,dP_xx);
   dP_x.GetDerivative(1,1,dP_xy);
   dP_y.GetDerivative(1,1,dP_yy);

   // for (int i = 0; i < d_x.Size(); i++)
   // {
   //    MA(i) = d_xx(i)*d_yy(i) - d_xy(i)*d_xy(i) - 1;
   // }

   // 11. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 22: metric = new TMOP_Metric_022(tauval); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
      case 211: metric = new TMOP_Metric_211; break;
      case 252: metric = new TMOP_Metric_252(tauval); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 352: metric = new TMOP_Metric_352(tauval); break;
      default: cout << "Unknown metric_id: " << metric_id << endl; return 3;
   }
   TargetConstructor::TargetType target_t;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      default: cout << "Unknown target_id: " << target_id << endl;
         delete metric; return 3;
   }
   TargetConstructor *target_c = new TargetConstructor(target_t);
   target_c->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c);

   // 12. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = fespace->GetFE(0)->GetGeomType();
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default: cout << "Unknown quad_type: " << quad_type << endl;
         delete he_nlf_integ; return 3;
   }
   cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;
   he_nlf_integ->SetIntegrationRule(*ir);

   // 13. Limit the node movement.
   // The limiting distances can be given by a general function of space.
   GridFunction dist(fespace);
   dist = 1.0;
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

   // 14. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights. Note that there are no
   //     command-line options for the weights and the type of the second
   //     metric; one should update those in the code.

   NonlinearForm MA(fespace);
   MA.AddDomainIntegrator(he_nlf_integ);

   //const double init_energy = MA.GetGridFunctionEnergy(*x);
   const double init_energy = 0.0;

   // 15. Visualize the starting mesh and metric values.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_metric(mesh_poly_deg, *metric, *target_c, *mesh, title, 0);
   }

   // 16. Fix all boundary nodes

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   MA.SetEssentialBC(ess_bdr);


   // 17. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL;
   const double linsol_rtol = 1e-12;
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver;
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver;
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = minres;
   }

   // 18. Compute the minimum det(J) of the starting mesh.
   tauval = infinity();
   const int NE = mesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *transf = mesh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir->IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   cout << "Minimum det(J) of the original mesh is " << tauval << endl;

   // 19. Finally, perform the nonlinear optimization.
   NewtonSolver *newton = NULL;
   if (tauval > 0.0)
   {
      tauval = 0.0;
      newton = new RelaxedNewtonSolver(*ir, fespace);
      cout << "The RelaxedNewtonSolver is used (as all det(J)>0)." << endl;
   }
   else
   {
      if ( (dim == 2 && metric_id != 22 && metric_id != 252) ||
           (dim == 3 && metric_id != 352) )
      {
         cout << "The mesh is inverted. Use an untangling metric." << endl;
         return 3;
      }
      tauval -= 0.01 * h0.Min(); // Slightly below minJ0 to avoid div by 0.
      newton = new DescentNewtonSolver(*ir, fespace);
      cout << "The DescentNewtonSolver is used (as some det(J)<0)." << endl;
   }
   newton->SetPreconditioner(*S);
   newton->SetMaxIter(newton_iter);
   newton->SetRelTol(newton_rtol);
   newton->SetAbsTol(0.0);
   newton->SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   newton->SetOperator(MA);
   newton->Mult(b, x->GetTrueVector());
   x->SetFromTrueVector();
   if (newton->GetConverged() == false)
   {
      cout << "NewtonIteration: rtol = " << newton_rtol << " not achieved."
           << endl;
   }
   delete newton;

   // 20. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized.mesh".
   {
      ofstream mesh_ofs("optimized.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }

   // 21. Compute the amount of energy decrease.
   const double fin_energy = MA.GetGridFunctionEnergy(*x);
   double metric_part = fin_energy;
   if (lim_const != 0.0)
   {
      lim_coeff.constant = 0.0;
      metric_part = MA.GetGridFunctionEnergy(*x);
      lim_coeff.constant = lim_const;
   }
   cout << "Initial strain energy: " << init_energy
        << " = metrics: " << init_energy
        << " + limiting term: " << 0.0 << endl;
   cout << "  Final strain energy: " << fin_energy
        << " = metrics: " << metric_part
        << " + limiting term: " << fin_energy - metric_part << endl;
   cout << "The strain energy decreased by: " << setprecision(12)
        << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;

   // 22. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_metric(mesh_poly_deg, *metric, *target_c, *mesh, title, 600);
   }

   // 23. Visualize the mesh displacement.
   if (visualization)
   {
      x0 -= *x;
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      x0.Save(sock);
      sock.send();
      sock << "window_title 'Displacements'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   // 24. Free the used memory.
   delete S;
   delete target_c;
   delete metric;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
