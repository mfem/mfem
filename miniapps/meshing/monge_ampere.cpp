#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;


class MA_Integrator : public NonlinearFormIntegrator
{
protected:
   FiniteElementSpace *fespace;
   GridFunction *P;
   Geometry::Type geom; 
   int own_integ;
   LinearFECollection lfec;
   IsoparametricTransformation T;
   NonlinearFormIntegrator *integ;

   Vector M;
   DenseMatrix J;

public:
   MA_Integrator(FiniteElementSpace *fespace = NULL, GridFunction P = NULL, int _own_integ = 1)
      : geom(Geometry::INVALID), own_integ(_own_integ) { }

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Tr,
                                   const Vector &elfun);
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);
   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun, DenseMatrix &elmat);
};


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

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file = "../../data/beam-quad.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
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
   args.AddOption(&lim_const, "-lc", "--limit-const", "Limiting constant.");
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

   // Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   cout << "Mesh curvature: ";
   if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // Define a finite element space on the mesh.
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec = new H1_FECollection(mesh_poly_deg, dim); }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);

   // Make the mesh curved based on the above finite element space.
   mesh->SetNodalFESpace(fespace);

   // Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // Get the mesh nodes as a finite element grid function in fespace.
   GridFunction *x = mesh->GetNodes();

   // Define a vector representing the minimal local mesh size in the mesh
   //    nodes. In addition, compute average mesh size and total volume.
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

   // Add a random perturbation to the nodes in the interior of the domain.
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

   // Save the starting (prior to the optimization) mesh to a file. 
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // Store the starting (prior to the optimization) positions.
   GridFunction x0(fespace);
   x0 = *x;

   int p = x->Size()/2;
   GridFunction Q(fespace);
   Q = *x;

   // Define fespace to defined P
   FiniteElementSpace *fespace_scalar = new FiniteElementSpace(mesh, fec, 1);
   GridFunction P(fespace_scalar);

   for (int i = 0; i < p; i++)
   {
      P(i) = 0.5*(std::pow(Q(i),2)+std::pow(Q(i+p),2));
   }
   P.SetTrueVector();
   P.SetFromTrueVector();

   // Set up integrator.
   const int geom_type = fespace_scalar->GetFE(0)->GetGeomType();
   const IntegrationRule *ir = &IntRules.Get(geom_type, quad_order);

   // Setup the final NonlinearForm.
   NonlinearForm MA(fespace_scalar);
   MA.AddDomainIntegrator(new MA_Integrator(fespace_scalar, P, 0));
   const double init_energy = 0.0;

   // Fix all boundary nodes
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   MA.SetEssentialBC(ess_bdr);

   // As we use the Newton method to solve the resulting nonlinear system,
   //  here we setup the linear solver for the system's Jacobian.
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

   // Compute the minimum det(J) of the starting mesh.
   double tauval = infinity();
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

   // Finally, perform the nonlinear optimization.
   NewtonSolver *newton = NULL;
   if (tauval > 0.0)
   {
      tauval = 0.0;
      newton = new RelaxedNewtonSolver(*ir, fespace);
      cout << "The RelaxedNewtonSolver is used (as all det(J)>0)." << endl;
   }
   else
   {
      cout << "The mesh is inverted. Use an untangling metric." << endl;
      return 3;
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
   newton->Mult(b, P.GetTrueVector());
   cout << "Before set" << endl;
   x->SetFromTrueVector();
   if (newton->GetConverged() == false)
   {
      cout << "NewtonIteration: rtol = " << newton_rtol << " not achieved."
           << endl;
   }
   delete newton;

   // Save the optimized mesh to a file.
   {
      ofstream mesh_ofs("optimized.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }

   // Compute the amount of energy decrease.
   const double fin_energy = MA.GetGridFunctionEnergy(*x);
   double metric_part = fin_energy;

   // Visualize the mesh displacement.
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

   // Free the used memory.
   delete S;
   delete fespace;
   delete fespace_scalar;
   delete fec;
   delete mesh;

   return 0;
}

// inline void MA_Integrator::SetSpace(const FiniteElementSpace &fespace_scalar)
// {
//    GridFunction P(fespace_scalar); 
// }

double MA_Integrator::GetElementEnergy(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
{
   double c = 0.0;
   return c;
   //return integ->GetElementEnergy(el, T, elfun);
}

void MA_Integrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   int dof = el.GetDof();
   //int dim = el.GetDim();
   int dim = 1;

   //GridFunction MAvalue(fespace);
   Vector MAvalue;
   MAvalue.SetSize(dof);
   GridFunction Px(fespace), Py(fespace), Pxx(fespace), Pxy(fespace), Pyy(fespace);

   P->GetDerivative(1,0,Px);
   P->GetDerivative(1,1,Py);
   Px.GetDerivative(1,0,Pxx);
   Px.GetDerivative(1,1,Pxy);
   Py.GetDerivative(1,1,Pyy);


   for (int i = 0; i < dof; i++)
   {
      MAvalue(i) = Pxx(i)*Pyy(i) - Pxy(i)*Pxy(i) - 1;
   }
}

void MA_Integrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, DenseMatrix &elmat)
{
   //J.SetSize(dof,dof); //Monge Ampere Jacobian (N x N)
   
   int dof = el.GetDof();
   //int dim = el.GetDim();
   int dim = 1;

   //GridFunction MAvalue(fespace);
   Vector MAvalue;
   MAvalue.SetSize(dof);
   GridFunction Px(fespace), Py(fespace), Pxx(fespace), Pxy(fespace), Pyy(fespace);

   P->GetDerivative(1,0,Px);
   P->GetDerivative(1,1,Py);
   Px.GetDerivative(1,0,Pxx);
   Px.GetDerivative(1,1,Pxy);
   Py.GetDerivative(1,1,Pyy);


   for (int i = 0; i < dof; i++)
   {
      MAvalue(i) = Pxx(i)*Pyy(i) - Pxy(i)*Pxy(i) - 1;
   }

   GridFunction J(fespace);
   J.GetGradient(Tr, MAvalue);
}
