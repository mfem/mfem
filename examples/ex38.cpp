//                                MFEM Example 34
//
// Compile with: make ex34
//
// Sample runs:  ex34
//               ex34 -r 1 -o 2
//               ex34 -r 1 -ni 30
//               ex34 -m ../data/square-disc.mesh
//
// Notice: depending on the mesh and finite element order, more Newton
// iterations might be necessary. If this is the case, use the -ni command line
// parameter to increase the maximum number of Newton's iterations.
//
// Description: We solve the following non-linear Poisson equation
//
//  (1) -Div(a(u).Grad(u)) = f
//
// with homogenous u = 0 boundary condition.
//
// We solve this problem using the Newton's method, where we find u such that
// g(u) = 0. The first step is to multiply by a test function and integrate by
// parts.
//
//  (2) (a(u).Grad(u), Grad(v)) = (f, v)
//
// Substituting u = u_0 + du into (2) gives
//
//  (3) (a(u_0 + du).Grad(u_0 + du), Grad(v)) = (f, v)
//
// Then, we use the Taylor expansion around u_0 of the non-linear term:
//
//  (4) a(u_0 + du) = a(u_0) + a'(u_0)du + O(du^2)
//
// substituting (4) into (3) gives
//
//  (5) ((a(u_0) + a'(u_0)du + O(du^2)).Grad(u_0 + du), Grad(v)) = (f, v)
//
// Replacing Grad(u_0 + du) by Grad(u_0) + Grad(du), we do the expansion:
//
//  (6) ((a(u_0) + a'(u_0)du + O(du^2)).(Grad(u_0) + Grad(du)), Grad(v)) = (f, v)
//
//  (7) ((a(u_0).Grad(u_0) + a'(u_0).du.Grad(u_0) + O(du^2).Grad(u_0) +
//       (a(u_0).Grad(du) + a'(u_0).du.Grad(du) + O(du^2).Grad(du))), Grad(v)) = (f, v)
//
// The quadratic terms in du are not required to converge and can be removed.
// We place all du terms on the lhs and u_0 terms on the rhs. The goal is to
// solve for the correction du.
//
//  (8) (a(u_0).Grad(du) + a'(u_0).du.Grad(u_0) +
//       a(u_0).Grad(u_0), Grad(v)) = (f, v)
//  (9) (a(u_0).Grad(du) + a'(u_0).du.Grad(u_0), Grad(v)) =
//      (f, v) - (a(u_0).Grad(u_0), Grad(v))
//
// At this point, we discretize the equation, in particular replacing du by
// du_h = Sum(d_j * phi_j), j=1..n
//
// (10) Sum(d_j.(a(u_h).Grad(phi_j) + a'(u_h).phi_j.Grad(u_h), Grad(phi_i)) =
//          (f, phi_i) - (a(u_h).Grad(u_h), Grad(phi_i)), j=1..n
//
// The rhs corresponds to the residue and is computed by the method
// Operator::AssembleElementVector(). The lhs is the bilinear term
// corresponding to the Jacobian matrix. This matrix is computed element wise
// by the method Operator::AssembleElementGrad(). The system Jdu = r is solved
// for the correction du to the solution u.
//
// References:
//
// [1] The Finite Element Method: Theory, Implementation, and Applications,
//     Mats G. Larson and Fredrik Bengzon, Springer, 2013

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Evaluate a(u)
double fct_a(double u)
{
   return (0.125 + u*u);
}

// Evaluate a'(u)
// FIXME: Could we use AutoDiff here?
double fct_diff_a(double u)
{
   return 2 * 0.125 * u;
}

// Define a coefficient that, given a grid function u, returns f(u)
class NonlinearCoefficient : public Coefficient
{
   GridFunction& gf;
   function<double(double)> f;

public:
   NonlinearCoefficient(GridFunction& gf_,
                        function<double(double)> f_);
   double Eval(ElementTransformation& T,
               const IntegrationPoint& ip);
};

class NLOperator : public Operator
{
private:
   NonlinearForm* N;
   mutable SparseMatrix* Jacobian;

public:
   NLOperator(NonlinearForm* N_, int size);
   virtual void Mult(const Vector& x, Vector& y) const;
   virtual Operator& GetGradient(const Vector& x) const;
};

class ProblemNLFIntegrator : public NonlinearFormIntegrator
{
private:
   FiniteElementSpace& fes;
   GridFunction gf_u;  // u
   Coefficient* f; // f on the rhs (source term)
   function<double(double)> func_a;
   function<double(double)> func_diff_a;

   Array<int> dofs;
   Vector shape;
   DenseMatrix dshape, dshapedxt, invdfdx;
   Vector vec, pointflux, dshape_grad_u;

public:
   ProblemNLFIntegrator(FiniteElementSpace& fes_,
                        Coefficient* f_,
                        function<double(double)> func_a_,
                        function<double(double)> func_diff_a_);
   virtual void AssembleElementVector(const FiniteElement& el,
                                      ElementTransformation& Tr,
                                      const Vector& elfun,
                                      Vector& elvect) override;
   virtual void AssembleElementGrad(const FiniteElement& el,
                                    ElementTransformation& Tr,
                                    const Vector& elfun,
                                    DenseMatrix& elmat) override;
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 2;
   int order = 1;
   int newton_iters = 30;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&newton_iters, "-ni", "--newton-iter",
                  "Maximum number of Newton iterations.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution, as specified on the
   // command line.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace fespace(&mesh, fec);
   int fe_size = fespace.GetTrueVSize();
   cout << "Number of finite element unknowns: "
        << fe_size << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Setup visualization as usual
   socketstream vis;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis.open(vishost, visport);
   }

   // 7. We create the non-linear coefficient and setup the solution vector.
   ConstantCoefficient u0(0.0);
   ConstantCoefficient f_coeff(-1.0);
   GridFunction sol(&fespace);
   //sol = 0.0;
   sol.ProjectCoefficient(u0);
   sol.SetTrueVector();

   // 8. The NonLinearForm integrator computes both the Jacobian matric and the
   //    residual vector computed at each Newton iterations. The existing
   //    values in the solution vectors at the ess_tdof_list indices represent the s.
   NonlinearForm nlf(&fespace);
   nlf.AddDomainIntegrator(new ProblemNLFIntegrator(fespace, &f_coeff, fct_a,
                                                    fct_diff_a));
   nlf.SetEssentialTrueDofs(ess_tdof_list);

   // 9. Here we setup two solvers. The MINRES solver finds the correction du
   //    to apply to the current solution u. The NewtonSolver is the outer newton
   //    iterating until the correction is below the specified tolerance, or
   //    the maximum number of iteration is reached.
   DSmoother J_prec(1);
   MINRESSolver J_minres;
   J_minres.SetPreconditioner(J_prec);
   J_minres.SetRelTol(1e-8);
   J_minres.SetAbsTol(0.0);
   J_minres.SetMaxIter(300);

   NLOperator N_oper(&nlf, fe_size);
   NewtonSolver newton_solver;
   newton_solver.SetRelTol(1e-4);
   newton_solver.SetAbsTol(0.0);
   newton_solver.SetMaxIter(newton_iters);
   newton_solver.SetSolver(J_minres);
   newton_solver.SetOperator(N_oper);
   newton_solver.SetPrintLevel(1);

   Vector zero;
   newton_solver.Mult(zero, sol);

   // 10. Show and/or save the the solution
   if (visualization)
   {
      vis << "solution\n" << mesh << sol;
   }
   sol.Save("ex34.gf");
   mesh.Save("ex34.mesh");

   return 0;
}

NonlinearCoefficient::NonlinearCoefficient(GridFunction &gf_,
                                           function<double (double)> f_)
   : gf(gf_), f(move(f_)) {}

double NonlinearCoefficient::Eval(ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   return f(gf.GetValue(T, ip));
}


ProblemNLFIntegrator::ProblemNLFIntegrator(FiniteElementSpace &fes_,
                                           Coefficient *f_, function<double (double)> func_a_,
                                           function<double (double)> func_diff_a_)
   : fes(fes_), gf_u(&fes), f(f_), func_a(func_a_),
     func_diff_a(func_diff_a_) {}

NLOperator::NLOperator(NonlinearForm *N_, int size)
   : Operator(size), N(N_), Jacobian(NULL) {}

void NLOperator::Mult(const Vector &x, Vector &y) const
{
   N->Mult(x, y);
}

Operator &NLOperator::GetGradient(const Vector &x) const
{
   Jacobian = dynamic_cast<SparseMatrix*>(&N->GetGradient(x));
   return *Jacobian;
}

void ProblemNLFIntegrator::AssembleElementVector(const FiniteElement &el,
                                                 ElementTransformation &Tr, const Vector &elfun, Vector &elvect)
{
   // Computes the residual vector
   int dof = el.GetDof();
   int dim = el.GetDim();
   shape.SetSize(dof);
   dshape.SetSize(dof, dim);
   invdfdx.SetSize(dim);
   vec.SetSize(dim);
   pointflux.SetSize(dim);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule* ir =
      &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + Tr.OrderW());

   // We only have one loop over gauss points. The loop over all dofs is
   // implicit and done using vector operators instead.
   fes.GetElementDofs(Tr.ElementNo, dofs);
   gf_u.SetSubVector(dofs, elfun);
   NonlinearCoefficient coeff(gf_u, func_a);

   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint& ip = ir->IntPoint(q);
      Tr.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      // Compute (f, v), v is shape function
      double fun_val = (*f).Eval(Tr, ip);
      double w = fun_val * ip.weight * Tr.Weight();
      add(elvect, w, shape, elvect);

      // Given u, compute (a(u).grad(u), grad(v)), v is shape function.
      // Based on DiffusionIntegrator::AssembleElementVector()
      // FIXME: what is the logic behind the multiplication with the jacobian
      // inverse twice?
      double a_u = coeff.Eval(Tr, ip);
      CalcAdjugate(Tr.Jacobian(), invdfdx);
      dshape.MultTranspose(elfun, vec);
      invdfdx.MultTranspose(vec, pointflux);
      double ww = a_u * ip.weight / Tr.Weight();
      pointflux *= ww;
      invdfdx.Mult(pointflux, vec);
      dshape.AddMult(vec, elvect);
   }
}

void ProblemNLFIntegrator::AssembleElementGrad(const FiniteElement &el,
                                               ElementTransformation &Tr, const Vector &elfun, DenseMatrix &elmat)
{
   // Computes the Jacobian matrix J.d = R
   // The parent computes d = [J]^-1*R
   int dof = el.GetDof();
   int dim = el.GetDim();
   dshapedxt.SetSize(dof, dim);
   dshape.SetSize(dof, dim);
   shape.SetSize(dof);
   invdfdx.SetSize(dim);
   vec.SetSize(dim);
   pointflux.SetSize(dim);
   dshape_grad_u.SetSize(dof);
   elmat.SetSize(dof);
   elmat = 0.0;

   const IntegrationRule* ir =
      &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + Tr.OrderW());

   fes.GetElementDofs(Tr.ElementNo, dofs);
   gf_u.SetSubVector(dofs, elfun);
   NonlinearCoefficient coeff(gf_u, func_a);
   NonlinearCoefficient dcoeff(gf_u, func_diff_a);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint& ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      // Compute (a(u).grad(du), grad(v)).
      // Based on DiffusionIntegrator::AssembleElementMatrix()
      // The inverse of a matrix is defined by:
      //   [A]^-1 = (1 / det(A)) * adj(A)
      // Tr.Weight() returns the determinant and the Tr.AdjugateJacobian() the
      // adjugate. The call Mult(dshape, Tr.AdjugateJacobian(), dshapedxt) IS
      // actually the same as el.CalcPhysDShape();
      double a_u = coeff.Eval(Tr, ip);
      double w = a_u * ip.weight / Tr.Weight();
      Mult(dshape, Tr.AdjugateJacobian(), dshapedxt);
      AddMult_a_AAt(w, dshapedxt, elmat);

      // Compute (a'(u).du.grad(u), grad(v)), v is shape function
      // a'(u): scalar value at the integration point (1 size)
      // grad(u): order 1 tensor of the gradient u at the integration point (dim size)
      // du: unkown scalar value, replaced by phi_j (dofs size)
      // grad(v): partial derivative of shape functions at the integration point
      // (dofs by dim size)
      double da_u = dcoeff.Eval(Tr, ip);
      gf_u.GetGradient(Tr, vec);
      dshape.Mult(vec, dshape_grad_u);
      double ww = da_u * ip.weight * Tr.Weight();
      AddMult_a_VWt(ww, dshape_grad_u, shape, elmat);
   }
}
