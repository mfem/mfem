// Compile with: make dan
//
// Sample runs:  ./dan
//               ./dan -m meshes/DShape.msh
//               ./dan -m meshes/DShape.msh -o 2
//
// After, run:
//               glvis -m mesh.mesh -g sol.gf
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int test();
double one_over_r_mu(const Vector & x, double & mu);

class PlasmaModel
{
private:
  double alpha;
  double beta;
  double lambda;
  double gamma;
  double mu0;
  double r0;
public:
  PlasmaModel(double & alpha_, double & beta_, double & lambda_, double & gamma_, double & mu0_, double & r0_) :
    alpha(alpha_), beta(beta_), lambda(lambda_), gamma(gamma_), mu0(mu0_), r0(r0) { }
  double S_p_prime(double & psi_N) const;
  double S_ff_prime(double & psi_N) const;
  double S_prime_p_prime(double & psi_N) const;
  double S_prime_ff_prime(double & psi_N) const;
  double get_mu() const {return mu0;}
  ~PlasmaModel() { }
};

class DiffusionIntegratorCoefficient : public Coefficient
{
private:
  // double mu;
  PlasmaModel *model;
public:
  // DiffusionIntegratorCoefficient(double & mu_) : mu(mu_) { }
  DiffusionIntegratorCoefficient(PlasmaModel *model_) : model(model_) { }
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~DiffusionIntegratorCoefficient() { }
};

// option:
//   1: r S_p' + S_ff' / mu r
//   2: (r S'_p' + S'_ff' / mu r) / (psi_bdp - psi_max)
//   3: - (r S'_p' + S'_ff' / mu r) (1 - psi_N) / (psi_bdp - psi_max)
//   4: - (r S'_p' + S'_ff' / mu r) psi_N / (psi_bdp - psi_max)
class NonlinearGridCoefficient : public Coefficient
{
private:
  const GridFunction *psi;
  const PlasmaModel *model;
  int option;
public:
  NonlinearGridCoefficient(const PlasmaModel *pm, int option_) : model(pm), option(option_) { }
  void set_grid_function(const GridFunction *psi_) {psi = psi_;}
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~NonlinearGridCoefficient() { }
};

int main(int argc, char *argv[])
{
   // Parse command line options.
   const char *mesh_file = "meshes/gs_simple_mesh.msh";
   int order = 1;
   double alpha = 2.0;
   double beta = 1.0;
   double lambda = 1.0;
   double gamma = 2.0;
   double mu = 1.0;
   double r0 = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&mu, "-mu", "--magnetic_permeability", "Magnetic permeability of a vaccuum");
   args.ParseCheck();

   // save options in model
   PlasmaModel model(alpha, beta, lambda, gamma, mu, r0);
   
   // unit tests
   test();

   // Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   mesh.UniformRefinement();

   // Define a finite element space on the mesh. Here we use H1 continuous
   // high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // Extract the list of all the boundary DOFs. These will be marked as
   // Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // Define the solution x as a finite element grid function in fespace. Set
   // the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // Set up the linear form b(.) corresponding to the right-hand side.
   LinearForm b(&fespace);
   // these are the unique element attributes used by the mesh
   Array<int> attribs(mesh.attributes);
   Vector coil_current(attribs.Max());
   coil_current = 0.0;
   for (int i = 0; i < attribs.Size(); ++i) {
     int attrib = attribs[i];
     if (attrib == 2000) {
       // exterior domain
     } else if (attrib == 1000) {
       // limiter domain
     } else {
       // coil domain
       coil_current(attrib-1) = 1.0;
     }
   }
   PWConstCoefficient coil_current_pw(coil_current);
   b.AddDomainIntegrator(new DomainLFIntegrator(coil_current_pw));
   b.Assemble();

   // Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   DiffusionIntegratorCoefficient acoeff(&model);
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(acoeff));
   a.Assemble();

   // Form the linear system A X = B. This includes eliminating boundary
   // conditions, applying AMR constraints, and other transformations.
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

   // Recover the solution x as a grid function and save to file. The output
   // can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   a.RecoverFEMSolution(X, b, x);

   NonlinearGridCoefficient nlgcoeff1(&model, 1);
   nlgcoeff1.set_grid_function(&x);
   LinearForm c(&fespace);
   c.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff1));
   c.Assemble();
   
   NonlinearGridCoefficient nlgcoeff2(&model, 2);
   nlgcoeff2.set_grid_function(&x);
   BilinearForm d(&fespace);
   d.AddDomainIntegrator(new MassIntegrator(nlgcoeff2));
   d.Assemble();
   
   x.Save("sol.gf");
   mesh.Save("mesh.mesh");

   //
   GridFunction y(&fespace);
   y= 0.0;
   a.Mult(x, y);
   y.Save("y.gf");
   

   return 0;
}

double PlasmaModel::S_p_prime(double & psi_N) const
{
  return lambda * beta * pow(1.0 - pow(psi_N, alpha), gamma) / r0;
}
double PlasmaModel::S_prime_p_prime(double & psi_N) const
{
  return - alpha * gamma * lambda * beta
    * pow(1.0 - pow(psi_N, alpha), gamma - 1.0)
    * pow(psi_N, alpha - 1.0) / r0;
}
double PlasmaModel::S_ff_prime(double & psi_N) const
{
  return lambda * (1.0 - beta) * mu0 * r0 * pow(1.0 - pow(psi_N, alpha), gamma);
}
double PlasmaModel::S_prime_ff_prime(double & psi_N) const
{
  return - alpha * gamma * lambda * (1.0 - beta) * mu0 * r0
    * pow(1.0 - pow(psi_N, alpha), gamma - 1.0)
    * pow(psi_N, alpha - 1.0);
}
double normalized_psi(double & psi, double & psi_max, double & psi_bdp)
{
  return max(0.0,
             min(1.0,
                 (psi - psi_max) / (psi_bdp - psi_max)));
}


int test()
{
  double a = 1.0;
  // double b = S_p_prime(a);
  // cout << "a" << a << "b" << b << endl;
  return 1;
}

double NonlinearGridCoefficient::Eval(ElementTransformation & T,
                                      const IntegrationPoint & ip)
{
  double psi_val;
  Mesh *gf_mesh = psi->FESpace()->GetMesh();
  int Component = 1;
  if (T.mesh == gf_mesh)
    {
      psi_val = psi->GetValue(T, ip, Component);
    }
  else
    {
      cout << "problem!!!" << endl;
      // IntegrationPoint coarse_ip;
      // ElementTransformation *coarse_T = RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
      // psi_val = psi->GetValue(*coarse_T, coarse_ip, Component);
      psi_val = 1.0;
    }

   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);

   double ri(x(0));
   // todo, need to compute these
   double psi_max = 1.0;
   double psi_bdp = 0.0;
   double psi_N = normalized_psi(psi_val, psi_max, psi_bdp);

   if (option == 1) {
     return ri * (model->S_p_prime(psi_N)) + (model->S_ff_prime(psi_N)) / (model->get_mu() * ri);
   } else {
     double coeff = 1.0;
     if (option == 2) {
       coeff = 1.0 / (psi_bdp - psi_max);
     } else if (option == 3) {
       coeff = - (1 - psi_N) / (psi_bdp - psi_max);
     } else if (option == 4) {
       coeff = - psi_N / (psi_bdp - psi_max);
     }
     
     return coeff * (ri * (model->S_prime_p_prime(psi_N))
                     + (model->S_prime_ff_prime(psi_N)) / (model->get_mu() * ri));
   }
}


double DiffusionIntegratorCoefficient::Eval(ElementTransformation & T,
                                            const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double ri(x(0));

   return 1.0 / (ri * model->get_mu());
}
