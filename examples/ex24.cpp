#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Constant variables
const double pi = M_PI;
//const double ϵ = 1.e-12;
static socketstream glvis;
const int  visport   = 19916;
const char vishost[] = "localhost";

// Parametrizations: Scherk, Enneper, Catenoid, Helicoid
void scherk(const Vector &x, Vector &p);
void enneper(const Vector &x, Vector &p);
void catenoid(const Vector &x, Vector &p);
void helicoid(const Vector &x, Vector &p);

// Surface mesh class
class SurfaceMesh: public Mesh
{
public:
   SurfaceMesh(socketstream &glvis,
               const int order,
               void (*parametrization)(const Vector &x, Vector &p),
               const int nx = 4,
               const int ny = 4,
               const double sx = 1.0,
               const double sy = 1.0,
               const int space_dim = 3,
               const Element::Type type = Element::QUADRILATERAL,
               const bool generate_edges = true,
               const bool space_filling_curve = true,
               const bool discontinuous = false):
      Mesh(nx, ny, type, generate_edges, sx, sy, space_filling_curve)
   {
      SetCurvature(order, discontinuous, space_dim, Ordering::byNODES);
      Transform(parametrization);
      //RemoveUnusedVertices();
      //RemoveInternalBoundaries();
      glvis << "mesh\n" << *this << flush;
   }
};

int main(int argc, char *argv[])
{
   int nx = 4;
   int ny = 4;
   int order = 4;
   int niter = 8;
   int ref_levels = 2;
   int parametrization = 0;
   bool visualization = true;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&parametrization, "-p", "--parametrization",
                  "Enable or disable parametrization .");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&ref_levels, "-r", "--ref-levels", "Refinement");
   args.AddOption(&niter, "-n", "--niter", "Number of iterations");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   if (visualization)
   {
      glvis.open(vishost, visport);
      glvis.precision(8);
   }

   // Initialize our surface mesh from command line option.
   Mesh *mesh = nullptr;
   if (parametrization==0)       { mesh = new SurfaceMesh(glvis, order, catenoid, nx, ny); }
   else if (parametrization==1)  { mesh = new SurfaceMesh(glvis, order, helicoid, nx, ny); }
   else if (parametrization==2)  { mesh = new SurfaceMesh(glvis, order, enneper, nx, ny); }
   else if (parametrization==3)  { mesh = new SurfaceMesh(glvis, order, scherk, nx, ny); }
   else { mfem_error("Not a valid parametrization, which should be in [0,3]"); }
   const int sdim = mesh->SpaceDimension();
   const int mdim = mesh->Dimension();

   //  Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++) { mesh->UniformRefinement(); }

   // Define a finite element space on the mesh.
   const H1_FECollection fec(order, mdim);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh, &fec, sdim);
   cout << "Number of finite element unknowns: " << fes->GetTrueVSize() << endl;

   // Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Define the solution vector x as a finite element grid function
   // and b as the right-hand side of the FEM linear system.
   GridFunction x(fes), b(fes);
   GridFunction *nodes = mesh->GetNodes();

   // Set up the bilinear form a(.,.) on the finite element space.
   BilinearForm a(fes);
   ConstantCoefficient one(1.0);
   a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));

   if (visualization)
   {
      glvis << "mesh\n" << *mesh << flush;
      glvis << "keys gAmaa\n";
      glvis << "window_size 800 800\n";
      glvis << "pause\n" << flush;
   }

   for (int iiter=0; iiter<niter; ++iiter)
   {
      b = 0.0;
      x = *nodes; // should only copy the BC
      a.Assemble();
      Vector B, X;
      OperatorPtr A;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M(static_cast<SparseMatrix&>(*A));
      PCG(*A, M, B, X, 1, 2000, 1e-12, 0.0);
      // Recover the solution as a finite element grid function.
      a.RecoverFEMSolution(X, b, x);
      *nodes = x;
      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         glvis << "mesh\n" << *mesh << flush;
         glvis << "pause\n" << flush;
      }
      a.Update();
   }
   // Free the used memory.
   delete fes;
   delete mesh;
   return 0;
}

// Parametrization of a Catenoid surface
void catenoid(const Vector &x, Vector &p)
{
   p.SetSize(3);
   const double a = 1.0;
   // u in [0,2π] and v in [-2π/3,2π/3]
   const double u = 2.0*pi*x[0];
   const double v = 2.0*pi*(2.0*x[1]-1.0)/3.0;
   p[0] = a*cos(u)*cosh(v);
   p[1] = a*sin(u)*cosh(v);
   p[2] = a*v;
}

// Parametrization of a Helicoid surface
void helicoid(const Vector &x, Vector &p)
{
   p.SetSize(3);
   const double a = 1.0;
   // u in [0,2π] and v in [-2π/3,2π/3]
   const double u = 2.0*pi*x[0];
   const double v = 2.0*pi*(2.0*x[1]-1.0)/3.0;
   p[0] = a*cos(u)*sinh(v);
   p[1] = a*sin(u)*sinh(v);
   p[2] = a*u;
}

// Parametrization of Enneper's surface
void enneper(const Vector &x, Vector &p)
{
   p.SetSize(3);
   // r in [0,1] and t in [−π, π]
   const double r = x[0];
   const double t = pi*(2.0*x[1]-1.0);
   const double third = 1./3.;
   const double u = r*cos(t);
   const double v = r*sin(t);
   p[0] = u - third*u*u*u + u*v*v;
   p[1] = v - third*v*v*v + u*u*v;
   p[2] = u*u - v*v;
}

// Parametrization of Scherk's surface
void scherk(const Vector &x, Vector &p)
{
   p.SetSize(3);
   const double alpha = 0.49;
   // (u,v) in [-απ, +απ]
   const double u = alpha*pi*(2.0*x[0]-1.0);
   const double v = alpha*pi*(2.0*x[1]-1.0);
   p[0] = u;
   p[1] = v;
   p[2] = log(cos(u)/cos(v));
}
