//                                MFEM Example 24
//
// Compile with: make ex24
//
// Sample runs:  ex24 -m ../data/mobius-strip -r 1
//               ex24 -p 0 -r 1
//               ex24 -p 1 -r 1
//               ex24 -p 2 -r 1
//               ex24 -p 3 -r 1
//
// Device sample runs:
//               ex24 -p 0 -pa

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Constant variables
const double pi = M_PI;
const double eps = 1.e-12;
static socketstream glvis;
const int  visport   = 19916;
const char vishost[] = "localhost";

// Parametrizations: Scherk, Enneper, Catenoid, Helicoid
void scherk(const Vector &x, Vector &p);
void enneper1(const Vector &x, Vector &p);
void enneper2(const Vector &x, Vector &p);
void helicoid(const Vector &x, Vector &p);
void catenoid(const Vector &x, Vector &p);
void catenoid_postfix(const int, const int, Mesh*);
void mollusc(const Vector &x, Vector &p);
void hold(const Vector &x, Vector &p);
void peach(const Vector &x, Vector &p);

// Surface mesh class *********************************************************
class SurfaceMesh: public Mesh
{
public:
   SurfaceMesh(const int order,
               void (*parametrization)(const Vector &x, Vector &p),
               void (*postfix)(const int nx, const int ny, Mesh*),
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
      if (parametrization) { Transform(parametrization); }
      if (postfix) { postfix(nx, ny, this); }
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
      SetCurvature(order, discontinuous, space_dim, Ordering::byVDIM);
      GridFunction &nodes = *GetNodes();
      for (int i = 0; i < nodes.Size(); i++)
      { if (std::abs(nodes(i)) < eps) { nodes(i) = 0.0; } }
   }
};

// Surface solver class *******************************************************
class SurfaceSolver
{
protected:
   const bool pa, visualization, wait;
   const int niter;
   const int sdim;
   Mesh *mesh;
   Vector X, B;
   OperatorPtr A;
   FiniteElementSpace *fes;
   BilinearForm a;
   Array<int> bc;
   GridFunction x, b, *nodes, solution;
   ConstantCoefficient one;
public:
   virtual ~SurfaceSolver() {}
   SurfaceSolver(const bool p, const bool v,
                 const int n, const bool w,
                 Mesh *m, FiniteElementSpace *f,
                 Array<int> ess_tdof_list):
      pa(p), visualization(v), wait(w), niter(n),
      sdim(m->SpaceDimension()), mesh(m), fes(f),
      a(fes), bc(ess_tdof_list), x(fes), b(fes),
      nodes(mesh->GetNodes()), solution(*nodes), one(1.0) { }
   virtual void Solve() { MFEM_ABORT("Not implemented!"); }
};

// ****************************************************************************
class ComponentSolver: public SurfaceSolver
{
public:
   ComponentSolver(const bool pa,
                   const bool vis,
                   const int niter,
                   const bool wait,
                   Mesh *mesh,
                   FiniteElementSpace *fes,
                   Array<int> bc):
      SurfaceSolver(pa, vis, niter, wait, mesh, fes, bc) { }

   void Solve()
   {
      if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      for (int iiter=0; iiter<niter; ++iiter)
      {
         a.Assemble();
         solution = *nodes;
         for (int i=0; i<sdim; ++i)
         {
            x = b = 0.0;
            GetComponent(solution, x, i);
            a.FormLinearSystem(bc, x, b, A, X, B);
            if (!pa)
            {
               // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
               GSSmoother M(static_cast<SparseMatrix&>(*A));
               PCG(*A, M, B, X, 3, 2000, eps, 0.0);
            }
            else { CG(*A, B, X, 3, 2000, eps, 0.0); }
            // Recover the solution as a finite element grid function.
            a.RecoverFEMSolution(X, b, x);
            SetComponent(solution, x, i);
         }
         *nodes = solution;
         // Send the solution by socket to a GLVis server.
         if (visualization)
         {
            glvis << "mesh\n" << *mesh << flush;
            if (wait) { glvis << "pause\n" << flush; }
         }
         a.Update();
      }
   }
private:
   void SetComponent(GridFunction &phi, const GridFunction &phi_i, const int d)
   {
      // ASSUME phi IS ORDERED byNODES!
      const int ndof = fes->GetNDofs();
      for (int i = 0; i < ndof; i++)
      { phi[d*ndof + i] = phi_i[i]; }
   }

   void GetComponent(const GridFunction &phi, GridFunction &phi_i, const int d)
   {
      // ASSUME phi IS ORDERED byNODES!
      const int ndof = fes->GetNDofs();
      for (int i = 0; i < ndof; i++)
      { phi_i[i] = phi[d*ndof + i]; }
   }
};

// ****************************************************************************
class VectorSolver: public SurfaceSolver
{
public:
   VectorSolver(const bool pa,
                const bool vis,
                const int niter,
                const bool wait,
                Mesh *mesh,
                FiniteElementSpace *fes,
                Array<int> bc):
      SurfaceSolver(pa, vis, niter, wait, mesh, fes, bc) {}

   void Solve()
   {
      if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
      for (int iiter=0; iiter<niter; ++iiter)
      {
         a.Assemble();
         b = 0.0;
         x = *nodes; // should only copy the BC
         Vector B, X;
         OperatorPtr A;
         a.FormLinearSystem(bc, x, b, A, X, B);
         if (!pa)
         {
            // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
            GSSmoother M(static_cast<SparseMatrix&>(*A));
            PCG(*A, M, B, X, 3, 2000, eps, 0.0);
         }
         else { CG(*A, B, X, 3, 2000, eps, 0.0); }
         // Recover the solution as a finite element grid function.
         a.RecoverFEMSolution(X, b, x);
         *nodes = x;
         // Send the solution by socket to a GLVis server.
         if (visualization)
         {
            glvis << "mesh\n" << *mesh << flush;
            if (wait) { glvis << "pause\n" << flush; }
         }
         a.Update();
      }
   }
};

// ****************************************************************************
int main(int argc, char *argv[])
{
   int nx = 4;
   int ny = 4;
   int order = 4;
   int niter = 4;
   bool pa = false;
   int ref_levels = 1;
   bool wait = false;
   bool components = true;
   int parametrization = 0;
   bool visualization = true;
   const char *keys = "gAaa";
   const char *device_config = "cpu";
   const char *mesh_file = "../data/mobius-strip.mesh";

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&parametrization, "-p", "--parametrization",
                  "Enable or disable parametrization .");
   args.AddOption(&wait, "-w", "--wait", "-no-w", "--no-wait",
                  "Enable or disable a GLVis pause.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&ref_levels, "-r", "--ref-levels", "Refinement");
   args.AddOption(&niter, "-n", "--niter", "Number of iterations");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&keys, "-k", "--keys", "GLVis configuration keys.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&components, "-c", "--components", "-no-c",
                  "--no-components", "Enable or disable the 'by component' solver");

   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);
   MFEM_VERIFY(components || !pa, "Vector solver does not support PA yet!");

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   if (visualization)
   {
      glvis.open(vishost, visport);
      glvis.precision(8);
   }

   // Initialize our surface mesh from command line option.
   Mesh *mesh = nullptr;
   if (parametrization<0)
   {
      const int refine = 1;
      const int generate_edges = 1;
      mesh = new Mesh(mesh_file, generate_edges, refine);
   }
   else if (parametrization==0)
   { mesh = new SurfaceMesh(order, catenoid, catenoid_postfix, nx, ny); }
   else if (parametrization==1)
   { mesh = new SurfaceMesh(order, helicoid, nullptr, nx, ny); }
   else if (parametrization==2)
   { mesh = new SurfaceMesh(order, enneper2, nullptr, nx, ny); }
   else if (parametrization==3)
   { mesh = new SurfaceMesh(order, scherk, nullptr, nx, ny); }
   else if (parametrization==4)
   { mesh = new SurfaceMesh(order, mollusc, nullptr, nx, ny); }
   else if (parametrization==5)
   { mesh = new SurfaceMesh(order, hold, nullptr, nx, ny); }
   else if (parametrization==6)
   { mesh = new SurfaceMesh(order, peach, nullptr, nx, ny); }
   else { mfem_error("Not a valid parametrization, p should be in ]-infty, 6]"); }
   const bool discontinuous = false;
   const int mdim = mesh->Dimension();
   const int sdim = mesh->SpaceDimension();
   cout << "mesh dimension: " << mdim << " mesh space dimension: " << sdim << endl;
   mesh->SetCurvature(order, discontinuous, sdim, Ordering::byNODES);

   //  Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++) { mesh->UniformRefinement(); }

   // Define a finite element space on the mesh.
   const H1_FECollection fec(order, mdim);
   FiniteElementSpace *sfes = new FiniteElementSpace(mesh, &fec);
   FiniteElementSpace *vfes = new FiniteElementSpace(mesh, &fec, sdim);
   cout << "Number of finite element unknowns: " << vfes->GetTrueVSize() << endl;

   // Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> v_ess_tdof_list, s_ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      sfes->GetEssentialTrueDofs(ess_bdr, s_ess_tdof_list);
      vfes->GetEssentialTrueDofs(ess_bdr, v_ess_tdof_list);
   }

   if (visualization)
   {
      //glvis << "solution\n" << *mesh << s <<flush;
      glvis << "mesh\n" << *mesh << flush;
      glvis << "keys " << keys << "\n";
      glvis << "window_size 800 800\n";
      if (wait) { glvis << "pause\n" << flush; }
   }

   SurfaceSolver *solver;
   if (components)
   {
      solver = new ComponentSolver(pa, visualization, niter, wait,
                                   mesh, sfes, s_ess_tdof_list);
   }
   else
   {
      solver = new VectorSolver(pa, visualization, niter, wait,
                                mesh, vfes, v_ess_tdof_list);
   }
   solver->Solve();

   // Free the used memory.
   delete sfes;
   delete vfes;
   delete mesh;
   return 0;
}

// Parametrization of a Catenoid surface
void catenoid(const Vector &x, Vector &p)
{
   p.SetSize(3);
   // u in [0,2π] and v in [-2π/3,2π/3]
   const double u = 2.0*pi*x[0];
   const double v = 2.0*pi*(2.0*x[1]-1.0)/3.0;
   p[0] = cos(u)*cosh(v);
   p[1] = sin(u)*cosh(v);
   p[2] = v;
}

// Postfix of the Catenoid surface
void catenoid_postfix(const int nx, const int ny, Mesh *mesh)
{
   Array<int> v2v(mesh->GetNV());
   for (int i = 0; i < v2v.Size(); i++) { v2v[i] = i; }
   // identify vertices on vertical lines
   for (int j = 0; j <= ny; j++)
   {
      const int v_old = nx + j * (nx + 1);
      const int v_new =      j * (nx + 1);
      v2v[v_old] = v_new;
   }
   // renumber elements
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Element *el = mesh->GetElement(i);
      int *v = el->GetVertices();
      const int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      { v[j] = v2v[v[j]]; }
   }
   // renumber boundary elements
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      Element *el = mesh->GetBdrElement(i);
      int *v = el->GetVertices();
      const int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      { v[j] = v2v[v[j]]; }
   }
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
void enneper1(const Vector &x, Vector &p)
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
void enneper2(const Vector &x, Vector &p)
{
   p.SetSize(3);
   // (u,v) in [-2, +2]
   const double u = 2.0*(2.0*x[0]-1.0);
   const double v = 2.0*(2.0*x[1]-1.0);
   p[0] = +u - u*u*u/3.0 + u*v*v;
   p[1] = -v - u*u*v + v*v*v/3.0;
   p[2] =  u*u - v*v;
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

// Mollusc shell model
void mollusc(const Vector &x, Vector &p)
{
   p.SetSize(3);
   // u in [0,2π] and v in [-15, 6]
   const double u = 2.0*pi*x[0];
   const double v = 21.0*x[1]-15.0;
   p[0] = +1.0*pow(1.16,v)*cos(v)*(1.0+cos(u));
   p[1] = -1.0*pow(1.16,v)*sin(v)*(1.0+cos(u));
   p[2] = -2.0*pow(1.16,v)*(1.0+sin(u));
}

// **********************************
void hold(const Vector &x, Vector &p)
{
   p.SetSize(3);
   // u in [0,2π] and v in [0,1]
   const double u = 2.0*pi*x[0];
   const double v = x[1];
   p[0] = cos(u)*(1.0 + 0.3*sin(5.*u + pi*v));
   p[1] = sin(u)*(1.0 + 0.3*sin(5.*u + pi*v));
   p[2] = v;
}

// ***********************************
void peach(const Vector &X, Vector &p)
{
   p = X;
   const double x = 2.0*X[0]-1.0;
   const double y = X[1];
   const double h = 1.0;
   const double r = sqrt(x*x + y*y);
   const double t = (x==0.0) ? pi/2.0 :
                    (y==0.0 && x>0.0) ? 0. :
                    (y==0.0 && x<0.0) ? pi : acos(x/r);
   const double sqrtx = sqrt(1.0 + x*x);
   const double sqrty = sqrt(1.0 + y*y);
   const bool yaxis = pi/4.0<t && t < 3.0*pi/4.0;
   const double R = yaxis?sqrtx:sqrty;
   const double gamma = r/R;
   p[0] = gamma * cos(t);
   p[1] = gamma * sin(t);
   p[2] = h*(1.0 - gamma);
}
