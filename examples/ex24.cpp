//                                MFEM Example 24
//
// Compile with: make ex24
//
// Sample runs:  ex24
//               ex24 -c
//               ex24 -p 0
//               ex24 -p 0 -c
//               ex24 -p 1
//               ex24 -p 1 -c
//               ex24 -p 2
//               ex24 -p 2 -c
//               ex24 -p 3
//               ex24 -p 3 -c
//               ex24 -p 4
//               ex24 -p 4 -c
//               ex24 -p 5
//               ex24 -p 5 -c
//               ex24 -p 6
//               ex24 -p 6 -c
//
// Device sample runs:
//               ex24 -pa
//               ex24 -pa -c
//               ex24 -p 0 -pa
//               ex24 -p 0 -pa -c
//               ex24 -p 1 -pa
//               ex24 -p 1 -pa -c
//               ex24 -p 2 -pa
//               ex24 -p 2 -pa -c
//               ex24 -p 3 -pa
//               ex24 -p 3 -pa -c
//               ex24 -p 4 -pa
//               ex24 -p 4 -pa -c
//               ex24 -p 5 -pa
//               ex24 -p 5 -pa -c
//               ex24 -p 6 -pa
//               ex24 -p 6 -pa -c

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

// Forward declaration
void SnapNodes(Mesh &mesh);

// Surface mesh class
template<class Type>
class Surface: public Mesh
{
protected:
   const int Order, Nx, Ny;
   const double Sx = 1.0;
   const double Sy = 1.0;
   const int SpaceDim = 3;
   const Element::Type ElType = Element::QUADRILATERAL;
   const bool GenerateEdges = true;
   const bool SpaceFillingCurves = true;
   const bool Discontinuous = false;
   Type *S;
public:
   Surface(const int order,
           const int nx = 4,
           const int ny = 4,
           const double sx = 1.0,
           const double sy = 1.0,
           const int space_dim = 3,
           const Element::Type type = Element::QUADRILATERAL,
           const bool edges = true,
           const bool space_filling_curves = true,
           const bool discontinuous = false):
      Mesh(nx, ny, type, edges, sx, sy, space_filling_curves),
      Order(order), Nx(nx), Ny(ny), Sx(sx), Sy(sy),
      SpaceDim(space_dim), ElType(type), GenerateEdges(edges),
      SpaceFillingCurves(space_filling_curves), Discontinuous(discontinuous),
      S(static_cast<Type*>(this))
   {
      EnsureNodes();
      S->Prefix();
      S->Equation();
      S->Postfix();
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
      SetCurvature(order, discontinuous, space_dim, Ordering::byVDIM);
      GridFunction &nodes = *GetNodes();
      for (int i = 0; i < nodes.Size(); i++)
      { if (std::abs(nodes(i)) < eps) { nodes(i) = 0.0; } }
   }
   void Equation() { Transform(Type::Parametrization); }
   void Prefix()
   {
      SetCurvature(Order, Discontinuous, SpaceDim, Ordering::byNODES);
   }
   void Postfix() { }
};

// Helicoid surface
struct Helicoid: public Surface<Helicoid>
{
   Helicoid(int order, int nx, int ny): Surface(order, nx, ny) {}
   static void Parametrization(const Vector &x, Vector &p)
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
};

// Catenoid surface
struct Catenoid: public Surface<Catenoid>
{
   Catenoid(int order, int nx, int ny): Surface(order, nx, ny) {}
   static void Parametrization(const Vector &x, Vector &p)
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
   void Postfix()
   {
      Array<int> v2v(GetNV());
      for (int i = 0; i < v2v.Size(); i++) { v2v[i] = i; }
      // identify vertices on vertical lines
      for (int j = 0; j <= Ny; j++)
      {
         const int v_old = Nx + j * (Nx + 1);
         const int v_new =      j * (Nx + 1);
         v2v[v_old] = v_new;
      }
      // renumber elements
      for (int i = 0; i < GetNE(); i++)
      {
         Element *el = GetElement(i);
         int *v = el->GetVertices();
         const int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         { v[j] = v2v[v[j]]; }
      }
      // renumber boundary elements
      for (int i = 0; i < GetNBE(); i++)
      {
         Element *el = GetBdrElement(i);
         int *v = el->GetVertices();
         const int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         { v[j] = v2v[v[j]]; }
      }
   }
};

// Enneper's surface
struct Enneper: public Surface<Enneper>
{
   Enneper(int order, int nx, int ny): Surface(order, nx, ny) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // (u,v) in [-2, +2]
      const double u = 2.0*(2.0*x[0]-1.0);
      const double v = 2.0*(2.0*x[1]-1.0);
      p[0] = +u - u*u*u/3.0 + u*v*v;
      p[1] = -v - u*u*v + v*v*v/3.0;
      p[2] =  u*u - v*v;
   }
};

// Parametrization of Scherk's surface
struct Scherk: public Surface<Scherk>
{
   Scherk(int order, int nx, int ny): Surface(order, nx, ny) {}
   static void Parametrization(const Vector &x, Vector &p)
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
};

// Shell surface model
struct Shell: public Surface<Shell>
{
   Shell(int order, int nx, int ny): Surface(order, nx, ny) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [-15, 6]
      const double u = 2.0*pi*x[0];
      const double v = 21.0*x[1]-15.0;
      p[0] = +1.0*pow(1.16,v)*cos(v)*(1.0+cos(u));
      p[1] = -1.0*pow(1.16,v)*sin(v)*(1.0+cos(u));
      p[2] = -2.0*pow(1.16,v)*(1.0+sin(u));
   }
};

// Hold surface
struct Hold: public Surface<Hold>
{
   Hold(int order, int nx, int ny): Surface(order, nx, ny) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [0,1]
      const double u = 2.0*pi*x[0];
      const double v = x[1];
      p[0] = cos(u)*(1.0 + 0.3*sin(5.*u + pi*v));
      p[1] = sin(u)*(1.0 + 0.3*sin(5.*u + pi*v));
      p[2] = v;
   }
};

// 1/4th Peach street model
// Could set BC: ess_bdr[0] = false; with attribute set to '1' from postfix
struct QPeach: public Surface<QPeach>
{
   QPeach(int order, int nx, int ny): Surface(order, nx, ny) {}
   static void Parametrization(const Vector &X, Vector &p)
   {
      p = X;
      const double x = 2.0*X[0]-1.0;
      const double y = X[1];
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
      p[2] = fabs(t)<eps?1.0:0.0;
      p[2] = 1.0 - gamma;
   }
   void Prefix()
   {
      SetCurvature(1, Discontinuous, SpaceDim, Ordering::byNODES);
   }
   void Postfix()
   {
      PrintCharacteristics();
      for (int i = 0; i < GetNBE(); i++)
      {
         Element *el = GetBdrElement(i);
         const int fn = GetBdrElementEdgeIndex(i);
         MFEM_VERIFY(!FaceIsTrueInterior(fn),"");
         Array<int> vertices;
         GetFaceVertices(fn, vertices);
         const GridFunction *nodes = GetNodes();
         Vector nval;
         double R[2], X[2][3];
         for (int v = 0; v < 2; v++)
         {
            R[v] = 0.0;
            const int iv = vertices[v];
            for (int d = 0; d < 3; d++)
            {
               nodes->GetNodalValues(nval, d+1);
               const double x = X[v][d] = nval[iv];
               if (d < 2) { R[v] += x*x; }
            }
         }
         if (fabs(X[0][1])<=eps &&
             fabs(X[1][1])<=eps && (R[0]>0.1 || R[1]>0.1))
         { el->SetAttribute(1); }
         else { el->SetAttribute(2); }
         //ofstream mesh_ofs("out.mesh");
         //mesh_ofs.precision(8);
         //Print(mesh_ofs);
      }
   }
};

// Full Peach street model
struct FPeach: public Surface<FPeach>
{
   FPeach(int order, int nx, int ny): Surface(order, nx, ny) {}
   static void Parametrization(const Vector &X, Vector &p)
   {
      p = X;
      const double x = 2.0*X[0]-1.0;
      const double y = X[1];
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
      p[2] = fabs(t)<eps?1.0:0.0;
      p[2] = 1.0 - gamma;
   }
   void Prefix()
   {
      SetCurvature(1, Discontinuous, SpaceDim, Ordering::byNODES);
   }
   void Postfix()
   {
      PrintCharacteristics();
      for (int i = 0; i < GetNBE(); i++)
      {
         Element *el = GetBdrElement(i);
         const int fn = GetBdrElementEdgeIndex(i);
         MFEM_VERIFY(!FaceIsTrueInterior(fn),"");
         Array<int> vertices;
         GetFaceVertices(fn, vertices);
         const GridFunction *nodes = GetNodes();
         Vector nval;
         double R[2], X[2][3];
         for (int v = 0; v < 2; v++)
         {
            R[v] = 0.0;
            const int iv = vertices[v];
            for (int d = 0; d < 3; d++)
            {
               nodes->GetNodalValues(nval, d+1);
               const double x = X[v][d] = nval[iv];
               if (d < 2) { R[v] += x*x; }
            }
         }
         if (fabs(X[0][1])<=eps &&
             fabs(X[1][1])<=eps && (R[0]>0.1 || R[1]>0.1))
         { el->SetAttribute(1); }
         else { el->SetAttribute(2); }
         //ofstream mesh_ofs("out.mesh");
         //mesh_ofs.precision(8);
         //Print(mesh_ofs);
      }
   }
};

// Visualize some solution on the given mesh
static void Visualize(Mesh *mesh, const int order, const bool pause,
                      const char *keys = NULL,
                      const int width = 0, const int height = 0)
{
   const H1_FECollection fec(2, 2);
   FiniteElementSpace *sfes = new FiniteElementSpace(mesh, &fec);
   GridFunction K(sfes);
   const int NE = mesh->GetNE();
   const Element::Type type = Element::QUADRILATERAL;
   const IntegrationRule *ir = &IntRules.Get(type, order);
   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *tr = mesh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         tr->SetIntPoint(&ir->IntPoint(j));
         K(i) = tr->Jacobian().Weight();
      }
   }
   //glvis << "mesh\n" << *mesh << flush;
   //glvis.precision(8);
   glvis << "solution\n" << *mesh << K << flush;
   if (keys) { glvis << "keys " << keys << "\n"; }
   if (width * height > 0)
   { glvis << "window_size " << width << " " << height <<"\n" << flush; }
   if (pause) { glvis << "pause\n" << flush; }
}

// Surface solver class
template<class Type>
class Solver
{
protected:
   bool pa, visualization, pause;
   int niter, sdim, order;
   Mesh *mesh;
   Vector X, B;
   OperatorPtr A;
   FiniteElementSpace *fes;
   BilinearForm a;
   Array<int> bc;
   GridFunction x, b, *nodes, solution;
   ConstantCoefficient one;
   Type *solver;
public:
   Solver(const bool p, const bool v,
          const int n, const bool w,
          const int o, Mesh *m,
          FiniteElementSpace *f,
          Array<int> dbc):
      pa(p), visualization(v), pause(w), niter(n),
      sdim(m->SpaceDimension()), order(o), mesh(m), fes(f),
      a(fes), bc(dbc), x(fes), b(fes),
      nodes(mesh->GetNodes()), solution(*nodes), one(1.0),
      solver(static_cast<Type*>(this)) { Solve(); }
   void Solve() { solver->Solve(); }
};

// Surface solver 'by compnents'
class ByComponent: public Solver<ByComponent>
{
public:
   ByComponent(const bool pa,  const bool vis,
               const int niter, const bool wait,
               const int order, Mesh *mesh,
               FiniteElementSpace *fes,
               Array<int> dbc):
      Solver(pa, vis, niter, wait, order, mesh, fes, dbc) { }
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
            b = 0.0;
            GetComponent(*nodes, x, i);
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
         if (visualization) { Visualize(mesh, order, pause); }
         a.Update();
      }
   }
private:
   void SetComponent(GridFunction &X, const GridFunction &Xi, const int d)
   {
      const int ndof = fes->GetNDofs();
      for (int i = 0; i < ndof; i++)
      { X[d*ndof + i] = Xi[i]; }
   }

   void GetComponent(const GridFunction &X, GridFunction &Xi, const int d)
   {
      const int ndof = fes->GetNDofs();
      for (int i = 0; i < ndof; i++)
      { Xi[i] = X[d*ndof + i]; }
   }
};

// Surface solver 'by vector'
class ByVector: public Solver<ByVector>
{
public:
   ByVector(const bool pa, const bool vis,
            const int niter, const bool pause,
            const int order, Mesh *mesh,
            FiniteElementSpace *fes,
            Array<int> bc):
      Solver(pa, vis, niter, pause, order, mesh, fes, bc) {}
   void Solve()
   {
      if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
      for (int iiter=0; iiter<niter; ++iiter)
      {
         a.Assemble();
         b = 0.0;
         x = *nodes; // should only copy the BC
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
         if (visualization) { Visualize(mesh, order, pause); }
         a.Update();
      }
   }
};

int main(int argc, char *argv[])
{
   int nx = 4;
   int ny = 4;
   int order = 2;
   int niter = 4;
   int surface = -1;
   int ref_levels = 2;
   bool pa = true;
   bool vis = true;
   bool amr = false;
   bool byc = false;
   bool wait = false;
   const char *keys = "gAaaa";
   const char *device_config = "cpu";
   const char *mesh_file = "../data/mobius-strip.mesh";

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&surface, "-s", "--surface",
                  "Choice of the surface.");
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
   args.AddOption(&amr, "-amr", "--adaptive-mesh-refinement", "-no-amr",
                  "--no-adaptive-mesh-refinement", "Enable AMR.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&keys, "-k", "--keys", "GLVis configuration keys.");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable visualization.");
   args.AddOption(&byc, "-c", "--components", "-no-c", "--no-components",
                  "Enable or disable the 'by component' solver");

   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);
   MFEM_VERIFY(!amr, "WIP");

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Initialize GLVis server if 'visualization' is set.
   if (vis) { glvis.open(vishost, visport); }

   // Initialize our surface mesh from command line option.
   Mesh *mesh = nullptr;
   if (surface < 0) { mesh = new Mesh(mesh_file, true); }
   else if (surface == 0) { mesh = new Catenoid(order, nx, ny); }
   else if (surface == 1) { mesh = new Helicoid(order, nx, ny); }
   else if (surface == 2) { mesh = new Enneper(order, nx, ny); }
   else if (surface == 3) { mesh = new Scherk(order, nx, ny); }
   else if (surface == 4) { mesh = new Shell(order, nx, ny); }
   else if (surface == 5) { mesh = new Hold(order, nx, ny); }
   else if (surface == 6) { mesh = new QPeach(order, nx, ny); }
   else if (surface == 7) { mesh = new FPeach(order, nx, ny); }
   else { mfem_error("Not a valid surface, p should be in ]-infty, 7]"); }
   const bool discontinuous = false;
   const int mdim = mesh->Dimension();
   const int sdim = mesh->SpaceDimension();
   const int vdim = byc ? 1 : sdim;
   mesh->SetCurvature(order, discontinuous, sdim, Ordering::byNODES);

   //  Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++) { mesh->UniformRefinement(); }

   // Adaptive mesh refinement
   if (amr)
   {
      for (int l = 0; l < 1; l++)
      {
         mesh->RandomRefinement(0.5); // probability
      }
      SnapNodes(*mesh);
   }
   // Define a finite element space on the mesh.
   const H1_FECollection fec(order, mdim);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh, &fec, vdim);
   cout << "Number of true DOFs: " << fes->GetTrueVSize() << endl;

   // Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> dbc;
   if (mesh->bdr_attributes.Size())
   {
      Array<bool> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = true;
      fes->GetEssentialTrueDofs(ess_bdr, dbc);
   }

   // Send to GLVis the first mesh and set the 'keys' options.
   if (vis) { Visualize(mesh, order, wait, keys, 800, 800); }

   // Instanciate and launch the surface solver.
   if (byc) { ByComponent Solve(pa, vis, niter, wait, order, mesh, fes, dbc); }
   else        { ByVector Solve(pa, vis, niter, wait, order, mesh, fes, dbc); }

   // Free the used memory.
   delete fes;
   delete mesh;
   return 0;
}

void SnapNodes(Mesh &mesh)
{
   GridFunction &nodes = *mesh.GetNodes();
   Vector node(mesh.SpaceDimension());
   for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
   {
      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));
      }

      node /= node.Norml2();

      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
      }
   }
}
