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
#include "../general/dbg.hpp"
#include <cassert>
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

// Surface mesh class
template<class Surface>
class SurfaceMesh: public Mesh
{
protected:
   const int Order, Nx, Ny;
   const double Sx = 1.0;
   const double Sy = 1.0;
   const int SpaceDim = 3;
   const Element::Type Type = Element::QUADRILATERAL;
   const bool GenerateEdges = true;
   const bool SpaceFillingCurves = true;
   const bool Discontinuous = false;
public:
   SurfaceMesh(const int order,
               const int nx,
               const int ny,
               const double sx = 1.0,
               const double sy = 1.0,
               const int space_dim = 3,
               const Element::Type type = Element::QUADRILATERAL,
               const bool edges = true,
               const bool space_filling_curves = true,
               const bool discontinuous = false):
      Mesh(nx, ny, type, edges, sx, sy, space_filling_curves),
      Order(order), Nx(nx), Ny(ny), Sx(sx), Sy(sy),
      SpaceDim(space_dim), Type(type), GenerateEdges(edges),
      SpaceFillingCurves(space_filling_curves), Discontinuous(discontinuous)
   {
      EnsureNodes();
      Surface *S = static_cast<Surface*>(this);
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
   SurfaceMesh(const bool snap,
               const int order,
               const int dim,
               const int NVert,
               const int NElem,
               const int NBdrElem,
               const int space_dim):
      Mesh(dim, NVert, NElem, NBdrElem, space_dim),
      Order(order), Nx(NVert), Ny(NElem), Sx(1.0), Sy(1.0),
      SpaceDim(space_dim), Type(Element::QUADRILATERAL), GenerateEdges(true),
      SpaceFillingCurves(true), Discontinuous(false)
   {
      dbg("SurfaceMesh ex7");
      Surface *S = static_cast<Surface*>(this);
      S->Prefix();
      SetCurvature(order, Discontinuous, space_dim, Ordering::byVDIM);
   }
   void Equation() {dbg("Equation");}
   void Prefix() { dbg("Prefix"); }
   void Postfix() { dbg("Postfix");}
};

// Parametrization of a Helicoid surface
struct Helicoid: public SurfaceMesh<Helicoid>
{
   Helicoid(int order, int nx, int ny): SurfaceMesh(order, nx, ny) {}
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

// Parametrization of a Catenoid surface
struct Catenoid: public SurfaceMesh<Catenoid>
{
   Catenoid(int order, int nx, int ny): SurfaceMesh(order, nx, ny) {}
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

// Parametrization of Enneper's surface
struct Enneper: public SurfaceMesh<Enneper>
{
   Enneper(int order, int nx, int ny): SurfaceMesh(order, nx, ny) {}
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
struct Scherk: public SurfaceMesh<Scherk>
{
   Scherk(int order, int nx, int ny): SurfaceMesh(order, nx, ny) {}
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
struct Shell: public SurfaceMesh<Shell>
{
   Shell(int order, int nx, int ny): SurfaceMesh(order, nx, ny) {}
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
struct Hold: public SurfaceMesh<Hold>
{
   Hold(int order, int nx, int ny): SurfaceMesh(order, nx, ny) {}
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
struct QPeach: public SurfaceMesh<QPeach>
{
   QPeach(int order, int nx, int ny): SurfaceMesh(order, nx, ny) {}
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
class FPeach: public SurfaceMesh<FPeach>
{
private:
   H1_FECollection *Fec;
   FiniteElementSpace *NodalFes;
public:
   FPeach(H1_FECollection *fec, FiniteElementSpace *nodal_fes,
          int order, int nx, int ny):
      // dim:2, Nvert:8, Nelem:6, NBdrElem:0, space_dim:3
      SurfaceMesh(true, order, 2, 8, 6, 0, 3),
      Fec(fec), NodalFes(nodal_fes) {}
   void Prefix()
   {
      dbg("Prefix");
      const double quad_v[8][3] =
      {
         {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
         {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}
      };
      const int quad_e[6][4] =
      {
         {3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
         {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}
      };
      // Nx == Nvert
      dbg("Nvert: %d",Nx);
      for (int j = 0; j < Nx; j++)
      {
         AddVertex(quad_v[j]);
      }
      // Ny == NElem
      dbg("NElem: %d",Ny);
      for (int j = 0; j < Ny; j++)
      {
         int attribute = j + 1;
         AddQuad(quad_e[j], attribute);
      }
      FinalizeQuadMesh(1, 1, true);
      Fec = new H1_FECollection(Order, Dimension());
      NodalFes = new FiniteElementSpace(this, Fec, SpaceDimension());
      //SetCurvature(Order, Discontinuous, SpaceDim, Ordering::byNODES);
      //EnsureNodes();
      SetNodalFESpace(NodalFes);
      //UniformRefinement();
      SnapNodes();
   }
   void SnapNodes()
   {
      GridFunction &nodes = *GetNodes();
      Vector node(SpaceDimension());
      for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
      {
         for (int d = 0; d < SpaceDimension(); d++)
         {
            node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));
         }

         node /= node.Norml2();

         for (int d = 0; d < SpaceDimension(); d++)
         {
            nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
         }
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
         K(i) = tr->Weight();
      }
   }
   //glvis << "mesh\n" << *mesh << flush;
   glvis << "solution\n" << *mesh << K << flush;
   if (keys) { glvis << "keys " << keys << "\n"; }
   if (width * height > 0)
   { glvis << "window_size " << width << " " << height <<"\n" << flush; }
   if (pause) { glvis << "pause\n" << flush; }
}

// Surface solver class
class SurfaceSolver
{
protected:
   const bool pa, visualization, pause;
   const int niter, sdim, order;
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
                 const int o, Mesh *m,
                 FiniteElementSpace *f,
                 Array<int> ess_tdof_list):
      pa(p), visualization(v), pause(w), niter(n),
      sdim(m->SpaceDimension()), order(o), mesh(m), fes(f),
      a(fes), bc(ess_tdof_list), x(fes), b(fes),
      nodes(mesh->GetNodes()), solution(*nodes), one(1.0) { }
   virtual void Solve() { MFEM_ABORT("Not implemented!"); }
};

// Surface solver 'by compnents'
class ComponentSolver: public SurfaceSolver
{
public:
   ComponentSolver(const bool pa,
                   const bool vis,
                   const int niter,
                   const bool pause,
                   const int order,
                   Mesh *mesh,
                   FiniteElementSpace *fes,
                   Array<int> bc):
      SurfaceSolver(pa, vis, niter, pause, order, mesh, fes, bc) { }

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
class VectorSolver: public SurfaceSolver
{
public:
   VectorSolver(const bool pa,
                const bool vis,
                const int niter,
                const bool pause,
                const int order,
                Mesh *mesh,
                FiniteElementSpace *fes,
                Array<int> bc):
      SurfaceSolver(pa, vis, niter, pause, order, mesh, fes, bc) {}

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
   bool pa = true;
   int ref_levels = 1;
   bool components = false;
   int parametrization = 7;
   bool visualization = true;
   bool wait = false;
   const char *keys = "gAaaa";
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
                  "--no-visualization", "Enable or disable visualization.");
   args.AddOption(&components, "-c", "--components", "-no-c", "--no-components",
                  "Enable or disable the 'by component' solver");

   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Initialize GLVis server if 'visualization' is set.
   if (visualization)
   {
      glvis.open(vishost, visport);
      glvis.precision(8);
   }

   // Initialize our surface mesh from command line option.
   Mesh *mesh = nullptr;
   H1_FECollection *nodal_fec = nullptr;
   FiniteElementSpace *nodal_fes = nullptr;
   if (parametrization<0)
   {
      const int refine = 1;
      const int generate_edges = 1;
      mesh = new Mesh(mesh_file, generate_edges, refine);
   }
   else if (parametrization==0) { mesh = new Catenoid(order, nx, ny); }
   else if (parametrization==1) { mesh = new Helicoid(order, nx, ny); }
   else if (parametrization==2) { mesh = new Enneper(order, nx, ny); }
   else if (parametrization==3) { mesh = new Scherk(order, nx, ny); }
   else if (parametrization==4) { mesh = new Shell(order, nx, ny); }
   else if (parametrization==5) { mesh = new Hold(order, nx, ny); }
   else if (parametrization==6) { mesh = new QPeach(order, nx, ny); }
   else if (parametrization==7)
   {
      mesh = new FPeach(nodal_fec, nodal_fes, order, nx, ny);
   }
   else { mfem_error("Not a valid parametrization, p should be in ]-infty, 7]"); }
   const bool discontinuous = false;
   const int mdim = mesh->Dimension();
   const int sdim = mesh->SpaceDimension();
   dbg("SetCurvature");
   mesh->SetCurvature(order, discontinuous, sdim, Ordering::byNODES);

   //  Refine the mesh to increase the resolution.
   dbg("Refine");
   if (parametrization!=7)
   {
      for (int l = 0; l < ref_levels; l++) { mesh->UniformRefinement(); }
   }

   // Define a finite element space on the mesh.
   dbg("fec");
   const H1_FECollection fec(order, mdim);
   FiniteElementSpace *sfes = new FiniteElementSpace(mesh, &fec);
   dbg("vfes");
   FiniteElementSpace *vfes = new FiniteElementSpace(mesh, &fec, sdim);
   cout << "Number of true DOFs: " << vfes->GetTrueVSize() << endl;

   // Determine the list of true (i.e. conforming) essential boundary dofs.
   dbg("BC: %d", mesh->bdr_attributes.Size());
   Array<int> v_ess_tdof_list, s_ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<bool> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = true;
      sfes->GetEssentialTrueDofs(ess_bdr, s_ess_tdof_list);
      vfes->GetEssentialTrueDofs(ess_bdr, v_ess_tdof_list);
   }
   else
   {
      dbg("No bdr_attributes!");
      // GetEssentialTrueDofs:
      Array<int> ess_cdofs, ess_tdofs;
      const FiniteElementSpace *nfes = mesh->GetNodalFESpace();
      // GetEssentialVDofs:
      {
         Array<int> cdofs, dofs;
         ess_cdofs.SetSize(nfes->GetVSize());
         ess_cdofs = 0;
         for (int i = 0; i < nfes->GetNE(); i++)
         {
            nfes->GetElementDofs(i, cdofs);
            // Mark all components.
            // mark_dofs(vdofs, ess_vdofs);
            for (int i = 0; i < cdofs.Size(); i++)
            {
               int k = cdofs[i];
               //dbg(" %d",k);
               if (k < 0) { k = -1 - k; }
               Array<double> X(3);
               X = 0.0;
               mesh->GetNode(k, X);
               //dbg("(%f,%f,%f)", X[0], X[1], X[2]);
               if (fabs(X[0])<eps) { ess_cdofs[k] = -1; }
               else if (fabs(X[1])<eps) { ess_cdofs[k] = -1; }
               else { ess_cdofs[k] = 0; }
               dbg("ess_cdofs:\n"); ess_cdofs.Print();
            }
         }
      }
      dbg("v_ess_tdof_list:\n"); v_ess_tdof_list.Print();
      const SparseMatrix *R = nfes->GetConformingRestriction();
      if (!R) { ess_tdofs.MakeRef(ess_cdofs); }
      else { R->BooleanMult(ess_cdofs, ess_tdofs); }
      FiniteElementSpace::MarkerToList(ess_tdofs, v_ess_tdof_list);
   }

   dbg("visualization");
   // Send to GLVis the first mesh and set the 'keys' options.
   if (visualization) { Visualize(mesh, order, wait, keys, 800, 800); }

   // Instanciate and launch the surface solver.
   dbg("Solver");
   SurfaceSolver *s;
   if (components)
   {
      s = new ComponentSolver(pa, visualization, niter, wait,
                              order, mesh, sfes, s_ess_tdof_list);
   }
   else
   {
      s = new VectorSolver(pa, visualization, niter, wait,
                           order, mesh, vfes, v_ess_tdof_list);
   }
   s->Solve();

   // Free the used memory.
   delete sfes;
   delete vfes;
   delete mesh;
   return 0;
}
