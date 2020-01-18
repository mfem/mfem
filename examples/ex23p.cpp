//                                MFEM Example 23
//
// Compile with: make ex23
//
// Sample runs:  ex23 -m ../data/square-disc.mesh
//
// Device sample runs:
//               ex23 -pa -d cuda
//
// Description:  This example code
//               s=0: Catenoid
//               s=1: Helicoid
//               s=2: Enneper
//               s=3: Scherk
//               s=4: Shell
//               s=5: Hold
//               s=6: QPeach
//               s=7: FPeach
//               s=8: SlottedSphere


#include "mfem.hpp"
#include "../general/forall.hpp"
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
template<typename T = nullptr_t>
class Surface: public Mesh
{
protected:
   T *S;
   Array<int> &bc;
   int order, nx, ny, nr, sdim;
   ParMesh *pmesh = nullptr;
   H1_FECollection *fec = nullptr;
   ParFiniteElementSpace *pfes = nullptr;
public:

   // Reading from mesh file
   Surface(Array<int> &b, int order, const char *file, int nr, int sdim):
      Mesh(file, true), S(static_cast<T*>(this)),
      bc(b), order(order), nr(nr), sdim(sdim)
   {
      EnsureNodes();
      S->Postfix();
      S->Refine();
      GenFESpace();
      S->BoundaryConditions();
   }

   // Generate Quad surface mesh
   Surface(Array<int> &b, int order, int nx, int ny, int nr, int sdim):
      Mesh(nx, ny, Element::QUADRILATERAL, true, 1.0, 1.0, false),
      S(static_cast<T*>(this)),
      bc(b), order(order), nx(nx), ny(ny), nr(nr), sdim(sdim)
   {
      EnsureNodes();
      S->Prefix();
      S->Generate();
      S->Postfix();
      S->Refine();
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
      SetCurvature(order, false, 3, Ordering::byVDIM);
      GridFunction &nodes = *GetNodes();
      for (int i = 0; i < nodes.Size(); i++)
      { if (std::abs(nodes(i)) < eps) { nodes(i) = 0.0; } }
      SetCurvature(order, false, 3, Ordering::byNODES);
      GenFESpace();
      S->BoundaryConditions();
   }

   // Generated Cube surface mesh
   Surface(Array<int> &b, int order, int nr,
           int NVert, int NElem, int NBdrElem, int sdim):
      Mesh(2, NVert, NElem, NBdrElem, 3), S(static_cast<T*>(this)),
      bc(b), order(order), nx(NVert), ny(NElem), nr(nr), sdim(sdim)
   {
      S->Generate();
      S->Postfix();
      S->Refine();
      S->GenFESpace();
      S->BoundaryConditions();
   }

   ~Surface() { delete fec; delete pfes; }

   void Prefix() { SetCurvature(order, false, 3, Ordering::byNODES); }

   void Generate() { Transform(T::Parametrization); }

   void Postfix() { SetCurvature(order, false, 3, Ordering::byNODES); }

   void Refine() { for (int l = 0; l < nr; l++) { UniformRefinement(); } }

   void  BoundaryConditions()
   {
      if (bdr_attributes.Size())
      {
         Array<int> ess_bdr(bdr_attributes.Max());
         ess_bdr = 1;
         pfes->GetEssentialTrueDofs(ess_bdr, bc);
      }
   }

   void GenFESpace()
   {
      fec = new H1_FECollection(order, 2);
      pmesh = new ParMesh(MPI_COMM_WORLD, *this);
      pfes = new ParFiniteElementSpace(pmesh, fec, sdim);
   }

   ParMesh *Pmesh() const { return pmesh; }

   ParFiniteElementSpace *Pfes() const { return pfes; }
};

// Mesh file surface
struct MeshFromFile: public Surface<MeshFromFile>
{
   MeshFromFile(Array<int> &bc, int order, const char *file, int nr, int sdim):
      Surface(bc, order, file, nr, sdim) {}
};

// Catenoid surface
struct Catenoid: public Surface<Catenoid>
{
   Catenoid(Array<int> &bc, int order, int nx, int ny, int nr, int sdim):
      Surface(bc, order, nx, ny, nr, sdim) {}
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
      SetCurvature(order, false, 3, Ordering::byNODES);
      Array<int> v2v(GetNV());
      for (int i = 0; i < v2v.Size(); i++) { v2v[i] = i; }
      // identify vertices on vertical lines
      for (int j = 0; j <= ny; j++)
      {
         const int v_old = nx + j * (nx + 1);
         const int v_new =      j * (nx + 1);
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

// Helicoid surface
struct Helicoid: public Surface<Helicoid>
{
   Helicoid(Array<int> &bc, int order, int nx, int ny, int nr, int sdim):
      Surface(bc, order, nx, ny, nr, sdim) {}
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

// Enneper's surface
struct Enneper: public Surface<Enneper>
{
   Enneper(Array<int> &bc, int order, int nx, int ny, int nr, int sdim):
      Surface(bc, order, nx, ny, nr, sdim) {}
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
   Scherk(Array<int> &bc, int order, int nx, int ny, int nr, int sdim):
      Surface(bc, order, nx, ny, nr, sdim) {}
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
   Shell(Array<int> &bc, int order, int nx, int ny, int nr, int sdim):
      Surface(bc, order, nx, ny, nr, sdim) {}
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
   Hold(Array<int> &bc, int order, int nx, int ny, int nr, int sdim):
      Surface(bc, order, nx, ny, nr, sdim) {}
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
struct QPeach: public Surface<QPeach>
{
   QPeach(Array<int> &bc, int order, int nx, int ny, int nr, int sdim):
      Surface(bc, order, nx, ny, nr, sdim) {}
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
      p[2] = 1.0 - gamma;
   }
   void Prefix() { SetCurvature(1, false, 3, Ordering::byNODES); }
   void Postfix()
   {
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
         if (fabs(X[0][1])<=eps && fabs(X[1][1])<=eps &&
             (R[0]>0.1 || R[1]>0.1))
         { el->SetAttribute(1); }
         else { el->SetAttribute(2); }
      }
   }
};

// Full Peach street model
struct FPeach: public Surface<FPeach>
{
   FPeach(Array<int> &bc, int order, int nr, int sdim):
      Surface<FPeach>(bc, order, nr, 8, 6, 6, sdim) { }
   void Generate()
   {
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
      for (int j = 0; j < nx; j++) { AddVertex(quad_v[j]); }
      for (int j = 0; j < ny; j++) { AddQuad(quad_e[j], j+1); }
      for (int j = 0; j < ny; j++) { AddBdrQuad(quad_e[j], j+1); }
      FinalizeQuadMesh(1, 1, true);
      UniformRefinement();
      SetCurvature(order, false, 3, Ordering::byNODES);
      // Snap the nodes to the unit sphere
      const int mesh_sdim = SpaceDimension();
      GridFunction &nodes = *GetNodes();
      Vector node(mesh_sdim);
      for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
      {
         for (int d = 0; d < mesh_sdim; d++)
         { node(d) = nodes(nodes.FESpace()->DofToVDof(i, d)); }
         node /= node.Norml2();
         for (int d = 0; d < mesh_sdim; d++)
         { nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d); }
      }
   }

   void BoundaryConditions()
   {
      double X[3];
      Array<int> dofs;
      Array<int> ess_cdofs, ess_tdofs;
      ess_cdofs.SetSize(pfes->GetVSize());
      ess_cdofs = 0;
      for (int e = 0; e < pfes->GetNE(); e++)
      {
         pfes->GetElementDofs(e, dofs);
         for (int c = 0; c < dofs.Size(); c++)
         {
            int k = dofs[c];
            if (k < 0) { k = -1 - k; }
            GetNode(k, X);
            const bool halfX = fabs(X[0]) < eps && X[1] <= 0;
            const bool halfY = fabs(X[2]) < eps && X[1] >= 0;
            const bool is_on_bc = halfX || halfY;
            for (int d = 0; d < sdim; d++)
            { ess_cdofs[pfes->DofToVDof(k, d)] = is_on_bc; }
         }
      }
      const SparseMatrix *R = pfes->GetConformingRestriction();
      if (!R) { ess_tdofs.MakeRef(ess_cdofs); }
      else { R->BooleanMult(ess_cdofs, ess_tdofs); }
      ParFiniteElementSpace::MarkerToList(ess_tdofs, bc);
   }
};

// Full Peach street model
struct SlottedSphere: public Surface<SlottedSphere> //Mesh
{
   SlottedSphere(Array<int> &bc, int order, int nr, int sdim):
      Surface<SlottedSphere>(bc, order, nr, 0, 0, 0, sdim) { }
   void Generate()
   {
      const double delta = 0.15;
      static const int nv1d = 4;
      static const int nv = nv1d*nv1d*nv1d;
      static const int nel_per_face = (nv1d-1)*(nv1d-1);
      static const int nel_delete = 7*2;
      static const int nel_total = nel_per_face*6;
      static const int nel = nel_total - nel_delete;

      InitMesh(2, 3, nv, nel, 0);

      double vert1d[nv1d] = {-1.0, -delta, delta, 1.0};
      double quad_v[nv][3];

      for (int iv=0; iv<nv; ++iv)
      {
         int ix = iv % nv1d;
         int iy = (iv / nv1d) % nv1d;
         int iz = (iv / nv1d) / nv1d;

         quad_v[iv][0] = vert1d[ix];
         quad_v[iv][1] = vert1d[iy];
         quad_v[iv][2] = vert1d[iz];
      }

      int quad_e[nel_total][4];

      for (int ix=0; ix<nv1d-1; ++ix)
      {
         for (int iy=0; iy<nv1d-1; ++iy)
         {
            int el_offset = ix + iy*(nv1d-1);
            // x = 0
            quad_e[0*nel_per_face + el_offset][0] = nv1d*ix + nv1d*nv1d*iy;
            quad_e[0*nel_per_face + el_offset][1] = nv1d*(ix+1) + nv1d*nv1d*iy;
            quad_e[0*nel_per_face + el_offset][2] = nv1d*(ix+1) + nv1d*nv1d*(iy+1);
            quad_e[0*nel_per_face + el_offset][3] = nv1d*ix + nv1d*nv1d*(iy+1);
            // x = 1
            int x_off = nv1d-1;
            quad_e[1*nel_per_face + el_offset][3] = x_off + nv1d*ix + nv1d*nv1d*iy;
            quad_e[1*nel_per_face + el_offset][2] = x_off + nv1d*(ix+1) + nv1d*nv1d*iy;
            quad_e[1*nel_per_face + el_offset][1] = x_off + nv1d*(ix+1) + nv1d*nv1d*(iy+1);
            quad_e[1*nel_per_face + el_offset][0] = x_off + nv1d*ix + nv1d*nv1d*(iy+1);
            // y = 0
            quad_e[2*nel_per_face + el_offset][0] = nv1d*nv1d*iy + ix;
            quad_e[2*nel_per_face + el_offset][1] = nv1d*nv1d*iy + ix + 1;
            quad_e[2*nel_per_face + el_offset][2] = nv1d*nv1d*(iy+1) + ix + 1;
            quad_e[2*nel_per_face + el_offset][3] = nv1d*nv1d*(iy+1) + ix;
            // y = 1
            int y_off = nv1d*(nv1d-1);
            quad_e[3*nel_per_face + el_offset][0] = y_off + nv1d*nv1d*iy + ix;
            quad_e[3*nel_per_face + el_offset][1] = y_off + nv1d*nv1d*iy + ix + 1;
            quad_e[3*nel_per_face + el_offset][2] = y_off + nv1d*nv1d*(iy+1) + ix + 1;
            quad_e[3*nel_per_face + el_offset][3] = y_off + nv1d*nv1d*(iy+1) + ix;
            // z = 0
            quad_e[4*nel_per_face + el_offset][0] = nv1d*iy + ix;
            quad_e[4*nel_per_face + el_offset][1] = nv1d*iy + ix + 1;
            quad_e[4*nel_per_face + el_offset][2] = nv1d*(iy+1) + ix + 1;
            quad_e[4*nel_per_face + el_offset][3] = nv1d*(iy+1) + ix;
            // z = 1
            int z_off = nv1d*nv1d*(nv1d-1);
            quad_e[5*nel_per_face + el_offset][0] = z_off + nv1d*iy + ix;
            quad_e[5*nel_per_face + el_offset][1] = z_off + nv1d*iy + ix + 1;
            quad_e[5*nel_per_face + el_offset][2] = z_off + nv1d*(iy+1) + ix + 1;
            quad_e[5*nel_per_face + el_offset][3] = z_off + nv1d*(iy+1) + ix;
         }
      }

      // Delete on z = 0 face
      quad_e[4*nel_per_face + 1 + 0*(nv1d-1)][0] = -1;
      quad_e[4*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      quad_e[4*nel_per_face + 1 + 2*(nv1d-1)][0] = -1;
      // Delete on y = 0 face
      quad_e[2*nel_per_face + 1 + 0*(nv1d-1)][0] = -1;
      quad_e[2*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      // Delete on y = 1 face
      quad_e[3*nel_per_face + 1 + 0*(nv1d-1)][0] = -1;
      quad_e[3*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;

      // Delete on z = 1 face
      quad_e[5*nel_per_face + 0 + 1*(nv1d-1)][0] = -1;
      quad_e[5*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      quad_e[5*nel_per_face + 2 + 1*(nv1d-1)][0] = -1;
      // Delete on x = 0 face
      quad_e[0*nel_per_face + 1 + 2*(nv1d-1)][0] = -1;
      quad_e[0*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      // Delete on x = 1 face
      quad_e[1*nel_per_face + 1 + 2*(nv1d-1)][0] = -1;
      quad_e[1*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;

      for (int j = 0; j < nv; j++) { AddVertex(quad_v[j]); }
      for (int j = 0; j < nel_total; j++)
      {
         if (quad_e[j][0] >= 0)
         {
            AddQuad(quad_e[j], j+1);
         }
      }

      RemoveUnusedVertices();
      FinalizeQuadMesh(1, 1, true);
      FinalizeTopology();
      for (int l = 0; l < nr; l++) { UniformRefinement(); }
      SetCurvature(order, false, 3, Ordering::byNODES);

      // Snap the nodes to the unit sphere
      const int mesh_sdim = SpaceDimension();
      GridFunction &nodes = *GetNodes();
      Vector node(mesh_sdim);
      for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
      {
         for (int d = 0; d < mesh_sdim; d++)
         { node(d) = nodes(nodes.FESpace()->DofToVDof(i, d)); }
         node /= node.Norml2();
         for (int d = 0; d < mesh_sdim; d++)
         { nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d); }
      }
   }

   void BoundaryConditions()
   {
      if (bdr_attributes.Size())
      {
         Array<int> ess_bdr(bdr_attributes.Max());
         ess_bdr = 1;
         Nodes->FESpace()->GetEssentialTrueDofs(ess_bdr, bc);
      }
   }
};

// Visualize some solution on the given mesh
static void Visualize(ParMesh *pm, const int w, const int h,
                      const char *keys)
{
   glvis << "parallel " << pm->GetNRanks() << " " << pm->GetMyRank() << "\n";
   const GridFunction *x = pm->GetNodes();
   glvis << "solution\n" << *pm << *x;
   glvis << "window_size " << w << " " << h <<"\n";
   glvis << "keys " << keys << "\n";
   glvis << flush;
}

static void Visualize(ParMesh *pm,  const bool pause)
{
   glvis << "parallel " << pm->GetNRanks() << " " << pm->GetMyRank() << "\n";
   const GridFunction *x = pm->GetNodes();
   glvis << "solution\n" << *pm << *x;
   if (pause) { glvis << "pause\n"; }
   glvis << flush;
}

// Surface solver class
template<class Type>
class SurfaceSolver
{
protected:
   bool pa, vis, pause;
   int niter, sdim, order;
   ParMesh *pmesh;
   Vector X, B;
   OperatorPtr A;
   ParFiniteElementSpace *pfes;
   ParBilinearForm a;
   Array<int> &dbc;
   ParGridFunction x, b;
   ConstantCoefficient one;
   Type *solver;
public:
   SurfaceSolver(const bool p, const bool v,
                 const int n, const bool w,
                 const int o, ParMesh *pm,
                 ParFiniteElementSpace *pf,
                 Array<int> &bc):
      pa(p), vis(v), pause(w), niter(n),
      sdim(pf->GetVDim()), order(o), pmesh(pm), pfes(pf),
      a(pfes), dbc(bc), x(pfes), b(pfes), one(1.0),
      solver(static_cast<Type*>(this)) { Solve(); }
   void Solve() { solver->Solve(); }
   void ParCG(const Operator &A, const Vector &B, Vector &X,
              int print_iter = 0, int max_num_iter = 2000,
              double RTOLERANCE = 1e-12, double ATOLERANCE = 0.0)
   {
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetPrintLevel(print_iter);
      cg.SetMaxIter(max_num_iter);
      cg.SetRelTol(sqrt(RTOLERANCE));
      cg.SetAbsTol(sqrt(ATOLERANCE));
      cg.SetOperator(A);
      cg.Mult(B, X);
   }
};

// Surface solver 'by compnents'
class ByComponent: public SurfaceSolver<ByComponent>
{
public:
   ByComponent(const bool pa,  const bool vis,
               const int niter, const bool wait,
               const int order, ParMesh *pm,
               ParFiniteElementSpace *pf,
               Array<int> &bc):
      SurfaceSolver(pa, vis, niter, wait, order, pm, pf, bc) { }
   void Solve()
   {
      if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      for (int iiter=0; iiter<niter; ++iiter)
      {
         a.Assemble();
         ParGridFunction nodes;
         pmesh->GetNodes(nodes);
         GridFunction solution = nodes;
         for (int i=0; i < 3; ++i)
         {
            x = b = 0.0;
            GetComponent(nodes, x, i);
            a.FormLinearSystem(dbc, x, b, A, X, B);
            if (!pa)
            {
               // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
               GSSmoother M(static_cast<SparseMatrix&>(*A));
               PCG(*A, M, B, X, 3, 2000, eps, 0.0);
            }
            else
            {
               mfem_error("Should use parallel CG!");
               CG(*A, B, X, 3, 2000, eps, 0.0);
            }
            // Recover the solution as a finite element grid function.
            a.RecoverFEMSolution(X, b, x);
            SetComponent(solution, x, i);
         }
         nodes = solution;
         pmesh->SetNodes(nodes);
         // Send the solution by socket to a GLVis server.
         if (vis) { Visualize(pmesh, pause); }
         pmesh->DeleteGeometricFactors();
         a.Update();
      }
   }
public:
   void SetComponent(GridFunction &X, const GridFunction &Xi, const int d)
   {
      auto d_Xi = Xi.Read();
      auto d_X  = X.Write();
      const int ndof = pfes->GetNDofs();
      MFEM_FORALL(i, ndof, d_X[d*ndof + i] = d_Xi[i]; );
   }

   void GetComponent(const GridFunction &X, GridFunction &Xi, const int d)
   {
      auto d_X  = X.Read();
      auto d_Xi = Xi.Write();
      const int ndof = pfes->GetNDofs();
      MFEM_FORALL(i, ndof, d_Xi[i] = d_X[d*ndof + i]; );
   }
};

// Surface solver 'by vector'
class ByVector: public SurfaceSolver<ByVector>
{
public:
   ByVector(const bool pa, const bool vis,
            const int niter, const bool wait,
            const int order, ParMesh *pm,
            ParFiniteElementSpace *pf,
            Array<int> &bc):
      SurfaceSolver(pa, vis, niter, wait, order, pm, pf, bc) { }
   void Solve()
   {
      if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
      for (int iiter=0; iiter<niter; ++iiter)
      {
         a.Assemble();
         b = 0.0;
         pmesh->GetNodes(x);
         a.FormLinearSystem(dbc, x, b, A, X, B);

         if (!pa)
         {
            // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
            GSSmoother M(static_cast<SparseMatrix&>(*A));
            PCG(*A, M, B, X, 3, 2000, eps, 0.0);
         }
         else
         {
            ParCG(*A, B, X, 3, 8000, 1.e-14, 0.0);
         }
         // Recover the solution as a finite element grid function.
         a.RecoverFEMSolution(X, b, x);
         pmesh->SetNodes(x);
         // Send the solution by socket to a GLVis server.
         if (vis) { Visualize(pmesh, pause); }
         pmesh->DeleteGeometricFactors();
         a.Update();
      }
   }
};

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   int s = -1;
   int nx = 4;
   int ny = 4;
   int nr = 2;
   int order = 3;
   int niter = 4;
   bool c = false;
   bool pa = true;
   bool vis = false;
   bool amr = false;
   bool wait = false;
   const char *keys = "gAmaaa";
   const char *device_config = "cpu";
   const char *mesh_file = "../data/mobius-strip.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&s, "-s", "--surface", "Choice of the surface.");
   args.AddOption(&wait, "-w", "--wait", "-no-w", "--no-wait",
                  "Enable or disable a GLVis pause.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&nr, "-r", "--ref-levels", "Refinement");
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
   args.AddOption(&c, "-c", "--components", "-no-c", "--no-components",
                  "Enable or disable the 'by component' solver");

   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   if (myid == 0) { args.PrintOptions(cout); }
   MFEM_VERIFY(!amr, "AMR not yet supported!");

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // Initialize GLVis server if 'visualization' is set.
   if (vis)
   {
      glvis.open(vishost, visport);
      if (!glvis)
      {
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
         vis = false;
      }
      glvis.precision(8);
   }

   // Initialize our surface mesh from command line option.
   // Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> bc;
   Mesh *mesh = nullptr;
   const int sdim = c ? 1 : 3;
   if (s < 0)  { mesh = new MeshFromFile(bc, order, mesh_file, nr, sdim); }
   if (s == 0) { mesh = new Catenoid(bc, order, nx, ny, nr, sdim); }
   if (s == 1) { mesh = new Helicoid(bc, order, nx, ny, nr, sdim); }
   if (s == 2) { mesh = new Enneper(bc, order, nx, ny, nr, sdim); }
   if (s == 3) { mesh = new Scherk(bc, order, nx, ny, nr, sdim); }
   if (s == 4) { mesh = new Shell(bc, order, nx, ny, nr, sdim); }
   if (s == 5) { mesh = new Hold(bc, order, nx, ny, nr, sdim); }
   if (s == 6) { mesh = new QPeach(bc, order, nx, ny, nr, sdim); }
   if (s == 7) { mesh = new FPeach(bc, order, nr, sdim); }
   if (s == 8) { mesh = new SlottedSphere(bc, order, nr, sdim); }
   MFEM_VERIFY(mesh, "Not a valid surface number!");

   Surface<> &surface = *static_cast<Surface<>*>(mesh);

   ParMesh *pmesh = surface.Pmesh();
   ParFiniteElementSpace *pfes = surface.Pfes();

   // Send to GLVis the first mesh and set the 'keys' options.
   if (vis) { Visualize(pmesh, 800, 800, keys); }

   // Create and launch the surface solver.
   if (c) { ByComponent Solve(pa, vis, niter, wait, order, pmesh, pfes, bc); }
   else {   ByVector    Solve(pa, vis, niter, wait, order, pmesh, pfes, bc); }

   // Free the used memory.
   delete mesh;
   MPI_Finalize();
   return 0;
}
