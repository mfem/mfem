// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//               -----------------------
//               Minimal Surface Miniapp
//               -----------------------
//
// Description:  This example code
//               s=0: Catenoid
//               s=1: Helicoid
//               s=2: Enneper
//               s=3: Scherk
//               s=4: Hold
//               s=5: QPeach
//               s=6: FPeach
//               s=7: SlottedSphere
//               s=8: Costa
//               s=9: Shell
//
// Compile with: make mesh-minimal-surface
//
// Sample runs:  mesh-minimal-surface -vis
//
// Device sample runs:
//               mesh-minimal-surface -d cuda

#include "mfem.hpp"
#include "../../general/forall.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Constant variables
constexpr int DIM = 2;
constexpr int SDIM = 3;
constexpr double PI = M_PI;
constexpr double NRM = 1.e-4;
constexpr double EPS = 1.e-14;

// Static variables for GLVis.
static int NRanks, MyRank;
static socketstream glvis;
const int  visport = 19916;
const char vishost[] = "localhost";

// Surface mesh class
template<typename T = nullptr_t>
class Surface: public Mesh
{
protected:
   T *S;
   Array<int> &bc;
   int order, nx, ny, nr, vdim;
   Mesh *msh = nullptr;
   H1_FECollection *fec = nullptr;
   FiniteElementSpace *fes = nullptr;
public:

   // Reading from mesh file
   Surface(Array<int> &bc, int order, const char *file, int nr, int vdim):
      Mesh(file, true), S(static_cast<T*>(this)),
      bc(bc), order(order), nr(nr), vdim(vdim)
   {
      S->Postfix();
      S->Refine();
      GenFESpace();
      S->BoundaryConditions();
   }

   // Generate Quad surface mesh
   Surface(Array<int> &b, int order, int nx, int ny, int nr, int vdim):
      Mesh(nx, ny, Element::QUADRILATERAL, true, 1.0, 1.0, false),
      S(static_cast<T*>(this)),
      bc(b), order(order), nx(nx), ny(ny), nr(nr), vdim(vdim)
   {
      S->Prefix();
      S->Create();
      S->Postfix();
      S->Refine();
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
      SetCurvature(order, false, SDIM, Ordering::byVDIM);
      GridFunction &nodes = *GetNodes();
      for (int i = 0; i < nodes.Size(); i++)
      { if (std::abs(nodes(i)) < EPS) { nodes(i) = 0.0; } }
      SetCurvature(order, false, SDIM, Ordering::byNODES);
      GenFESpace();
      S->BoundaryConditions();
   }

   // Generated Cube surface mesh
   Surface(Array<int> &b, int order, int NV, int NE, int NBE,
           int nr, int vdim):
      Mesh(DIM, NV, NE, NBE, SDIM), S(static_cast<T*>(this)),
      bc(b), order(order), nx(NV), ny(NE), nr(nr), vdim(vdim)
   {
      S->Create();
      GenFESpace();
      S->BoundaryConditions();
   }

   ~Surface() { delete fec; delete msh; delete fes; }

   void Prefix() { SetCurvature(order, false, SDIM, Ordering::byNODES); }

   void Create() { Transform(T::Parametrization); }

   void Postfix() { SetCurvature(order, false, SDIM, Ordering::byNODES); }

   void Refine()
   {
      for (int l = 0; l < nr; l++) { UniformRefinement(); }
      // Adaptive mesh refinement
      //if (amr)  { for (int l = 0; l < 1; l++) { RandomRefinement(0.5); } }
      //PrintCharacteristics();
   }

   void BoundaryConditions()
   {
      if (bdr_attributes.Size())
      {
         Array<int> ess_bdr(bdr_attributes.Max());
         ess_bdr = 1;
         fes->GetEssentialTrueDofs(ess_bdr, bc);
      }
   }

   void GenFESpace()
   {
      fec = new H1_FECollection(order, DIM);
      msh = new Mesh(*this, true);
      fes = new FiniteElementSpace(msh, fec, vdim);
   }

   Mesh *GetMesh() const { return msh; }

   FiniteElementSpace *GetFESpace() const { return fes; }
};

// Default surface mesh file
struct MeshFromFile: public Surface<MeshFromFile>
{
   MeshFromFile(Array<int> &BC, int o, const char *file, int r, int d):
      Surface(BC, o, file, r, d) {}
};

// #0: Catenoid surface
struct Catenoid: public Surface<Catenoid>
{
   Catenoid(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) { }

   void Prefix()
   {
      SetCurvature(order, false, SDIM, Ordering::byNODES);
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
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
   }

   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(SDIM);
      // u in [0,2π] and v in [-2π/3,2π/3]
      const double u = 2.0*PI*x[0];
      const double v = 2.0*PI*(2.0*x[1]-1.0)/3.0;
      p[0] = 3.2*cos(u); // cos(u)*cosh(v);
      p[1] = 3.2*sin(u); // sin(u)*cosh(v);
      p[2] = v;
   }
};

// #1: Helicoid surface
struct Helicoid: public Surface<Helicoid>
{
   Helicoid(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) { }

   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(SDIM);
      const double a = 1.0;
      // u in [0,2π] and v in [-2π/3,2π/3]
      const double u = 2.0*PI*x[0];
      const double v = 2.0*PI*(2.0*x[1]-1.0)/3.0;
      p[0] = a*cos(u)*sinh(v);
      p[1] = a*sin(u)*sinh(v);
      p[2] = a*u;
   }
};

// #2: Enneper's surface
struct Enneper: public Surface<Enneper>
{
   Enneper(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) { }

   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(SDIM);
      // (u,v) in [-2, +2]
      const double u = 2.0*(2.0*x[0]-1.0);
      const double v = 2.0*(2.0*x[1]-1.0);
      p[0] = +u - u*u*u/3.0 + u*v*v;
      p[1] = -v - u*u*v + v*v*v/3.0;
      p[2] =  u*u - v*v;
   }
};

// #3: Parametrization of Scherk's doubly periodic surface
struct Scherk: public Surface<Scherk>
{
   Scherk(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) { }

   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(SDIM);
      const double alpha = 0.49;
      // (u,v) in [-απ, +απ]
      const double u = alpha*PI*(2.0*x[0]-1.0);
      const double v = alpha*PI*(2.0*x[1]-1.0);
      p[0] = u;
      p[1] = v;
      p[2] = log(cos(u)/cos(v));
   }
};

// #4: Hold surface
struct Hold: public Surface<Hold>
{
   Hold(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) { }

   void Prefix()
   {
      SetCurvature(order, false, SDIM, Ordering::byNODES);
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
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
   }

   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(SDIM);
      // u in [0,2π] and v in [0,1]
      const double u = 2.0*PI*x[0];
      const double v = x[1];
      p[0] = cos(u)*(1.0 + 0.3*sin(3.*u + PI*v));
      p[1] = sin(u)*(1.0 + 0.3*sin(3.*u + PI*v));
      p[2] = v;
   }
};

// #5: 1/4th Peach street model
struct QuarterPeach: public Surface<QuarterPeach>
{
   QuarterPeach(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) { }

   void Prefix() { SetCurvature(1, false, SDIM, Ordering::byNODES); }

   static void Parametrization(const Vector &X, Vector &p)
   {
      p = X;
      const double x = 2.0*X[0]-1.0;
      const double y = X[1];
      const double r = sqrt(x*x + y*y);
      const double t = (x==0.0) ? PI/2.0 :
                       (y==0.0 && x>0.0) ? 0. :
                       (y==0.0 && x<0.0) ? PI : acos(x/r);
      const double sqrtx = sqrt(1.0 + x*x);
      const double sqrty = sqrt(1.0 + y*y);
      const bool yaxis = PI/4.0<t && t < 3.0*PI/4.0;
      const double R = yaxis?sqrtx:sqrty;
      const double gamma = r/R;
      p[0] = gamma * cos(t);
      p[1] = gamma * sin(t);
      p[2] = 1.0 - gamma;
   }

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
         double R[2], X[2][SDIM];
         for (int v = 0; v < 2; v++)
         {
            R[v] = 0.0;
            const int iv = vertices[v];
            for (int d = 0; d < SDIM; d++)
            {
               nodes->GetNodalValues(nval, d+1);
               const double x = X[v][d] = nval[iv];
               if (d < 2) { R[v] += x*x; }
            }
         }
         if (fabs(X[0][1])<=EPS && fabs(X[1][1])<=EPS &&
             (R[0]>0.1 || R[1]>0.1))
         { el->SetAttribute(1); }
         else { el->SetAttribute(2); }
      }
   }
};

// #6: Full Peach street model
struct FullPeach: public Surface<FullPeach>
{
   FullPeach(Array<int> &BC, int o, int r, int d):
      Surface(BC, o, 8, 6, 6, r, d) { }

   void Create()
   {
      const double quad_v[8][SDIM] =
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
      EnsureNodes();
      UniformRefinement();
      for (int l = 0; l < nr; l++) { UniformRefinement(); }
      SetCurvature(order, false, SDIM, Ordering::byNODES);
      // Snap the nodes to the unit sphere
      const int sdim = SpaceDimension();
      GridFunction &nodes = *GetNodes();
      Vector node(sdim);
      for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
      {
         for (int d = 0; d < sdim; d++)
         { node(d) = nodes(nodes.FESpace()->DofToVDof(i, d)); }
         node /= node.Norml2();
         for (int d = 0; d < sdim; d++)
         { nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d); }
      }
   }

   void BoundaryConditions()
   {
      double X[3];
      Array<int> dofs;
      Array<int> ess_cdofs, ess_tdofs;
      ess_cdofs.SetSize(fes->GetVSize());
      ess_cdofs = 0;
      for (int e = 0; e < fes->GetNE(); e++)
      {
         fes->GetElementDofs(e, dofs);
         for (int c = 0; c < dofs.Size(); c++)
         {
            int k = dofs[c];
            if (k < 0) { k = -1 - k; }
            fes->GetMesh()->GetNode(k, X);
            const bool halfX = fabs(X[0]) < EPS && X[1] <= 0.0;
            const bool halfY = fabs(X[2]) < EPS && X[1] >= 0.0;
            const bool is_on_bc = halfX || halfY;
            for (int d = 0; d < vdim; d++)
            { ess_cdofs[fes->DofToVDof(k, d)] = is_on_bc; }
         }
      }
      const SparseMatrix *R = fes->GetRestrictionMatrix();
      if (!R) { ess_tdofs.MakeRef(ess_cdofs); }
      else { R->BooleanMult(ess_cdofs, ess_tdofs); }
      FiniteElementSpace::MarkerToList(ess_tdofs, bc);
   }
};

// #7: Full Peach street model
struct SlottedSphere: public Surface<SlottedSphere>
{
   SlottedSphere(Array<int> &BC, int o, int r, int d):
      Surface(BC, o, 64, 40, 0, r, d) { }
   void Create()
   {
      constexpr double delta = 0.15;
      constexpr int nv1d = 4;
      constexpr int nv = nv1d*nv1d*nv1d;
      constexpr int nel_per_face = (nv1d-1)*(nv1d-1);
      constexpr int nel_total = nel_per_face*6;
      const double vert1d[nv1d] = {-1.0, -delta, delta, 1.0};
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

      // Delete on x = 0 face
      quad_e[0*nel_per_face + 1 + 2*(nv1d-1)][0] = -1;
      quad_e[0*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      // Delete on x = 1 face
      quad_e[1*nel_per_face + 1 + 2*(nv1d-1)][0] = -1;
      quad_e[1*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      // Delete on y = 1 face
      quad_e[3*nel_per_face + 1 + 0*(nv1d-1)][0] = -1;
      quad_e[3*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      // Delete on z = 1 face
      quad_e[5*nel_per_face + 0 + 1*(nv1d-1)][0] = -1;
      quad_e[5*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      quad_e[5*nel_per_face + 2 + 1*(nv1d-1)][0] = -1;
      // Delete on z = 0 face
      quad_e[4*nel_per_face + 1 + 0*(nv1d-1)][0] = -1;
      quad_e[4*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;
      quad_e[4*nel_per_face + 1 + 2*(nv1d-1)][0] = -1;
      // Delete on y = 0 face
      quad_e[2*nel_per_face + 1 + 0*(nv1d-1)][0] = -1;
      quad_e[2*nel_per_face + 1 + 1*(nv1d-1)][0] = -1;

      for (int j = 0; j < nv; j++) { AddVertex(quad_v[j]); }
      for (int j = 0; j < nel_total; j++)
      {
         if (quad_e[j][0] < 0) { continue; }
         AddQuad(quad_e[j], j+1);
      }
      RemoveUnusedVertices();
      FinalizeQuadMesh(1, 1, true);
      EnsureNodes();
      FinalizeTopology();
      for (int l = 0; l < nr; l++) { UniformRefinement(); }
      SetCurvature(order, false, SDIM, Ordering::byNODES);
      // Snap the nodes to the unit sphere
      GridFunction &nodes = *GetNodes();
      Vector node(SDIM);
      for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
      {
         for (int d = 0; d < SDIM; d++)
         { node(d) = nodes(nodes.FESpace()->DofToVDof(i, d)); }
         node /= node.Norml2();
         for (int d = 0; d < SDIM; d++)
         { nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d); }
      }
   }
};

// #8: Shell surface model
struct Shell: public Surface<Shell>
{
   Shell(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [-15, 6]
      const double u = 2.0*PI*x[0];
      const double v = 21.0*x[1]-15.0;
      p[0] = +1.0*pow(1.16,v)*cos(v)*(1.0+cos(u));
      p[1] = -1.0*pow(1.16,v)*sin(v)*(1.0+cos(u));
      p[2] = -2.0*pow(1.16,v)*(1.0+sin(u));
   }
};

// #9: Costa minimal surface
#include <complex>
using cdouble = std::complex<double>;
#define I cdouble(0.0, 1.0)

// https://dlmf.nist.gov/20.2
cdouble EllipticTheta(const int a, const cdouble u, const cdouble q)
{
   cdouble J = 0.0;
   double delta = std::numeric_limits<double>::max();
   switch (a)
   {
      case 1:
         for (int n=0; delta > EPS; n+=1)
         {
            const cdouble j = pow(-1,n)*pow(q,n*(n+1.0))*sin((2.0*n+1.0)*u);
            delta = abs(j);
            J += j;
         }
         return 2.0*pow(q,0.25)*J;

      case 2:
         for (int n=0; delta > EPS; n+=1)
         {
            const cdouble j = pow(q,n*(n+1))*cos((2.0*n+1.0)*u);
            delta = abs(j);
            J += j;
         }
         return 2.0*pow(q,0.25)*J;
      case 3:
         for (int n=1; delta > EPS; n+=1)
         {
            const cdouble j = pow(q,n*n)*cos(2.0*n*u);
            delta = abs(j);
            J += j;
         }
         return 1.0 + 2.0*J;
      case 4:
         for (int n=1; delta > EPS; n+=1)
         {
            const cdouble j =pow(-1,n)*pow(q,n*n)*cos(2.0*n*u);
            delta = abs(j);
            J += j;
         }
         return 1.0 + 2.0*J;
   }
   return J;
}

// https://dlmf.nist.gov/23.6#E5
cdouble WeierstrassP(const cdouble z,
                     const cdouble w1 = 0.5,
                     const cdouble w3 = 0.5*I)
{
   const cdouble tau = w3/w1;
   const cdouble q = exp(I*M_PI*tau);
   const cdouble e1 = M_PI*M_PI/(12.0*w1*w1)*
                      (1.0*pow(EllipticTheta(2,0,q),4) +
                       2.0*pow(EllipticTheta(4,0,q),4));
   const cdouble u = M_PI*z / (2.0*w1);
   const cdouble P = M_PI * EllipticTheta(3,0,q)*EllipticTheta(4,0,q) *
                     EllipticTheta(2,u,q)/(2.0*w1*EllipticTheta(1,u,q));
   return P*P + e1;
}

cdouble EllipticTheta1Prime(const int k, const cdouble u, const cdouble q)
{
   cdouble J = 0.0;
   double delta = std::numeric_limits<double>::max();
   for (int n=0; delta > EPS; n+=1)
   {
      const double alpha = 2.0*n+1.0;
      const cdouble Dcosine = pow(alpha,k)*sin(k*M_PI/2.0 + alpha*u);
      const cdouble j = pow(-1,n)*pow(q,n*(n+1.0))*Dcosine;
      delta = abs(j);
      J += j;
   }
   return 2.0*pow(q,0.25)*J;
}

// Logarithmic Derivative of Theta Function 1
cdouble LogEllipticTheta1Prime(const cdouble u, const cdouble q)
{
   cdouble J = 0.0;
   double delta = std::numeric_limits<double>::max();
   for (int n=1; delta > EPS; n+=1)
   {
      cdouble q2n = pow(q, 2*n);
      if (abs(q2n) < EPS) { q2n = 0.0; }
      const cdouble j = q2n/(1.0-q2n)*sin(2.0*n*u);
      delta = abs(j);
      J += j;
   }
   return 1.0/tan(u) + 4.0*J;
}

// https://dlmf.nist.gov/23.6#E13
cdouble WeierstrassZeta(const cdouble z,
                        const cdouble w1 = 0.5,
                        const cdouble w3 = 0.5*I)
{
   const cdouble tau = w3/w1;
   const cdouble q = exp(I*M_PI*tau);
   const cdouble n1 = -M_PI*M_PI/(12.0*w1) *
                      (EllipticTheta1Prime(3,0,q)/
                       EllipticTheta1Prime(1,0,q));
   const cdouble u = M_PI*z / (2.0*w1);
   return z*n1/w1 + M_PI/(2.0*w1)*LogEllipticTheta1Prime(u,q);
}

// https://www.mathcurve.com/surfaces.gb/costa/costa.shtml
static double ALPHA = 1.0;
struct Costa: public Surface<Costa>
{
   Costa(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, 0, r, d)  { }

   void Create()
   {
      ALPHA = 1.0 / (double)nx;
      MFEM_VERIFY(nx>2 && ny>2, "");
      const int nXhalf = (nx%2)==0 ? 4 : 2;
      const int nYhalf = (ny%2)==0 ? 4 : 2;
      const int nxh = nXhalf + nYhalf;
      const int NVert = (nx+1) * (ny+1);
      const int NElem = nx*ny - 4 - nxh;
      const int NBdrElem = 0;
      InitMesh(DIM, SDIM, NVert, NElem, NBdrElem);
      // Sets vertices and the corresponding coordinates
      for (int j = 0; j <= ny; j++)
      {
         const double cy = ((double) j / ny) ;
         for (int i = 0; i <= nx; i++)
         {
            const double cx = ((double) i / nx);
            const double coords[SDIM] = {cx, cy, 0.0};
            AddVertex(coords);
         }
      }
      // Sets elements and the corresponding indices of vertices
      for (int j = 0; j < ny; j++)
      {
         for (int i = 0; i < nx; i++)
         {
            if (i == 0 && j == 0) { continue; }
            if (i+1 == nx && j == 0) { continue; }
            if (i == 0 && j+1 == ny) { continue; }
            if (i+1 == nx && j+1 == ny) { continue; }
            if ((j == 0 || j+1 == ny) && (abs(nx-(i<<1)-1)<=1)) { continue; }
            if ((i == 0 || i+1 == nx) && (abs(ny-(j<<1)-1)<=1)) { continue; }
            const int i0 = i   +     j*(nx+1);
            const int i1 = i+1 +     j*(nx+1);
            const int i2 = i+1 + (j+1)*(nx+1);
            const int i3 = i   + (j+1)*(nx+1);
            const int ind[4] = {i0, i1, i2, i3};
            AddQuad(ind);
         }
      }
      RemoveUnusedVertices();
      // generate_edges, refinement, fix_orientation
      FinalizeQuadMesh(true, 1, true);
      FinalizeTopology();
      for (int l = 0; l < nr; l++) { UniformRefinement(); }
      SetCurvature(order, false, SDIM, Ordering::byNODES);
      Transform(Parametrization);
   }

   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      const bool y_top = x[1] > 0.5;
      const bool x_top = x[0] > 0.5;
      double u = x[0];
      double v = x[1];
      if (y_top) { v = 1.0 - x[1]; }
      if (x_top) { u = 1.0 - x[0]; }
      const cdouble w = u + I*v;
      const cdouble w3 = I/2.;
      const cdouble w1 = 1./2.;
      const cdouble pw = WeierstrassP(w);
      const cdouble e1 = WeierstrassP(0.5);
      const cdouble zw = WeierstrassZeta(w);
      const cdouble dw = WeierstrassZeta(w-w1) - WeierstrassZeta(w-w3);
      p[0] = real(PI*(u+PI/(4.*e1))- zw +PI/(2.*e1)*(dw));
      p[1] = real(PI*(v+PI/(4.*e1))-I*zw-PI*I/(2.*e1)*(dw));
      p[2] = sqrt(PI/2.)*log(abs((pw-e1)/(pw+e1)));
      p *= ALPHA;
      if (y_top) { p[1] *= -1.0; }
      if (x_top) { p[0] *= -1.0; }
      const bool nan = isnan(p[0])||isnan(p[1])||isnan(p[2]);
      MFEM_VERIFY(!nan, "nan");
   }
};

// Visualize some solution on the given mesh
static void Visualize(Mesh *pm, const int w, const int h,
                      const char *keys)
{
   glvis << "parallel " << NRanks << " " << MyRank << "\n";
   const GridFunction *x = pm->GetNodes();
   glvis << "solution\n" << *pm << *x;
   glvis << "window_size " << w << " " << h <<"\n";
   glvis << "keys " << keys << "\n";
   glvis.precision(8);
   glvis << flush;
}

static void Visualize(Mesh *pm,  const bool pause)
{
   glvis << "parallel " << NRanks << " " << MyRank << "\n";
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
   bool pa, vis, pause, radial;
   int nmax, vdim, order;
   Mesh *pmesh;
   Vector X, B;
   OperatorPtr A;
   FiniteElementSpace *pfes;
   BilinearForm a;
   Array<int> &bc;
   GridFunction x, x0, b;
   ConstantCoefficient one;
   Type *solver;
   Solver *M;
   const int print_iter = -1, max_num_iter = 2000;
   const double lambda = 0.0, RTOLERANCE = EPS, ATOLERANCE = EPS*EPS;
public:
   SurfaceSolver(const bool pa, const bool vis,
                 const int n, const bool pause,
                 const int order, const double l, const bool radial,
                 Mesh *pmesh, FiniteElementSpace *pfes,
                 Array<int> &bc):
      pa(pa), vis(vis), pause(pause), radial(radial), nmax(n),
      vdim(pfes->GetVDim()), order(order), pmesh(pmesh), pfes(pfes),
      a(pfes), bc(bc), x(pfes), x0(pfes), b(pfes), one(1.0),
      solver(static_cast<Type*>(this)), M(nullptr), lambda(l) { }

   ~SurfaceSolver() { delete M; }

   void Solve()
   {
      if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL);}
      for (int i=0; i<nmax; ++i)
      {
         if (MyRank == 0)
         { mfem::out << "Linearized iteration " << i << ": "; }
         Update();
         a.Assemble();
         if (solver->Loop()) { break; }
      }
   }

   bool Converged(const double rnorm)
   {
      if (rnorm < NRM)
      {
         if (MyRank==0) { mfem::out << "Converged!" << endl; }
         return true;
      }
      return false;
   }

   bool ParAXeqB(bool by_component)
   {
      b = 0.0;
      a.FormLinearSystem(bc, x, b, A, X, B);
      CGSolver cg;
      cg.SetPrintLevel(print_iter);
      cg.SetMaxIter(max_num_iter);
      cg.SetRelTol(RTOLERANCE);
      cg.SetAbsTol(ATOLERANCE);
      if (!pa) { M = new GSSmoother((SparseMatrix&)(*A)); }
      if (M) { cg.SetPreconditioner(*M); }
      cg.SetOperator(*A);
      cg.Mult(B, X);
      a.RecoverFEMSolution(X, b, x);
      GridFunction *nodes = by_component ? &x0 : pfes->GetMesh()->GetNodes();
      x.HostRead();
      nodes->HostRead();
      double rnorm = nodes->DistanceTo(x) / nodes->Norml2();
      mfem::out << "rnorm = " << rnorm << endl;
      if (by_component)
      {
         MFEM_VERIFY(lambda == 0.0,"'By component' assumes lambda == 0.0");
         MFEM_VERIFY(!radial,"'By component' solver can't use 'radial option");
         return Converged(rnorm);
      }
      if (!radial) { add(lambda, *nodes, 1.0-lambda, x, *nodes); }
      else
      {
         GridFunction delta(pfes); // x = nodes + delta
         subtract(x,*nodes,delta);
         // position and delta vectors at point i
         Vector ni(3), di(3);
         for (int i = 0; i < delta.Size()/3; i++)
         {
            // extract local vectors
            const int ndof = pfes->GetNDofs();
            for (int d = 0; d < 3; d++)
            {
               ni(d) = (*nodes)(d*ndof + i);
               di(d) = delta(d*ndof + i);
            }
            // project the delta vector in radial direction
            const double ndotd = (ni*di) / (ni*ni);
            di.Set(ndotd,ni);
            // set global vectors
            for (int d = 0; d < 3; d++) { delta(d*ndof + i) = di(d); }
         }
         add(lambda, delta, 1.0-lambda, *nodes, *nodes);
      }
      return Converged(rnorm);
   }

   void Update()
   {
      if (vis) { Visualize(pmesh, pause); }
      pmesh->DeleteGeometricFactors();
      a.Update();
   }
};

// Surface solver 'by compnents'
class ByComponent: public SurfaceSolver<ByComponent>
{
public:
   void SetNodes(const GridFunction &Xi, const int c)
   {
      auto d_Xi = Xi.Read();
      auto d_nodes  = pfes->GetMesh()->GetNodes()->Write();
      const int ndof = pfes->GetNDofs();
      MFEM_FORALL(i, ndof, d_nodes[c*ndof + i] = d_Xi[i]; );
   }

   void GetNodes(GridFunction &Xi, const int c)
   {
      auto d_Xi = Xi.Write();
      const int ndof = pfes->GetNDofs();
      auto d_nodes  = pfes->GetMesh()->GetNodes()->Read();
      MFEM_FORALL(i, ndof, d_Xi[i] = d_nodes[c*ndof + i]; );
   }

public:
   ByComponent(bool PA, bool glvis, int n, bool wait, int o, double l, bool rad,
               Mesh *xmesh, FiniteElementSpace *xfes, Array<int> &BC):
      SurfaceSolver(PA, glvis, n, wait, o, l, rad, xmesh, xfes, BC)
   { a.AddDomainIntegrator(new DiffusionIntegrator(one)); }

   bool Loop()
   {
      bool cvg[3] {false};
      for (int c=0; c < 3; ++c)
      {
         this->GetNodes(x, c);
         x0 = x;
         cvg[c] = this->ParAXeqB(true);
         this->SetNodes(x, c);
      }
      const bool converged = cvg[0] && cvg[1] && cvg[2];
      return converged ? true : false;
   }
};

// Surface solver 'by vector'
class ByVector: public SurfaceSolver<ByVector>
{
public:
   ByVector(bool PA, bool glvis, int n, bool wait, int o, double l, bool rad,
            Mesh *xmsh, FiniteElementSpace *xfes, Array<int> &BC):
      SurfaceSolver(PA, glvis, n, wait, o, l, rad, xmsh, xfes, BC)
   { a.AddDomainIntegrator(new VectorDiffusionIntegrator(one)); }

   bool Loop()
   {
      x = *pfes->GetMesh()->GetNodes();
      bool converge = this->ParAXeqB(false);
      pmesh->SetNodes(x);
      return converge ? true : false;
   }
};

Mesh *NewMeshFromSurface(const int surface, Array<int> &bc,
                         int o, int x, int y, int r, int d)
{
   switch (surface)
   {
      case 0: return new Catenoid(bc, o, x, y, r, d);
      case 1: return new Helicoid(bc, o, x, y, r, d);
      case 2: return new Enneper(bc, o, x, y, r, d);
      case 3: return new Scherk(bc, o, x, y, r, d);
      case 4: return new Hold(bc, o, x, y, r, d);
      case 5: return new QuarterPeach(bc, o, x, y, r, d);
      case 6: return new FullPeach(bc, o, r, d);
      case 7: return new SlottedSphere(bc, o, r, d);
      case 8: return new Costa(bc, o, x, y, r, d);
      case 9: return new Shell(bc, o, x, y, r, d);
      default: ;
   }
   mfem_error("Unknown surface (0 <= surface <= 9)!");
   return nullptr;
}

int main(int argc, char *argv[])
{
   NRanks = 1; MyRank = 0;
   // Parse command-line options.
   int o = 4;
   int x = 4;
   int y = 4;
   int r = 2;
   int nmax = 16;
   int surface = -1;
   bool pa = true;
   bool vis = true;
   bool amr = false;
   bool wait = false;
   bool rad = false;
   double lambda = 0.0;
   bool solve_by_components = false;
   const char *keys = "gAmmaaa";
   const char *device_config = "cpu";
   const char *mesh_file = "../../data/mobius-strip.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&wait, "-w", "--wait", "-no-w", "--no-wait",
                  "Enable or disable a GLVis pause.");
   args.AddOption(&rad, "-rad", "--radial", "-no-rad", "--no-radial",
                  "Enable or disable radial constraints in solver.");
   args.AddOption(&x, "-x", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&y, "-y", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&o, "-o", "--order", "Finite element order.");
   args.AddOption(&r, "-r", "--ref-levels", "Refinement");
   args.AddOption(&nmax, "-n", "--niter-max", "Max number of iterations");
   args.AddOption(&surface, "-s", "--surface", "Choice of the surface.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&lambda, "-l", "--lambda", "Lambda step toward solution.");
   args.AddOption(&amr, "-amr", "--adaptive-mesh-refinement", "-no-amr",
                  "--no-adaptive-mesh-refinement", "Enable AMR.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&keys, "-k", "--keys", "GLVis configuration keys.");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable visualization.");
   args.AddOption(&solve_by_components, "-c", "--components",
                  "-no-c", "--no-components",
                  "Enable or disable the 'by component' solver");

   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   if (MyRank == 0) { args.PrintOptions(cout); }
   MFEM_VERIFY(!amr, "AMR not yet supported!");

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (MyRank == 0) { device.Print(); }

   // Initialize GLVis server if 'visualization' is set.
   if (vis) { vis = glvis.open(vishost, visport) == 0; }

   // Initialize our surface mesh from command line option and determine
   // the list of true (i.e. conforming) essential boundary dofs.
   Mesh *mesh;
   Array<int> bc;
   const int d = solve_by_components ? 1 : 3;
   mesh = (surface < 0) ? new MeshFromFile(bc, o, mesh_file, r, d) :
          NewMeshFromSurface(surface, bc, o, x, y, r, d);
   MFEM_VERIFY(mesh, "Not a valid surface number!");

   // Grab back the pmesh & pfes from the Surface object.
   Surface<> &S = *static_cast<Surface<>*>(mesh);
   Mesh *pmesh = S.GetMesh();
   FiniteElementSpace *pfes = S.GetFESpace();

   // Send to GLVis the first mesh and set the 'keys' options.
   if (vis) { Visualize(pmesh, 800, 800, keys); }

   // Create and launch the surface solver.
   if (solve_by_components)
   { ByComponent(pa, vis, nmax, wait, o, lambda, rad, pmesh, pfes, bc).Solve(); }
   else
   { ByVector(pa, vis, nmax, wait, o, lambda, rad, pmesh, pfes, bc).Solve(); }

   // Free the used memory.
   delete mesh;
   return 0;
}
