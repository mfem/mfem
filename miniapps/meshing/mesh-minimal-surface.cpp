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
//               s=4: Shell
//               s=5: Hold
//               s=6: QPeach
//               s=7: FPeach
//               s=8: SlottedSphere
//
// Compile with: make mesh-minimal-surface
//
// Sample runs:  mesh-minimal-surface -vis
//
// Device sample runs:
//               mesh-minimal-surface -d cuda

#include "mfem.hpp"
#include "general/forall.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Constant variables
const double pi = M_PI;
const double eps = 1.e-24;
const int  visport = 19916;
const char vishost[] = "localhost";

// Static variables for GLVis.
static socketstream glvis;
static int NRanks, MyRank;

// Use MFEM's sequential classes and constructs for X-MPI ones, if needed.
#ifndef XMesh
#define XMesh Mesh
#define XGridFunction GridFunction
#define XBilinearForm BilinearForm
#define XMeshConstructor(this) Mesh(*this)
#define XFiniteElementSpace FiniteElementSpace
#define XInit(num_procs, myid) { num_procs=1; myid=0; }
#define XCGArguments
#define XPreconditioner new GSSmoother((SparseMatrix&)(*A))
#define XFinalize()
#endif

// Surface mesh class
template<typename T = nullptr_t>
class Surface: public Mesh
{
protected:
   T *S;
   Array<int> &bc;
   int order, nx, ny, nr, vdim;
   XMesh *pmesh = nullptr;
   H1_FECollection *fec = nullptr;
   XFiniteElementSpace *pfes = nullptr;
public:

   // Reading from mesh file
   Surface(Array<int> &bc, int order, const char *file, int nr, int vdim):
      Mesh(file, true), S(static_cast<T*>(this)),
      bc(bc), order(order), nr(nr), vdim(vdim)
   {
      EnsureNodes();
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
      EnsureNodes();
      S->Prefix();
      S->Create();
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
           int NVert, int NElem, int NBdrElem, int vdim):
      Mesh(2, NVert, NElem, NBdrElem, 3), S(static_cast<T*>(this)),
      bc(b), order(order), nx(NVert), ny(NElem), nr(nr), vdim(vdim)
   {
      S->Create();
      S->Postfix();
      S->Refine();
      S->GenFESpace();
      S->BoundaryConditions();
   }

   ~Surface() { delete fec; delete pmesh; delete pfes; }

   void Prefix() { SetCurvature(order, false, 3, Ordering::byNODES); }

   void Create() { Transform(T::Parametrization); }

   void Postfix() { SetCurvature(order, false, 3, Ordering::byNODES); }

   void Refine()
   {
      for (int l = 0; l < nr; l++) { UniformRefinement(); }
      // Adaptive mesh refinement
      //if (amr)  { for (int l = 0; l < 1; l++) { RandomRefinement(0.5); } }
      //PrintCharacteristics();
   }

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
      pmesh = new XMeshConstructor(this);
      pfes = new XFiniteElementSpace(pmesh, fec, vdim);
   }

   XMesh *Pmesh() const { return pmesh; }

   XFiniteElementSpace *Pfes() const { return pfes; }
};

// Mesh file surface
struct MeshFromFile: public Surface<MeshFromFile>
{
   MeshFromFile(Array<int> &BC, int o, const char *file, int r, int d):
      Surface(BC, o, file, r, d) {}
};

// Catenoid surface
struct Catenoid: public Surface<Catenoid>
{
   Catenoid(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [-2π/3,2π/3]
      const double u = 2.0*pi*x[0];
      const double v = 2.0*pi*(2.0*x[1]-1.0)/3.0;
      //p[0] = cos(u)*cosh(v);
      //p[1] = sin(u)*cosh(v);
      p[0] = 3.2*cos(u);
      p[1] = 3.2*sin(u);
      p[2] = v;
   }
   // Prefix of the Catenoid surface
   void Prefix()
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
      RemoveUnusedVertices();
      RemoveInternalBoundaries();
   }
};

// Helicoid surface
struct Helicoid: public Surface<Helicoid>
{
   Helicoid(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
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
   Enneper(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
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

// Parametrization of Scherk's doubly periodic surface
struct Scherk: public Surface<Scherk>
{
   Scherk(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
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
   Shell(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
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
   Hold(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(3);
      // u in [0,2π] and v in [0,1]
      const double u = 2.0*pi*x[0];
      const double v = x[1];
      p[0] = cos(u)*(1.0 + 0.3*sin(3.*u + pi*v));
      p[1] = sin(u)*(1.0 + 0.3*sin(3.*u + pi*v));
      p[2] = v;
   }
   void Prefix()
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

      RemoveUnusedVertices();
      RemoveInternalBoundaries();
   }
};

// 1/4th Peach street model
struct QPeach: public Surface<QPeach>
{
   QPeach(Array<int> &BC, int o, int x, int y, int r, int d):
      Surface(BC, o, x, y, r, d) {}
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
   FPeach(Array<int> &BC, int o, int r, int d):
      Surface<FPeach>(BC, o, r, 8, 6, 6, d) { }
   void Create()
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
            const bool halfX = fabs(X[0]) < eps && X[1] <= 0.0;
            const bool halfY = fabs(X[2]) < eps && X[1] >= 0.0;
            const bool is_on_bc = halfX || halfY;
            for (int d = 0; d < vdim; d++)
            { ess_cdofs[pfes->DofToVDof(k, d)] = is_on_bc; }
         }
      }
      const SparseMatrix *R = pfes->GetConformingRestriction();
      if (!R) { ess_tdofs.MakeRef(ess_cdofs); }
      else { R->BooleanMult(ess_cdofs, ess_tdofs); }
      XFiniteElementSpace::MarkerToList(ess_tdofs, bc);
   }
};

// Full Peach street model
struct SlottedSphere: public Surface<SlottedSphere>
{
   SlottedSphere(Array<int> &BC, int o, int r, int d):
      Surface<SlottedSphere>(BC, o, r, 0, 0, 0, d) { }
   void Create()
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
};


// Visualize some solution on the given mesh
static void Visualize(XMesh *pm, const int w, const int h,
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

static void Visualize(XMesh *pm,  const bool pause)
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
   bool pa, vis, pause;
   int niter, vdim, order;
   XMesh *pmesh;
   Vector X, B;
   OperatorPtr A;
   XFiniteElementSpace *pfes;
   XBilinearForm a;
   Array<int> &bc;
   XGridFunction x, b;
   ConstantCoefficient one;
   Type *solver;
   Solver *M;
   const int print_iter = 3, max_num_iter = 2000;
   const double RTOLERANCE = eps, ATOLERANCE = 0.0;
public:
   SurfaceSolver(const bool pa, const bool vis,
                 const int niter, const bool pause,
                 const int order, XMesh *pmesh,
                 XFiniteElementSpace *pfes,
                 Array<int> &bc):
      pa(pa), vis(vis), pause(pause), niter(niter),
      vdim(pfes->GetVDim()), order(order),
      pmesh(pmesh), pfes(pfes), a(pfes), bc(bc), x(pfes), b(pfes), one(1.0),
      solver(static_cast<Type*>(this)), M(nullptr) { }
   ~SurfaceSolver() { delete M; }
   void Solve()
   {
      if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL);}
      for (int iiter=0; iiter<niter; ++iiter)
      { a.Assemble(); solver->Loop(); Update();}
   }
   void ParAXeqB()
   {
      b = 0.0;
      a.FormLinearSystem(bc, x, b, A, X, B);
      CGSolver cg XCGArguments;
      cg.SetPrintLevel(print_iter);
      cg.SetMaxIter(max_num_iter);
      cg.SetRelTol(sqrt(RTOLERANCE));
      cg.SetAbsTol(sqrt(ATOLERANCE));
      if (!pa) { M = XPreconditioner; }
      if (M) { cg.SetPreconditioner(*M); }
      cg.SetOperator(*A);
      cg.Mult(B, X);
      a.RecoverFEMSolution(X, b, x);
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
   ByComponent(bool PA,  bool glvis, int n, bool wait, int o,
               XMesh *xmesh, XFiniteElementSpace *xfes, Array<int> &BC):
      SurfaceSolver(PA, glvis, n, wait, o, xmesh, xfes, BC)
   { a.AddDomainIntegrator(new DiffusionIntegrator(one)); }
   void Loop()
   {
      for (int c=0; c < 3; ++c)
      {
         this->GetNodes(x, c);
         this->ParAXeqB();
         this->SetNodes(x, c);
      }
   }
};

// Surface solver 'by vector'
class ByVector: public SurfaceSolver<ByVector>
{
public:
   ByVector(bool PA, bool glvis, int n, bool wait, int o,
            XMesh *xmsh, XFiniteElementSpace *xfes, Array<int> &BC):
      SurfaceSolver(PA, glvis, n, wait, o, xmsh, xfes, BC)
   { a.AddDomainIntegrator(new VectorDiffusionIntegrator(one)); }
   void Loop()
   {
      x = *pfes->GetMesh()->GetNodes();
      this->ParAXeqB();
      pmesh->SetNodes(x);
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
      case 4: return new Shell(bc, o, x, y, r, d);
      case 5: return new Hold(bc, o, x, y, r, d);
      case 6: return new QPeach(bc, o, x, y, r, d);
      case 7: return new FPeach(bc, o, r, d);
      case 8: return new SlottedSphere(bc, o, r, d);
      default: ;
   }
   mfem_error("Unknown surface (0 <= surface <= 8)!");
   return nullptr;
}

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   XInit(num_procs, myid);
   NRanks = num_procs; MyRank = myid;

   // Parse command-line options.
   int x = 4;
   int y = 4;
   int r = 2;
   int o = 3;
   int niter = 10;
   int surface = -1;
   bool pa = true;
   bool vis = false;
   bool amr = false;
   bool wait = false;
   bool solve_by_components = false;
   const char *keys = "gAmmaaa";
   const char *device_config = "cpu";
   const char *mesh_file = "../../data/mobius-strip.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&wait, "-w", "--wait", "-no-w", "--no-wait",
                  "Enable or disable a GLVis pause.");
   args.AddOption(&x, "-x", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&y, "-y", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&o, "-o", "--order", "Finite element order.");
   args.AddOption(&r, "-r", "--ref-levels", "Refinement");
   args.AddOption(&niter, "-n", "--niter", "Number of iterations");
   args.AddOption(&surface, "-s", "--surface", "Choice of the surface.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
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
   if (myid == 0) { args.PrintOptions(cout); }
   MFEM_VERIFY(!amr, "AMR not yet supported!");

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

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
   XMesh *pmesh = S.Pmesh();
   XFiniteElementSpace *pfes = S.Pfes();

   // Send to GLVis the first mesh and set the 'keys' options.
   if (vis) { Visualize(pmesh, 800, 800, keys); }

   // Create and launch the surface solver.
   if (solve_by_components)
   { ByComponent(pa, vis, niter, wait, o, pmesh, pfes, bc).Solve(); }
   else { ByVector(pa, vis, niter, wait, o, pmesh, pfes, bc).Solve(); }

   // Free the used memory.
   delete mesh;
   XFinalize();
   return 0;
}
