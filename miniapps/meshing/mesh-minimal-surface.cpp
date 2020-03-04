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

#include <cassert>
#include "mfem.hpp"
#include "../../general/forall.hpp"

using namespace std;
using namespace mfem;

// Constant variables
constexpr int DIM = 2;
constexpr int SDIM = 3;
constexpr double PI = M_PI;
constexpr double NRM = 1.e-4;
constexpr double EPS = 1.e-14;
constexpr double AMR_PROB = 0.25;
constexpr Element::Type QUAD = Element::QUADRILATERAL;

// Static variables for GLVis
static int NRanks, MyRank;
static socketstream glvis;
constexpr int GLVIZ_W = 1024;
constexpr int GLVIZ_H = 1024;
constexpr int  visport = 19916;
constexpr char vishost[] = "localhost";

// Options and data for the solver
struct Opt
{
   Array<int> bc;
   int order = 3;
   int nx = 6;
   int ny = 6;
   int refine = 2;
   int max_iter = 32;
   int surface = -1;
   int vdim = 3;
   bool pa = true;
   bool vis = true;
   bool amr = false;
   bool wait = false;
   bool radial = false;
   double lambda = 0.0;
   bool solve_by_components = false;
   const char *keys = "gAmmaaa";
   const char *device_config = "cpu";
   const char *mesh_file = "../../data/mobius-strip.mesh";
   Mesh *mesh = nullptr;
   FiniteElementSpace *fes = nullptr;
};

// Surface mesh class
template<typename T = nullptr_t>
class Surface: public Mesh
{
protected:
   T *S;
   Opt &opt;
   Mesh *msh = nullptr;
   H1_FECollection *fec = nullptr;
   FiniteElementSpace *fes = nullptr;
public:

   // Reading from mesh file
   Surface(Opt &opt, const char *file): Mesh(file, true),
      S(static_cast<T*>(this)), opt(opt) { Postflow(); }

   // Generate 2D quad surface mesh
   Surface(Opt &opt): Mesh(opt.nx, opt.ny, QUAD, true, 1.0, 1.0, false),
      S(static_cast<T*>(this)), opt(opt) { Preflow(); Postflow(); }

   // Generate 2D generic surface mesh
   Surface(Opt &opt, int NV, int NE, int NBE): Mesh(DIM, NV, NE, NBE, SDIM),
      S(static_cast<T*>(this)), opt(opt) { Preflow(); Postflow(); }

   void Preflow() { S->Prefix(); S->Create(); }

   void Postflow()
   {
      S->Postfix();
      S->Refine();
      S->Snap();
      this->GenFESpace();
      S->BC();
   }

   ~Surface() { delete fec; delete msh; delete fes; }

   void Prefix() { SetCurvature(opt.order, false, SDIM, Ordering::byNODES); }

   void Create() { Transform(T::Parametrization); }

   void Postfix() { SetCurvature(opt.order, false, SDIM, Ordering::byNODES); }

   void Refine()
   {
      for (int l = 0; l < opt.refine; l++) { UniformRefinement(); }
      if (opt.amr) { RandomRefinement(AMR_PROB); }
   }

   void Snap()
   {
      GridFunction &nodes = *GetNodes();
      for (int i = 0; i < nodes.Size(); i++)
      { if (std::abs(nodes(i)) < EPS) { nodes(i) = 0.0; } }
   }

   void SnapToUnitSphere() // Snap the nodes to the unit sphere
   {
      Vector node(SDIM);
      GridFunction &nodes = *GetNodes();
      for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
      {
         for (int d = 0; d < SDIM; d++)
         { node(d) = nodes(nodes.FESpace()->DofToVDof(i, d)); }
         node /= node.Norml2();
         for (int d = 0; d < SDIM; d++)
         { nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d); }
      }
   }

   void GenFESpace()
   {
      fec = new H1_FECollection(opt.order, DIM);
      msh = new Mesh(*this, true);
      fes = new FiniteElementSpace(msh, fec, opt.vdim);
   }

   void BC()
   {
      if (bdr_attributes.Size())
      {
         Array<int> ess_bdr(bdr_attributes.Max());
         ess_bdr = 1;
         fes->GetEssentialTrueDofs(ess_bdr, opt.bc);
      }
   }

   Mesh *GetMesh() const { return msh; }

   FiniteElementSpace *GetFESpace() const { return fes; }
};

// Default surface mesh file
struct MeshFromFile: public Surface<MeshFromFile>
{ MeshFromFile(Opt &opt): Surface(opt, opt.mesh_file) { } };

// #0: Catenoid surface
struct Catenoid: public Surface<Catenoid>
{
   Catenoid(Opt &opt): Surface(opt) { }

   void Prefix()
   {
      SetCurvature(opt.order, false, SDIM, Ordering::byNODES);
      Array<int> v2v(GetNV());
      for (int i = 0; i < v2v.Size(); i++) { v2v[i] = i; }
      // identify vertices on vertical lines
      for (int j = 0; j <= opt.ny; j++)
      {
         const int v_old = opt.nx + j * (opt.nx + 1);
         const int v_new =          j * (opt.nx + 1);
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
      // u in [0,2π] and v in [-π/6,π/6]
      const double u = 2.0*PI*x[0];
      const double v = PI*(x[1]-0.5)/3.;
      p[0] = cos(u);
      p[1] = sin(u);
      p[2] = v;
   }
};

// #1: Helicoid surface
struct Helicoid: public Surface<Helicoid>
{
   Helicoid(Opt &opt): Surface(opt) { }

   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(SDIM);
      // u in [0,2π] and v in [-2π/3,2π/3]
      const double u = 2.0*PI*x[0];
      const double v = 2.0*PI*(2.0*x[1]-1.0)/3.0;
      p[0] = sin(u)*v;
      p[1] = cos(u)*v;
      p[2] = u;
   }
};

// #2: Enneper's surface
struct Enneper: public Surface<Enneper>
{
   Enneper(Opt &opt): Surface(opt) { }

   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(SDIM);
      // (u,v) in [-2, +2]
      const double u = 4.0*(x[0]-0.5);
      const double v = 4.0*(x[1]-0.5);
      p[0] = +u - u*u*u/3.0 + u*v*v;
      p[1] = -v - u*u*v + v*v*v/3.0;
      p[2] = u*u - v*v;
   }
};

// #3: Scherk's doubly periodic surface
struct Scherk: public Surface<Scherk>
{
   Scherk(Opt &opt): Surface(opt) { }

   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(SDIM);
      const double alpha = 0.49;
      // (u,v) in [-απ, +απ]
      const double u = alpha*PI*(2.0*x[0]-1.0);
      const double v = alpha*PI*(2.0*x[1]-1.0);
      p[0] = u;
      p[1] = v;
      p[2] = log(cos(v)/cos(u));
   }
};

// #4: Hold surface
struct Hold: public Surface<Hold>
{
   Hold(Opt &opt): Surface(opt) { }

   void Prefix()
   {
      SetCurvature(opt.order, false, SDIM, Ordering::byNODES);
      Array<int> v2v(GetNV());
      for (int i = 0; i < v2v.Size(); i++) { v2v[i] = i; }
      // identify vertices on vertical lines
      for (int j = 0; j <= opt.ny; j++)
      {
         const int v_old = opt.nx + j * (opt.nx + 1);
         const int v_new =          j * (opt.nx + 1);
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
   QuarterPeach(Opt &opt): Surface(opt) { }

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
   static constexpr int NV = 8;
   static constexpr int NE = 6;
   static constexpr int NBE = 6;

   FullPeach(Opt &opt): Surface(opt, NV, NE, NBE) { opt.max_iter = 4; }

   void Prefix()
   {
      const double quad_v[NV][SDIM] =
      {
         {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
         {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}
      };
      const int quad_e[NE][4] =
      {
         {3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
         {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}

      };
      for (int j = 0; j < NV; j++)  { AddVertex(quad_v[j]); }
      for (int j = 0; j < NE; j++)  { AddQuad(quad_e[j], j+1); }
      for (int j = 0; j < NBE; j++) { AddBdrQuad(quad_e[j], j+1); }

      RemoveUnusedVertices();
      FinalizeQuadMesh(true, 0, true);
      FinalizeTopology();
      UniformRefinement();
   }

   void Create() { /* no Transform call */ }

   void Snap() { SnapToUnitSphere(); }

   void BC()
   {
      double X[SDIM];
      Array<int> dofs;
      Array<int> ess_cdofs, ess_tdofs;
      ess_cdofs.SetSize(fes->GetVSize());
      ess_cdofs = 0;
      fes->GetMesh()->GetNodes()->HostRead();
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
            for (int d = 0; d < 3; d++)
            { ess_cdofs[fes->DofToVDof(k, d)] = is_on_bc; }
         }
      }
      const SparseMatrix *R = fes->GetRestrictionMatrix();
      if (!R) { ess_tdofs.MakeRef(ess_cdofs); }
      else { R->BooleanMult(ess_cdofs, ess_tdofs); }
      FiniteElementSpace::MarkerToList(ess_tdofs, opt.bc);
   }
};

// #7: Full Peach street model
struct SlottedSphere: public Surface<SlottedSphere>
{
   SlottedSphere(Opt &opt): Surface(opt, 64, 40, 0) { opt.max_iter = 4; }

   void Prefix()
   {
      constexpr double delta = 0.15;
      constexpr int NV1D = 4;
      constexpr int NV = NV1D*NV1D*NV1D;
      constexpr int NEPF = (NV1D-1)*(NV1D-1);
      constexpr int NE = NEPF*6;
      const double V1D[NV1D] = {-1.0, -delta, delta, 1.0};
      double QV[NV][3];
      for (int iv=0; iv<NV; ++iv)
      {
         int ix = iv % NV1D;
         int iy = (iv / NV1D) % NV1D;
         int iz = (iv / NV1D) / NV1D;

         QV[iv][0] = V1D[ix];
         QV[iv][1] = V1D[iy];
         QV[iv][2] = V1D[iz];
      }
      int QE[NE][4];
      for (int ix=0; ix<NV1D-1; ++ix)
      {
         for (int iy=0; iy<NV1D-1; ++iy)
         {
            int el_offset = ix + iy*(NV1D-1);
            // x = 0
            QE[0*NEPF + el_offset][0] = NV1D*ix + NV1D*NV1D*iy;
            QE[0*NEPF + el_offset][1] = NV1D*(ix+1) + NV1D*NV1D*iy;
            QE[0*NEPF + el_offset][2] = NV1D*(ix+1) + NV1D*NV1D*(iy+1);
            QE[0*NEPF + el_offset][3] = NV1D*ix + NV1D*NV1D*(iy+1);
            // x = 1
            int x_off = NV1D-1;
            QE[1*NEPF + el_offset][3] = x_off + NV1D*ix + NV1D*NV1D*iy;
            QE[1*NEPF + el_offset][2] = x_off + NV1D*(ix+1) + NV1D*NV1D*iy;
            QE[1*NEPF + el_offset][1] = x_off + NV1D*(ix+1) + NV1D*NV1D*(iy+1);
            QE[1*NEPF + el_offset][0] = x_off + NV1D*ix + NV1D*NV1D*(iy+1);
            // y = 0
            QE[2*NEPF + el_offset][0] = NV1D*NV1D*iy + ix;
            QE[2*NEPF + el_offset][1] = NV1D*NV1D*iy + ix + 1;
            QE[2*NEPF + el_offset][2] = NV1D*NV1D*(iy+1) + ix + 1;
            QE[2*NEPF + el_offset][3] = NV1D*NV1D*(iy+1) + ix;
            // y = 1
            int y_off = NV1D*(NV1D-1);
            QE[3*NEPF + el_offset][0] = y_off + NV1D*NV1D*iy + ix;
            QE[3*NEPF + el_offset][1] = y_off + NV1D*NV1D*iy + ix + 1;
            QE[3*NEPF + el_offset][2] = y_off + NV1D*NV1D*(iy+1) + ix + 1;
            QE[3*NEPF + el_offset][3] = y_off + NV1D*NV1D*(iy+1) + ix;
            // z = 0
            QE[4*NEPF + el_offset][0] = NV1D*iy + ix;
            QE[4*NEPF + el_offset][1] = NV1D*iy + ix + 1;
            QE[4*NEPF + el_offset][2] = NV1D*(iy+1) + ix + 1;
            QE[4*NEPF + el_offset][3] = NV1D*(iy+1) + ix;
            // z = 1
            int z_off = NV1D*NV1D*(NV1D-1);
            QE[5*NEPF + el_offset][0] = z_off + NV1D*iy + ix;
            QE[5*NEPF + el_offset][1] = z_off + NV1D*iy + ix + 1;
            QE[5*NEPF + el_offset][2] = z_off + NV1D*(iy+1) + ix + 1;
            QE[5*NEPF + el_offset][3] = z_off + NV1D*(iy+1) + ix;
         }
      }
      // Delete on x = 0 face
      QE[0*NEPF + 1 + 2*(NV1D-1)][0] = -1;
      QE[0*NEPF + 1 + 1*(NV1D-1)][0] = -1;
      // Delete on x = 1 face
      QE[1*NEPF + 1 + 2*(NV1D-1)][0] = -1;
      QE[1*NEPF + 1 + 1*(NV1D-1)][0] = -1;
      // Delete on y = 1 face
      QE[3*NEPF + 1 + 0*(NV1D-1)][0] = -1;
      QE[3*NEPF + 1 + 1*(NV1D-1)][0] = -1;
      // Delete on z = 1 face
      QE[5*NEPF + 0 + 1*(NV1D-1)][0] = -1;
      QE[5*NEPF + 1 + 1*(NV1D-1)][0] = -1;
      QE[5*NEPF + 2 + 1*(NV1D-1)][0] = -1;
      // Delete on z = 0 face
      QE[4*NEPF + 1 + 0*(NV1D-1)][0] = -1;
      QE[4*NEPF + 1 + 1*(NV1D-1)][0] = -1;
      QE[4*NEPF + 1 + 2*(NV1D-1)][0] = -1;
      // Delete on y = 0 face
      QE[2*NEPF + 1 + 0*(NV1D-1)][0] = -1;
      QE[2*NEPF + 1 + 1*(NV1D-1)][0] = -1;
      for (int j = 0; j < NV; j++) { AddVertex(QV[j]); }
      for (int j = 0; j < NE; j++)
      {
         if (QE[j][0] < 0) { continue; }
         AddQuad(QE[j], j+1);
      }
      RemoveUnusedVertices();
      FinalizeQuadMesh(1, 1, true);
      EnsureNodes();
      FinalizeTopology();
   }

   void Create() { /* no Transform call */ }

   void Snap() { SnapToUnitSphere(); }
};

// #8: Costa minimal surface
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
static double ALPHA[3] {0.0};
struct Costa: public Surface<Costa>
{
   Costa(Opt &opt): Surface(opt) { }

   void Prefix()
   {
      const int nx = opt.nx, ny = opt.ny;
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
      FinalizeQuadMesh(true, 0, true);
      FinalizeTopology();
      SetCurvature(opt.order, false, SDIM, Ordering::byNODES);
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
      if (y_top) { p[1] *= -1.0; }
      if (x_top) { p[0] *= -1.0; }
      const bool nan = isnan(p[0])||isnan(p[1])||isnan(p[2]);
      MFEM_VERIFY(!nan, "nan");
      ALPHA[0] = fmax(p[0], ALPHA[0]);
      ALPHA[1] = fmax(p[1], ALPHA[1]);
      ALPHA[2] = fmax(p[2], ALPHA[2]);
   }

   void Snap()
   {
      Vector node(SDIM);
      MFEM_VERIFY(ALPHA[0] > 0.0,"");
      MFEM_VERIFY(ALPHA[1] > 0.0,"");
      MFEM_VERIFY(ALPHA[2] > 0.0,"");
      GridFunction &nodes = *GetNodes();
      const double phi = (1.0 + sqrt(5.0)) / 2.0;
      for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
      {
         for (int d = 0; d < SDIM; d++)
         {
            const double alpha = d==2 ? phi : 1.0;
            const int vdof = nodes.FESpace()->DofToVDof(i, d);
            nodes(vdof) /= alpha * ALPHA[d];
         }
      }
   }
};

// #9: Shell surface model
struct Shell: public Surface<Shell>
{
   Shell(Opt &opt): Surface(opt) { opt.max_iter = 0; }

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

// Visualize some solution on the given mesh
static void Visualize(const Opt &opt, const int w, const int h)
{
   glvis << "parallel " << NRanks << " " << MyRank << "\n";
   const GridFunction *x = opt.mesh->GetNodes();
   glvis << "solution\n" << *opt.mesh << *x;
   glvis << "window_size " << w << " " << h <<"\n";
   glvis << "keys " << opt.keys << "\n";
   glvis.precision(8);
   glvis << flush;
}

static void Visualize(const Opt &opt)
{
   glvis << "parallel " << NRanks << " " << MyRank << "\n";
   const GridFunction *x = opt.mesh->GetNodes();
   glvis << "solution\n" << *opt.mesh << *x;
   if (opt.wait) { glvis << "pause\n"; }
   glvis << flush;
}

// Surface solver class
template<class Type> class SurfaceSolver
{
protected:
   const Opt &opt;
   Mesh *mesh;
   Vector X, B;
   OperatorPtr A;
   FiniteElementSpace *fes;
   BilinearForm a;
   GridFunction x, x0, b;
   ConstantCoefficient one;
   Type *solver;
   Solver *M;
   const int print_iter = -1, max_num_iter = 2000;
   const double RTOLERANCE = EPS, ATOLERANCE = EPS*EPS;
public:
   SurfaceSolver(const Opt &opt):
      opt(opt), mesh(opt.mesh), fes(opt.fes),
      a(fes), x(fes), x0(fes), b(fes), one(1.0),
      solver(static_cast<Type*>(this)), M(nullptr) { }

   ~SurfaceSolver() { delete M; }

   void Solve()
   {
      if (opt.pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL);}
      for (int i=0; i < opt.max_iter; ++i)
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

   bool ParAXeqB()
   {
      const bool by_component = opt.solve_by_components;
      b = 0.0;
      a.FormLinearSystem(opt.bc, x, b, A, X, B);
      CGSolver cg;
      cg.SetPrintLevel(print_iter);
      cg.SetMaxIter(max_num_iter);
      cg.SetRelTol(RTOLERANCE);
      cg.SetAbsTol(ATOLERANCE);
      if (!opt.pa) { M = new GSSmoother((SparseMatrix&)(*A)); }
      if (M) { cg.SetPreconditioner(*M); }
      cg.SetOperator(*A);
      cg.Mult(B, X);
      a.RecoverFEMSolution(X, b, x);
      GridFunction *nodes = by_component ? &x0 : fes->GetMesh()->GetNodes();
      x.HostRead();
      nodes->HostRead();
      double rnorm = nodes->DistanceTo(x) / nodes->Norml2();
      mfem::out << "rnorm = " << rnorm << endl;
      const double lambda = opt.lambda;
      if (opt.radial)
      {
         GridFunction delta(fes);
         subtract(x, *nodes, delta); // Δ = x - nodes
         // position and Δ vectors at point i
         Vector ni(SDIM), di(SDIM);
         for (int i = 0; i < delta.Size()/SDIM; i++)
         {
            // extract local vectors
            const int ndof = fes->GetNDofs();
            for (int d = 0; d < SDIM; d++)
            {
               ni(d) = (*nodes)(d*ndof + i);
               di(d) = delta(d*ndof + i);
            }
            // project the delta vector in radial direction
            const double ndotd = (ni*di) / (ni*ni);
            di.Set(ndotd,ni);
            // set global vectors
            for (int d = 0; d < SDIM; d++) { delta(d*ndof + i) = di(d); }
         }
         add(*nodes, delta, *nodes);
      }
      // x = λ*nodes + (1-λ)*x
      add(lambda, *nodes, (1.0-lambda), x, x);
      return Converged(rnorm);
   }

   void Update()
   {
      if (opt.vis) { Visualize(opt); }
      mesh->DeleteGeometricFactors();
      a.Update();
   }
};

// Surface solver 'by vector'
class ByVector: public SurfaceSolver<ByVector>
{
public:
   ByVector(Opt &opt): SurfaceSolver(opt)
   { a.AddDomainIntegrator(new VectorDiffusionIntegrator(one)); }

   bool Loop()
   {
      x = *fes->GetMesh()->GetNodes();
      bool converge = this->ParAXeqB();
      mesh->SetNodes(x);
      return converge ? true : false;
   }
};

// Surface solver 'by compnents'
class ByComponent: public SurfaceSolver<ByComponent>
{
public:
   void SetNodes(const GridFunction &Xi, const int c)
   {
      auto d_Xi = Xi.Read();
      auto d_nodes  = fes->GetMesh()->GetNodes()->Write();
      const int ndof = fes->GetNDofs();
      MFEM_FORALL(i, ndof, d_nodes[c*ndof + i] = d_Xi[i]; );
   }

   void GetNodes(GridFunction &Xi, const int c)
   {
      auto d_Xi = Xi.Write();
      const int ndof = fes->GetNDofs();
      auto d_nodes  = fes->GetMesh()->GetNodes()->Read();
      MFEM_FORALL(i, ndof, d_Xi[i] = d_nodes[c*ndof + i]; );
   }

public:
   ByComponent(Opt &opt): SurfaceSolver(opt)
   { a.AddDomainIntegrator(new DiffusionIntegrator(one)); }

   bool Loop()
   {
      bool cvg[SDIM] {false};
      for (int c=0; c < SDIM; ++c)
      {
         this->GetNodes(x, c);
         x0 = x;
         cvg[c] = this->ParAXeqB();
         this->SetNodes(x, c);
      }
      const bool converged = cvg[0] && cvg[1] && cvg[2];
      return converged ? true : false;
   }
};


Mesh *CreateSurface(Opt &opt)
{
   Mesh *S = nullptr;
   const int s = opt.surface;
   if (s< 0) { S = new MeshFromFile(opt); }
   if (s==0) { S = new Catenoid(opt); }
   if (s==1) { S = new Helicoid(opt); }
   if (s==2) { S = new Enneper(opt); }
   if (s==3) { S = new Scherk(opt); }
   if (s==4) { S = new Hold(opt); }
   if (s==5) { S = new QuarterPeach(opt); }
   if (s==6) { S = new FullPeach(opt); }
   if (s==7) { S = new SlottedSphere(opt); }
   if (s==8) { S = new Costa(opt); }
   if (s==9) { S = new Shell(opt); }
   MFEM_VERIFY(S, "Unknown surface (0 <= surface <= 9)!");
   // Grab back the mesh & fes from the Surface object.
   opt.mesh = static_cast<Surface<>*>(S)->GetMesh();
   opt.fes = static_cast<Surface<>*>(S)->GetFESpace();
   return S;
}

int main(int argc, char *argv[])
{
   NRanks = 1; MyRank = 0;
   // Parse command-line options.
   Opt opt;
   OptionsParser args(argc, argv);
   args.AddOption(&opt.mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&opt.wait, "-w", "--wait", "-no-w", "--no-wait",
                  "Enable or disable a GLVis pause.");
   args.AddOption(&opt.radial, "-rad", "--radial", "-no-rad", "--no-radial",
                  "Enable or disable radial constraints in solver.");
   args.AddOption(&opt.nx, "-x", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&opt.ny, "-y", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&opt.order, "-o", "--order", "Finite element order.");
   args.AddOption(&opt.refine, "-r", "--ref-levels", "Refinement");
   args.AddOption(&opt.max_iter, "-n", "--niter-max", "Max number of iterations");
   args.AddOption(&opt.surface, "-s", "--surface", "Choice of the surface.");
   args.AddOption(&opt.pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&opt.lambda, "-l", "--lambda", "Lambda step toward solution.");
   args.AddOption(&opt.amr, "-a", "--amr", "-no-a", "--no-amr", "Enable AMR.");
   args.AddOption(&opt.device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&opt.keys, "-k", "--keys", "GLVis configuration keys.");
   args.AddOption(&opt.vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable visualization.");
   args.AddOption(&opt.solve_by_components, "-c", "--components",
                  "-no-c", "--no-components",
                  "Enable or disable the 'by component' solver");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   MFEM_VERIFY(opt.lambda >= 0.0 && opt.lambda <=1.0,"");
   opt.vdim = opt.solve_by_components ? 1 : SDIM;
   if (MyRank == 0) { args.PrintOptions(cout); }

   // Initialize hardware devices.
   Device device(opt.device_config);
   if (MyRank == 0) { device.Print(); }

   // Initialize GLVis server if 'visualization' is set.
   if (opt.vis) { opt.vis = glvis.open(vishost, visport) == 0; }

   // Create our surface mesh from command line option.
   Mesh *S = CreateSurface(opt);

   // Send to GLVis the first mesh.
   if (opt.vis) { Visualize(opt, GLVIZ_W, GLVIZ_H); }

   // Create and launch the surface solver.
   if (opt.solve_by_components) { ByComponent(opt).Solve(); }
   else { ByVector(opt).Solve(); }

   // Free the used memory.
   delete S;
   return 0;
}
