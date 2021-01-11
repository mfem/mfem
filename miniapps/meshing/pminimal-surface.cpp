// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//     --------------------------------------------------------------------
//     Minimal Surface Miniapp: Compute minimal surfaces - Parallel Version
//     --------------------------------------------------------------------
//
// This miniapp solves Plateau's problem: the Dirichlet problem for the minimal
// surface equation.
//
// Two problems can be run:
//
//  - Problem 0 solves the minimal surface equation of parametric surfaces.
//              The surface (-s) option allow the selection of different
//              parametrization:
//                s=0: Uses the given mesh from command line options
//                s=1: Catenoid
//                s=2: Helicoid
//                s=3: Enneper
//                s=4: Hold
//                s=5: Costa
//                s=6: Shell
//                s=7: Scherk
//                s=8: FullPeach
//                s=9: QuarterPeach
//               s=10: SlottedSphere
//
//  - Problem 1 solves the minimal surface equation of the form z=f(x,y),
//              for the Dirichlet problem, using Picard iterations:
//              -div( q grad u) = 0, with q(u) = (1 + |∇u|²)^{-1/2}
//
// Compile with: make pminimal-surface
//
// Sample runs:  mpirun -np 4 pminimal-surface
//               mpirun -np 4 pminimal-surface -a
//               mpirun -np 4 pminimal-surface -c
//               mpirun -np 4 pminimal-surface -c -a
//               mpirun -np 4 pminimal-surface -no-pa
//               mpirun -np 4 pminimal-surface -no-pa -a
//               mpirun -np 4 pminimal-surface -no-pa -a -c
//               mpirun -np 4 pminimal-surface -p 1
//
// Device sample runs:
//               mpirun -np 4 pminimal-surface -d debug
//               mpirun -np 4 pminimal-surface -d debug -a
//               mpirun -np 4 pminimal-surface -d debug -c
//               mpirun -np 4 pminimal-surface -d debug -c -a
//               mpirun -np 4 pminimal-surface -d  cuda
//               mpirun -np 4 pminimal-surface -d  cuda -a
//               mpirun -np 4 pminimal-surface -d  cuda -c
//               mpirun -np 4 pminimal-surface -d  cuda -c -a
//               mpirun -np 4 pminimal-surface -d  cuda -no-pa
//               mpirun -np 4 pminimal-surface -d  cuda -no-pa -a
//               mpirun -np 4 pminimal-surface -d  cuda -no-pa -c
//               mpirun -np 4 pminimal-surface -d  cuda -no-pa -c -a

#include "mfem.hpp"
#include "../../general/forall.hpp"

using namespace mfem;

// Constant variables
constexpr int DIM = 2;
constexpr int SDIM = 3;
constexpr double PI = M_PI;
constexpr double NRM = 1.e-4;
constexpr double EPS = 1.e-14;
constexpr Element::Type QUAD = Element::QUADRILATERAL;
constexpr double NL_DMAX = std::numeric_limits<double>::max();

// Static variables for GLVis
static socketstream glvis;
constexpr int GLVIZ_W = 1024;
constexpr int GLVIZ_H = 1024;
constexpr int  visport = 19916;
constexpr char vishost[] = "localhost";

// Context/Options for the solver
struct Opt
{
   int sz, id;
   int pb = 0;
   int nx = 6;
   int ny = 6;
   int order = 3;
   int refine = 2;
   int niters = 8;
   int surface = 5;
   bool pa = true;
   bool vis = true;
   bool amr = false;
   bool wait = false;
   bool print = false;
   bool radial = false;
   bool by_vdim = false;
   bool snapshot = false;
   // bool vis_mesh = false; // Not supported by GLVis
   double tau = 1.0;
   double lambda = 0.1;
   double amr_threshold = 0.6;
   const char *keys = "gAm";
   const char *device_config = "cpu";
   const char *mesh_file = "../../data/mobius-strip.mesh";
   void (*Tptr)(const Vector&, Vector&) = nullptr;
};

class Surface: public Mesh
{
protected:
   Opt &opt;
   ParMesh *mesh;
   Array<int> bc;
   H1_FECollection *fec;
   ParFiniteElementSpace *fes;
public:
   // Reading from mesh file
   Surface(Opt &opt, const char *file): Mesh(file, true), opt(opt) { }

   // Generate 2D empty surface mesh
   Surface(Opt &opt, bool): Mesh(), opt(opt) { }

   // Generate 2D quad surface mesh
   Surface(Opt &opt): Mesh(opt.nx, opt.ny, QUAD, true), opt(opt) { }

   // Generate 2D generic surface mesh
   Surface(Opt &opt, int nv, int ne, int nbe):
      Mesh(DIM, nv, ne, nbe, SDIM), opt(opt) { }

   void Create()
   {
      if (opt.surface > 0)
      {
         Prefix();
         Transform();
      }
      Postfix();
      Refine();
      Snap();
      fec = new H1_FECollection(opt.order, DIM);
      if (opt.amr) { EnsureNCMesh(); }
      mesh = new ParMesh(MPI_COMM_WORLD, *this);
      fes = new ParFiniteElementSpace(mesh, fec, opt.by_vdim ? 1 : SDIM);
      BoundaryConditions();
   }

   int Solve()
   {
      // Initialize GLVis server if 'visualization' is set
      if (opt.vis) { opt.vis = glvis.open(vishost, visport) == 0; }
      // Send to GLVis the first mesh
      if (opt.vis) { Visualize(opt, mesh, GLVIZ_W, GLVIZ_H); }
      // Create and launch the surface solver
      if (opt.by_vdim)
      {
         ByVDim(*this, opt).Solve();
      }
      else
      {
         ByNodes(*this, opt).Solve();
      }
      if (opt.vis && opt.snapshot)
      {
         opt.keys = "Sq";
         Visualize(opt, mesh, mesh->GetNodes());
      }
      return 0;
   }

   ~Surface()
   {
      if (opt.vis) { glvis.close(); }
      delete mesh; delete fec; delete fes;
   }

   virtual void Prefix()
   {
      SetCurvature(opt.order, false, SDIM, Ordering::byNODES);
   }

   virtual void Transform() { if (opt.Tptr) { Mesh::Transform(opt.Tptr); } }

   virtual void Postfix()
   {
      SetCurvature(opt.order, false, SDIM, Ordering::byNODES);
   }

   virtual void Refine()
   {
      for (int l = 0; l < opt.refine; l++)
      {
         UniformRefinement();
      }
   }

   virtual void Snap()
   {
      GridFunction &nodes = *GetNodes();
      for (int i = 0; i < nodes.Size(); i++)
      {
         if (std::abs(nodes(i)) < EPS)
         {
            nodes(i) = 0.0;
         }
      }
   }

   void SnapNodesToUnitSphere()
   {
      Vector node(SDIM);
      GridFunction &nodes = *GetNodes();
      for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
      {
         for (int d = 0; d < SDIM; d++)
         {
            node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));
         }
         node /= node.Norml2();
         for (int d = 0; d < SDIM; d++)
         {
            nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
         }
      }
   }

   virtual void BoundaryConditions()
   {
      if (bdr_attributes.Size())
      {
         Array<int> ess_bdr(bdr_attributes.Max());
         ess_bdr = 1;
         bc.HostReadWrite();
         fes->GetEssentialTrueDofs(ess_bdr, bc);
      }
   }

   // Initialize visualization of some given mesh
   static void Visualize(Opt &opt, const Mesh *mesh,
                         const int w, const int h,
                         const GridFunction *sol = nullptr)
   {
      const GridFunction &solution = sol ? *sol : *mesh->GetNodes();
      glvis << "parallel " << opt.sz << " " << opt.id << "\n";
      glvis << "solution\n" << *mesh << solution;
      glvis.precision(8);
      glvis << "window_size " << w << " " << h << "\n";
      glvis << "keys " << opt.keys << "\n";
      opt.keys = nullptr;
      if (opt.wait) { glvis << "pause\n"; }
      glvis << std::flush;
   }

   // Visualize some solution on the given mesh
   static void Visualize(const Opt &opt, const Mesh *mesh,
                         const GridFunction *sol = nullptr)
   {
      glvis << "parallel " << opt.sz << " " << opt.id << "\n";
      const GridFunction &solution = sol ? *sol : *mesh->GetNodes();
      glvis << "solution\n" << *mesh << solution;
      if (opt.wait) { glvis << "pause\n"; }
      if (opt.snapshot && opt.keys) { glvis << "keys " << opt.keys << "\n"; }
      glvis << std::flush;
   }

   using Mesh::Print;
   static void Print(const Opt &opt, ParMesh *mesh, const GridFunction *sol)
   {
      const char *mesh_file = "surface.mesh";
      const char *sol_file = "sol.gf";
      if (!opt.id)
      {
         mfem::out << "Printing " << mesh_file << ", " << sol_file << std::endl;
      }

      std::ostringstream mesh_name;
      mesh_name << mesh_file << "." << std::setfill('0') << std::setw(6) << opt.id;
      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      mesh_ofs.close();

      std::ostringstream sol_name;
      sol_name << sol_file << "." << std::setfill('0') << std::setw(6) << opt.id;
      std::ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      sol->Save(sol_ofs);
      sol_ofs.close();
   }

   // Surface Solver class
   class Solver
   {
   protected:
      Opt &opt;
      Surface &S;
      CGSolver cg;
      OperatorPtr A;
      ParBilinearForm a;
      ParGridFunction x, x0, b;
      ConstantCoefficient one;
      mfem::Solver *M = nullptr;
      const int print_iter = -1, max_num_iter = 2000;
      const double RTOLERANCE = EPS, ATOLERANCE = EPS*EPS;
   public:
      Solver(Surface &S, Opt &opt): opt(opt), S(S), cg(MPI_COMM_WORLD),
         a(S.fes), x(S.fes), x0(S.fes), b(S.fes), one(1.0)
      {
         cg.SetRelTol(RTOLERANCE);
         cg.SetAbsTol(ATOLERANCE);
         cg.SetMaxIter(max_num_iter);
         cg.SetPrintLevel(print_iter);
      }

      ~Solver() { delete M; }

      void Solve()
      {
         constexpr bool converged = true;
         if (opt.pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL);}
         for (int i=0; i < opt.niters; ++i)
         {
            if (opt.amr) { Amr(); }
            if (opt.vis) { Surface::Visualize(opt, S.mesh); }
            if (!opt.id) { mfem::out << "Iteration " << i << ": "; }
            S.mesh->DeleteGeometricFactors();
            a.Update();
            a.Assemble();
            if (Step() == converged) { break; }
         }
         if (opt.print) { Surface::Print(opt, S.mesh, S.mesh->GetNodes()); }
      }

      virtual bool Step() = 0;

   protected:
      bool Converged(const double rnorm) { return rnorm < NRM; }

      bool ParAXeqB()
      {
         b = 0.0;
         Vector X, B;
         a.FormLinearSystem(S.bc, x, b, A, X, B);
         if (!opt.pa) { M = new HypreBoomerAMG; }
         if (M) { cg.SetPreconditioner(*M); }
         cg.SetOperator(*A);
         cg.Mult(B, X);
         a.RecoverFEMSolution(X, b, x);
         const bool by_vdim = opt.by_vdim;
         GridFunction *nodes = by_vdim ? &x0 : S.fes->GetMesh()->GetNodes();
         x.HostReadWrite();
         nodes->HostRead();
         double rnorm = nodes->DistanceTo(x) / nodes->Norml2();
         double glob_norm;
         MPI_Allreduce(&rnorm, &glob_norm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         rnorm = glob_norm;
         if (!opt.id) { mfem::out << "rnorm = " << rnorm << std::endl; }
         const double lambda = opt.lambda;
         if (by_vdim)
         {
            MFEM_VERIFY(!opt.radial,"'VDim solver can't use radial option!");
            return Converged(rnorm);
         }
         if (opt.radial)
         {
            GridFunction delta(S.fes);
            subtract(x, *nodes, delta); // delta = x - nodes
            // position and Δ vectors at point i
            Vector ni(SDIM), di(SDIM);
            for (int i = 0; i < delta.Size()/SDIM; i++)
            {
               // extract local vectors
               const int ndof = S.fes->GetNDofs();
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
         // x = lambda*nodes + (1-lambda)*x
         add(lambda, *nodes, (1.0-lambda), x, x);
         return Converged(rnorm);
      }

      void Amr()
      {
         MFEM_VERIFY(opt.amr_threshold >= 0.0 && opt.amr_threshold <= 1.0, "");
         Mesh *mesh = S.mesh;
         Array<Refinement> amr;
         const int NE = mesh->GetNE();
         DenseMatrix Jadjt, Jadj(DIM, SDIM);
         for (int e = 0; e < NE; e++)
         {
            double minW = +NL_DMAX;
            double maxW = -NL_DMAX;
            ElementTransformation *eTr = mesh->GetElementTransformation(e);
            const Geometry::Type &type = mesh->GetElement(e)->GetGeometryType();
            const IntegrationRule *ir = &IntRules.Get(type, opt.order);
            const int NQ = ir->GetNPoints();
            for (int q = 0; q < NQ; q++)
            {
               eTr->SetIntPoint(&ir->IntPoint(q));
               const DenseMatrix &J = eTr->Jacobian();
               CalcAdjugate(J, Jadj);
               Jadjt = Jadj;
               Jadjt.Transpose();
               const double w = Jadjt.Weight();
               minW = std::fmin(minW, w);
               maxW = std::fmax(maxW, w);
            }
            if (std::fabs(maxW) != 0.0)
            {
               const double rho = minW / maxW;
               MFEM_VERIFY(rho <= 1.0, "");
               if (rho < opt.amr_threshold) { amr.Append(Refinement(e)); }
            }
         }
         if (amr.Size()>0)
         {
            mesh->GetNodes()->HostReadWrite();
            mesh->GeneralRefinement(amr);
            S.fes->Update();
            x.HostReadWrite();
            x.Update();
            a.Update();
            b.HostReadWrite();
            b.Update();
            S.BoundaryConditions();
         }
      }
   };

   // Surface solver 'by vector'
   class ByNodes: public Solver
   {
   public:
      ByNodes(Surface &S, Opt &opt): Solver(S, opt)
      { a.AddDomainIntegrator(new VectorDiffusionIntegrator(one)); }

      bool Step()
      {
         x = *S.fes->GetMesh()->GetNodes();
         bool converge = ParAXeqB();
         S.mesh->SetNodes(x);
         return converge ? true : false;
      }
   };

   // Surface solver 'by ByVDim'
   class ByVDim: public Solver
   {
   public:
      void SetNodes(const GridFunction &Xi, const int c)
      {
         auto d_Xi = Xi.Read();
         auto d_nodes  = S.fes->GetMesh()->GetNodes()->Write();
         const int ndof = S.fes->GetNDofs();
         MFEM_FORALL(i, ndof, d_nodes[c*ndof + i] = d_Xi[i]; );
      }

      void GetNodes(GridFunction &Xi, const int c)
      {
         auto d_Xi = Xi.Write();
         const int ndof = S.fes->GetNDofs();
         auto d_nodes  = S.fes->GetMesh()->GetNodes()->Read();
         MFEM_FORALL(i, ndof, d_Xi[i] = d_nodes[c*ndof + i]; );
      }

      ByVDim(Surface &S, Opt &opt): Solver(S, opt)
      { a.AddDomainIntegrator(new DiffusionIntegrator(one)); }

      bool Step()
      {
         bool cvg[SDIM] {false};
         for (int c=0; c < SDIM; ++c)
         {
            GetNodes(x, c);
            x0 = x;
            cvg[c] = ParAXeqB();
            SetNodes(x, c);
         }
         const bool converged = cvg[0] && cvg[1] && cvg[2];
         return converged ? true : false;
      }
   };
};

// #0: Default surface mesh file
struct MeshFromFile: public Surface
{
   MeshFromFile(Opt &opt): Surface(opt, opt.mesh_file) { }
};

// #1: Catenoid surface
struct Catenoid: public Surface
{
   Catenoid(Opt &opt): Surface((opt.Tptr = Parametrization, opt)) { }

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
         {
            v[j] = v2v[v[j]];
         }
      }
      // renumber boundary elements
      for (int i = 0; i < GetNBE(); i++)
      {
         Element *el = GetBdrElement(i);
         int *v = el->GetVertices();
         const int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
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

// #2: Helicoid surface
struct Helicoid: public Surface
{
   Helicoid(Opt &opt): Surface((opt.Tptr = Parametrization, opt)) { }

   static void Parametrization(const Vector &x, Vector &p)
   {
      p.SetSize(SDIM);
      // u in [0,2π] and v in [-2π/3,2π/3]
      const double u = 2.0*PI*x[0];
      const double v = 2.0*PI*(2.0*x[1]-1.0)/3.0;
      p(0) = sin(u)*v;
      p(1) = cos(u)*v;
      p(2) = u;
   }
};

// #3: Enneper's surface
struct Enneper: public Surface
{
   Enneper(Opt &opt): Surface((opt.Tptr = Parametrization, opt)) { }

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

// #4: Hold surface
struct Hold: public Surface
{
   Hold(Opt &opt): Surface((opt.Tptr = Parametrization, opt)) { }

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
         {
            v[j] = v2v[v[j]];
         }
      }
      // renumber boundary elements
      for (int i = 0; i < GetNBE(); i++)
      {
         Element *el = GetBdrElement(i);
         int *v = el->GetVertices();
         const int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
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

// #5: Costa minimal surface
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
static double ALPHA[4] {0.0};
struct Costa: public Surface
{
   Costa(Opt &opt): Surface((opt.Tptr = Parametrization, opt), false) { }

   void Prefix()
   {
      ALPHA[3] = opt.tau;
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
      FinalizeQuadMesh(false, 0, true);
      FinalizeTopology();
      SetCurvature(opt.order, false, SDIM, Ordering::byNODES);
   }

   static void Parametrization(const Vector &X, Vector &p)
   {
      const double tau = ALPHA[3];
      Vector x = X;
      x -= +0.5;
      x *= tau;
      x -= -0.5;

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
      const bool nan = std::isnan(p[0]) || std::isnan(p[1]) || std::isnan(p[2]);
      MFEM_VERIFY(!nan, "nan");
      ALPHA[0] = std::fmax(p[0], ALPHA[0]);
      ALPHA[1] = std::fmax(p[1], ALPHA[1]);
      ALPHA[2] = std::fmax(p[2], ALPHA[2]);
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

// #6: Shell surface model
struct Shell: public Surface
{
   Shell(Opt &opt): Surface((opt.niters = 1, opt.Tptr = Parametrization, opt)) { }

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

// #7: Scherk's doubly periodic surface
struct Scherk: public Surface
{
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

   Scherk(Opt &opt): Surface((opt.Tptr = Parametrization, opt)) { }
};

// #8: Full Peach street model
struct FullPeach: public Surface
{
   static constexpr int NV = 8;
   static constexpr int NE = 6;

   FullPeach(Opt &opt):
      Surface((opt.niters = std::min(4, opt.niters), opt), NV, NE, 0) { }

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

      FinalizeQuadMesh(false, 0, true);
      FinalizeTopology(false);
      UniformRefinement();
      SetCurvature(opt.order, false, SDIM, Ordering::byNODES);
   }

   void Snap() { SnapNodesToUnitSphere(); }

   void BoundaryConditions()
   {
      Vector X(SDIM);
      Array<int> dofs;
      Array<int> ess_vdofs, ess_tdofs;
      ess_vdofs.SetSize(fes->GetVSize());
      MFEM_VERIFY(fes->GetVSize() >= fes->GetTrueVSize(),"");
      ess_vdofs = 0;
      DenseMatrix PointMat;
      mesh->GetNodes()->HostRead();
      for (int e = 0; e < fes->GetNE(); e++)
      {
         fes->GetElementDofs(e, dofs);
         const IntegrationRule &ir = fes->GetFE(e)->GetNodes();
         ElementTransformation *eTr = mesh->GetElementTransformation(e);
         eTr->Transform(ir, PointMat);
         Vector one(dofs.Size());
         for (int dof = 0; dof < dofs.Size(); dof++)
         {
            one = 0.0;
            one[dof] = 1.0;
            const int k = dofs[dof];
            MFEM_ASSERT(k >= 0, "");
            PointMat.Mult(one, X);
            const bool halfX = std::fabs(X[0]) < EPS && X[1] <= 0.0;
            const bool halfY = std::fabs(X[2]) < EPS && X[1] >= 0.0;
            const bool is_on_bc = halfX || halfY;
            for (int c = 0; c < SDIM; c++)
            { ess_vdofs[fes->DofToVDof(k, c)] = is_on_bc; }
         }
      }
      const SparseMatrix *R = fes->GetRestrictionMatrix();
      if (!R)
      {
         ess_tdofs.MakeRef(ess_vdofs);
      }
      else
      {
         R->BooleanMult(ess_vdofs, ess_tdofs);
      }
      bc.HostReadWrite();
      ParFiniteElementSpace::MarkerToList(ess_tdofs, bc);
   }
};

// #9: 1/4th Peach street model
struct QuarterPeach: public Surface
{
   QuarterPeach(Opt &opt): Surface((opt.Tptr = Parametrization, opt)) { }

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
         if (std::fabs(X[0][1])<=EPS && std::fabs(X[1][1])<=EPS &&
             (R[0]>0.1 || R[1]>0.1))
         { el->SetAttribute(1); }
         else { el->SetAttribute(2); }
      }
   }
};

// #10: Slotted sphere mesh
struct SlottedSphere: public Surface
{
   SlottedSphere(Opt &opt): Surface((opt.niters = 4, opt), 64, 40, 0) { }

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
      FinalizeQuadMesh(false, 0, true);
      EnsureNodes();
      FinalizeTopology();
   }

   void Snap() { SnapNodesToUnitSphere(); }
};

static int Problem0(Opt &opt)
{
   // Create our surface mesh from command line options
   Surface *S = nullptr;
   switch (opt.surface)
   {
      case 0: S = new MeshFromFile(opt); break;
      case 1: S = new Catenoid(opt); break;
      case 2: S = new Helicoid(opt); break;
      case 3: S = new Enneper(opt); break;
      case 4: S = new Hold(opt); break;
      case 5: S = new Costa(opt); break;
      case 6: S = new Shell(opt); break;
      case 7: S = new Scherk(opt); break;
      case 8: S = new FullPeach(opt); break;
      case 9: S = new QuarterPeach(opt); break;
      case 10: S = new SlottedSphere(opt); break;
      default: MFEM_ABORT("Unknown surface (surface <= 10)!");
   }
   S->Create();
   S->Solve();
   delete S;
   return 0;
}

// Problem 1: solve the Dirichlet problem for the minimal surface equation
//            of the form z=f(x,y), using Picard iterations.
static double u0(const Vector &x) { return sin(3.0 * PI * (x[1] + x[0])); }

enum {NORM, AREA};

static double qf(const int order, const int ker, Mesh &m,
                 FiniteElementSpace &fes, GridFunction &u)
{
   const Geometry::Type type = m.GetElementBaseGeometry(0);
   const IntegrationRule &ir(IntRules.Get(type, order));
   const QuadratureInterpolator *qi(fes.GetQuadratureInterpolator(ir));

   const int NE(m.GetNE());
   const int ND(fes.GetFE(0)->GetDof());
   const int NQ(ir.GetNPoints());
   const int flags = GeometricFactors::JACOBIANS|GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = m.GetGeometricFactors(ir, flags);

   const int D1D = fes.GetFE(0)->GetOrder() + 1;
   const int Q1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
   MFEM_VERIFY(ND == D1D*D1D, "");
   MFEM_VERIFY(NQ == Q1D*Q1D, "");

   Vector Eu(ND*NE), grad_u(DIM*NQ*NE), sum(NE*NQ), one(NE*NQ);
   qi->SetOutputLayout(QVectorLayout::byVDIM);
   const ElementDofOrdering e_ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *G(fes.GetElementRestriction(e_ordering));
   G->Mult(u, Eu);
   qi->Derivatives(Eu, grad_u);

   auto W = Reshape(ir.GetWeights().Read(), Q1D, Q1D);
   auto J = Reshape(geom->J.Read(), Q1D, Q1D, DIM, DIM, NE);
   auto detJ = Reshape(geom->detJ.Read(), Q1D, Q1D, NE);
   auto grdU = Reshape(grad_u.Read(), DIM, Q1D, Q1D, NE);
   auto S = Reshape(sum.Write(), Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const double w = W(qx, qy);
            const double J11 = J(qx, qy, 0, 0, e);
            const double J12 = J(qx, qy, 1, 0, e);
            const double J21 = J(qx, qy, 0, 1, e);
            const double J22 = J(qx, qy, 1, 1, e);
            const double det = detJ(qx, qy, e);
            const double area = w * det;
            const double gu0 = grdU(0, qx, qy, e);
            const double gu1 = grdU(1, qx, qy, e);
            const double tgu0 = (J22*gu0 - J12*gu1)/det;
            const double tgu1 = (J11*gu1 - J21*gu0)/det;
            const double ngu = tgu0*tgu0 + tgu1*tgu1;
            const double s = (ker == AREA) ? sqrt(1.0 + ngu) :
            (ker == NORM) ? ngu : 0.0;
            S(qx, qy, e) = area * s;
         }
      }
   });
   one = 1.0;
   return sum * one;
}

static int Problem1(Opt &opt)
{
   const int order = opt.order;
   Mesh smesh(opt.nx, opt.ny, QUAD);
   smesh.SetCurvature(opt.order, false, DIM, Ordering::byNODES);
   for (int l = 0; l < opt.refine; l++) { smesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   const H1_FECollection fec(order, DIM);
   ParFiniteElementSpace fes(&mesh, &fec);
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   ParGridFunction uold(&fes), u(&fes), b(&fes);
   FunctionCoefficient u0_fc(u0);
   u.ProjectCoefficient(u0_fc);
   if (opt.vis) { opt.vis = glvis.open(vishost, visport) == 0; }
   if (opt.vis) { Surface::Visualize(opt, &mesh, GLVIZ_W, GLVIZ_H, &u); }
   Vector B, X;
   OperatorPtr A;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(EPS);
   cg.SetAbsTol(EPS*EPS);
   cg.SetMaxIter(400);
   cg.SetPrintLevel(0);
   ParGridFunction eps(&fes);
   for (int i = 0; i < opt.niters; i++)
   {
      b = 0.0;
      uold = u;
      ParBilinearForm a(&fes);
      if (opt.pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      const double q_uold = qf(order, AREA, mesh, fes, uold);
      MFEM_VERIFY(std::fabs(q_uold) > EPS,"");
      ConstantCoefficient q_uold_cc(1.0/sqrt(q_uold));
      a.AddDomainIntegrator(new DiffusionIntegrator(q_uold_cc));
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      a.RecoverFEMSolution(X, b, u);
      subtract(u, uold, eps);
      const double norm = sqrt(std::fabs(qf(order, NORM, mesh, fes, eps)));
      const double area = qf(order, AREA, mesh, fes, u);
      if (!opt.id)
      {
         mfem::out << "Iteration " << i << ", norm: " << norm
                   << ", area: " << area << std::endl;
      }
      if (opt.vis) { Surface::Visualize(opt, &mesh, &u); }
      if (opt.print) { Surface::Print(opt, &mesh, &u); }
      if (norm < NRM) { break; }
   }
   return 0;
}

int main(int argc, char *argv[])
{
   Opt opt;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &opt.id);
   MPI_Comm_size(MPI_COMM_WORLD, &opt.sz);
   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&opt.pb, "-p", "--problem", "Problem to solve.");
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
   args.AddOption(&opt.niters, "-n", "--niter-max", "Max number of iterations");
   args.AddOption(&opt.surface, "-s", "--surface", "Choice of the surface.");
   args.AddOption(&opt.pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&opt.tau, "-t", "--tau", "Costa scale factor.");
   args.AddOption(&opt.lambda, "-l", "--lambda", "Lambda step toward solution.");
   args.AddOption(&opt.amr, "-a", "--amr", "-no-a", "--no-amr", "Enable AMR.");
   args.AddOption(&opt.amr_threshold, "-at", "--amr-threshold", "AMR threshold.");
   args.AddOption(&opt.device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&opt.keys, "-k", "--keys", "GLVis configuration keys.");
   args.AddOption(&opt.vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable visualization.");
   args.AddOption(&opt.by_vdim, "-c", "--solve-byvdim",
                  "-no-c", "--solve-bynodes",
                  "Enable or disable the 'ByVdim' solver");
   args.AddOption(&opt.print, "-print", "--print", "-no-print", "--no-print",
                  "Enable or disable result output (files in mfem format).");
   args.AddOption(&opt.snapshot, "-ss", "--snapshot", "-no-ss", "--no-snapshot",
                  "Enable or disable GLVis snapshot.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(mfem::out); MPI_Finalize(); return 1; }
   MFEM_VERIFY(opt.lambda >= 0.0 && opt.lambda <= 1.0,"");
   if (!opt.id) { args.PrintOptions(mfem::out); }

   // Initialize hardware devices
   Device device(opt.device_config);
   if (!opt.id) { device.Print(); }

   if (opt.pb == 0) { Problem0(opt); }

   if (opt.pb == 1) { Problem1(opt); }

   MPI_Finalize();
   return 0;
}
