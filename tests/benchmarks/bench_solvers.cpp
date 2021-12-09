// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bench.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "fem/linearform.hpp"
#include "linalg/dtensor.hpp"

#define MFEM_NVTX_COLOR Orange
#include "general/nvtx.hpp"

#define MFEM_DEBUG_COLOR 226
#include "general/debug.hpp"

#include "linalg/wamg.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

/*
  This benchmark contains the implementation of the CEED's solver problems.
*/

namespace mfem
{

using LOR_AMG= LORSolver<HypreBoomerAMG>;

////////////////////////////////////////////////////////////////////////////////
static MPI_Session *mpi;
static int config_dev_size = 4; // default 4 GPU per node

static bool config_use_caliper = false;

static bool config_debug = false;
static int config_max_nic = 4;
static int config_max_nip = 4;
static int config_max_nif = 500;

enum MGSpecification
{
   INPUT = 0,
   INCREMENT = 1,
   DIVIDE_BY_2 = 2
};
static const char *config_mg_spec = nullptr;
static int config_mg_spec_select = MGSpecification::DIVIDE_BY_2;

////////////////////////////////////////////////////////////////////////////////
// The different fine preconditioners the tests can use
enum FinePrecondType
{
   None = 0,
   Jacobi,
   LORBatch,
   MGJacobi,
   MGFAHypre,
   MGLORHypre,
   MGLORBatch,
   MGInner,
   MGWavelets,
   MGFAWavelets,
   MGLEGACYWavelets
};

////////////////////////////////////////////////////////////////////////////////
namespace precond
{

////////////////////////////////////////////////////////////////////////////////
struct SolverConfig
{
   enum SolverType
   {
      JACOBI = 0,
      FA_HYPRE,
      LOR_HYPRE,
      LOR_BATCH,
      FA_AMGX,
      LOR_AMGX,
      FULL_WAMG,
      LEGACY_WAMG,
      WAMG
   };
   SolverType type = JACOBI;
   const char *amgx_config_file = "amgx/amgx.json";
   bool inner_cg = false; //<-- use inner CG iteration for coarse solver
   SolverConfig() = default;
   SolverConfig(SolverType type) : type(type) { }
};

////////////////////////////////////////////////////////////////////////////////
class LORDiffusion
{
   ParMesh MakeLORMesh(ParMesh &mesh_ho, int order)
   {
      MFEM_NVTX;
      ParMesh mesh_lor_tensor =
         ParMesh::MakeRefined(mesh_ho, order, BasisType::GaussLobatto);
      return mesh_lor_tensor;
   }

   ParMesh mesh;
   H1_FECollection fec;
   ParFiniteElementSpace fes;
   ParBilinearForm a;
   OperatorPtr A;
   IntegrationRules irs;

public:
   LORDiffusion(ParMesh &mesh_ho, int order,
                Coefficient &coeff,
                Array<int> &ess_dofs):
      mesh(MakeLORMesh(mesh_ho, order)),
      fec(1, mesh.Dimension()),
      fes(&mesh, &fec),
      a(&fes),
      irs(0, Quadrature1D::GaussLobatto)
   {
      MFEM_NVTX;
      // LOR system uses collocated (2-point Gauss-Lobatto) quadrature
      int dim = mesh.Dimension();
      Geometry::Type geom = mesh.GetElementBaseGeometry(0);
      const IntegrationRule &ir = irs.Get(geom, 1);
      if (geom == Geometry::SQUARE || geom == Geometry::CUBE)
      { MFEM_VERIFY(ir.Size() == pow(2,dim), "Wrong quadrature rule"); }

      DiffusionIntegrator *integ = new DiffusionIntegrator(coeff, &ir);
      a.AddDomainIntegrator(integ);
      a.Assemble();
      a.FormSystemMatrix(ess_dofs, A);
   }

   OperatorPtr& GetA() { return A; }
};

////////////////////////////////////////////////////////////////////////////////
class DiffusionMultigrid : public GeometricMultigrid
{
   AssemblyLevel GetCoarseAssemblyLevel(SolverConfig config)
   {
      switch (config.type)
      {
         case SolverConfig::JACOBI:
         case SolverConfig::LOR_HYPRE:
         case SolverConfig::LOR_BATCH:
         case SolverConfig::LOR_AMGX:
            return AssemblyLevel::PARTIAL;

         case SolverConfig::WAMG:
            return AssemblyLevel::PARTIAL;

         case SolverConfig::FULL_WAMG:
            return AssemblyLevel::FULL;

         case SolverConfig::LEGACY_WAMG:
            return AssemblyLevel::LEGACY;

         default: return AssemblyLevel::LEGACY;
      }
   }

   AssemblyLevel GetCoarseAssemblyLevel(SolverConfig config,
                                        std::string &al_str)
   {
      const AssemblyLevel assembly_level = GetCoarseAssemblyLevel(config);
      switch (assembly_level)
      {
         case AssemblyLevel::LEGACY: al_str = std::string("LEGACY"); break;
         case AssemblyLevel::FULL: al_str = std::string("FULL"); break;
         case AssemblyLevel::ELEMENT: al_str = std::string("ELEMENT"); break;
         case AssemblyLevel::PARTIAL: al_str = std::string("PARTIAL"); break;
         default: al_str = std::string("UNKNOWN ?!");
      }
      return assembly_level;
   }

   bool NeedsLOR(SolverConfig config)
   {
      switch (config.type)
      {
         case SolverConfig::LOR_HYPRE:
         case SolverConfig::LOR_AMGX: return true;
         default: return false;
      }
   }
   const bool GLL;
   Coefficient &coeff;
   IntegrationRules irs;
   std::unique_ptr<LORDiffusion> lor_diffusion;
   OperatorPtr A_coarse;
   std::shared_ptr<Solver> coarse_solver, coarse_precond;

public:
   DiffusionMultigrid(bool GLL,
                      ParFiniteElementSpaceHierarchy& hierarchy,
                      Coefficient &coeff_,
                      Array<int>& ess_bdr,
                      SolverConfig coarse_solver_config)
      : GeometricMultigrid(hierarchy),
        GLL(GLL),
        coeff(coeff_),
        irs(0, Quadrature1D::GaussLegendre)
   {
      MFEM_NVTX;
      int nlevels = hierarchy.GetNumLevels();
      ConstructCoarseOperatorAndSolver(coarse_solver_config,
                                       hierarchy.GetFESpaceAtLevel(0),
                                       ess_bdr);
      for (int i=1; i<nlevels; ++i)
      {
         ConstructOperatorAndSmoother(hierarchy.GetFESpaceAtLevel(i), ess_bdr);
      }
   }

   void ConstructBilinearForm(ParFiniteElementSpace &pfes,
                              Array<int> &ess_bdr,
                              AssemblyLevel asm_lvl)
   {
      MFEM_NVTX;
      dbg("%s", asm_lvl==AssemblyLevel::PARTIAL ? "PARTIAL" :
          asm_lvl==AssemblyLevel::LEGACY ? "LEGACY" :
          asm_lvl==AssemblyLevel::FULL ? "FULL" :
          "???");
      ParBilinearForm* form = new ParBilinearForm(&pfes);
      dbg("ParBilinearForm: %dx%d",form->Height(),form->Width());
      form->SetAssemblyLevel(asm_lvl);

      const int p = pfes.GetOrder(0);
      const int dim = pfes.GetMesh()->Dimension();
      // Integration rule for high-order problem: (p+2)^d Gauss-Legendre points
      const int int_order = 2*p + (GLL?-1:3); // = 2*(p+2) - 1
      Geometry::Type geom = pfes.GetMesh()->GetElementBaseGeometry(0);
      const IntegrationRule &ir = irs.Get(geom, int_order);
      MFEM_VERIFY(ir.Size() == pow(p+2,dim), "Wrong quadrature");

      DiffusionIntegrator *integ = new DiffusionIntegrator(coeff, &ir);
      form->AddDomainIntegrator(integ);
      form->Assemble();
      bfs.Append(form);

      essentialTrueDofs.Append(new Array<int>());
      pfes.GetEssentialTrueDofs(ess_bdr, *essentialTrueDofs.Last());
   }

   void ConstructOperatorAndSmoother(ParFiniteElementSpace& fespace,
                                     Array<int>& ess_bdr)
   {
      MFEM_NVTX;
      ConstructBilinearForm(fespace, ess_bdr, AssemblyLevel::PARTIAL);

      OperatorPtr opr;
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
      opr.SetOperatorOwner(false);

      Vector diag(fespace.GetTrueVSize());
      bfs.Last()->AssembleDiagonal(diag);

      Solver* smoother =
         new OperatorChebyshevSmoother(*opr.Ptr(),  diag,
                                       *essentialTrueDofs.Last(), 2,
                                       fespace.GetParMesh()->GetComm());

      AddLevel(opr.Ptr(), smoother, true, true);
   }

   void ConstructCoarseOperatorAndSolver(SolverConfig config,
                                         ParFiniteElementSpace &pfes,
                                         Array<int>& ess_bdr)
   {
      dbg();
      MFEM_NVTX;
      const int print_level = config_debug ? 3: -1;
      std::string al_str;
      ConstructBilinearForm(pfes, ess_bdr,
                            GetCoarseAssemblyLevel(config, al_str));
      dbg("AssemblyLevel: %s",al_str.c_str());
      BilinearForm &a = *bfs.Last();
      Array<int> &ess_dofs = *essentialTrueDofs.Last();
      a.FormSystemMatrix(*essentialTrueDofs.Last(), A_coarse);

      ///  FULL => BC NOT DONE !!!
      if (GetCoarseAssemblyLevel(config) == AssemblyLevel::FULL)
      {
         dbg("FULL => BC NOT DONE, doing NOW !!!");
         SparseMatrix &mat = a.SpMat();
         //SparseMatrix &mat_e = a.SpMatElim();
         assert(A_coarse.Type() == Type::ANY_TYPE);
         const int remove_zeros = 0;
         a.Finalize(remove_zeros);
         OperatorHandle p_mat(Operator::Hypre_ParCSR),
                        p_mat_e(Operator::Hypre_ParCSR);
         MFEM_VERIFY(p_mat.Ptr() == NULL && p_mat_e.Ptr() == NULL,
                     "The ParBilinearForm must be updated with Update() before "
                     "re-assembling the ParBilinearForm.");
         //a.ParallelAssemble(p_mat, mat);
         p_mat.Clear();
         OperatorHandle dA(p_mat.Type()), Ph(p_mat.Type()), hdA;
         dA.MakeSquareBlockDiag(pfes.GetComm(),
                                pfes.GlobalVSize(),
                                pfes.GetDofOffsets(), &mat);
         Ph.ConvertFrom(pfes.Dof_TrueDof_Matrix());
         p_mat.MakePtAP(dA, Ph);
         p_mat_e.EliminateRowsCols(p_mat, ess_dofs);
         A_coarse = p_mat;
         p_mat.SetOperatorOwner(false);
      }

      OperatorPtr A_prec;
      if (NeedsLOR(config))
      {
         dbg("NeedsLOR!");
         ParMesh &mesh = *pfes.GetParMesh();
         int order = pfes.GetOrder(0); // <-- Assume uniform p
         lor_diffusion.reset(new LORDiffusion(mesh, order, coeff, ess_dofs));
         A_prec = lor_diffusion->GetA();
      }
      else
      {
         dbg("A_prec = A_coarse");
         A_prec = A_coarse;
      }

      switch (config.type)
      {
         case SolverConfig::JACOBI:
            coarse_precond.reset(new OperatorJacobiSmoother(a, ess_dofs));
            break;
         case SolverConfig::FA_HYPRE:
         case SolverConfig::LOR_HYPRE:
         {
            dbg("[precond] HYPRE");
            HypreBoomerAMG *amg = new HypreBoomerAMG(*A_prec.As<HypreParMatrix>());
            amg->SetPrintLevel(0);
            Vector b(amg->Height());
            Vector x(amg->Height());
            b = 0.0;
            x = 0.0;
            amg->Mult(b, x); // <-- Force setup. Ugly hack.
            coarse_precond.reset(amg);
            break;
         }
         case SolverConfig::LOR_BATCH:
         {
            dbg("[precond] LOR BATCH");
            LOR_AMG *lor_amg = new LOR_AMG(a, ess_dofs);
            lor_amg->GetSolver().SetPrintLevel(0);
            coarse_precond.reset(lor_amg);
            break;
         }
#ifdef MFEM_USE_AMGX
         case SolverConfig::FA_AMGX:
         case SolverConfig::LOR_AMGX:
         {
            dbg("[precond] AMGX");
            AmgXSolver *amg = new AmgXSolver;
            amg->ReadParameters(config.amgx_config_file, AmgXSolver::EXTERNAL);
            amg->InitExclusiveGPU(MPI_COMM_WORLD);
            amg->SetOperator(*A_prec.As<HypreParMatrix>());
            coarse_precond.reset(amg);
            break;
         }
#endif
         case SolverConfig::FULL_WAMG:
         case SolverConfig::LEGACY_WAMG:
         {
            const int smoother_order = 1;
            Vector diag = *new Vector(pfes.GetTrueVSize());
            a.AssembleDiagonal(diag);
            dbg("[precond] FA_WAMG %dx%d", A_prec->Height(), A_prec->Width());
            // A_prec is a LEGACY/FULL A_coarse ParHypreMatrix
            wargs_t args(A_prec, diag, ess_dofs,
                         smoother_order, config_max_nic, print_level);
            coarse_precond.reset(new faWAMG(pfes, Wavelet::HAAR, args));
            break;
         }
         case SolverConfig::WAMG:
         {
            const int smoother_order = 1;
            Vector diag = *new Vector(pfes.GetTrueVSize());
            a.AssembleDiagonal(diag);
            dbg("[precond] WAMG %dx%d", A_prec->Height(), A_prec->Width());
            wargs_t args(A_prec, diag, ess_dofs,
                         smoother_order, config_max_nic, print_level);
            coarse_precond.reset(new WAMG(pfes, Wavelet::HAAR, args));
            break;
         }

         default:
            MFEM_ABORT("Not available.")
      }

      if (config.inner_cg)
      {
         CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
         cg->SetMaxIter(config_max_nip);
         cg->SetRelTol(1e-8);
         cg->SetAbsTol(1e-8);
         cg->SetPrintLevel(print_level);
         cg->SetOperator(*A_prec);
         cg->SetPreconditioner(*coarse_precond);
         cg->iterative_mode = false;
         coarse_solver.reset(cg);
      }
      else
      {
         coarse_solver = coarse_precond;
         assert(coarse_solver != nullptr);
      }

      AddLevel(A_coarse.Ptr(), coarse_solver.get(), false, false);
   }
};

////////////////////////////////////////////////////////////////////////////////
struct MultigridLevel
{
   enum Type { P_MG, H_MG };
   Type type;
   int order;
   MultigridLevel(Type type_, int order_) : type(type_), order(order_) { }
   static MultigridLevel p(int order_) { return MultigridLevel(P_MG, order_); }
   static MultigridLevel h() { return MultigridLevel(H_MG, 0); }
};

////////////////////////////////////////////////////////////////////////////////
std::vector<MultigridLevel> *WalkMGSpecification(int &coarse_order,
                                                 const int order)
{
   int p;
   coarse_order = 1;
   std::vector<MultigridLevel> *mg_refinements = new std::vector<MultigridLevel>;
   for (p = coarse_order+1; p <= order; p += 1)
   {
      mg_refinements->push_back(MultigridLevel::p(p));
   }
   assert(p-1==order);
   return mg_refinements;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<MultigridLevel> *Div2MGSpecification(int &coarse_order,
                                                 const int order)
{
   int p;
   coarse_order = 1;
   std::vector<MultigridLevel> *mg_refinements = new std::vector<MultigridLevel>;
   for (p = coarse_order*2; p <= order; p *= 2)
   {
      mg_refinements->push_back(MultigridLevel::p(p));
   }
   if (p != 2*order)
   {
      mg_refinements->push_back(MultigridLevel::p(order));
   }
   return mg_refinements;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<MultigridLevel> *ParseMGSpecification(int &coarse_order,
                                                  int &order,
                                                  int ref_par = 0)
{
   int h_ref = ref_par;
   std::vector<MultigridLevel> *mg_refinements = new std::vector<MultigridLevel>;
   std::istringstream mg_stream(config_mg_spec);
   std::string ref;
   mg_stream >> coarse_order;
   int prev_order = order = coarse_order;
   while (mg_stream >> ref)
   {
      if (ref == "r")
      {
         mg_refinements->push_back(MultigridLevel::h());
         ++h_ref;
      }
      else
      {
         try { order = stoi(ref); }
         catch (...)
         {
            MFEM_ABORT("Multigrid refinement must either be an integer or "
                       "the character `r`");
         }
         MFEM_VERIFY(order > 0, "Orders must be positive");
         MFEM_VERIFY(order > prev_order, "Orders must be sorted");
         mg_refinements->push_back(MultigridLevel::p(order));
         prev_order = order;
      }
   }
   return mg_refinements;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<MultigridLevel> *GetMGSelector(const bool mg_solver,
                                           const int p,
                                           int &mg_fine_order,
                                           int &mg_coarse_order)
{
   mg_fine_order = p;
   std::vector<MultigridLevel> *specs = nullptr;

   if (!mg_solver) { specs = new std::vector<MultigridLevel>; }
   else
   {
      switch (config_mg_spec_select)
      {
         case INPUT: specs = ParseMGSpecification(mg_coarse_order, mg_fine_order); break;
         case INCREMENT: specs = WalkMGSpecification(mg_coarse_order, p); break;
         case DIVIDE_BY_2: specs= Div2MGSpecification(mg_coarse_order, p); break;
         default: MFEM_ABORT("Unhandled MG specification");
      }
   }
   assert(specs);
   return specs;
};

} // namespace precond

////////////////////////////////////////////////////////////////////////////////
namespace kershaw
{

// 1D transformation at the right boundary.
double right(const double eps, const double x)
{
   return (x <= 0.5) ? (2.-eps) * x : 1. + eps*(x-1.);
}

// 1D transformation at the left boundary
double left(const double eps, const double x) { return 1.-right(eps,1.-x); }

// Transition from a value of "a" for x=0, to a value of "b" for x=1.
// Smoothness is controlled by the parameter "s", taking values 0, 1, or 2.
double step(const double a, const double b, const double x, const int s)
{
   if (x <= 0.) { return a; }
   if (x >= 1.) { return b; }
   switch (s)
   {
      case 0: return a + (b-a) * (x);
      case 1: return a + (b-a) * (x*x*(3.-2.*x));
      case 2: return a + (b-a) * (x*x*x*(x*(6.*x-15.)+10.));
      default: MFEM_ABORT("Smoothness values: 0, 1, or 2.");
   }
   return 0.0;
}

// 3D version of a generalized Kershaw mesh transformation, see D. Kershaw,
// "Differencing of the diffusion equation in Lagrangian hydrodynamic codes",
// JCP, 39:375–395, 1981.
//
// The input mesh should be Cartesian nx x ny x nz with nx divisible by 6 and
// ny, nz divisible by 2.
//
// The eps parameters are in (0, 1]. Uniform mesh is recovered for epsy=epsz=1.
void kershaw(const double epsy, const double epsz, const int smoothness,
             const double x, const double y, const double z,
             double &X, double &Y, double &Z)
{
   X = x;

   const int layer = 6.0*x;
   const double lambda = (x-layer/6.0)*6;

   // The x-range is split in 6 layers going from left-to-left, left-to-right,
   // right-to-left (2 layers), left-to-right and right-to-right yz-faces.
   switch (layer)
   {
      case 0:
         Y = left(epsy, y);
         Z = left(epsz, z);
         break;
      case 1:
      case 4:
         Y = step(left(epsy, y), right(epsy, y), lambda, smoothness);
         Z = step(left(epsz, z), right(epsz, z), lambda, smoothness);
         break;
      case 2:
         Y = step(right(epsy, y), left(epsy, y), lambda/2.0, smoothness);
         Z = step(right(epsz, z), left(epsz, z), lambda/2.0, smoothness);
         break;
      case 3:
         Y = step(right(epsy, y), left(epsy, y), (1.0+lambda)/2.0, smoothness);
         Z = step(right(epsz, z), left(epsz, z), (1.0+lambda)/2.0, smoothness);
         break;
      default:
         Y = right(epsy, y);
         Z = right(epsz, z);
         break;
   }
}

struct Transformation : VectorCoefficient
{
   double epsy, epsz;
   int dim, s;
   Transformation(int dim, double epsy, double epsz, int s = 0):
      VectorCoefficient(dim),
      epsy(epsy),
      epsz(epsz),
      dim(dim),
      s(s) { }

   using VectorCoefficient::Eval;

   void Eval(Vector &V,
             ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      double xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      if (dim == 1)
      {
         V[0] = xyz[0]; // no transformation in 1D
      }
      else if (dim == 2)
      {
         double z = 0, zt;
         kershaw(epsy, epsz, s, xyz[0], xyz[1], z, V[0], V[1], zt);
      }
      else // dim == 3
      {
         kershaw(epsy, epsz, s, xyz[0], xyz[1], xyz[2], V[0], V[1], V[2]);
      }
   }
};

} // namespace kershaw

////////////////////////////////////////////////////////////////////////////////
namespace rhs
{

double s(int k, double x) { return sin(2*M_PI*k*x); }

double u(int k, double x)
{
   const double skx = s(k,x);
   const double sgn = skx < 0.0 ? -1.0 : 1.0;
   return exp(-1.0/skx/skx)*sgn;
}

double u_xx(int k, double x)
{
   const double kpix = k*M_PI*x;
   const double csc_2kpix = 1.0/sin(2.*kpix);
   const double sgn = sin(2.*kpix) < 0.0 ? -1.0 : 1.0;
   return 2.*exp(-csc_2kpix*csc_2kpix)*k*k*M_PI*M_PI
          *(1. + 6.*cos(4.*kpix) + cos(8.*kpix))
          *pow(csc_2kpix,6.)
          *sgn;
}

double w(int n, double x)
{
   double wkx = 0.0;
   const double xx = 2.0*x - 1.0; // transform from [0,1] to [-1,1]
   for (int j=0; j<n; ++j)
   {
      const int k = pow(3, j);
      wkx += u(k, xx);
   }
   return wkx;
}

double w_xx(int n, double x)
{
   double wkx = 0.0;
   const double xx = 2.0*x - 1.0; // transform from [0,1] to [-1,1]
   if (xx == 0.0) { return 0.0; }
   for (int j=0; j<n; ++j)
   {
      const int k = pow(3, j);
      wkx += 4.0*u_xx(k, xx); // factor of four from reference interval transf.
   }
   return wkx;
}

struct ExactSolution: Coefficient
{
   int dim, n;
   ExactSolution(int dim, int n=0) : dim(dim), n(n) { }
   using Coefficient::Eval;
   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      double xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      if (dim == 1) { return w(n, xyz[0]); }
      if (dim == 2) { return w(n, xyz[0])*w(n, xyz[1]); }
      else { return w(n, xyz[0])*w(n, xyz[1])*w(n, xyz[2]); }
   }
};

struct RhsCoefficient: Coefficient
{
   int dim, n;
   RhsCoefficient(int dim, int n=0) : dim(dim), n(n) { }
   using mfem::Coefficient::Eval;
   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      double xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      const double x = xyz[0];
      const double y = xyz[1];
      const double z = xyz[2];
      if (dim == 1) { return -w_xx(n,x); }
      if (dim == 2) { return -w_xx(n,x) * w(n,y) - w(n,x) * w_xx(n,y); }
      else
      {
         return - w_xx(n,x) *    w(n,y) *    w(n,z)
                -    w(n,x) * w_xx(n,y) *    w(n,z)
                -    w(n,x) *    w(n,y) * w_xx(n,z);
      }
   }
};

} // namespace rhs

////////////////////////////////////////////////////////////////////////////////
namespace cg
{

////////////////////////////////////////////////////////////////////////////////
struct CGMonitor : IterativeSolverMonitor
{
   double initial_nrm;
   void MonitorResidual(int it, double norm, const Vector &r, bool final)
   {
      MFEM_CONTRACT_VAR(norm);
      if (mpi->Root() && (it == 0 || final))
      {
         std::string prefix = final ? "Final" : "Initial";
         const double nrm = r*r;
         mfem::out << "   " << prefix
                   << " l2 norm of residual: " << nrm << '\n';
         if (it == 0) { initial_nrm = nrm; }
         if (final)
         {
            mfem::out << "   Relative residual: "
                      << nrm / initial_nrm
                      << std::endl;
         }
      }
   }
};

} // namespace cg

////////////////////////////////////////////////////////////////////////////////
namespace ceed
{

////////////////////////////////////////////////////////////////////////////////
struct BakeOff
{
   static constexpr int dim = 3, N = 6;
   const bool mg_solver;
   const int num_procs, myid;
   const int preconditioner, refinements, smoothness, p, q;
   precond::SolverConfig solver_config;
   const double epsy, epsz;
   const bool rhs_1;
   const int rhs_n;

   StopWatch sw_setup, sw_solve;
   double t_setup, t_solve;

   std::function<ParMesh()> GetCoarseKershawMesh = [&]()
   {
      Mesh serial_mesh;
      const bool sfc_ordering = true;
      const bool generate_edges = false;

      if (dim == 3)
      {
         serial_mesh = Mesh::MakeCartesian3D(N,N,N,Element::HEXAHEDRON,
                                             1.0,1.0,1.0,sfc_ordering);
      }
      if (dim == 2)
      {
         serial_mesh = Mesh::MakeCartesian2D(N,N,Element::QUADRILATERAL,
                                             generate_edges,1.0,1.0,
                                             sfc_ordering);
      }

      // Set the curvature of the initial mesh
      if (smoothness > 0) { serial_mesh.SetCurvature(2*smoothness+1); }

      // Kershaw transformation
      kershaw::Transformation kt(dim, epsy, epsz, smoothness);
      serial_mesh.Transform(kt);

      ParMesh coarse_pmesh(MPI_COMM_WORLD, serial_mesh);
      serial_mesh.Clear();

      // perform refinements if requested
      for (int i=0; i<refinements; i++) { coarse_pmesh.UniformRefinement(); }

      const bool ordering_t = false;
      if (ordering_t)
      {
         Array<int> elem_ordering;
         coarse_pmesh.GetHilbertElementOrdering(elem_ordering);
         coarse_pmesh.ReorderElements(elem_ordering);
      }
      return coarse_pmesh;
   };

   ParMesh pmesh;

   int mg_nlevels, mg_fine_order, mg_coarse_order ;
   std::vector<FiniteElementCollection*> mg_fe_collections;
   std::vector<precond::MultigridLevel> *mg_refinements;
   std::function<ParFiniteElementSpaceHierarchy*()> GetFESpaceHierarchy = [&]()
   {
      MFEM_DEVICE_SYNC;
      sw_setup.Start();
      // Coarse fec, fes & hierarchy
      const int btype = BasisType::GaussLobatto;
      mg_fe_collections.push_back(new H1_FECollection(mg_coarse_order, dim, btype));
      ParFiniteElementSpace *fes_coarse
         = new ParFiniteElementSpace(&pmesh, mg_fe_collections.back());
      const bool ownM = false, ownFES = true;
      ParFiniteElementSpaceHierarchy *pfes_h =
         new ParFiniteElementSpaceHierarchy(&pmesh, fes_coarse, ownM, ownFES);

      std::for_each(mg_refinements->begin(), mg_refinements->end(),
                    [&](const precond::MultigridLevel &lvl)
      {
         if (lvl.type == precond::MultigridLevel::H_MG)
         {
            pfes_h->AddUniformlyRefinedLevel();
         }
         else
         {
            mg_fe_collections.push_back(new H1_FECollection(lvl.order, dim));
            pfes_h->AddOrderRefinedLevel(mg_fe_collections.back());
         }
      });

      mg_nlevels = pfes_h->GetNumLevels();
      MFEM_DEVICE_SYNC;
      sw_setup.Stop();
      return pfes_h;
   };
   ParFiniteElementSpaceHierarchy *mg_hierarchy;

   H1_FECollection p_fec;
   ParFiniteElementSpace p_fes, mg_fes;
   ParMesh mg_pmesh;

   const Geometry::Type cube_t;
   IntegrationRules IntRulesGLL;
   const IntegrationRule *irGLL;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   rhs::RhsCoefficient rhs;
   const HYPRE_Int dofs;

   ParGridFunction x;
   ParBilinearForm a;
   ParLinearForm b;
   Vector diag;

   int niter = 0;

   BakeOff(int preconditioner, int refinements, int smoothness,
           double epsy, double epsz,
           bool rhs_1, int rhs_n,
           int p, int vdim, bool GLL):
      mg_solver(preconditioner >= FinePrecondType::MGJacobi),
      num_procs(mpi->WorldSize()),
      myid(mpi->WorldRank()),
      preconditioner(preconditioner),
      refinements(refinements),
      smoothness(smoothness),
      p(p),
      q(2*p + (GLL?-1:3)),
      epsy(epsy),
      epsz(epsz),
      rhs_1(rhs_1),
      rhs_n(rhs_n),
      t_setup((sw_setup.Clear(), 0.0)),
      t_solve((sw_solve.Clear(), 0.0)),
      pmesh(GetCoarseKershawMesh()),
      mg_nlevels(0), // set in GetFESpaceHierarchy
      mg_fine_order(0), // set in GetMGSelector
      mg_coarse_order(0), // set in GetMGSelector
      mg_fe_collections(), // set in GetFESpaceHierarchy
      mg_refinements(mg_solver ?
                     precond::GetMGSelector(mg_solver, p,
                                            mg_fine_order,
                                            mg_coarse_order) :
                     nullptr),
      mg_hierarchy(mg_solver ? GetFESpaceHierarchy() : nullptr),
      p_fec(p, dim, BasisType::GaussLobatto),
      p_fes(&pmesh, &p_fec, vdim),
      mg_fes(mg_solver ? mg_hierarchy->GetFinestFESpace() : p_fes),
      mg_pmesh(mg_solver ? *mg_fes.GetParMesh() : pmesh),
      cube_t(Geometry::CUBE),
      IntRulesGLL(0, Quadrature1D::GaussLegendre),
      irGLL(&IntRulesGLL.Get(cube_t, q)),
      ir(&IntRules.Get(cube_t, q)),

      one(1.0),
      rhs(dim, rhs_n),
      dofs(mg_fes.GlobalTrueVSize()),
      x(&mg_fes),
      a(&mg_fes),
      b(&mg_fes),
      diag(mg_fes.GetTrueVSize())
   {
      x = 0.0;
   }

   ~BakeOff() { }

   virtual void benchmark() = 0;

   double MDofs() const { return 1e-6 * dofs; }

   double TSetup() { return sw_setup.RealTime(); }
   double TSolve() { return sw_solve.RealTime(); }
};

/// Bake-off Solver (BPS)
template<typename BFI, int VDIM = 1, bool GLL = false>
struct SolverProblem: public BakeOff
{
   const int max_it = config_max_nif;
   const double rtol = 1e-8;
   const int print_lvl = config_debug ? 3 : -1;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   OperatorHandle A;
   Vector B, X;
   CGSolver cg;
   cg::CGMonitor monitor;
   std::unique_ptr<Solver> M;

   SolverProblem(int preconditioner,
                 int refinements,
                 int smoothness,
                 double epsy, double epsz,
                 bool rhs_1, int rhs_n,
                 int order):
      BakeOff(preconditioner, refinements, smoothness,
              epsy, epsz,
              rhs_1, rhs_n,
              order,
              VDIM, GLL),
      ess_bdr(mg_solver ?
              mg_pmesh.bdr_attributes.Max():
              pmesh.bdr_attributes.Max()),
      cg(MPI_COMM_WORLD)
   {
      MFEM_NVTX;
      ess_bdr = 1;
      mg_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      if (rhs_1) { b.AddDomainIntegrator(new DomainLFIntegrator(one)); }
      else { b.AddDomainIntegrator(new DomainLFIntegrator(rhs)); }
      {
         NVTX("b");
         b.SetAssemblyLevel(LinearAssemblyLevel::FULL);
         b.Assemble();
      }

      MFEM_DEVICE_SYNC;
      sw_setup.Start();

      if (!mg_solver)
      {
         dbg("a.Assemble");
         NVTX("a.Assemble");
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         a.AddDomainIntegrator(new BFI(one, GLL?irGLL:ir));
         a.Assemble();
         a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
         a.AssembleDiagonal(diag);
      }
      // should be an option
      const int smoother_order = 1;
      wargs_t args {A, diag, ess_tdof_list, smoother_order,
                    config_max_nic, print_lvl};

      //// Setup phase
      auto SetMGPrecond = [&](const char *header,
                              precond::SolverConfig::SolverType type,
                              const bool inner_cg)
      {
         NVTX(header);
         assert(p == mg_fine_order);
         solver_config.type = type;
         solver_config.inner_cg = inner_cg;
         precond::DiffusionMultigrid *DMG =
            new precond::DiffusionMultigrid(GLL, *mg_hierarchy,
                                            one, ess_bdr, solver_config);
         M.reset(DMG);
         DMG->SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);
         DMG->FormFineLinearSystem(x, b, A, X, B);
      };

      switch (preconditioner)
      {
         case None:
         {
            dbg("None");
            break;
         }
         case Jacobi:
         {
            dbg("Jacobi");
            NVTX("Jacobi");
            M.reset(new OperatorJacobiSmoother(a,ess_tdof_list));
            break;
         }
         case LORBatch:
         {
            dbg("LORBatch");
            NVTX("LORBatch");
            LOR_AMG *lor_amg = new LOR_AMG(a, ess_tdof_list);
            lor_amg->GetSolver().SetPrintLevel(0);
            M.reset(lor_amg);
            break;
         }
         case MGJacobi:
         {
            SetMGPrecond("MGJacobi", precond::SolverConfig::JACOBI, true);
            break;
         }
         case MGFAHypre:
         {
            SetMGPrecond("MGFAHypre", precond::SolverConfig::FA_HYPRE, false);
            break;
         }
         case MGLORHypre:
         {
            SetMGPrecond("MGLORHypre", precond::SolverConfig::LOR_HYPRE, false);
            break;
         }
         case MGLORBatch:
         {
            SetMGPrecond("MGLORBatch", precond::SolverConfig::LOR_BATCH, false);
            break;
         }
         case MGInner:
         {
            dbg("MGInner");
            NVTX("MGInner");
            SetMGPrecond("MGInner", precond::SolverConfig::JACOBI, true);
            break;
         }
         case MGFAWavelets:
         {
            dbg("MGFAWavelets");
            NVTX("MGFAWavelets");
            SetMGPrecond("MGFAWavelets", precond::SolverConfig::FULL_WAMG, true);
            break;
         }
         case MGLEGACYWavelets:
         {
            dbg("MGLEGACYWavelets");
            NVTX("MGLEGACYWavelets");
            SetMGPrecond("MGLEGACYWavelets", precond::SolverConfig::LEGACY_WAMG, true);
            break;
         }
         case MGWavelets:
         {
            dbg("MGWavelets");
            NVTX("MGWavelets");
            SetMGPrecond("MGWavelets", precond::SolverConfig::WAMG, true);
            break;
         }
         default: MFEM_ABORT("Unknown preconditioner");
      }
      MFEM_DEVICE_SYNC;
      sw_setup.Stop();

      assert(A.Ptr());
      cg.SetOperator(*A);
      cg.SetRelTol(rtol);
      cg.SetMaxIter(max_it);
      cg.iterative_mode = false;
      cg.SetAbsTol(sqrt(1e-16));
      cg.SetPrintLevel(print_lvl);
      if (M.get()) { cg.SetPreconditioner(*M); }
      if (config_debug) { cg.SetMonitor(monitor); }
   }

   void benchmark() override
   {
      MFEM_NVTX;
      MFEM_DEVICE_SYNC;
      sw_solve.Start();
      cg.Mult(B,X);
      MFEM_DEVICE_SYNC;
      sw_solve.Stop();
      const bool converged = cg.GetConverged();
      const int num_iterations = cg.GetNumIterations();
      // make sure we have the same number of iterations
      if (niter > 0) { MFEM_VERIFY(std::abs(num_iterations-niter)<4, ""); }
      else { niter = converged ? num_iterations : -1; }
      // Reset the solve timer if it has not converged
      if (!converged) { sw_solve.Clear(); }
   }
};

} // namespace ceed

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,6,1)

// The different preconditioners the tests can use
#define P_PRECONDS {None,Jacobi,MGJacobi,MGFAHypre,MGLORHypre}

// The different epsilons dividers
#define P_EPSILONS {1,2,3}

// The different refinements
#define P_REFINEMENTS bm::CreateDenseRange(0,3,1)

/// Bake-off Solvers (BPSs)
/// Smoothness in 0, 1 or 2
/// refinements set from state.range(2)
/// epsy = epsz, set from state.range(1)
#define BakeOff_Solver(i,Kernel,Prcd)\
static void BPS##i##_##Prcd(bm::State &state){\
   const int refinements = state.range(2);\
   const int smoothness = 0;\
   const double eps = 1.0/state.range(1);\
   const bool rhs_1 = true;\
   const bool rhs_n = 3;\
   const int order = state.range(0);\
   const int preconditioner = Prcd;\
   ceed::SolverProblem<Kernel##Integrator> ker\
      (preconditioner, refinements, smoothness, eps,eps, rhs_1,rhs_n, order);\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["Ndofs"] = bm::Counter(ker.dofs);\
   state.counters["Epsilon"] = bm::Counter(ker.epsy);\
   state.counters["Niters"] = bm::Counter(ker.niter);\
   state.counters["P"] = bm::Counter(order);\
   state.counters["Tsetup"] = bm::Counter(ker.TSetup());\
   state.counters["Tsolve"] = bm::Counter(ker.TSolve(), bm::Counter::kAvgIterations);}\
BENCHMARK(BPS##i##_##Prcd)\
    -> ArgsProduct({P_ORDERS,P_EPSILONS,P_REFINEMENTS})\
    -> Unit(bm::kMillisecond);

/// BPS3: scalar PCG with stiffness matrix, q=p+2
BakeOff_Solver(3,Diffusion,None)
BakeOff_Solver(3,Diffusion,Jacobi)
BakeOff_Solver(3,Diffusion,LORBatch)
BakeOff_Solver(3,Diffusion,MGJacobi)
BakeOff_Solver(3,Diffusion,MGFAHypre)
BakeOff_Solver(3,Diffusion,MGLORHypre)
BakeOff_Solver(3,Diffusion,MGLORBatch)
BakeOff_Solver(3,Diffusion,MGInner)
BakeOff_Solver(3,Diffusion,MGWavelets)
BakeOff_Solver(3,Diffusion,MGFAWavelets)
BakeOff_Solver(3,Diffusion,MGLEGACYWavelets)

} // namespace mfem

/**
 * @brief main entry point
 * --benchmark_context=device=cuda
 * --benchmark_filter=BPS3_\(None\|Jacobi\|MGJacobi\)/4/2/1
 */
int main(int argc, char *argv[])
{
   MFEM_NVTX;
#ifdef MFEM_USE_MPI
   mfem::MPI_Session main_mpi(argc, argv);
   mpi = &main_mpi;
#endif

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string config_device = "cpu";

   if (bmi::global_context != nullptr)
   {
      bmi::FindInContext("device", config_device); // device=cuda
      bmi::FindInContext("debug", config_debug); // debug=true
      bmi::FindInContext("nic", config_max_nic);
      bmi::FindInContext("nip", config_max_nip);
      bmi::FindInContext("nif", config_max_nif);
      bmi::FindInContext("mg_spec", config_mg_spec); // mg_spec="1 2"
      bmi::FindInContext("mg_select", config_mg_spec_select); // mg_select=2
      bmi::FindInContext("caliper", config_use_caliper);
      bmi::FindInContext("dev_size", config_dev_size);
   }

   const int mpi_rank = mpi->WorldRank();
   const int mpi_size = mpi->WorldSize();
   const int dev = mpi_rank % config_dev_size;
   dbg("[MPI] rank: %d/%d, using device #%d", 1+mpi_rank, mpi_size, dev);

   Device device(config_device.c_str(), dev);
   if (mpi->Root()) { device.Print(); }

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }

#ifdef MFEM_USE_CALIPER
   cali::ConfigManager caliper;
   if (use_caliper)
   {
      caliper.add("runtime-report(max_column_width=0)");
      caliper.start();
   }
#else
   MFEM_CONTRACT_VAR(config_use_caliper);
#endif

#ifndef MFEM_USE_MPI
   bm::RunSpecifiedBenchmarks(&CR);
#else
   if (mpi->Root()) { bm::RunSpecifiedBenchmarks(&CR); }
   else { bm::RunSpecifiedBenchmarks(NoReporter()); }
#endif

#ifdef MFEM_USE_CALIPER
   if (use_caliper)
   {
      caliper.flush();
      fflush(0);
   }
#endif
   return 0;
}

#endif // MFEM_USE_BENCHMARK
