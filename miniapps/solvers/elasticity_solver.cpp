// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

/// Finite element spaces concerning Elasticity solvers.
/// The main usage of this class is to collect data needed for the solver.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace mfem;

/// Exact solution, u and p, and r.h.s., f and g.
void u_exact(const Vector & x, Vector & u);
double p_exact(const Vector & x);
void f_exact(const Vector & x, Vector & f);
double g_exact(const Vector & x);
double natural_bc(const Vector & x);

/// Check Neumann BC
bool IsAllNeumannBoundary(const Array<int>& ess_bdr_attr);

/// Parameters for any general solver
struct IterSolveParameters
{
   int print_level = 0;
   int max_iter = 500;
   double abs_tol = 1e-12;
   double rel_tol = 1e-9;
};

/// Parameters for the Elasticity problem
struct ElastParameters : IterSolveParameters
{
   bool use_nodal_space = true;
   bool reorder_space = true;
};

// FESpaces for the elasticity problem
class ElasticitySpaces
{
   std::unique_ptr<FiniteElementCollection> fec_;
   L2_FECollection l2_fec_;
   std::unique_ptr<ParFiniteElementSpace> fes_;
   std::unique_ptr<ParFiniteElementSpace> l2_fes_;

   const Array<int>& ess_bdr_attr_;
   Array<int> all_bdr_attr_;

   // TODO Maybe
   std::unique_ptr<ParFiniteElementSpace> coarse_fes_;
   std::unique_ptr<ParFiniteElementSpace> coarse_l2_fes_;

   // TODO
   // std::vector<SparseMatrix> el_l2dof_;

   // int level_;
   // ElasticityData data_;

   // void MakeDofRelationTables(int level);
   // void DataFinalize();
public:
   ElasticitySpaces(int order, int num_refine, ParMesh *mesh,
                    const Array<int>& ess_attr,
                    const ElastParameters& param);

   /** This should be called each time when the mesh (where the FE spaces are
       defined) is refined. The spaces will be updated, and the prolongation for
       the spaces and other data needed for the div-free solver are stored. */
   // void CollectDFSData();

   // const DFSData& GetDFSData() const { return data_; }
   ParFiniteElementSpace* GetVh() const { return fes_.get(); }
   ParFiniteElementSpace* GetWh() const { return l2_fes_.get(); }
};

ElasticitySpaces::ElasticitySpaces(
   int order, int num_refine, ParMesh *mesh,
   const Array<int>& ess_attr,
   const ElastParameters& param)
   : l2_fec_(order, mesh->Dimension()), ess_bdr_attr_(ess_attr)
{
   if (mesh->GetElement(0)->GetType() == Element::TETRAHEDRON && order)
   {
      mfem_error("DFSDataCollector: High order spaces on tetrahedra are not supported");
   }

   int dim = mesh->Dimension();
   if (param.use_nodal_space)
   {
      fec_.reset(NULL);
      fes_.reset((ParFiniteElementSpace *)mesh->GetNodes()->FESpace());
   }
   else
   {
      fec_.reset(new H1_FECollection(order, dim));
      if (param.reorder_space)
      {
         fes_.reset(new ParFiniteElementSpace(mesh, fec_.get(), dim, Ordering::byNODES));
      }
      else
      {
         fes_.reset(new ParFiniteElementSpace(mesh, fec_.get(), dim, Ordering::byVDIM));
      }
   }

   all_bdr_attr_.SetSize(ess_attr.Size(), 1);
   l2_fes_.reset(new ParFiniteElementSpace(mesh, &l2_fec_));

   coarse_fes_.reset(new ParFiniteElementSpace(*fes_));
   coarse_l2_fes_.reset(new ParFiniteElementSpace(*l2_fes_));
}

/** Wrapper for assembling the discrete Elasticity problem (ex2p)
                     [ M  B^T ] [u] = [f]
                     [ B   0  ] [p] = [g]
    where:
       M = \int_\Omega u_h \cdot v_h dx + \int_\Omega (k e(u_h)) \cdot e(v_h) dx,
       B = \int_\Omega (div_h u_h) q_h dx,
       // TODO
       f = \int_\Omega f_exact v_h dx + \int_D natural_bc v_h dS,
       g = 0, \int_\Omega g_exact q_h dx,
       u_h, v_h \in V_h (Vector Lagrange finite element space),
       q_h \in W_h (piecewise discontinuous polynomials),
       D: subset of the boundary where natural boundary condition is imposed. */
class ElasticityProblem
{
   OperatorPtr M_;
   OperatorPtr B_;
   Vector rhs_;
   Vector ess_data_;
   ParGridFunction u_;
   ParGridFunction p_;
   ParMesh mesh_;
   shared_ptr<ParBilinearForm> mVarf_;
   shared_ptr<ParMixedBilinearForm> bVarf_;
   VectorFunctionCoefficient ucoeff_;
   FunctionCoefficient pcoeff_;
   ElasticitySpaces elas_spaces_;
   PWConstCoefficient mass_coeff;
   const IntegrationRule *irs_[Geometry::NumGeom];
public:
   ElasticityProblem(Mesh &mesh, int num_refines, int order, const char *coef_file,
                     Array<int> &ess_bdr, ElastParameters param);

   HypreParMatrix& GetM() { return *M_.As<HypreParMatrix>(); }
   HypreParMatrix& GetB() { return *B_.As<HypreParMatrix>(); }
   const Vector& GetRHS() { return rhs_; }
   const Vector& GetEssentialBC() { return ess_data_; }
   // const DFSData& GetDFSData() const { return dfs_spaces_.GetDFSData(); }
   void ShowError(const Vector &sol, bool verbose);
   void VisualizeSolution(const Vector &sol, string tag);
   shared_ptr<ParBilinearForm> GetMform() const { return mVarf_; }
   shared_ptr<ParMixedBilinearForm> GetBform() const { return bVarf_; }
};

ElasticityProblem::ElasticityProblem(Mesh &mesh, int num_refs, int order,
                                     const char *coef_file, Array<int> &ess_bdr,
                                     ElastParameters param)
   : mesh_(MPI_COMM_WORLD, mesh), ucoeff_(mesh.Dimension(), u_exact),
     pcoeff_(p_exact), elas_spaces_(order, num_refs, &mesh_, ess_bdr, param),
     mass_coeff()
{
   for (int l = 0; l < num_refs; l++)
   {
      mesh_.UniformRefinement();
      // dfs_spaces_.CollectDFSData();
   }

   Vector coef_vector(mesh.GetNE());
   coef_vector = 1.0;
   if (std::strcmp(coef_file, ""))
   {
      ifstream coef_str(coef_file);
      coef_vector.Load(coef_str, mesh.GetNE());
   }

   mass_coeff.UpdateConstants(coef_vector);
   VectorFunctionCoefficient fcoeff(mesh_.Dimension(), f_exact);
   FunctionCoefficient natcoeff(natural_bc);
   FunctionCoefficient gcoeff(g_exact);

   u_.SetSpace(elas_spaces_.GetVh());
   p_.SetSpace(elas_spaces_.GetWh());
   p_ = 0.0;
   u_ = 0.0;
   u_.ProjectBdrCoefficientNormal(ucoeff_, ess_bdr);

   // RHS
   ParLinearForm fform(elas_spaces_.GetVh());
   fform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   // TODO
   // fform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   fform.Assemble();

   ParLinearForm gform(elas_spaces_.GetWh());
   gform.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform.Assemble();

   // Bilinear forms
   mVarf_ = make_shared<ParBilinearForm>(elas_spaces_.GetVh());
   bVarf_ = make_shared<ParMixedBilinearForm>(elas_spaces_.GetVh(),
                                              elas_spaces_.GetWh());
   // TODO  Check ownership
   // Coefficients
   Vector lambda(mesh.attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);

   Vector mu(mesh.attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   // TODO check values
   mVarf_->AddDomainIntegrator(new VectorFEMassIntegrator(mass_coeff));
   // elast_int:  div + sym_grad
   mVarf_->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
   mVarf_->ComputeElementMatrices();
   mVarf_->Assemble();
   mVarf_->EliminateEssentialBC(ess_bdr, u_, fform);
   mVarf_->Finalize();
   M_.Reset(mVarf_->ParallelAssemble());

   bVarf_->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf_->Assemble();
   bVarf_->EliminateTrialDofs(ess_bdr, u_, gform);
   bVarf_->Finalize();
   B_.Reset(bVarf_->ParallelAssemble());

   rhs_.SetSize(M_->NumRows() + B_->NumRows());
   Vector rhs_block0(rhs_.GetData(), M_->NumRows());
   Vector rhs_block1(rhs_.GetData()+M_->NumRows(), B_->NumRows());
   fform.ParallelAssemble(rhs_block0);
   gform.ParallelAssemble(rhs_block1);

   ess_data_.SetSize(M_->NumRows() + B_->NumRows());
   ess_data_ = 0.0;
   Vector ess_data_block0(ess_data_.GetData(), M_->NumRows());
   u_.ParallelProject(ess_data_block0);

   int order_quad = max(2, 2*order+1);
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs_[i] = &(IntRules.Get(i, order_quad));
   }
}

void ElasticityProblem::ShowError(const Vector& sol, bool verbose)
{
   u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
   p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

   double err_u  = u_.ComputeL2Error(ucoeff_, irs_);
   double norm_u = ComputeGlobalLpNorm(2, ucoeff_, mesh_, irs_);
   double err_p  = p_.ComputeL2Error(pcoeff_, irs_);
   double norm_p = ComputeGlobalLpNorm(2, pcoeff_, mesh_, irs_);

   if (!verbose) { return; }
   cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
   cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
}

void ElasticityProblem::VisualizeSolution(const Vector& sol, string tag)
{
   int num_procs, myid;
   MPI_Comm_size(mesh_.GetComm(), &num_procs);
   MPI_Comm_rank(mesh_.GetComm(), &myid);

   u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
   p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

   const char vishost[] = "localhost";
   const int  visport   = 19916;
   socketstream u_sock(vishost, visport);
   u_sock << "parallel " << num_procs << " " << myid << "\n";
   u_sock.precision(8);
   u_sock << "solution\n" << mesh_ << u_ << "window_title 'Velocity ("
          << tag << " solver)'" << endl;
   MPI_Barrier(mesh_.GetComm());
   socketstream p_sock(vishost, visport);
   p_sock << "parallel " << num_procs << " " << myid << "\n";
   p_sock.precision(8);
   p_sock << "solution\n" << mesh_ << p_ << "window_title 'Pressure ("
          << tag << " solver)'" << endl;
}

int main(int argc, char *argv[])
{
#ifdef HYPRE_USING_GPU
   cout << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this miniapp\n"
        << "is NOT supported with the GPU version of hypre.\n\n";
   return 242;
#endif

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   StopWatch chrono;
   auto ResetTimer = [&chrono]() { chrono.Clear(); chrono.Start(); };

   // Parse command-line options.
   const char *mesh_file = "../../data/beam-hex.mesh";
   const char *coef_file = "";
   const char *ess_bdr_attr_file = "";
   int order = 0;
   int par_ref_levels = 2;
   bool show_error = false;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&par_ref_levels, "-r", "--ref",
                  "Number of parallel refinement steps.");
   args.AddOption(&coef_file, "-c", "--coef",
                  "Coefficient file to use.");
   args.AddOption(&ess_bdr_attr_file, "-eb", "--ess-bdr",
                  "Essential boundary attribute file to use.");
   args.AddOption(&show_error, "-se", "--show-error", "-no-se",
                  "--no-show-error",
                  "Show or not show approximation error.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   ElastParameters param;

   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   if (Mpi::Root() && par_ref_levels == 0)
   {
      std::cout << "WARNING: DivFree solver is equivalent to BDPMinresSolver "
                << "when par_ref_levels == 0.\n";
   }

   // Initialize the mesh, boundary attributes, and solver parameters
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int ser_ref_lvls =
      (int)ceil(log(Mpi::WorldSize()/mesh->GetNE())/log(2.)/dim);
   for (int i = 0; i < ser_ref_lvls; ++i)
   {
      mesh->UniformRefinement();
   }

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   if (std::strcmp(ess_bdr_attr_file, ""))
   {
      ifstream ess_bdr_attr_str(ess_bdr_attr_file);
      ess_bdr.Load(mesh->bdr_attributes.Max(), ess_bdr_attr_str);
   }
   if (IsAllNeumannBoundary(ess_bdr))
   {
      if (Mpi::Root())
      {
         cout << "\nSolution is not unique when Neumann boundary condition is "
              << "imposed on the entire boundary. \nPlease provide a different "
              << "boundary condition.\n";
      }
      delete mesh;
      return 0;
   }

   string line = "**********************************************************\n";

   ResetTimer();

   // Generate components of the saddle point problem
   ElasticityProblem elast(*mesh, par_ref_levels, order, coef_file, ess_bdr, param);
   // HypreParMatrix& M = darcy.GetM();
   // HypreParMatrix& B = darcy.GetB();
   delete mesh;

   return 0;
}

// TODO
// Exact solutions (for the Darcy problem lol)
void u_exact(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   double zi(x.Size() == 3 ? x(2) : 0.0);

   u(0) = - exp(xi)*sin(yi)*cos(zi);
   u(1) = - exp(xi)*cos(yi)*cos(zi);
   if (x.Size() == 3)
   {
      u(2) = exp(xi)*sin(yi)*sin(zi);
   }
}

double p_exact(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   double zi(x.Size() == 3 ? x(2) : 0.0);
   return exp(xi)*sin(yi)*cos(zi);
}

void f_exact(const Vector & x, Vector & f)
{
   f = 0.0;
}

double g_exact(const Vector & x)
{
   if (x.Size() == 3) { return -p_exact(x); }
   return 0;
}

double natural_bc(const Vector & x)
{
   return (-p_exact(x));
}

bool IsAllNeumannBoundary(const Array<int>& ess_bdr_attr)
{
   for (int attr : ess_bdr_attr) { if (attr == 0) { return false; } }
   return true;
}
