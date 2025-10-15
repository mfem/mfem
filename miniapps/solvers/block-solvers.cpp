// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//          ----------------------------------------------------------
//          Block Solvers Miniapp: Compare Saddle Point System Solvers
//          ----------------------------------------------------------
//
// This miniapp compares various linear solvers for the saddle point system
// obtained from mixed finite element discretization of the simple mixed Darcy
// problem in ex5p
//
//                            k*u + grad p = f
//                           - div u      = g
//
// with natural boundary condition -p = <given pressure>. We use a given exact
// solution (u,p) and compute the corresponding r.h.s. (f,g). We discretize
// with Raviart-Thomas finite elements (velocity u) and piecewise discontinuous
// polynomials (pressure p).
//
// The solvers being compared include:
//    1. MINRES preconditioned by a block diagonal preconditioner
//    2. The divergence free solver (couple and decoupled modes)
//    3. The Bramble-Pasciak solver (using BPCG or regular PCG)
//
// We recommend viewing example 5 before viewing this miniapp.
//
// Sample runs:
//
//    mpirun -np 8 block-solvers -r 2 -o 0
//    mpirun -np 8 block-solvers -m anisotropic.mesh -c anisotropic.coeff -eb anisotropic.bdr
//
//
// NOTE:  The coefficient file (provided through -c) defines a piecewise constant
//        scalar coefficient k. The number of entries in this file should equal
//        to the number of "element attributes" in the mesh file. The value of
//        the coefficient in elements with the i-th attribute is given by the
//        i-th entry of the coefficient file.
//
//
// NOTE:  The essential boundary attribute file (provided through -eb) defines
//        which attributes to impose essential boundary condition (on u). The
//        number of entries in this file should equal to the number of "boundary
//        attributes" in the mesh file. If the i-th entry of the file is nonzero
//        (respectively 0), essential (respectively natural) boundary condition
//        will be imposed on boundary with the i-th attribute.

#include <fstream>
#include <iostream>
#include <functional>

#include "mfem.hpp"
#include "bramble_pasciak.hpp"
#include "div_free_solver.hpp"

using namespace std;
using namespace mfem;
using namespace blocksolvers;

// Exact solution, u and p, and r.h.s., f and g.
void u_exact(const Vector & x, Vector & u);
real_t p_exact(const Vector & x);
void f_exact(const Vector & x, Vector & f);
real_t g_exact(const Vector & x);
real_t natural_bc(const Vector & x);

/** Wrapper for assembling the discrete Darcy problem (ex5p)
                     [ M  B^T ] [u] = [f]
                     [ B   0  ] [p] = [g]
    where:
       M = \int_\Omega (k u_h) \cdot v_h dx,
       B = -\int_\Omega (div_h u_h) q_h dx,
       f = \int_\Omega f_exact v_h dx + \int_D natural_bc v_h dS,
       g = \int_\Omega g_exact q_h dx,
       u_h, v_h \in R_h (Raviart-Thomas finite element space),
       q_h \in W_h (piecewise discontinuous polynomials),
       D: subset of the boundary where natural boundary condition is imposed. */
class DarcyProblem
{
   OperatorPtr M_, B_;
   Vector rhs_, ess_data_;
   ParGridFunction u_, p_;
   ParMesh mesh_;
   DFSSpaces dfs_spaces_;
   std::function<bool (int)> refine_fn = [&](int num_refs)
   {
      for (int l = 0; l < num_refs; l++)
      {
         mesh_.UniformRefinement();
         dfs_spaces_.CollectDFSData();
      }
      return true;
   };
   const bool dfs_refine_;
   ParBilinearForm mVarf_;
   ParMixedBilinearForm bVarf_;
   VectorFunctionCoefficient ucoeff_;
   FunctionCoefficient pcoeff_;
   PWConstCoefficient mass_coeff;
   const IntegrationRule *irs_[Geometry::NumGeom];
public:
   DarcyProblem(Mesh &mesh, int num_refines, int order, const char *coef_file,
                Array<int> &ess_bdr, DFSParameters param);

   const HypreParMatrix& GetM() const { return *M_.As<HypreParMatrix>(); }
   const HypreParMatrix& GetB() const { return *B_.As<HypreParMatrix>(); }
   const Vector& GetRHS() { return rhs_; }
   const Vector& GetEssentialBC() { return ess_data_; }
   const DFSData& GetDFSData() const { return dfs_spaces_.GetDFSData(); }
   void ShowError(const Vector &sol, bool verbose);
   void VisualizeSolution(const Vector &sol, std::string tag, int visport = 19916);
   ParBilinearForm& GetMform() { return mVarf_; }
   ParMixedBilinearForm& GetBform() { return bVarf_; }
};

DarcyProblem::DarcyProblem(Mesh &mesh, int num_refs, int order,
                           const char *coef_file, Array<int> &ess_bdr,
                           DFSParameters dfs_param)
   : mesh_(MPI_COMM_WORLD, mesh),
     dfs_spaces_(order, num_refs, &mesh_, ess_bdr, dfs_param),
     dfs_refine_(refine_fn(num_refs)),
     mVarf_(dfs_spaces_.GetHdivFES()),
     bVarf_(dfs_spaces_.GetHdivFES(), dfs_spaces_.GetL2FES()),
     ucoeff_(mesh.Dimension(), u_exact),
     pcoeff_(p_exact),
     mass_coeff()
{
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

   u_.SetSpace(dfs_spaces_.GetHdivFES());
   p_.SetSpace(dfs_spaces_.GetL2FES());
   p_ = 0.0;
   u_ = 0.0;
   u_.ProjectBdrCoefficientNormal(ucoeff_, ess_bdr);

   ParLinearForm fform(dfs_spaces_.GetHdivFES());
   fform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(natcoeff));
   fform.Assemble();

   ParLinearForm gform(dfs_spaces_.GetL2FES());
   gform.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform.Assemble();

   mVarf_.AddDomainIntegrator(new VectorFEMassIntegrator(mass_coeff));
   mVarf_.ComputeElementMatrices();
   mVarf_.Assemble();
   mVarf_.EliminateEssentialBC(ess_bdr, u_, fform);

   mVarf_.Finalize();
   M_.Reset(mVarf_.ParallelAssemble());

   bVarf_.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf_.Assemble();
   bVarf_.SpMat() *= -1.0;
   bVarf_.EliminateTrialEssentialBC(ess_bdr, u_, gform);
   bVarf_.Finalize();
   B_.Reset(bVarf_.ParallelAssemble());

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

void DarcyProblem::ShowError(const Vector& sol, bool verbose)
{
   u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
   p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

   real_t err_u  = u_.ComputeL2Error(ucoeff_, irs_);
   real_t norm_u = ComputeGlobalLpNorm(2, ucoeff_, mesh_, irs_);
   real_t err_p  = p_.ComputeL2Error(pcoeff_, irs_);
   real_t norm_p = ComputeGlobalLpNorm(2, pcoeff_, mesh_, irs_);

   if (!verbose) { return; }
   mfem::out << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
   mfem::out << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
}

void DarcyProblem::VisualizeSolution(const Vector& sol, string tag,
                                     int visport)
{
   int num_procs, myid;
   MPI_Comm_size(mesh_.GetComm(), &num_procs);
   MPI_Comm_rank(mesh_.GetComm(), &myid);

   u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
   p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

   const char vishost[] = "localhost";
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

bool IsAllNeumannBoundary(const Array<int>& ess_bdr_attr)
{
   for (int attr : ess_bdr_attr) { if (attr == 0) { return false; } }
   return true;
}

int main(int argc, char *argv[])
{
#ifdef HYPRE_USING_GPU
   mfem::out << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this miniapp\n"
             << "is NOT supported with the GPU version of hypre.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   StopWatch chrono;

   // Parse command-line options.
   const char *mesh_file = "../../data/beam-hex.mesh";
   const char *coef_file = "";
   const char *ess_bdr_attr_file = "";
   int order = 0;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   bool show_error = false;
   bool visualization = false;

   DFSParameters param;
   int visport = 19916;
   BPSParameters bps_param;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of serial refinement steps.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
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
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (Mpi::Root()) { args.ParseCheck(); }

   if (Mpi::Root() && par_ref_levels == 0)
   {
      mfem::out << "WARNING: DivFree solver is equivalent to BDPMinresSolver "
                << "when par_ref_levels == 0.\n";
   }

   // Initialize the mesh, boundary attributes, and solver parameters
   Mesh *mesh = new Mesh(mesh_file, 1, 1);

   int dim = mesh->Dimension();

   if (Mpi::Root())
   {
      mfem::out << "Number of serial refinements:   " << ser_ref_levels << "\n"
                << "Number of parallel refinements: " << par_ref_levels << "\n";
   }

   for (int i = 0; i < ser_ref_levels; ++i)
   {
      mesh->UniformRefinement();
   }

   if (Mpi::Root() && Mpi::WorldSize() > mesh->GetNE())
   {
      mfem::out << "\nWARNING: Number of processors is greater than the number of "
                << "elements in the mesh.\n"
                << "Number of processors: " << Mpi::WorldSize() << "\n"
                << "Number of elements:   " << mesh->GetNE() << "\n\n";
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
         mfem::out << "\nSolution is not unique when Neumann boundary condition is "
                   << "imposed on the entire boundary. \nPlease provide a different "
                   << "boundary condition.\n";
      }
      delete mesh;
      return 0;
   }

   string line = "**********************************************************\n";

   chrono.Restart();

   // Generate components of the saddle point problem
   DarcyProblem darcy(*mesh, par_ref_levels, order, coef_file, ess_bdr, param);
   const HypreParMatrix &M = darcy.GetM();
   const HypreParMatrix &B = darcy.GetB();
   const DFSData& DFS_data = darcy.GetDFSData();
   delete mesh;

   if (Mpi::Root())
   {
      mfem::out << line << "System assembled in " << chrono.RealTime() << "s.\n";
      mfem::out << "Dimension of the physical space: " << dim << "\n";
      mfem::out << "Size of the discrete Darcy system: " << M.M() + B.M() << "\n";
      if (par_ref_levels > 0)
      {
         mfem::out << "Dimension of the divergence free subspace: "
                   << DFS_data.C.back().Ptr()->NumCols() << "\n\n";
      }
   }

   // Setup various solvers for the discrete problem
   std::map<const DarcySolver*, real_t> setup_time;
   chrono.Restart();
   BDPMinresSolver bdp(M, B, param);
   setup_time[&bdp] = chrono.RealTime();

   chrono.Restart();
   DivFreeSolver dfs_dm(M, B, DFS_data);
   setup_time[&dfs_dm] = chrono.RealTime();

   chrono.Restart();
   const_cast<bool&>(DFS_data.param.coupled_solve) = true;
   DivFreeSolver dfs_cm(M, B, DFS_data);
   setup_time[&dfs_cm] = chrono.RealTime();

   chrono.Restart();
   BramblePasciakSolver bp_bpcg(darcy.GetMform(), darcy.GetBform(), bps_param);
   setup_time[&bp_bpcg] = chrono.RealTime();

   chrono.Restart();
   bps_param.use_bpcg = false;
   BramblePasciakSolver bp_pcg(darcy.GetMform(), darcy.GetBform(), bps_param);
   setup_time[&bp_pcg] = chrono.RealTime();

   std::map<const DarcySolver*, std::string> solver_to_name;
   solver_to_name[&bdp] = "Block-diagonal-preconditioned MINRES";
   solver_to_name[&dfs_dm] = "Divergence free (decoupled mode)";
   solver_to_name[&dfs_cm] = "Divergence free (coupled mode)";
   solver_to_name[&bp_bpcg] = "Bramble Pasciak CG (using BPCG)";
   solver_to_name[&bp_pcg] = "Bramble Pasciak CG (using regular PCG)";

   // Solve the problem using all solvers
   for (const auto& solver_pair : solver_to_name)
   {
      auto& solver = solver_pair.first;
      auto& name = solver_pair.second;

      Vector sol = darcy.GetEssentialBC();

      chrono.Restart();
      solver->Mult(darcy.GetRHS(), sol);
      chrono.Stop();

      if (Mpi::Root())
      {
         mfem::out << line << name << " solver:\n   Setup time: "
                   << setup_time[solver] << "s.\n   Solve time: "
                   << chrono.RealTime() << "s.\n   Total time: "
                   << setup_time[solver] + chrono.RealTime() << "s.\n"
                   << "   Iteration count: " << solver->GetNumIterations() <<"\n\n";
      }
      if (show_error && std::strcmp(coef_file, "") == 0)
      {
         darcy.ShowError(sol, Mpi::Root());
      }
      else if (show_error && Mpi::Root())
      {
         mfem::out << "Exact solution is unknown for coefficient '" << coef_file
                   << "'.\nApproximation error is computed in this case!\n\n";
      }

      if (visualization) { darcy.VisualizeSolution(sol, name, visport); }

   }

   return 0;
}

void u_exact(const Vector & x, Vector & u)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(x.Size() == 3 ? x(2) : 0.0);

   u(0) = - exp(xi)*sin(yi)*cos(zi);
   u(1) = - exp(xi)*cos(yi)*cos(zi);
   if (x.Size() == 3)
   {
      u(2) = exp(xi)*sin(yi)*sin(zi);
   }
}

real_t p_exact(const Vector & x)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(x.Size() == 3 ? x(2) : 0.0);
   return exp(xi)*sin(yi)*cos(zi);
}

void f_exact(const Vector & x, Vector & f)
{
   f = 0.0;
}

real_t g_exact(const Vector & x)
{
   if (x.Size() == 3) { return -p_exact(x); }
   return 0;
}

real_t natural_bc(const Vector & x)
{
   return (-p_exact(x));
}
