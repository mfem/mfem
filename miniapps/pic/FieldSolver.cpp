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

#include "FieldSolver.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "ParticleMover.hpp"

namespace
{
   constexpr mfem::real_t EPSILON = 1.0;

   mfem::real_t ComputeGlobalSum(mfem::ParLinearForm& lf)
   {
      std::unique_ptr<mfem::HypreParVector> lf_true(lf.ParallelAssemble());
      const mfem::real_t local_sum = lf_true->Sum();
      mfem::real_t global_sum = 0.0;
      MPI_Allreduce(&local_sum, &global_sum, 1,
                    mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM,
                    lf.ParFESpace()->GetComm());
      return global_sum;
   }
}

using namespace std;
using namespace mfem;
using namespace mfem::common;

FieldSolver::FieldSolver(ParFiniteElementSpace* phi_fes,
                         ParFiniteElementSpace* E_fes,
                         FindPointsGSLIB& E_finder_, real_t diffusivity,
                         bool precompute_neutralizing_const_,
                         int efield_output_interval_,
                         int phi_output_interval_,
                         int rho_output_interval_,
                         int field_sample_resolution_)
    : precompute_neutralizing_const(precompute_neutralizing_const_),
      E_finder(E_finder_),
      b(phi_fes),
      efield_output_interval(efield_output_interval_),
      phi_output_interval(phi_output_interval_),
      rho_output_interval(rho_output_interval_)
{
   MFEM_VERIFY(field_sample_resolution_ > 0,
               "field sample resolution must be positive.");
   efield_sample_nx = field_sample_resolution_;
   efield_sample_ny = field_sample_resolution_;

   ParMesh* pmesh = phi_fes->GetParMesh();
   real_t local_domain_volume = 0.0;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      local_domain_volume += pmesh->GetElementVolume(i);
   }
   MPI_Allreduce(&local_domain_volume, &domain_volume, 1, MPI_DOUBLE, MPI_SUM,
                 phi_fes->GetParMesh()->GetComm());

   {
      ParBilinearForm dm(phi_fes);
      ConstantCoefficient epsilon(EPSILON);
      dm.AddDomainIntegrator(new DiffusionIntegrator(epsilon));
      dm.Assemble();
      dm.Finalize();
      diffusion_matrix = dm.ParallelAssemble();
   }

   {
      ParBilinearForm dm(phi_fes);
      dm.AddDomainIntegrator(new DiffusionIntegrator);
      dm.Assemble();
      dm.Finalize();
      HypreParMatrix* temp_diffusion_matrix = dm.ParallelAssemble();

      // Build mass matrix M
      ParBilinearForm m(phi_fes);
      m.AddDomainIntegrator(new MassIntegrator());
      m.Assemble();
      m.Finalize();
      HypreParMatrix* M = m.ParallelAssemble();

      // Build discrete K^4 using the Poisson stiffness matrix (diffusion_matrix)
      HypreParMatrix* K2 =
         ParMult(temp_diffusion_matrix, temp_diffusion_matrix);
      HypreParMatrix* K4 = ParMult(K2, K2);

      // Form the p=4 hyper-diffusion operator: M + diffusivity * K^4
      M_plus_cK_matrix = Add(1.0, *M, diffusivity, *K4);

      delete K4;
      delete K2;
      delete temp_diffusion_matrix;
      delete M;
   }

   {
      grad_interpolator = new ParDiscreteLinearOperator(phi_fes, E_fes);
      grad_interpolator->AddDomainInterpolator(new GradientInterpolator);
      grad_interpolator->Assemble();
   }

   InitializeFieldSamplingGrid(phi_fes);
}

FieldSolver::~FieldSolver()
{
   delete diffusion_matrix;
   delete M_plus_cK_matrix;
   delete precomputed_neutralizing_lf;
   delete grad_interpolator;
}

const ParLinearForm& FieldSolver::ComputeNeutralizingRHS(
   ParFiniteElementSpace* pfes, const ParticleVector& Q, MPI_Comm comm)
{
   const int npt = Q.Size();
   const Array<unsigned int>& code = E_finder.GetCode();

   if (!precompute_neutralizing_const || precomputed_neutralizing_lf == nullptr)
   {
      real_t local_sum = 0.0;
      for (int p = 0; p < npt; ++p)
      {
         MFEM_ASSERT(code[p] != 2, "Particle " << p << " not found.");
         local_sum += Q(p);
      }

      real_t global_sum = 0.0;
      MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

      neutralizing_const = -global_sum / domain_volume;
      if (Mpi::Root())
      {
         cout << "Total charge: " << global_sum
              << ", Domain volume: " << domain_volume
              << ", Neutralizing constant: " << neutralizing_const << endl;
         if (precompute_neutralizing_const)
         {
            cout << "Further updates will use this precomputed neutralizing "
                    "constant."
                 << endl;
         }
      }

      delete precomputed_neutralizing_lf;
      precomputed_neutralizing_lf = new ParLinearForm(pfes);
      *precomputed_neutralizing_lf = 0.0;
      ConstantCoefficient neutralizing_coeff(neutralizing_const);
      precomputed_neutralizing_lf->AddDomainIntegrator(
         new DomainLFIntegrator(neutralizing_coeff));
      precomputed_neutralizing_lf->Assemble();
   }
   return *precomputed_neutralizing_lf;
}

void FieldSolver::DepositCharge(ParFiniteElementSpace* pfes,
                                const ParticleVector& Q, ParLinearForm& b)
{
   const int npt = Q.Size();
   ParMesh* pmesh = pfes->GetParMesh();
   const int dim = pmesh->SpaceDimension();
   int curr_rank;
   MPI_Comm_rank(pmesh->GetComm(), &curr_rank);

   // 0: inside, 1: boundary, 2: not found.
   const Array<unsigned int>& code = E_finder.GetCode();
   const Array<unsigned int>& proc = E_finder.GetProc();
   const Array<unsigned int>& elem = E_finder.GetElem();
   const Vector& rref = E_finder.GetReferencePosition();

   Array<int> dofs;

   for (int p = 0; p < npt; ++p)
   {
      MFEM_ASSERT(code[p] != 2, "Particle " << p << " not found.");

      MFEM_ASSERT((int)proc[p] == curr_rank,
                  "Particle " << p << " found in element owned by rank "
                              << proc[p] << " but current rank is " << curr_rank
                              << "." << endl
                              << "You must call redistribute everytime before "
                                 "updating the density grid function.");
      const int e = elem[p];

      IntegrationPoint ip;
      ip.Set(rref.GetData() + dim * p, dim);

      const FiniteElement& fe = *pfes->GetFE(e);
      const int ldofs = fe.GetDof();

      Vector shape(ldofs);
      fe.CalcShape(ip, shape);

      pfes->GetElementDofs(e, dofs);

      const real_t q_p = Q(p);
      b.AddElementVector(dofs, q_p, shape);
   }
}

void FieldSolver::UpdatePhiGridFunction(ParticleSet& particles,
                                        ParGridFunction& phi_gf,
                                        ParGridFunction& rho_gf)
{
   ParFiniteElementSpace* pfes = phi_gf.ParFESpace();

   ParticleVector& Q = particles.Field(ParticleMover::CHARGE);

   MPI_Comm comm = pfes->GetComm();
   b = ComputeNeutralizingRHS(pfes, Q, comm);
   if (Mpi::Root())
   {
      cout << "Total charge A: " << ComputeGlobalSum(b) << "\t";
   }
   else { ComputeGlobalSum(b); }

   DepositCharge(pfes, Q, b);
   if (Mpi::Root())
   {
      cout << "Total charge B: " << ComputeGlobalSum(b) << "\t";
   }
   else { ComputeGlobalSum(b); }

   DiffuseRHS(b, rho_gf);
   if (Mpi::Root())
   {
      cout << "Total charge C: " << ComputeGlobalSum(b) << endl;
   }
   else { ComputeGlobalSum(b); }

   HypreParVector B(pfes);
   b.ParallelAssemble(B);

   phi_gf = 0.0;
   HypreParVector Phi_true(pfes);
   Phi_true = 0.0;

   HyprePCG solver(diffusion_matrix->GetComm());
   solver.SetOperator(*diffusion_matrix);
   solver.SetTol(1e-12);
   solver.SetMaxIter(200);
   solver.SetPrintLevel(0);

   HypreBoomerAMG prec(*diffusion_matrix);
   prec.SetPrintLevel(0);
   solver.SetPreconditioner(prec);

   OrthoSolver ortho(comm);
   ortho.SetSolver(solver);
   ortho.Mult(B, Phi_true);

   phi_gf.Distribute(Phi_true);
}

void FieldSolver::UpdateEGridFunction(ParGridFunction& phi_gf,
                                      ParGridFunction& E_gf)
{
   grad_interpolator->Mult(phi_gf, E_gf);
   E_gf.Neg();
}

bool FieldSolver::ShouldWriteSample(int interval, int timestep) const
{
   return interval >= 0 &&
          (interval == 0 || (timestep > 0 && timestep % interval == 0));
}

void FieldSolver::SaveFieldSamples(const ParGridFunction& E_gf,
                                   const ParGridFunction& phi_gf,
                                   const ParGridFunction& rho_gf, int timestep)
{
   if (ShouldWriteSample(efield_output_interval, timestep) &&
       efield_last_output_step != timestep)
   {
      SampleAndWriteEField(E_gf, timestep);
      efield_last_output_step = timestep;
   }

   if (ShouldWriteSample(phi_output_interval, timestep) &&
       phi_last_output_step != timestep)
   {
      SampleAndWriteScalarField(phi_gf, "phi", timestep);
      phi_last_output_step = timestep;
   }

   if (ShouldWriteSample(rho_output_interval, timestep) &&
       rho_last_output_step != timestep)
   {
      SampleAndWriteScalarField(rho_gf, "rho", timestep);
      rho_last_output_step = timestep;
   }
}

void FieldSolver::InitializeFieldSamplingGrid(ParFiniteElementSpace* sample_fes)
{
   ParMesh* pmesh = sample_fes->GetParMesh();
   if (pmesh->SpaceDimension() != 2)
   {
      efield_sampling_enabled = false;
      if (Mpi::Root())
      {
         cout << "Skipping field " << efield_sample_nx << "x"
              << efield_sample_ny << " sampling: only supported in 2D." << endl;
      }
      return;
   }

   Vector bb_min, bb_max;
   pmesh->GetBoundingBox(bb_min, bb_max, 1);
   const real_t xmin = bb_min(0);
   const real_t xmax = bb_max(0);
   const real_t ymin = bb_min(1);
   const real_t ymax = bb_max(1);

   const real_t dx = (xmax - xmin) / efield_sample_nx;
   const real_t dy = (ymax - ymin) / efield_sample_ny;

   efield_sample_npts = efield_sample_nx * efield_sample_ny;
   efield_sample_points.SetSize(2 * efield_sample_npts);
   efield_sample_values.SetSize(2 * efield_sample_npts);
   scalar_sample_values.SetSize(efield_sample_npts);

   // byNODES ordering: [x0..xN-1, y0..yN-1]
   // Sample at cell centers, not domain boundaries.
   for (int j = 0; j < efield_sample_ny; ++j)
   {
      for (int i = 0; i < efield_sample_nx; ++i)
      {
         const int p = j * efield_sample_nx + i;
         efield_sample_points[p] = xmin + (i + 0.5) * dx;
         efield_sample_points[efield_sample_npts + p] = ymin + (j + 0.5) * dy;
      }
   }

   efield_sampling_enabled = true;
}

void FieldSolver::SampleAndWriteEField(const ParGridFunction& E_gf,
                                       int timestep)
{
   if (!efield_sampling_enabled) { return; }

   // byVDIM ordering: [Ex0,Ey0, Ex1,Ey1, ...]
   E_finder.Interpolate(efield_sample_points, E_gf, efield_sample_values,
                        Ordering::byNODES, Ordering::byVDIM);

   if (!Mpi::Root()) { return; }

   ostringstream filename;
   filename << "E_field_samples_" << efield_sample_nx << "x" << efield_sample_ny
            << "_step_" << setw(6) << setfill('0') << timestep << ".csv";

   ofstream out(filename.str());
   out << setfill(' ') << setprecision(17);
   out << "step,i,j,x,y,Ex,Ey\n";
   for (int j = 0; j < efield_sample_ny; ++j)
   {
      for (int i = 0; i < efield_sample_nx; ++i)
      {
         const int p = j * efield_sample_nx + i;
         const real_t x = efield_sample_points[p];
         const real_t y = efield_sample_points[efield_sample_npts + p];
         const real_t ex = efield_sample_values[2 * p];
         const real_t ey = efield_sample_values[2 * p + 1];
         out << timestep << "," << i << "," << j << "," << x << "," << y << ","
             << ex << "," << ey << "\n";
      }
   }
}

void FieldSolver::SampleAndWriteScalarField(const ParGridFunction& field,
                                            const std::string& field_name,
                                            int timestep)
{
   if (!efield_sampling_enabled) { return; }

   E_finder.Interpolate(efield_sample_points, field, scalar_sample_values,
                        Ordering::byNODES, Ordering::byNODES);

   if (!Mpi::Root()) { return; }

   ostringstream filename;
   filename << field_name << "_field_samples_" << efield_sample_nx << "x"
            << efield_sample_ny << "_step_" << setw(6) << setfill('0')
            << timestep << ".csv";

   ofstream out(filename.str());
   out << setfill(' ') << setprecision(17);
   out << "step,i,j,x,y," << field_name << "\n";
   for (int j = 0; j < efield_sample_ny; ++j)
   {
      for (int i = 0; i < efield_sample_nx; ++i)
      {
         const int p = j * efield_sample_nx + i;
         const real_t x = efield_sample_points[p];
         const real_t y = efield_sample_points[efield_sample_npts + p];
         const real_t value = scalar_sample_values[p];
         out << timestep << "," << i << "," << j << "," << x << "," << y
             << "," << value << "\n";
      }
   }
}

void FieldSolver::DiffuseRHS(ParLinearForm& b, ParGridFunction& rho_gf)
{
   HypreParVector* B = b.ParallelAssemble();
   ParFiniteElementSpace* pfes = rho_gf.ParFESpace();

   // Warm-start from the previous rho state to reduce linear iterations.
   HypreParVector Rho_true(pfes);
   rho_gf.GetTrueDofs(Rho_true);

   HyprePCG solver(M_plus_cK_matrix->GetComm());
   solver.SetOperator(*M_plus_cK_matrix);
   solver.SetTol(1e-12);
   solver.SetMaxIter(4000);
   solver.SetPrintLevel(0);
   solver.iterative_mode = true;

   HypreBoomerAMG prec(*M_plus_cK_matrix);
   prec.SetPrintLevel(0);
   solver.SetPreconditioner(prec);

   solver.Mult(*B, Rho_true);
   delete B;

   rho_gf.SetFromTrueDofs(Rho_true);

   b = 0.0;

   b.GetDLFI()->DeleteAll();

   GridFunctionCoefficient rho_coeff(&rho_gf);
   b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
   b.Assemble();
}

real_t FieldSolver::ComputeFieldEnergy(const ParGridFunction& E_gf) const
{
   const ParFiniteElementSpace* fes = E_gf.ParFESpace();
   const ParMesh* pmesh = fes->GetParMesh();

   const int order = fes->GetMaxElementOrder();
   const int qorder = std::max(2, 2 * order + 1);

   const IntegrationRule* irs[Geometry::NumGeom];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      irs[g] = &IntRules.Get(g, qorder);
   }

   Vector zero(pmesh->Dimension());
   zero = 0.0;
   VectorConstantCoefficient zero_vec(zero);

   const real_t E_l2 = E_gf.ComputeL2Error(zero_vec, irs);
   return 0.5 * EPSILON * E_l2 * E_l2;
}