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
//           -----------------------------------------------------
//           2D2V Particle-In-Cell (PIC) Simulation
//           -----------------------------------------------------
//
// This miniapp performs a 2D2V (2 spatial dimensions, 2 velocity dimensions)
// Particle-In-Cell simulation of multiple charged particles subject to
// electric field forces.
//
//                           dp/dt = q E
//
// The method used is explicit time integration with a leap-frog scheme.
//
// The electric field is computed from the particle charge distribution using
// a Poisson solver. The particle trajectories are computed within a periodic
// 2D domain.
//
// Solution process (per timestep, repeating steps 1-6):
//   (1) Deposit charge from particles to grid via Dirac delta function
//       to form the RHS of the Poisson equation
//   (2) Solve Poisson equation (-Δφ = ρ - ρ_0) to compute potential φ, where
//       ρ_0 is a constant neutralizing term that enforces global charge
//       neutrality.
//   (3) Compute electric field E = -∇φ from the potential
//   (4) Interpolate E-field to particle positions
//   (5) Push particles using leap-frog scheme (update momentum and position)
//   (6) Redistribute particles across processors
//
// Compile with: make electrostatic-2d2v
//
// Sample runs:
//
//   Linear Landau damping test case (Ricketson & Hu, 2025):
//      mpirun -n 4 ./electrostatic-2d2v -rdf 1 -npt 409600 -k 0.2855993321 -a 0.05 -nt 200 -nx 32 -ny 32 -O 1 -q 0.001181640625 -m 0.001181640625 -ocf 1000 -dt 0.1

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../../general/text.hpp"
#include "../common/fem_extras.hpp"
#include "../common/particles_extras.hpp"
#include "../common/pfem_extras.hpp"
#include "mfem.hpp"

#define EPSILON 1  // ε_0

using namespace std;
using namespace mfem;
using namespace mfem::common;

struct PICContext
{
   // mesh related parameters
   int order = 1;
   int nx = 100;
   int ny = 100;
   real_t L_x = 1.0;

   int ordering = 1;
   int npt = 1000;
   real_t q = 1.0;
   real_t m = 1.0;

   real_t k = 1.0;
   real_t alpha = 0.1;

   real_t dt = 1e-2;
   real_t t_init = 0.0;

   int nt = 1000;
   int redist_freq = 1e6;
   int output_csv_freq = 1;

   bool visualization = true;
   int visport = 19916;
   bool reproduce = true;
} ctx;

/// This class implements explicit time integration for charged particles
/// in an electric field using ParticleSet.
class ParticleMover
{
public:
   enum Fields
   {
      MASS,    // vdim = 1
      CHARGE,  // vdim = 1
      MOM,     // vdim = dim
      EFIELD  // vdim = dim
   };

protected:
   /// Pointers to E field GridFunctions
   ParGridFunction* E_gf;

   /// FindPointsGSLIB object for E field mesh
   FindPointsGSLIB& E_finder;

   /// ParticleSet of charged particles
   std::unique_ptr<ParticleSet> charged_particles;

   /// Temporary vectors for particle computation
   mutable Vector pm_, pp_;

   /// Get values of a ParGridFunction at given particle coordinates
   static void GetValues(const ParticleVector& coords, FindPointsGSLIB& finder,
                         ParGridFunction& gf, ParticleVector& pv);

   /// Single particle Boris step
   void ParticleStep(Particle& part, real_t& dt,
                     real_t L_x, bool zeroth_step = false);

public:
   ParticleMover(MPI_Comm comm, ParGridFunction* E_gf_, FindPointsGSLIB& E_finder_,
       int num_particles,
       Ordering::Type pdata_ordering);

   /// Initialize charged particles with given parameters
   void InitializeChargedParticles(const real_t& k, const real_t& alpha,
                                   real_t m, real_t q, real_t L_x,
                                   bool reproduce = false);

   /// Interpolate E field to particles
   void InterpolateE();

   /// Find Particles in mesh corresponding to E and field
   void FindParticles();

   /// Advance particles one time step using Boris algorithm
   void Step(real_t& t, real_t& dt, real_t L_x, bool zeroth_step = false);

   /// Redistribute particles across processors
   void Redistribute();

   /// Get reference to ParticleSet
   ParticleSet& GetParticles() { return *charged_particles; }
};

class FieldSolver
{
private:
   real_t domain_volume;
   real_t neutralizing_const;
   ParLinearForm* precomputed_neutralizing_lf = nullptr;
   bool neutralizing_const_computed = false;
   bool precompute_neutralizing_const = false;
   // Diffusion matrix
   HypreParMatrix* DiffusionMatrix;
   FindPointsGSLIB& E_finder;
   int visport;
   bool visualization;
   socketstream vis_e;
   socketstream vis_phi;

public:
   FieldSolver(ParGridFunction& phi_gf, FindPointsGSLIB& E_finder_,
                       int visport_, bool visualization_,
                       bool precompute_neutralizing_const_ = false);

   ~FieldSolver();

   /// Update the phi_gf grid function from the particles.
   /// Solve periodic Poisson: DiffusionMatrix * phi = (rho - <rho>)
   /// with zero-mean enforcement via OrthoSolver.
   void UpdatePhiGridFunction(ParticleSet& particles, ParGridFunction& phi_gf,
                              ParGridFunction& E_gf);

   /// Output energy (kinetic, field and total) with stdout and csv output
   void TotalEnergyValidation(const ParticleSet& particles,
                              const ParGridFunction& E_gf);
};

/// Prints the program's logo to the given output stream
void display_banner(ostream& os);

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   // Mesh parameters
   int dim = 2;

   if (Mpi::Root()) { display_banner(cout); }

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.order, "-O", "--order",
                  "Finite element polynomial degree");
   args.AddOption(&ctx.nx, "-nx", "--num-x",
                  "Number of elements in the x direction.");
   args.AddOption(&ctx.ny, "-ny", "--num-y",
                  "Number of elements in the y direction.");
   args.AddOption(&ctx.q, "-q", "--charge", "Particle charge.");
   args.AddOption(&ctx.m, "-m", "--mass", "Particle mass.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time Step.");
   args.AddOption(&ctx.t_init, "-ti", "--initial-time", "Initial Time.");
   args.AddOption(&ctx.nt, "-nt", "--num-timesteps", "Number of timesteps.");
   args.AddOption(&ctx.npt, "-npt", "--num-particles",
                  "Total number of particles.");
   args.AddOption(&ctx.k, "-k", "--k", "K parameter for initial distribution.");
   args.AddOption(&ctx.alpha, "-a", "--alpha",
                  "Alpha parameter for initial distribution.");
   args.AddOption(&ctx.ordering, "-o", "--ordering",
                  "Ordering of particle data. 0 = byNODES, 1 = byVDIM.");
   args.AddOption(&ctx.redist_freq, "-rdf", "--redist-freq",
                  "Redistribution and update E_gf frequency.");
   args.AddOption(&ctx.output_csv_freq, "-ocf", "--output-csv-freq",
                  "Output CSV frequency.");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&ctx.reproduce, "-rep", "--reproduce", "-no-rep",
                  "--no-reproduce",
                  "Enable or disable reproducible random seed.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }
   ctx.L_x = 2.0 * M_PI / ctx.k;

   ParGridFunction* E_gf = nullptr;

   // build up E_gf
   // 1. make a 2D Cartesian Mesh
   Mesh serial_mesh(Mesh::MakeCartesian2D(
                       ctx.nx, ctx.ny, Element::QUADRILATERAL, false, ctx.L_x, ctx.L_x));
   std::vector<Vector> translations = {Vector({ctx.L_x, 0.0}),
                                       Vector({0.0, ctx.L_x})
                                      };
   Mesh periodic_mesh(Mesh::MakePeriodic(
                         serial_mesh, serial_mesh.CreatePeriodicVertexMapping(translations)));
   // 2. parallelize the mesh
   ParMesh mesh(MPI_COMM_WORLD, periodic_mesh);
   serial_mesh.Clear();    // the serial mesh is no longer needed
   periodic_mesh.Clear();  // the periodic mesh is no longer needed

   // 3. Build the E_finder
   mesh.EnsureNodes();
   FindPointsGSLIB E_finder(mesh);

   // 4. Define a finite element space on the parallel mesh
   H1_FECollection sca_fec(ctx.order, dim);
   ParFiniteElementSpace sca_fespace(&mesh, &sca_fec);
   ND_FECollection vec_fec(ctx.order, dim);
   ParFiniteElementSpace vec_fespace(&mesh, &vec_fec);

   // 5. Prepare an empty phi_gf and E_gf for later use
   ParGridFunction phi_gf(&sca_fespace);
   E_gf = new ParGridFunction(&vec_fespace);
   phi_gf = 0.0;  // Initialize phi_gf to zero
   *E_gf = 0.0;   // Initialize E_gf to zero

   // 6. Build the grid function updates
   FieldSolver field_solver(phi_gf, E_finder, ctx.visport,
                                  ctx.visualization, true);
   Ordering::Type ordering_type =
      ctx.ordering == 0 ? Ordering::byNODES : Ordering::byVDIM;

   // 7. Initialize ParticleMover
   int num_particles = ctx.npt / size + (rank < (ctx.npt % size) ? 1 : 0);
   ParticleMover particle_mover(MPI_COMM_WORLD, E_gf, E_finder, num_particles, ordering_type);
   particle_mover.InitializeChargedParticles(ctx.k, ctx.alpha, ctx.m,
                                  ctx.q, ctx.L_x, ctx.reproduce);

   real_t t = ctx.t_init;
   real_t dt = ctx.dt;

   // set up timer
   auto start_time = std::chrono::high_resolution_clock::now();
   for (int step = 1; step <= ctx.nt; step++)
   {
      // Redistribute
      if (ctx.redist_freq > 0 && (step % ctx.redist_freq == 0 || step == 1) &&
          particle_mover.GetParticles().GetGlobalNParticles() > 0)
      {
         // Redistribute
         particle_mover.Redistribute();

         // Update phi_gf from particles
         field_solver.UpdatePhiGridFunction(particle_mover.GetParticles(), phi_gf, *E_gf);

         field_solver.TotalEnergyValidation(particle_mover.GetParticles(), *E_gf);
      }

      if (step == 1)
      {
         real_t neg_half_dt = -dt / 2.0;
         // Perform a "zeroth" step to move p half step backward
         particle_mover.Step(t, neg_half_dt, ctx.L_x,true);
      }
      // Step the ParticleMover
      particle_mover.Step(t, dt, ctx.L_x);
      if (Mpi::Root())
      {
         mfem::out << "Step: " << step << " | Time: " << t;
         // Print timing information every 100 steps
         if (step % 10 == 0)
         {
            std::chrono::duration<double> elapsed =
               std::chrono::high_resolution_clock::now() - start_time;
            mfem::out << " | Time per step: " << elapsed.count() / step;
         }
         mfem::out << endl;
      }
      // Output particle data to CSV
      if (step % ctx.output_csv_freq == 0 || step == 1)
      {
         std::string csv_prefix = "PIC_Part_";
         Array<int> field_idx{2}, tag_idx;
         std::string file_name =
            csv_prefix + mfem::to_padded_string(step, 6) + ".csv";
         particle_mover.GetParticles().PrintCSV(file_name.c_str(), field_idx, tag_idx);
      }
   }

   // Clean up
   delete E_gf;
}

void ParticleMover::GetValues(const ParticleVector& coords, FindPointsGSLIB& E_finder,
                    ParGridFunction& gf, ParticleVector& pv)
{
   Mesh &mesh = *gf.FESpace()->GetMesh();
   E_finder.Interpolate(gf, pv);
   Ordering::Reorder(pv, pv.GetVDim(), gf.FESpace()->GetOrdering(),
                     pv.GetOrdering());
}

void ParticleMover::ParticleStep(Particle& part, real_t& dt, real_t L_x, bool zeroth_step)
{
   Vector& x = part.Coords();
   real_t m = part.FieldValue(MASS);
   real_t q = part.FieldValue(CHARGE);
   Vector& p = part.Field(MOM);
   Vector& e = part.Field(EFIELD);
   // Compute half of the contribution from q E
   add(p, 0.5 * dt * q, e, pm_);

   // --- Simplified update: no magnetic field ---
   pp_ = pm_;  // only include E contribution

   // Update the momentum (full electric contribution)
   add(pp_, 0.5 * dt * q, e, p);

   if (zeroth_step) { return; }

   // Update the position
   x.Add(dt / m, p);

   // periodic boundary: wrap around using ctx mesh extents
   for (int d = 0; d < x.Size(); d++)
   {
      x(d) = std::fmod(x(d), L_x);
      if (x(d) < 0.0) { x(d) += L_x; }
   }
}

ParticleMover::ParticleMover(MPI_Comm comm, ParGridFunction* E_gf_, FindPointsGSLIB& E_finder_,
         int num_particles,
         Ordering::Type pdata_ordering)
   : E_gf(E_gf_), E_finder(E_finder_)
{
   MFEM_VERIFY(E_gf, "Must pass an E field to ParticleMover.");

   int dim = E_gf->ParFESpace()->GetMesh()->SpaceDimension();

   pm_.SetSize(dim);
   pp_.SetSize(dim);

   // Create particle set: 2 scalars of mass and charge,
   // 2 vectors of size space dim for momentum and e field
   Array<int> field_vdims({1, 1, dim, dim});
   charged_particles = std::make_unique<ParticleSet>(
                          comm, num_particles, dim, field_vdims, 1, pdata_ordering);
}

void ParticleMover::InitializeChargedParticles(const real_t& k,
                                     const real_t& alpha, real_t m, real_t q,
                                     real_t L_x, bool reproduce)
{
   int rank;
   MPI_Comm_rank(charged_particles->GetComm(), &rank);
   // use time-based seed for randomness
   std::mt19937 gen(
      reproduce ? rank : (rank + static_cast<unsigned int>(time(nullptr))));
   std::uniform_real_distribution<> real_dist(0.0, 1.0);
   std::normal_distribution<> norm_dist(0.0, 1.0);

   int dim = charged_particles->Coords().GetVDim();
   MFEM_VERIFY(alpha >= -1.0 && alpha < 1.0,
               "Alpha should be in range [-1, 1).");
   MFEM_VERIFY(k != 0.0, "k must be nonzero for displacement initialization.");

   ParticleVector& X = charged_particles->Coords();
   ParticleVector& P = charged_particles->Field(ParticleMover::MOM);
   ParticleVector& M = charged_particles->Field(ParticleMover::MASS);
   ParticleVector& Q = charged_particles->Field(ParticleMover::CHARGE);

   for (int i = 0; i < charged_particles->GetNParticles(); i++)
   {
      // Initialize momentum
      for (int d = 0; d < dim; d++) { P(i, d) = m * norm_dist(gen); }

      // Uniform positions (no accept-reject)
      for (int d = 0; d < dim; d++) { X(i, d) = real_dist(gen) * L_x; }

      // Displacement along x for perturbation ~ cos(k x)
      for (int d = 0; d < dim; d++)
      {
         real_t x = X(i, d);
         x -= (alpha / k) * std::sin(k * x);

         // periodic wrap to [0, L_x)
         x = std::fmod(x, L_x);
         if (x < 0) { x += L_x; }

         X(i, d) = x;
      }

      // Initialize mass + charge
      M(i) = m;
      Q(i) = q;
   }
   FindParticles();
}

void ParticleMover::InterpolateE()
{
   ParticleVector& X = charged_particles->Coords();
   ParticleVector& E = charged_particles->Field(EFIELD);

   // Interpolate E-field onto particles
   GetValues(X, E_finder, *E_gf, E);
}

void ParticleMover::FindParticles()
{
   ParticleVector &X = charged_particles->Coords();
   E_finder.FindPoints(X, X.GetOrdering());
}

void ParticleMover::Step(real_t& t, real_t& dt, real_t L_x, bool zeroth_step)
{
   InterpolateE();
   // Individually step each particle:
   if (charged_particles->IsParticleRefValid())
   {
      for (int i = 0; i < charged_particles->GetNParticles(); i++)
      {
         Particle p = charged_particles->GetParticleRef(i);
         ParticleStep(p, dt, L_x, zeroth_step);
      }
   }
   else
   {
      for (int i = 0; i < charged_particles->GetNParticles(); i++)
      {
         Particle p = charged_particles->GetParticle(i);
         ParticleStep(p, dt, L_x, zeroth_step);
         charged_particles->SetParticle(i, p);
      }
   }

   FindParticles();

   if (zeroth_step) { return; }

   // Update time
   t += dt;
}

void ParticleMover::Redistribute()
{
   charged_particles->Redistribute(E_finder.GetProc());
   FindParticles();
}

FieldSolver::FieldSolver(ParGridFunction& phi_gf,
                                         FindPointsGSLIB& E_finder_,
                                         int visport_, bool visualization_,
                                         bool precompute_neutralizing_const_)
   : precompute_neutralizing_const(precompute_neutralizing_const_),
     E_finder(E_finder_),
     visport(visport_),
     visualization(visualization_),
     vis_e("localhost", visport_),
     vis_phi("localhost", visport_)
{
   // compute domain volume
   ParMesh* pmesh = phi_gf.ParFESpace()->GetParMesh();
   real_t local_domain_volume = 0.0;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      local_domain_volume += pmesh->GetElementVolume(i);
   }
   MPI_Allreduce(&local_domain_volume, &domain_volume, 1, MPI_DOUBLE,
                 MPI_SUM, phi_gf.ParFESpace()->GetParMesh()->GetComm());

   ParFiniteElementSpace* pfes = phi_gf.ParFESpace();

   {
      // Par bilinear form for the gradgrad matrix
      ParBilinearForm dm(pfes);
      ConstantCoefficient epsilon(EPSILON);  // ε_0
      dm.AddDomainIntegrator(
         new DiffusionIntegrator(epsilon));  // ∫ ∇φ_i · ∇φ_j

      dm.Assemble();
      dm.Finalize();

      DiffusionMatrix = dm.ParallelAssemble();  // global gradgrad matrix
   }
}

FieldSolver::~FieldSolver()
{
   delete DiffusionMatrix;
   delete precomputed_neutralizing_lf;
}

void FieldSolver::UpdatePhiGridFunction(ParticleSet& particles,
                                                ParGridFunction& phi_gf,
                                                ParGridFunction& E_gf)
{
   {
      // FE space / mesh
      ParFiniteElementSpace* pfes = phi_gf.ParFESpace();
      ParMesh* pmesh = pfes->GetParMesh();
      const int dim = pmesh->Dimension();

      // Particle data
      ParticleVector& X = particles.Coords();  // coordinates (vdim x npt)
      ParticleVector& Q = particles.Field(ParticleMover::CHARGE);  // charges (1 x npt)

      const int npt = particles.GetNParticles();
      MFEM_VERIFY(X.GetVDim() == dim, "Unexpected particle coordinate layout.");
      MFEM_VERIFY(Q.GetVDim() == 1,
                  "Charge field must be scalar per particle.");

      // --------------------------------------------------------
      // 1) Build positions in byVDIM ordering: (XYZ,XYZ,...)
      // --------------------------------------------------------
      Vector point_pos(X.GetData(), dim * npt);  // alias underlying storage

      // --------------------------------------------------------
      // 2) Locate particles with FindPointsGSLIB
      // --------------------------------------------------------
      const Array<unsigned int>& code =
         E_finder.GetCode();  // 0: inside, 1: boundary, 2: not found
      const Array<unsigned int>& proc = E_finder.GetProc();  // owning MPI rank
      const Array<unsigned int>& elem = E_finder.GetElem();  // local element id
      const Vector& rref = E_finder.GetReferencePosition();  // (r,s,t) byVDIM

      // --------------------------------------------------------
      // 3) Make RHS and pre-subtract averaged charge density for zero-mean RHS
      // --------------------------------------------------------

      MPI_Comm comm = pfes->GetComm();

      if (!precompute_neutralizing_const || !neutralizing_const_computed)
      {
         // compute neutralizing constant
         real_t local_sum = 0.0;
         for (int p = 0; p < npt; ++p)
         {
            // Skip particles not successfully found
            if (code[p] == 2)  // not found
            {
               continue;
            }

            local_sum += Q(p);
         }

         real_t global_sum = 0.0;
         MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

         neutralizing_const = -global_sum / domain_volume;
         neutralizing_const_computed = true;
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
         neutralizing_const_computed = true;
         delete precomputed_neutralizing_lf;
         precomputed_neutralizing_lf = new ParLinearForm(pfes);
         *precomputed_neutralizing_lf = 0.0;
         ConstantCoefficient neutralizing_coeff(neutralizing_const);
         precomputed_neutralizing_lf->AddDomainIntegrator(
            new DomainLFIntegrator(neutralizing_coeff));
         precomputed_neutralizing_lf->Assemble();
      }
      ParLinearForm b(pfes);
      // start with precomputed neutralizing contribution
      b = *precomputed_neutralizing_lf;

      // --------------------------------------------------------
      // 4) Deposit q_p * phi_i(x_p) into a ParLinearForm (RHS b)
      //      b_i = sum_p q_p * φ_i(x_p)
      // --------------------------------------------------------
      int myid;
      MPI_Comm_rank(pmesh->GetComm(), &myid);

      Array<int> dofs;

      for (int p = 0; p < npt; ++p)
      {
         // Skip particles not successfully found
         if (code[p] == 2)  // not found
         {
            continue;
         }

         // Raise error if particle is not on the current rank
         if ((int)proc[p] != myid)
         {
            // raise error
            MFEM_ABORT("Particle "
                       << p << " found in element owned by rank " << proc[p]
                       << " but current rank is " << myid << "." << endl
                       << "You must call redistribute everytime before "
                       "updating the density grid function.");
         }
         const int e = elem[p];

         // Reference coordinates for this particle (r,s[,t]) with byVDIM layout
         IntegrationPoint ip;
         ip.Set(rref.GetData() + dim * p, dim);

         const FiniteElement& fe = *pfes->GetFE(e);
         const int ldofs = fe.GetDof();

         Vector shape(ldofs);
         fe.CalcShape(ip, shape);  // φ_i(x_p) in this element

         pfes->GetElementDofs(e, dofs);  // local dof indices

         const real_t q_p = Q(p);

         // Add q_p * φ_i(x_p) to b_i
         b.AddElementVector(dofs, q_p, shape);
      }

      // Assemble to a global true-dof RHS vector compatible with MassMatrix
      HypreParVector* B = b.ParallelAssemble();  // owns new vector on heap

      // ------------------------------------------------------------------
      // 5) Solve A * phi = B with zero-mean enforcement via OrthoSolver
      // ------------------------------------------------------------------
      MFEM_VERIFY(DiffusionMatrix != nullptr,
                  "DiffusionMatrix must be precomputed.");

      phi_gf = 0.0;
      HypreParVector Phi_true(pfes);
      Phi_true = 0.0;

      HyprePCG solver(DiffusionMatrix->GetComm());
      solver.SetOperator(*DiffusionMatrix);
      solver.SetTol(1e-12);
      solver.SetMaxIter(200);
      solver.SetPrintLevel(0);

      HypreBoomerAMG prec(*DiffusionMatrix);
      prec.SetPrintLevel(0);
      solver.SetPreconditioner(prec);

      OrthoSolver ortho(comm);
      ortho.SetSolver(solver);
      ortho.Mult(*B, Phi_true);

      // Map true-dof solution back to the ParGridFunction
      phi_gf.Distribute(Phi_true);
      delete B;
   }

   {
      // 1.a make the RHS bilinear form
      ParMixedBilinearForm b_bi(phi_gf.ParFESpace(), E_gf.ParFESpace());
      ConstantCoefficient neg_one_coef(-1.0);
      b_bi.AddDomainIntegrator(new MixedVectorGradientIntegrator(neg_one_coef));
      b_bi.Assemble();
      b_bi.Finalize();
      // 1.b form linear form from bilinear form
      ParLinearForm b(E_gf.ParFESpace());
      b = 0.0;
      b_bi.Mult(phi_gf, b);
      // Convert to true-dof (parallel) vector
      HypreParVector* B = b.ParallelAssemble();

      // 2. make the bilinear form
      ParBilinearForm a(E_gf.ParFESpace());
      ConstantCoefficient one_coef(1.0);
      a.AddDomainIntegrator(new VectorFEMassIntegrator(one_coef));
      a.Assemble();
      a.Finalize();
      // Parallel operator (HypreParMatrix)
      HypreParMatrix* A = a.ParallelAssemble();

      // 3. solve for E_gf
      CGSolver M_solver(E_gf.ParFESpace()->GetComm());
      M_solver.iterative_mode = false;
      M_solver.SetRelTol(1e-12);
      M_solver.SetAbsTol(0.0);
      M_solver.SetMaxIter(1e5);
      M_solver.SetPrintLevel(0);
      M_solver.SetOperator(*A);

      HypreParVector X(E_gf.ParFESpace()->GetComm(),
                       E_gf.ParFESpace()->GlobalTrueVSize(),
                       E_gf.ParFESpace()->GetTrueDofOffsets());
      X = 0.0;
      M_solver.Mult(*B, X);
      E_gf.SetFromTrueDofs(X);
      delete A;
      delete B;
   }

   if (visualization)
   {
      common::VisualizeField(vis_e, "localhost", visport, E_gf, "E_field",
                             0, 0, 500, 500);
      common::VisualizeField(vis_phi, "localhost", visport, phi_gf, "Potential",
                             500, 0, 500, 500);
   }
}

void FieldSolver::TotalEnergyValidation(const ParticleSet& particles,
                                                const ParGridFunction& E_gf)
{
   const ParticleVector& P = particles.Field(ParticleMover::MOM);
   const ParticleVector& M = particles.Field(ParticleMover::MASS);

   real_t kinetic_energy = 0.0;
   for (int p = 0; p < particles.GetNParticles(); ++p)
   {
      real_t p_square_p = 0.0;
      for (int d = 0; d < P.GetVDim(); ++d) { p_square_p += P(p, d) * P(p, d); }
      kinetic_energy += 0.5 * p_square_p / M(p);
   }

   // ---- Field energy: 0.5 * ∫ ||E||^2 dx ----
   const ParFiniteElementSpace* fes = E_gf.ParFESpace();
   const ParMesh* pmesh = fes->GetParMesh();

   const int order = fes->GetMaxElementOrder();
   const int qorder = std::max(2, 2 * order + 1);

   const IntegrationRule* irs[Geometry::NumGeom];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      irs[g] = &IntRules.Get(g, qorder);
   }

   real_t global_field_energy = 0.0;

   // IMPORTANT: ND/RT use VectorFiniteElement even if fes->GetVDim() == 1
   if (fes->GetFE(0)->GetRangeType() == FiniteElement::VECTOR)
   {
      Vector zero(pmesh->Dimension());
      zero = 0.0;
      VectorConstantCoefficient zero_vec(zero);

      const real_t E_l2 = E_gf.ComputeL2Error(zero_vec, irs);
      global_field_energy = 0.5 * EPSILON * E_l2 * E_l2;
   }
   else
   {
      ConstantCoefficient zero_s(0.0);
      const real_t E_l2 = E_gf.ComputeL2Error(zero_s, irs);
      global_field_energy = 0.5 * EPSILON * E_l2 * E_l2;
   }

   // reduce kinetic energy and field energy
   real_t global_kinetic_energy = 0.0;
   MPI_Allreduce(&kinetic_energy, &global_kinetic_energy, 1, MPI_DOUBLE,
                 MPI_SUM, fes->GetComm());
   if (Mpi::Root())
   {
      cout << "Kinetic energy: " << global_kinetic_energy << "\t";
      cout << "Field energy: " << global_field_energy << "\t";
      cout << "Total energy: " << global_kinetic_energy + global_field_energy
           << endl;
   }
   // write to a csv
   if (Mpi::Root())
   {
      std::ofstream energy_file("energy.csv", std::ios::app);
      energy_file << setprecision(10) << global_kinetic_energy << ","
                  << global_field_energy << ","
                  << global_kinetic_energy + global_field_energy << "\n";
   }
}

void display_banner(ostream& os)
{
   os << R"(
      ██████╗░██╗░█████╗░
      ██╔══██╗██║██╔══██╗
      ██████╔╝██║██║░░╚═╝
      ██╔═══╝░██║██║░░██╗
      ██║░░░░░██║╚█████╔╝
      ╚═╝░░░░░╚═╝░╚════╝░
         )"
      << endl
      << flush;
}
