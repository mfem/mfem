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
//           Particle-In-Cell (PIC) Simulation (2D/3D)
//           -----------------------------------------------------
//
// This miniapp performs a Particle-In-Cell simulation (supports 2D or 3D
// spatial dimensions) of multiple charged particles subject to electric
// field forces.
//
//                           dp/dt = q E
//
// The method used is explicit time integration with a leap-frog scheme.
//
// The electric field is computed from the particle charge distribution using
// a Poisson solver. The particle trajectories are computed within a periodic
// domain (2D or 3D).
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
// Compile with: make electrostatic-pic
//
// Sample runs:
//
//   2D2V Linear Landau damping test case (Ricketson & Hu, 2025):
//      mpirun -n 4 ./electrostatic-pic -rdi 1 -npt 409600 -k 0.2855993321 -a 0.05 -nt 200 -nx 32 -ny 32 -O 1 -q 0.001181640625 -m 0.001181640625 -oci 1000 -dt 0.1
//   3D3V Linear Landau damping test case (Zheng et al., 2025):
//      mpirun -n 128 ./electrostatic-pic -dim 3 -rdi 1 -npt 40960000 -k 0.5 -a 0.01 -nt 100 -nx 32 -ny 32 -nz 32 -O 1 -q 0.00004844730731 -m 0.00004844730731 -oci 1000 -dt 0.1 -no-vis
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

#define EPSILON 1 // ε_0

using namespace std;
using namespace mfem;
using namespace mfem::common;

struct PICContext
{
   int dim = 2;      ///< Spatial dimension.
   int order = 1;    ///< FE order for spatial discretization.
   int nx = 100;     ///< Number of grid cells in x-direction.
   int ny = 100;     ///< Number of grid cells in y-direction.
   int nz = 100;     ///< Number of grid cells in z-direction.
   real_t L_x = 1.0; ///< Domain length in x-direction.

   int ordering = 1; ///< Ordering of particles.
   int npt = 1000;   ///< Number of particles.
   real_t q = 1.0;   ///< Particle charge.
   real_t m = 1.0;   ///< Particle mass.

   real_t k = 1.0;     ///< Wave number (Landau damping init).
   real_t alpha = 0.1; ///< Perturbation amplitude (Landau damping init).

   real_t dt = 1e-2; ///< Time step size.

   int nt = 1000;               ///< Number of time steps to run.
   int redist_interval = 5;     ///< Redistribution and update E_gf interval.
   int output_csv_interval = 1; ///< Interval for outputting CSV data files.

   bool visualization = true; ///< Enable visualization.
   int visport = 19916;       ///< Port number for visualization server.
   bool reproduce = true;     ///< Enable reproducible results.
} ctx;

/** This class implements explicit time integration for charged particles
    in an electric field using ParticleSet. */
class ParticleMover
{
public:
   enum Fields
   {
      MASS,   // vdim = 1
      CHARGE, // vdim = 1
      MOM,    // vdim = dim
      EFIELD  // vdim = dim
   };

protected:
   /// Pointers to E field GridFunctions
   ParGridFunction *E_gf;

   /// FindPointsGSLIB object for E field mesh
   FindPointsGSLIB &E_finder;

   /// ParticleSet of charged particles
   std::unique_ptr<ParticleSet> charged_particles;

   /// Temporary vectors for particle computation
   mutable Vector pm_, pp_;

public:
   ParticleMover(MPI_Comm comm,
                 ParGridFunction *E_gf_, FindPointsGSLIB &E_finder_,
                 int num_particles,
                 Ordering::Type pdata_ordering);

   /// Initialize charged particles with given parameters
   void InitializeChargedParticles(const real_t &k, const real_t &alpha,
                                   real_t m, real_t q, real_t L_x,
                                   bool reproduce = false);

   /// Find Particles in mesh corresponding to E and field
   void FindParticles();

   /// Advance particles one time step using Boris algorithm
   void Step(real_t &t, real_t dt, real_t L_x, bool first_step = false);

   /// Redistribute particles across processors
   void Redistribute();

   /// Get reference to ParticleSet
   ParticleSet &GetParticles() { return *charged_particles; }

   /// Compute kinetic energy from particles (MPI-reduced).
   real_t ComputeKineticEnergy() const;
};

/** Field solver responsible for updating the electrostatic potential and field
    from the particle charge density. Assembles and solves the periodic Poisson
    problem, computes the electric field via a discrete gradient operator, and
    provides utilities for field diagnostics (e.g. global field energy) and
    optional visualization output. */
class FieldSolver
{
private:
   real_t domain_volume;
   real_t neutralizing_const;
   ParLinearForm *precomputed_neutralizing_lf = nullptr;
   bool precompute_neutralizing_const = false;
   // Diffusion matrix
   HypreParMatrix *diffusion_matrix;
   // Gradient operator for computing E = -∇φ
   ParDiscreteLinearOperator *grad_interpolator;
   FindPointsGSLIB &E_finder;

public:
   FieldSolver(ParFiniteElementSpace *phi_fes, ParFiniteElementSpace *E_fes,
               FindPointsGSLIB &E_finder_,
               bool precompute_neutralizing_const_ = false);

   ~FieldSolver();

   /** Update the phi_gf grid function from the particles.
       Solve periodic Poisson: diffusion_matrix * phi = (rho - <rho>)
       with zero-mean enforcement via OrthoSolver. */
   void UpdatePhiGridFunction(ParticleSet &particles, ParGridFunction &phi_gf,
                              ParGridFunction &E_gf);

   /// Compute (global) field energy: 0.5 * ∫ ||E||^2 dx
   real_t ComputeFieldEnergy(const ParGridFunction &E_gf) const;
};

/// Prints the program's logo to the given output stream
void display_banner(ostream &os);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_ranks = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   if (Mpi::Root())
   {
      display_banner(cout);
   }

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.dim, "-dim", "--dimension",
                  "Spatial dimension (2 or 3)");
   args.AddOption(&ctx.order, "-O", "--order",
                  "Finite element polynomial degree");
   args.AddOption(&ctx.nx, "-nx", "--num-x",
                  "Number of elements in the x direction.");
   args.AddOption(&ctx.ny, "-ny", "--num-y",
                  "Number of elements in the y direction.");
   args.AddOption(&ctx.nz, "-nz", "--num-z",
                  "Number of elements in the z direction.");
   args.AddOption(&ctx.q, "-q", "--charge", "Particle charge.");
   args.AddOption(&ctx.m, "-m", "--mass", "Particle mass.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time Step.");
   args.AddOption(&ctx.nt, "-nt", "--num-timesteps", "Number of timesteps.");
   args.AddOption(&ctx.npt, "-npt", "--num-particles",
                  "Total number of particles.");
   args.AddOption(&ctx.k, "-k", "--k",
                  "Wave number for initial distribution.");
   args.AddOption(&ctx.alpha, "-a", "--alpha",
                  "Perturbation amplitude for initial distribution.");
   args.AddOption(&ctx.ordering, "-o", "--ordering",
                  "Ordering of particle data. 0 = byNODES, 1 = byVDIM.");
   args.AddOption(&ctx.redist_interval, "-rdi", "--redist-interval",
                  "Redistribution and update E_gf interval. Disabled if < 0.");
   args.AddOption(&ctx.output_csv_interval, "-oci", "--output-csv-interval",
                  "Output CSV interval. Disabled if < 0.");
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
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   // Assert that dimension is 2 or 3
   MFEM_VERIFY(ctx.dim == 2 || ctx.dim == 3,
               "Dimension must be 2 or 3, got " << ctx.dim);

   ctx.L_x = 2.0 * M_PI / ctx.k;

   // build up E_gf
   // 1. make a Cartesian Mesh (2D or 3D)
   Mesh serial_mesh;
   std::vector<Vector> translations;

   if (ctx.dim == 2)
   {
      serial_mesh = Mesh(Mesh::MakeCartesian2D(
          ctx.nx, ctx.ny, Element::QUADRILATERAL, false,
          ctx.L_x, ctx.L_x));
      translations = {Vector({ctx.L_x, 0.0}),
                      Vector({0.0, ctx.L_x})};
   }
   else // ctx.dim == 3
   {
      serial_mesh = Mesh(Mesh::MakeCartesian3D(
          ctx.nx, ctx.ny, ctx.nz, Element::HEXAHEDRON,
          ctx.L_x, ctx.L_x, ctx.L_x));
      translations = {Vector({ctx.L_x, 0.0, 0.0}),
                      Vector({0.0, ctx.L_x, 0.0}),
                      Vector({0.0, 0.0, ctx.L_x})};
   }

   Mesh periodic_mesh(Mesh::MakePeriodic(
       serial_mesh,
       serial_mesh.CreatePeriodicVertexMapping(
           translations)));
   // 2. parallelize the mesh
   ParMesh mesh(MPI_COMM_WORLD, periodic_mesh);
   serial_mesh.Clear();   // the serial mesh is no longer needed
   periodic_mesh.Clear(); // the periodic mesh is no longer needed

   // 3. Build the E_finder
   mesh.EnsureNodes();
   FindPointsGSLIB E_finder(mesh);

   // 4. Define a finite element space on the parallel mesh
   H1_FECollection phi_fec(ctx.order, ctx.dim);
   ParFiniteElementSpace phi_fespace(&mesh, &phi_fec);
   ND_FECollection E_fec(ctx.order, ctx.dim);
   ParFiniteElementSpace E_fespace(&mesh, &E_fec);

   // 5. Prepare an empty phi_gf and E_gf for later use
   ParGridFunction phi_gf(&phi_fespace);
   ParGridFunction E_gf(&E_fespace);
   phi_gf = 0.0; // Initialize phi_gf to zero
   E_gf = 0.0;   // Initialize E_gf to zero

   // 6. Build the grid function updates
   FieldSolver field_solver(&phi_fespace, &E_fespace, E_finder, true);
   Ordering::Type ordering_type =
       ctx.ordering == 0 ? Ordering::byNODES : Ordering::byVDIM;

   // 7. Initialize ParticleMover
   int num_particles = ctx.npt / num_ranks +
                       (rank < (ctx.npt % num_ranks) ? 1 : 0);
   ParticleMover particle_mover(MPI_COMM_WORLD,
                                &E_gf, E_finder,
                                num_particles, ordering_type);
   particle_mover.InitializeChargedParticles(ctx.k, ctx.alpha,
                                             ctx.m, ctx.q,
                                             ctx.L_x, ctx.reproduce);

   real_t t = 0;
   real_t dt = ctx.dt;

   // set up timer
   mfem::StopWatch sw;
   sw.Start();
   for (int step = 1; step <= ctx.nt; step++)
   {
      // Step the FieldSolver
      if (ctx.redist_interval > 0 &&
          (step % ctx.redist_interval == 0 || step == 1) &&
          particle_mover.GetParticles().GetGlobalNParticles() > 0)
      {
         // Redistribute
         particle_mover.Redistribute();

         // Update phi_gf from particles
         field_solver.UpdatePhiGridFunction(particle_mover.GetParticles(),
                                            phi_gf,
                                            E_gf);

         // Visualize fields if requested
         if (ctx.visualization)
         {
            static socketstream vis_e, vis_phi;
            common::VisualizeField(vis_e, "localhost", ctx.visport,
                                   E_gf, "E_field",
                                   0, 0, 500, 500);
            common::VisualizeField(vis_phi, "localhost", ctx.visport,
                                   phi_gf, "Potential",
                                   500, 0, 500, 500);
         }

         // Compute energies
         real_t kinetic_energy = particle_mover.ComputeKineticEnergy();
         real_t field_energy = field_solver.ComputeFieldEnergy(E_gf);

         // Output energies
         if (Mpi::Root())
         {
            cout << "Kinetic energy: " << kinetic_energy << "\t";
            cout << "Field energy: " << field_energy << "\t";
            cout << "Total energy: " << kinetic_energy + field_energy
                 << endl;
         }
         // write to a csv
         if (Mpi::Root())
         {
            std::ofstream energy_file("energy.csv", std::ios::app);
            energy_file << setprecision(10) << kinetic_energy << ","
                        << field_energy << ","
                        << kinetic_energy + field_energy << "\n";
         }
      }

      // Step the ParticleMover
      particle_mover.Step(t, dt, ctx.L_x, step == 1);
      if (Mpi::Root())
      {
         mfem::out << "Step: " << step << " | Time: " << t;
         // Print timing information every 10 steps
         if (step % 10 == 0)
         {
            mfem::out << " | Time per step: " << sw.RealTime() / step;
         }
         mfem::out << endl;
      }
      // Output particle data to CSV
      if (ctx.output_csv_interval > 0 &&
          (step % ctx.output_csv_interval == 0 || step == 1))
      {
         std::string csv_prefix = "PIC_Part_";
         Array<int> field_idx{2}, tag_idx;
         std::string file_name =
             csv_prefix + mfem::to_padded_string(step, 6) + ".csv";
         particle_mover.GetParticles().PrintCSV(file_name.c_str(),
                                                field_idx, tag_idx);
      }
   }
}

ParticleMover::ParticleMover(MPI_Comm comm, ParGridFunction *E_gf_,
                             FindPointsGSLIB &E_finder_,
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
       comm, num_particles, dim,
       field_vdims, 1, pdata_ordering);
}

void ParticleMover::InitializeChargedParticles(const real_t &k,
                                               const real_t &alpha,
                                               real_t m, real_t q,
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
   MFEM_VERIFY(k != 0.0,
               "k must be nonzero for displacement initialization.");

   ParticleVector &X = charged_particles->Coords();
   ParticleVector &P = charged_particles->Field(ParticleMover::MOM);
   ParticleVector &M = charged_particles->Field(ParticleMover::MASS);
   ParticleVector &Q = charged_particles->Field(ParticleMover::CHARGE);

   for (int i = 0; i < charged_particles->GetNParticles(); i++)
   {
      // Initialize momentum
      for (int d = 0; d < dim; d++)
      {
         P(i, d) = m * norm_dist(gen);
      }

      // Uniform positions (no accept-reject)
      for (int d = 0; d < dim; d++)
      {
         X(i, d) = real_dist(gen) * L_x;
      }

      // Displacement along x for perturbation ~ cos(k x)
      for (int d = 0; d < dim; d++)
      {
         real_t x = X(i, d);
         x -= (alpha / k) * std::sin(k * x);

         // periodic wrap to [0, L_x)
         x = std::fmod(x, L_x);
         if (x < 0)
         {
            x += L_x;
         }

         X(i, d) = x;
      }

      // Initialize mass + charge
      M(i) = m;
      Q(i) = q;
   }
   FindParticles();
}

void ParticleMover::FindParticles()
{
   E_finder.FindPoints(charged_particles->Coords());
}

void ParticleMover::Step(real_t &t, real_t dt, real_t L_x, bool first_step)
{
   // Update E field at particles
   ParticleVector &E = charged_particles->Field(EFIELD);
   E_finder.Interpolate(*E_gf, E, E.GetOrdering());

   // Extract particle data
   ParticleVector &X = charged_particles->Coords();
   ParticleVector &P = charged_particles->Field(MOM);
   ParticleVector &M = charged_particles->Field(MASS);
   ParticleVector &Q = charged_particles->Field(CHARGE);

   // Periodic boundary: wrap coordinates to [0, L_x)
   const int npt = charged_particles->GetNParticles();
   const int dim = X.GetVDim();

   for (int particle = 0; particle < npt; ++particle)
   {
      for (int d = 0; d < dim; ++d)
      {
         P(particle, d) +=
             (first_step ? dt / 2.0 : dt) *
             Q(particle) * E(particle, d);
      }
   }

   for (int particle = 0; particle < npt; ++particle)
   {
      for (int d = 0; d < dim; ++d)
      {
         X(particle, d) += dt / M(particle) * P(particle, d);
         while (X(particle, d) > L_x)
         {
            X(particle, d) -= L_x;
         }
         while (X(particle, d) < 0.0)
         {
            X(particle, d) += L_x;
         }
      }
   }

   FindParticles();

   // Update time
   t += dt;
}

void ParticleMover::Redistribute()
{
   charged_particles->Redistribute(E_finder.GetProc());
   FindParticles();
}

real_t ParticleMover::ComputeKineticEnergy() const
{
   const ParticleVector &P = charged_particles->Field(MOM);
   const ParticleVector &M = charged_particles->Field(MASS);

   real_t kinetic_energy = 0.0;
   for (int p = 0; p < charged_particles->GetNParticles(); ++p)
   {
      real_t p_square_p = 0.0;
      for (int d = 0; d < P.GetVDim(); ++d)
      {
         p_square_p += P(p, d) * P(p, d);
      }
      kinetic_energy += 0.5 * p_square_p / M(p);
   }

   real_t global_kinetic_energy = 0.0;
   MPI_Allreduce(&kinetic_energy, &global_kinetic_energy, 1, MPI_DOUBLE,
                 MPI_SUM, charged_particles->GetComm());
   return global_kinetic_energy;
}

FieldSolver::FieldSolver(ParFiniteElementSpace *phi_fes,
                         ParFiniteElementSpace *E_fes,
                         FindPointsGSLIB &E_finder_,
                         bool precompute_neutralizing_const_)
    : precompute_neutralizing_const(precompute_neutralizing_const_),
      E_finder(E_finder_)
{
   // compute domain volume
   ParMesh *pmesh = phi_fes->GetParMesh();
   real_t local_domain_volume = 0.0;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      local_domain_volume += pmesh->GetElementVolume(i);
   }
   MPI_Allreduce(&local_domain_volume, &domain_volume, 1, MPI_DOUBLE,
                 MPI_SUM, phi_fes->GetParMesh()->GetComm());

   {
      // Par bilinear form for the gradgrad matrix
      ParBilinearForm dm(phi_fes);
      ConstantCoefficient epsilon(EPSILON); // ε_0
      dm.AddDomainIntegrator(
          new DiffusionIntegrator(epsilon)); // ∫ ∇φ_i · ∇φ_j

      dm.Assemble();
      dm.Finalize();

      diffusion_matrix = dm.ParallelAssemble(); // global gradgrad matrix
   }

   {
      // Compute E = -∇φ using DiscreteLinearOperator
      grad_interpolator = new ParDiscreteLinearOperator(phi_fes, E_fes);
      grad_interpolator->AddDomainInterpolator(new GradientInterpolator);
      grad_interpolator->Assemble();
   }
}

FieldSolver::~FieldSolver()
{
   delete diffusion_matrix;
   delete precomputed_neutralizing_lf;
   delete grad_interpolator;
}

void FieldSolver::UpdatePhiGridFunction(ParticleSet &particles,
                                        ParGridFunction &phi_gf,
                                        ParGridFunction &E_gf)
{
   {
      // FE space / mesh
      ParFiniteElementSpace *pfes = phi_gf.ParFESpace();
      ParMesh *pmesh = pfes->GetParMesh();
      const int dim = pmesh->Dimension();

      // Particle data: X - coordinates (dim x npt), Q - charges (1 x npt)
      ParticleVector &X = particles.Coords();
      ParticleVector &Q = particles.Field(ParticleMover::CHARGE);

      const int npt = particles.GetNParticles();
      MFEM_VERIFY(X.GetVDim() == dim,
                  "Unexpected particle coordinate layout.");

      MFEM_VERIFY(Q.GetVDim() == 1,
                  "Charge field must be scalar per particle.");
      // --------------------------------------------------------
      // 1) Locate particles with FindPointsGSLIB
      // --------------------------------------------------------
      // 0: inside, 1: boundary, 2: not found
      const Array<unsigned int> &code =
          E_finder.GetCode();
      const Array<unsigned int> &proc = E_finder.GetProc(); // owning MPI rank
      const Array<unsigned int> &elem = E_finder.GetElem(); // local element id
      const Vector &rref = E_finder.GetReferencePosition(); // (r,s,t) byVDIM

      // --------------------------------------------------------
      // 2) Make RHS and pre-subtract averaged charge density for zero-mean RHS
      // --------------------------------------------------------

      MPI_Comm comm = pfes->GetComm();

      if (!precompute_neutralizing_const ||
          precomputed_neutralizing_lf == nullptr)
      {
         // compute neutralizing constant
         real_t local_sum = 0.0;
         for (int p = 0; p < npt; ++p)
         {
            // Skip particles not successfully found
            if (code[p] == 2) // not found
            {
               MFEM_ABORT("Particle " << p << " not found.");
            }

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
      ParLinearForm b(pfes);
      // start with precomputed neutralizing contribution
      b = *precomputed_neutralizing_lf;

      // --------------------------------------------------------
      // 3) Deposit q_p * phi_i(x_p) into a ParLinearForm (RHS b)
      //      b_i = sum_p q_p * φ_i(x_p)
      // --------------------------------------------------------
      int curr_rank;
      MPI_Comm_rank(pmesh->GetComm(), &curr_rank);

      Array<int> dofs;

      for (int p = 0; p < npt; ++p)
      {
         // Skip particles not successfully found
         if (code[p] == 2) // not found
         {
            continue;
         }

         // Raise error if particle is not on the current rank
         if ((int)proc[p] != curr_rank)
         {
            // raise error
            MFEM_ABORT("Particle "
                       << p << " found in element owned by rank " << proc[p]
                       << " but current rank is " << curr_rank << "." << endl
                       << "You must call redistribute everytime before "
                          "updating the density grid function.");
         }
         const int e = elem[p];

         // Reference coordinates for this particle (r,s[,t]) with byVDIM layout
         IntegrationPoint ip;
         ip.Set(rref.GetData() + dim * p, dim);

         const FiniteElement &fe = *pfes->GetFE(e);
         const int ldofs = fe.GetDof();

         Vector shape(ldofs);
         fe.CalcShape(ip, shape); // φ_i(x_p) in this element

         pfes->GetElementDofs(e, dofs); // local dof indices

         const real_t q_p = Q(p);

         // Add q_p * φ_i(x_p) to b_i
         b.AddElementVector(dofs, q_p, shape);
      }

      // Assemble to a global true-dof RHS vector compatible with MassMatrix
      HypreParVector B(pfes);
      b.ParallelAssemble(B);

      // ------------------------------------------------------------------
      // 4) Solve A * phi = B with zero-mean enforcement via OrthoSolver
      // ------------------------------------------------------------------
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

      // Map true-dof solution back to the ParGridFunction
      phi_gf.Distribute(Phi_true);
   }

   {
      // Compute ∇φ using precomputed gradient operator
      grad_interpolator->Mult(phi_gf, E_gf);
      // Scale by -1 to get E = -∇φ
      E_gf.Neg();
   }
}

real_t FieldSolver::ComputeFieldEnergy(const ParGridFunction &E_gf) const
{
   // ---- Field energy: 0.5 * ∫ ||E||^2 dx ----
   const ParFiniteElementSpace *fes = E_gf.ParFESpace();
   const ParMesh *pmesh = fes->GetParMesh();

   const int order = fes->GetMaxElementOrder();
   const int qorder = std::max(2, 2 * order + 1);

   const IntegrationRule *irs[Geometry::NumGeom];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      irs[g] = &IntRules.Get(g, qorder);
   }

   real_t field_energy = 0.0;

   Vector zero(pmesh->Dimension());
   zero = 0.0;
   VectorConstantCoefficient zero_vec(zero);

   const real_t E_l2 = E_gf.ComputeL2Error(zero_vec, irs);
   field_energy = 0.5 * EPSILON * E_l2 * E_l2;

   return field_energy;
}

void display_banner(ostream &os)
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
