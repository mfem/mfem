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
//           Lorentz Miniapp:  Simple Lorentz Force Particle Mover
//           -----------------------------------------------------
//
// This miniapp computes the trajectories of a set of charged particles subject to
// Lorentz forces.
//
//                           dp/dt = q (E + v x B)
//
// The method used is the explicit Boris algortihm which conserves phase space
// volume for long term accuracy.
//
// The electric and magnetic fields are read from VisItDataCollection objects
// such as those produced by the Volta and Tesla miniapps. It is notable that
// these two fields do not need to be defined on the same mesh. At least
// one of either an electric field or a magnetic field must be provided. The
// particles' location and momentum are randomly initialized within a bounding
// box specified by command line input.
//
// This miniapp demonstrates the use of ParticleSet with FindPointsGSLIB. When
// particles leave both domains, they are subject to removal. Redistribution of
// particle data between MPI ranks is also demonstrated.
//
// Note that the VisItDataCollection objects must have been stored using the
// parallel format e.g. visit_dc.SetFormat(DataCollection::PARALLEL_FORMAT);.
// Without this optional format specifier the vector field lookups will fail.
//
// Compile with: make lorentz
//
// Sample runs:
//

#include "mfem.hpp"
#include "../common/particles_extras.hpp"
#include "../../general/text.hpp"

#include "../electromagnetics/electromagnetics.hpp"
#include <fstream>
#include <iostream>

// add timer
#include <chrono>
#define EPSILON 1 // ε_0

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::electromagnetics;

struct LorentzContext
{
   // mesh related parameters
   int order = 1;
   int nx = 100;
   int ny = 100;
   real_t L_x = 1.0;

   struct DColl
   {
      string coll_name;
      string field_name;
      int cycle;
      int pad_digits_cycle;
      int pad_digits_rank;
   };
   DColl E{"", "E", 10, 6, 6};
   DColl B{"", "B", 10, 6, 6};

   int ordering = 1;
   int npt = 1;
   real_t q = 1.0;
   real_t m = 1.0;

   real_t k = 1;
   real_t alpha = 0.1;

   real_t dt = 1e-2;
   real_t t0 = 0.0;
   int nt = 1000;
   int redist_freq = 1e6;
   int redist_mesh = 0;
   int rm_lost_freq = 1;

   bool visualization = true;
   int visport = 19916;
   int vis_tail_size = 5;
   int vis_freq = 50;
} ctx;

/// This class implements the Boris algorithm as described in the
/// article `Why is Boris algorithm so good?` by H. Qin et al in
/// Physics of Plasmas, Volume 20 Issue 8, August 2013,
/// https://doi.org/10.1063/1.4818428.
class Boris
{
public:
   enum Fields
   {
      MASS,   // vdim = 1
      CHARGE, // vdim = 1
      MOM,    // vdim = dim
      EFIELD, // vdim = dim
      BFIELD, // vdim = dim
      SIZE
   };

protected:
   GridFunction *E_gf;
   GridFunction *B_gf;

   FindPointsGSLIB E_finder;
   FindPointsGSLIB B_finder;

   std::unique_ptr<ParticleSet> charged_particles;

   mutable Vector pxB_, pm_, pp_;

   static void GetValues(const MultiVector &coords, FindPointsGSLIB &finder,
                         GridFunction &gf, MultiVector &pv);
   void ParticleStep(Particle &part, real_t &dt, bool zeroth_step = false);

public:
   Boris(MPI_Comm comm, GridFunction *E_gf_, GridFunction *B_gf_,
         int num_particles, Ordering::Type pdata_ordering);
   void InterpolateEB();
   void Step(real_t &t, real_t &dt, bool zeroth_step = false);
   void RemoveLostParticles();
   void Redistribute(int redist_mesh); // 0 = E field, 1 = B field
   ParticleSet &GetParticles() { return *charged_particles; }
};

class GridFunctionUpdates
{
private:
   real_t domain_volume;
   real_t neutralizing_const_1;
   real_t neutralizing_const_2;
   ParLinearForm *vol_lf = nullptr;
   ParLinearForm *precomputed_neutralizing_lf = nullptr;
   bool neutralizing_const_computed_1 = false;
   bool neutralizing_const_computed_2 = false;
   bool use_precomputed_neutralizing_const = false;
   // Diffusion matrix
   HypreParMatrix *DiffusionMatrix;

public:
   // Update the phi_gf grid function from the particles.
   void UpdatePhiGridFunction(ParticleSet &particles, ParGridFunction &phi_gf, ParGridFunction &E_gf);
   void TotalEnergyValidation(const ParticleSet &particles, const ParGridFunction &E_gf);
   // constructor
   GridFunctionUpdates(ParGridFunction &phi_gf, bool use_precomputed_neutralizing_const_ = false)
       : use_precomputed_neutralizing_const(use_precomputed_neutralizing_const_)
   {
      // compute domain volume
      ParMesh *pmesh = phi_gf.ParFESpace()->GetParMesh();
      real_t local_domain_volume = 0.0;
      for (int i = 0; i < pmesh->GetNE(); i++)
         local_domain_volume += pmesh->GetElementVolume(i);
      MPI_Allreduce(&local_domain_volume, &domain_volume, 1, MPI_DOUBLE, MPI_SUM,
                    phi_gf.ParFESpace()->GetParMesh()->GetComm());

      ParFiniteElementSpace *pfes = phi_gf.ParFESpace();

      { // Par bilinear form for the gradgrad matrix
         ParBilinearForm dm(pfes);
         ConstantCoefficient epsilon(EPSILON);                     // ε_0
         dm.AddDomainIntegrator(new DiffusionIntegrator(epsilon)); // ∫ ∇φ_i · ∇φ_j

         dm.Assemble();
         dm.Finalize();

         DiffusionMatrix = dm.ParallelAssemble(); // global gradgrad matrix
      }
   }
   ~GridFunctionUpdates()
   {
      delete DiffusionMatrix;
      delete precomputed_neutralizing_lf;
      delete vol_lf;
   }
};

// Prints the program's logo to the given output stream
void display_banner(ostream &os);

// Open the named VisItDataCollection and read the named field.
// Returns pointers to the two new objects.
int ReadGridFunction(std::string coll_name, std::string field_name,
                     int pad_digits_cycle, int pad_digits_rank, int cycle,
                     std::unique_ptr<VisItDataCollection> &dc, ParGridFunction *&gf);

// Initialize particles from user input.
void InitializeChargedParticles(ParticleSet &charged_particles,
                                const real_t &k, const real_t &alpha,
                                real_t m, real_t q, real_t L_x);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   // Mesh parameters
   int dim = 2;

   if (Mpi::Root())
   {
      display_banner(cout);
   }

   OptionsParser args(argc, argv);

   // Field variables (moved into ctx)
   args.AddOption(&ctx.order, "-O", "--order", "Finite element polynomial degree");

   // Mesh parameters (moved into ctx)
   args.AddOption(&ctx.nx, "-nx", "--num-x", "Number of elements in the x direction.");
   args.AddOption(&ctx.ny, "-ny", "--num-y", "Number of elements in the y direction.");

   args.AddOption(&ctx.redist_freq, "-rdf", "--redist-freq",
                  "Redistribution frequency.");
   args.AddOption(&ctx.redist_mesh, "-rdm", "--redistribution-mesh",
                  "Particle domain mesh for redistribution. 0 for E field mesh. 1 for B field mesh.");
   args.AddOption(&ctx.rm_lost_freq, "-rmf", "--remove-lost-freq",
                  "Remove lost particles frequency.");
   args.AddOption(&ctx.ordering, "-o", "--ordering",
                  "Ordering of particle data. 0 = byNODES, 1 = byVDIM.");
   args.AddOption(&ctx.npt, "-npt", "--num-particles",
                  "Total number of particles.");
   args.AddOption(&ctx.m, "-m", "--mass", "Particles' mass.");
   args.AddOption(&ctx.q, "-q", "--charge", "Particles' charge.");
   args.AddOption(&ctx.k, "-k", "--k", "K parameter for initial distribution.");
   args.AddOption(&ctx.alpha, "-a", "--alpha", "Alpha parameter for initial distribution.");

   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time Step.");
   args.AddOption(&ctx.t0, "-t0", "--initial-time", "Initial Time.");
   args.AddOption(&ctx.nt, "-nt", "--num-timesteps", "Number of timesteps.");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.vis_tail_size, "-vt", "--vis-tail-size",
                  "GLVis visualization trajectory truncation tail size.");
   args.AddOption(&ctx.vis_freq, "-vf", "--vis-freq",
                  "GLVis visualization frequency.");
   args.AddOption(&ctx.visport, "-p", "--send-port", "Socket for GLVis.");

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
   ctx.L_x = 2.0 * M_PI / ctx.k;

   std::unique_ptr<VisItDataCollection> E_dc, B_dc;
   ParGridFunction *E_gf = nullptr, *B_gf = nullptr;

   // build up E_gf, B_gf can remain nullptr
   // 1. make a 2D Cartesian Mesh
   Mesh serial_mesh(Mesh::MakeCartesian2D(ctx.nx, ctx.ny, Element::QUADRILATERAL, false, ctx.L_x, ctx.L_x));
   std::vector<Vector> translations = {Vector({ctx.L_x, 0.0}), Vector({0.0, ctx.L_x})};
   Mesh periodic_mesh(Mesh::MakePeriodic(serial_mesh, serial_mesh.CreatePeriodicVertexMapping(translations)));
   // 2. parallelize the mesh
   ParMesh mesh(MPI_COMM_WORLD, periodic_mesh);
   serial_mesh.Clear();   // the serial mesh is no longer needed
   periodic_mesh.Clear(); // the periodic mesh is no longer needed
   // 3. Define a finite element space on the parallel mesh
   H1_FECollection sca_fec(ctx.order, dim);
   ParFiniteElementSpace sca_fespace(&mesh, &sca_fec);
   ND_FECollection vec_fec(ctx.order, dim);
   ParFiniteElementSpace vec_fespace(&mesh, &vec_fec);

   // 4. Prepare an empty phi_gf and E_gf for later use
   ParGridFunction phi_gf(&sca_fespace);
   E_gf = new ParGridFunction(&vec_fespace);
   phi_gf = 0.0; // Initialize phi_gf to zero
   *E_gf = 0.0;  // Initialize E_gf to zero

   // 7. Build the grid function updates
   GridFunctionUpdates gf_updates(phi_gf, true);
   Ordering::Type ordering_type = ctx.ordering == 0 ? Ordering::byNODES : Ordering::byVDIM;

   // Initialize Boris
   int num_particles = ctx.npt / size + (rank < (ctx.npt % size) ? 1 : 0);
   Boris boris(MPI_COMM_WORLD, E_gf, B_gf, num_particles, ordering_type);
   InitializeChargedParticles(boris.GetParticles(),
                              ctx.k, ctx.alpha, ctx.m, ctx.q, ctx.L_x);
   boris.InterpolateEB(); // Interpolate E and B field onto updated particle positions

   real_t t = ctx.t0;
   real_t dt = ctx.dt;
   // Setup visualization
   char vishost[] = "localhost";
   socketstream pre_redist_sock, post_redist_sock;
   std::unique_ptr<ParticleTrajectories> traj_vis;
   if (ctx.visualization)
   {
      traj_vis = std::make_unique<ParticleTrajectories>(boris.GetParticles(),
                                                        ctx.vis_tail_size, vishost, ctx.visport, "Particle Trajectories", 0, 0, 800,
                                                        800, "ba", 0.75 * ctx.L_x);
   }

   // set up timer
   auto start_time = std::chrono::high_resolution_clock::now();
   for (int step = 1; step <= ctx.nt; step++)
   {
      // Redistribute
      if (ctx.redist_freq > 0 && (step % ctx.redist_freq == 0 || step == 1) &&
          boris.GetParticles().GetGlobalNP() > 0)
      {
         // Redistribute
         boris.Redistribute(ctx.redist_mesh);

         // Update phi_gf from particles
         gf_updates.UpdatePhiGridFunction(boris.GetParticles(), phi_gf, *E_gf);

         gf_updates.TotalEnergyValidation(boris.GetParticles(), *E_gf);
      }
      
      if (step == 1)
      {
         real_t neg_half_dt = -dt / 2.0;
         // Perform a "zeroth" step to move p half step backward
         boris.Step(t, neg_half_dt, true);
      }
      // Step the Boris algorithm
      boris.Step(t, dt);
      if (Mpi::Root())
      {
         mfem::out << "Step: " << step << " | Time: " << t;
         // Print timing information every 100 steps
         if (step % 10 == 0)
         {
            std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start_time;
            mfem::out << " | Time per step: " << elapsed.count() / step;
         }
         mfem::out << endl;
      }

      // Visualize trajectories
      if (ctx.visualization && step % ctx.vis_freq == 0)
      {
         traj_vis->Visualize();
      }

      // Remove lost particles
      if (step % ctx.rm_lost_freq == 0)
      {
         boris.RemoveLostParticles();
         std::string csv_prefix = "Lorentz_Part_";
         Array<int> field_idx{2}, tag_idx;
         std::string file_name = csv_prefix + mfem::to_padded_string(step, 6) + ".csv";
         boris.GetParticles().PrintCSV(file_name.c_str(), field_idx, tag_idx);
      }

   }
}

void Boris::GetValues(const MultiVector &coords, FindPointsGSLIB &finder,
                      GridFunction &gf, MultiVector &pv)
{
   Mesh &mesh = *gf.FESpace()->GetMesh();
   mesh.EnsureNodes();
   finder.FindPoints(mesh, coords, coords.GetOrdering());
   finder.Interpolate(gf, pv);
   Ordering::Reorder(pv, pv.GetVDim(), gf.FESpace()->GetOrdering(),
                     pv.GetOrdering());
}

void Boris::ParticleStep(Particle &part, real_t &dt, bool zeroth_step)
{
   Vector &x = part.Coords();
   real_t m = part.FieldValue(MASS);
   real_t q = part.FieldValue(CHARGE);
   Vector &p = part.Field(MOM);
   Vector &e = part.Field(EFIELD);
   Vector &b = part.Field(BFIELD);

   // Compute half of the contribution from q E
   add(p, 0.5 * dt * q, e, pm_);

   // --- Magnetic field contribution (q p x B) disabled ---
   // const real_t B2 = b * b;

   // const real_t a1 = 4.0 * dt * q * m;
   // pm_.cross3D(b, pxB_);
   // pp_.Set(a1, pxB_);

   // const real_t a2 = 4.0 * m * m -
   //                   dt * dt * q * q * B2;
   // pp_.Add(a2, pm_);

   // const real_t a3 = 2.0 * dt * dt * q * q * (b * p);
   // pp_.Add(a3, b);

   // const real_t a4 = 4.0 * m * m +
   //                   dt * dt * q * q * B2;
   // pp_ /= a4;

   // --- Simplified update: no magnetic field ---
   pp_ = pm_; // only include E contribution

   // Update the momentum (full electric contribution)
   add(pp_, 0.5 * dt * q, e, p);

   if (zeroth_step)
      return;

   // Update the position
   x.Add(dt / m, p);

   // periodic boundary: wrap around using ctx mesh extents
   x(0) = fmod(x(0), ctx.L_x);
   if (x(0) < 0.0)
      x(0) += ctx.L_x;
   x(1) = fmod(x(1), ctx.L_x);
   if (x(1) < 0.0)
      x(1) += ctx.L_x;
}

Boris::Boris(MPI_Comm comm, GridFunction *E_gf_, GridFunction *B_gf_,
             int num_particles, Ordering::Type pdata_ordering)
    : E_gf(E_gf_),
      B_gf(B_gf_),
      E_finder(comm),
      B_finder(comm)
{
   MFEM_VERIFY(E_gf || B_gf, "Must pass an E field or B field to Boris.");

   Mesh *E_mesh = E_gf ? E_gf->FESpace()->GetMesh() : nullptr;
   Mesh *B_mesh = B_gf ? B_gf->FESpace()->GetMesh() : nullptr;
   if (E_mesh && B_mesh)
   {
      int E_dim = E_mesh->SpaceDimension();
      int B_dim = B_mesh->SpaceDimension();
      MFEM_VERIFY(E_dim == B_dim,
                  "E mesh and B mesh must have the same spatial dimension.");
   }
   if (E_gf)
   {
      E_mesh->EnsureNodes();
      E_finder.Setup(*E_mesh);
   }
   if (B_gf)
   {
      B_mesh->EnsureNodes();
      B_finder.Setup(*B_mesh);
   }

   int dim = E_mesh ? E_mesh->SpaceDimension() : B_mesh->SpaceDimension();

   pxB_.SetSize(dim);
   pm_.SetSize(dim);
   pp_.SetSize(dim);

   // Create particle set: 2 scalars of mass and charge, 3 vectors of size space dim for momentum, e field, and b field
   Array<int> field_vdims({1, 1, dim, dim, dim});
   charged_particles = std::make_unique<ParticleSet>(comm, ctx.npt, dim,
                                                     field_vdims, 1, pdata_ordering);
}

void Boris::InterpolateEB()
{
   MultiVector &X = charged_particles->Coords();
   MultiVector &E = charged_particles->Field(EFIELD);
   MultiVector &B = charged_particles->Field(BFIELD);

   // Interpolate E-field + B-field onto particles
   if (E_gf)
   {
      GetValues(X, E_finder, *E_gf, E);
   }
   else
   {
      E = 0.0;
   }
   if (B_gf)
   {
      GetValues(X, B_finder, *B_gf, B);
   }
   else
   {
      B = 0.0;
   }
}

void Boris::Step(real_t &t, real_t &dt, bool zeroth_step)
{
   // Individually step each particle:
   if (charged_particles->ParticleRefValid())
   {
      for (int i = 0; i < charged_particles->GetNP(); i++)
      {
         Particle p = charged_particles->GetParticleRef(i);
         ParticleStep(p, dt, zeroth_step);
      }
   }
   else
   {
      for (int i = 0; i < charged_particles->GetNP(); i++)
      {
         Particle p = charged_particles->GetParticle(i);
         ParticleStep(p, dt, zeroth_step);
         charged_particles->SetParticle(i, p);
      }
   }
   if (zeroth_step)
      return;

   // Interpolate E and B field onto new locations of particles
   InterpolateEB();

   // Update time
   t += dt;
}

void Boris::RemoveLostParticles()
{
   Array<int> lost_idxs;
   const Array<int> E_lost = E_finder.GetPointsNotFoundIndices();
   const Array<int> B_lost = B_finder.GetPointsNotFoundIndices();

   for (const int &elem : E_lost)
   {
      lost_idxs.Union(elem);
   }

   for (const int &elem : B_lost)
   {
      lost_idxs.Union(elem);
   }

   charged_particles->RemoveParticles(lost_idxs);
}

void Boris::Redistribute(int redist_mesh)
{
   if (redist_mesh == 0)
   {
      charged_particles->Redistribute(E_finder.GetProc());
   }
   else
   {
      charged_particles->Redistribute(B_finder.GetProc());
   }
}

void display_banner(ostream &os)
{
   os << "   ____                                __          "
      << endl
      << "  |    |    ___________   ____   _____/  |_________"
      << endl
      << "  |    |   /  _ \\_  __ \\_/ __ \\ /    \\   __\\___   /"
      << endl
      << "  |    |__(  <_> )  | \\/\\  ___/|   |  \\  |  /    / "
      << endl
      << "  |_______ \\____/|__|    \\___  >___|  /__| /_____ \\"
      << endl
      << "          \\/                 \\/     \\/           \\/"
      << endl
      << flush;
}

int ReadGridFunction(std::string coll_name, std::string field_name,
                     int pad_digits_cycle, int pad_digits_rank, int cycle,
                     std::unique_ptr<VisItDataCollection> &dc, ParGridFunction *&gf)
{
   dc = std::make_unique<VisItDataCollection>(MPI_COMM_WORLD, coll_name);
   dc->SetPadDigitsCycle(pad_digits_cycle);
   dc->SetPadDigitsRank(pad_digits_rank);
   dc->Load(cycle);

   if (dc->Error() != DataCollection::No_Error)
   {
      mfem::err << "Error loading VisIt data collection: "
                << coll_name << endl;
      return 1;
   }

   if (dc->HasField(field_name))
   {
      gf = dc->GetParField(field_name);
   }

   return 0;
}

void InitializeChargedParticles(ParticleSet &charged_particles,
                                const real_t &k, const real_t &alpha,
                                real_t m, real_t q, real_t L_x)
{
   int rank;
   MPI_Comm_rank(charged_particles.GetComm(), &rank);
   std::mt19937 gen(rank);
   std::uniform_real_distribution<> real_dist(0.0, 1.0);
   std::normal_distribution<> norm_dist(0.0, 1.0);

   int dim = charged_particles.Coords().GetVDim();
   // assert alpha in [0, 1/d)
   MFEM_VERIFY(alpha >= -1.0 && alpha < 1.0, "Alpha should be in range [-1, 1).");

   MultiVector &X = charged_particles.Coords();
   MultiVector &P = charged_particles.Field(Boris::MOM);
   MultiVector &M = charged_particles.Field(Boris::MASS);
   MultiVector &Q = charged_particles.Field(Boris::CHARGE);

   for (int i = 0; i < charged_particles.GetNP(); i++)
   {
      for (int d = 0; d < dim; d++)
         // Initialize momentum
         P(i, d) = m * norm_dist(gen);

      for (int d = 0; d < dim; d++)
      {
         while (true)
         {
            X(i, d) = real_dist(gen) * L_x;
            double w = 1.0 + alpha * std::cos(k * X(i, d)); // should be >= 0 if |alpha|<=1

            if (real_dist(gen) < w / (1.0 + std::abs(alpha)))
               break;
         }
      }
      // Initialize mass + charge
      M(i) = m;
      Q(i) = q;
   }
}

// Solve periodic Poisson: DiffusionMatrix * phi = (rho - <rho>)
// with zero-mean enforcement via OrthoSolver.
void GridFunctionUpdates::UpdatePhiGridFunction(ParticleSet &particles,
                                                ParGridFunction &phi_gf,
                                                ParGridFunction &E_gf)
{
   { // FE space / mesh
      ParFiniteElementSpace *pfes = phi_gf.ParFESpace();
      ParMesh *pmesh = pfes->GetParMesh();
      const int dim = pmesh->Dimension();

      // Particle data
      MultiVector &X = particles.Coords();             // coordinates (vdim x npt)
      MultiVector &Q = particles.Field(Boris::CHARGE); // charges (1 x npt)
      Ordering::Type ordering_type = X.GetOrdering();

      const int npt = X.GetNumVectors();
      MFEM_VERIFY(X.GetVDim() == dim, "Unexpected particle coordinate layout.");
      MFEM_VERIFY(Q.GetVDim() == 1, "Charge field must be scalar per particle.");

      // ------------------------------------------------------------------------
      // 1) Build positions in byVDIM ordering: (XYZ,XYZ,...)
      // ------------------------------------------------------------------------
      Vector point_pos(X.GetData(), dim * npt); // alias underlying storage

      // ------------------------------------------------------------------------
      // 2) Locate particles with FindPointsGSLIB
      // ------------------------------------------------------------------------
      FindPointsGSLIB finder(pmesh->GetComm());
      finder.Setup(*pmesh);
      finder.FindPoints(point_pos, ordering_type);

      const Array<unsigned int> &code = finder.GetCode(); // 0: inside, 1: boundary, 2: not found
      const Array<unsigned int> &proc = finder.GetProc(); // owning MPI rank
      const Array<unsigned int> &elem = finder.GetElem(); // local element id
      const Vector &rref = finder.GetReferencePosition(); // (r,s,t) byVDIM

      // ------------------------------------------------------------------------
      // 3) Make RHS and pre-subtract averaged charge density => enforce zero-mean RHS
      // ------------------------------------------------------------------------

      MPI_Comm comm = pfes->GetComm();

      if (!use_precomputed_neutralizing_const || !neutralizing_const_computed_1)
      {
         // compute neutralizing constant
         real_t local_sum = 0.0;
         for (int p = 0; p < npt; ++p)
         {
            // Skip particles not successfully found
            if (code[p] == 2) // not found
            {
               continue;
            }

            local_sum += Q(p);
         }

         real_t global_sum = 0.0;
         MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

         neutralizing_const_1 = -global_sum / domain_volume;
         if (Mpi::Root())
         {
            cout << "Total charge: " << global_sum << ", Domain volume: " << domain_volume << ", Neutralizing constant: " << neutralizing_const_1 << endl;
            if (use_precomputed_neutralizing_const)
            {
               cout << "Further updates will use this precomputed neutralizing constant." << endl;
            }
         }
         neutralizing_const_computed_1 = true;
         delete precomputed_neutralizing_lf;
         precomputed_neutralizing_lf = new ParLinearForm(pfes);
         *precomputed_neutralizing_lf = 0.0;
         ConstantCoefficient neutralizing_coeff(neutralizing_const_1);
         precomputed_neutralizing_lf->AddDomainIntegrator(new DomainLFIntegrator(neutralizing_coeff));
         precomputed_neutralizing_lf->Assemble();
      }
      ParLinearForm b(pfes);
      b = *precomputed_neutralizing_lf; // start with precomputed neutralizing contribution

      // ------------------------------------------------------------------------
      // 4) Deposit q_p * phi_i(x_p) into a ParLinearForm (RHS b)
      //      b_i = sum_p q_p * φ_i(x_p)
      // ------------------------------------------------------------------------
      int myid;
      MPI_Comm_rank(pmesh->GetComm(), &myid);

      Array<int> dofs;

      for (int p = 0; p < npt; ++p)
      {
         // Skip particles not successfully found
         if (code[p] == 2) // not found
         {
            continue;
         }

         // Raise error if particle is not on the current rank
         if ((int)proc[p] != myid)
         {
            // raise error
            MFEM_ABORT("Particle " << p << " found in element owned by rank "
                                   << proc[p] << " but current rank is " << myid << "." << endl
                                   << "You must call redistribute everytime before updating the density grid function.");
            continue;
         }
         const int e = elem[p];

         // Reference coordinates for this particle (r,s[,t]) with byVDIM layout
         IntegrationPoint ip;
         if (dim == 1)
         {
            ip.x = rref(p);
         }
         else if (dim == 2)
         {
            ip.Set2(rref[2 * p + 0], rref[2 * p + 1]);
         }
         else // dim == 3
         {
            ip.Set3(rref[3 * p + 0], rref[3 * p + 1], rref[3 * p + 2]);
         }

         const FiniteElement &fe = *pfes->GetFE(e);
         const int ldofs = fe.GetDof();

         Vector shape(ldofs);
         fe.CalcShape(ip, shape); // φ_i(x_p) in this element

         pfes->GetElementDofs(e, dofs); // local dof indices

         const real_t q_p = Q(p);

         // Add q_p * φ_i(x_p) to b_i
         b.AddElementVector(dofs, q_p, shape);
      }
      if (!use_precomputed_neutralizing_const || !neutralizing_const_computed_2)
      {
         ParGridFunction one_gf(pfes);
         one_gf = 1.0;

         // Compute b(1)
         double local_b1 = b(one_gf);
         double global_b1 = 0.0;
         MPI_Allreduce(&local_b1, &global_b1, 1, MPI_DOUBLE, MPI_SUM, comm);

         // Build v_i = ∫ phi_i dx  (linear form for coefficient 1)
         vol_lf = new ParLinearForm(pfes);
         *vol_lf = 0.0;
         ConstantCoefficient one(1.0);
         vol_lf->AddDomainIntegrator(new DomainLFIntegrator(one));
         vol_lf->Assemble();

         // Compute ∫ 1 dx consistently (same discrete measure)
         double local_V = (*vol_lf)(one_gf);
         double global_V = 0.0;
         MPI_Allreduce(&local_V, &global_V, 1, MPI_DOUBLE, MPI_SUM, comm);

         neutralizing_const_2 = global_b1 / global_V; // amount to subtract

         // b <- b - neutralizing_const_2 * vol_lf  so that b(1) becomes exactly 0 (up to roundoff in this step)
         neutralizing_const_computed_2 = true;
         if (Mpi::Root())
         {
            cout << "RHS total before neutralization: " << global_b1 << ", Domain volume (discrete): " << global_V << ", Neutralizing constant 2: " << neutralizing_const_2 << endl;
         }
      }
      b.Add(-neutralizing_const_2, *vol_lf);

      // Assemble to a global true-dof RHS vector compatible with MassMatrix
      HypreParVector *B = b.ParallelAssemble(); // owns new vector on heap

      // ------------------------------------------------------------------
      // 5) Solve A * phi = B with zero-mean enforcement via OrthoSolver
      // ------------------------------------------------------------------
      MFEM_VERIFY(DiffusionMatrix != nullptr, "DiffusionMatrix must be precomputed.");

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
      HypreParVector *B = b.ParallelAssemble();

      // 2. make the bilinear form
      ParBilinearForm a(E_gf.ParFESpace());
      ConstantCoefficient one_coef(1.0);
      a.AddDomainIntegrator(new VectorFEMassIntegrator(one_coef));
      a.Assemble();
      a.Finalize();
      // Parallel operator (HypreParMatrix)
      HypreParMatrix *A = a.ParallelAssemble();

      // 3. solve for E_gf
      CGSolver M_solver(E_gf.ParFESpace()->GetComm());
      M_solver.iterative_mode = false;
      M_solver.SetRelTol(1e-12);
      M_solver.SetAbsTol(0.0);
      M_solver.SetMaxIter(1e5);
      M_solver.SetPrintLevel(0);
      M_solver.SetOperator(*A);

      HypreParVector X(E_gf.ParFESpace()->GetComm(), E_gf.ParFESpace()->GlobalTrueVSize(), E_gf.ParFESpace()->GetTrueDofOffsets());
      X = 0.0;
      M_solver.Mult(*B, X);
      E_gf.SetFromTrueDofs(X);
      delete A;
      delete B;
   }

   if (ctx.visualization)
   {
      static socketstream sol_sock;
      static bool init = false;
      static ParMesh *pmesh = E_gf.ParFESpace()->GetParMesh();

      int num_procs = Mpi::WorldSize();
      int myid_vis = Mpi::WorldRank();
      char vishost[] = "localhost";
      int visport = ctx.visport;

      if (!init)
      {
         sol_sock.open(vishost, visport);
         if (sol_sock)
         {
            init = true;
         }
      }
      if (init)
      {
         sol_sock << "parallel " << num_procs << " " << myid_vis << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n"
                  << *pmesh << E_gf << std::flush;
      }
   }
   if (ctx.visualization)
   {
      static socketstream sol_sock;
      static bool init = false;
      static ParMesh *pmesh = phi_gf.ParFESpace()->GetParMesh();

      int num_procs = Mpi::WorldSize();
      int myid_vis = Mpi::WorldRank();
      char vishost[] = "localhost";
      int visport = ctx.visport;

      if (!init)
      {
         sol_sock.open(vishost, visport);
         if (sol_sock)
         {
            init = true;
         }
      }
      if (init)
      {
         sol_sock << "parallel " << num_procs << " " << myid_vis << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n"
                  << *pmesh << phi_gf << std::flush;
      }
   }
}
class GreenFunctionCoefficient : public Coefficient
{
public:
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      // G(x, y) = \sum_{(m,n)\neq(0,0)} \frac{(-1)^{m+n}}{4\pi^2(m^2+n^2)} \cos\big(2\pi(mx+ny)\big).
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t val = 0.0;
      const int M_max = 15;
      const int N_max = 15;

      for (int m = -M_max; m <= M_max; ++m)
      {
         for (int n = -N_max; n <= N_max; ++n)
         {
            if (m == 0 && n == 0)
               continue;

            int parity = (m + n) & 1; // even/odd, works for negatives
            real_t numerator = (parity == 0) ? 1.0 : -1.0;

            real_t denominator = 4.0 * M_PI * M_PI * (m * m + n * n);
            real_t angle = 2.0 * M_PI * (m * x[0] + n * x[1]);

            val += numerator * cos(angle) / denominator;
         }
      }

      return val;
   }
};

void GridFunctionUpdates::TotalEnergyValidation(const ParticleSet &particles,
                                                const ParGridFunction &E_gf)
{
   const MultiVector &P = particles.Field(Boris::MOM);
   const MultiVector &M = particles.Field(Boris::MASS);

   real_t kinetic_energy = 0.0;
   for (int p = 0; p < particles.GetNP(); ++p)
   {
      real_t p_square_p = 0.0;
      for (int d = 0; d < P.GetVDim(); ++d)
      {
         p_square_p += P(p, d) * P(p, d);
      }
      kinetic_energy += 0.5 * p_square_p / M(p);
   }

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
   MPI_Allreduce(&kinetic_energy, &global_kinetic_energy, 1, MPI_DOUBLE, MPI_SUM,
                 fes->GetComm());
   if (Mpi::Root())
   {
      cout << "Kinetic energy: " << global_kinetic_energy << "\t";
      cout << "Field energy: " << global_field_energy << "\t";
      cout << "Total energy: " << global_kinetic_energy + global_field_energy << endl;
   }
   // write to a csv
   if (Mpi::Root())
   {
      std::ofstream energy_file("energy.csv", std::ios::app);
      energy_file << setprecision(10) << global_kinetic_energy << "," << global_field_energy << "," << global_kinetic_energy + global_field_energy << "\n";
   }
}
