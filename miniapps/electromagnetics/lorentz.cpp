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
// This miniapp computes the trajectories of a set of charged particles subject
// to Lorentz forces.
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
// particles' locations and momenta are randomly initialized within a bounding
// box specified by command line input.
//
// This miniapp demonstrates the use of ParticleSet with FindPointsGSLIB. When
// particles leave either domains, they are subject to removal. Redistribution
// of particle data between MPI ranks is also demonstrated.
//
// Note that the VisItDataCollection objects must have been stored using the
// parallel format e.g. visit_dc.SetFormat(DataCollection::PARALLEL_FORMAT);.
// Without this optional format specifier the vector field lookups will fail.
//
// Compile with: make lorentz
//
// Sample runs:
//
//   Particles accelerating in a constant electric field
//      mpirun -np 4 volta -m ../../data/inline-hex.mesh -dbcs '1 6' -dbcv '0 1'
//      mpirun -np 4 lorentz -er Volta-AMR-Parallel -npt 100 -xmin '0.0 0.0 0.0' -xmax '1.0 1.0 1.0' -pmin '1 0 0' -pmax '1 0 0' -rdf 0 -vt 0 -nt 100
//
//   Particles accelerating in a constant magnetic field
//      mpirun -np 4 tesla -m ../../data/inline-hex.mesh -ubbc '0 0 1'
//      mpirun -np 4 lorentz -br Tesla-AMR-Parallel -npt 10 -xmin '0.0 0.0 0.0' -xmax '1.0 1.0 1.0' -pmin '0 0.1 0.05' -pmax '0 0.4 0.1' -nt 1000 -rdf 0 -vt 0
//
//   Magnetic mirror effect near a charged sphere and a bar magnet
//      mpirun -np 4 volta -m ../../data/ball-nurbs.mesh -dbcs 1 -cs '0 0 0 0.1 2e-11' -rs 2 -maxit 4
//      mpirun -np 4 tesla -m ../../data/fichera.mesh -maxit 4 -rs 3 -bm '-0.1 -0.1 -0.1 0.1 0.1 0.1 0.1 -1e10'
//      mpirun -np 4 lorentz -er Volta-AMR-Parallel -ec 4 -br Tesla-AMR-Parallel -bc 4 -q -10 -dt 1e-4 -nt 2000 -npt 500 -vt 10 -rdf 500 -rdm 1 -vf 10 -pmin '-8 -4 4' -pmax '-8 -4 4' -xmin '-1 -1 -1' -xmax '1 1 1'
//      mpirun -np 4 lorentz -er Volta-AMR-Parallel -ec 4 -br Tesla-AMR-Parallel -bc 4 -q -10 -dt 1e-3 -npt 1 -vt 650 -rdf 500 -rdm 1 -vf 2 -pmin '-8 -4 4' -pmax '-8 -4 4' -xmin '0.8 0 0' -xmax '0.8 0 0' -nt 1300

#include "mfem.hpp"
#include "../common/particles_extras.hpp"

#include "electromagnetics.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::electromagnetics;

struct LorentzContext
{
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

   int ordering = 1;               // 0 - byNODES, 1 - byVDIM
   int npt = 1;                    // total number of particles
   real_t q = 1.0;                 // particle charge
   real_t m = 1.0;                 // particle mass
   Vector x_min{-1.0,-1.0,-1.0};   // initial position min
   Vector x_max{1.0,1.0,1.0};      // initial position max
   Vector p_min{-1.0,-1.0,-1.0};   // initial momentum min
   Vector p_max{1.0,1.0,1.0};      // initial momentum max
   real_t dt = 1e-2;               // time step
   int nt = 1000;                  // number of timesteps
   int redist_interval = 5;        // redistribution interval
   int redist_mesh = 0;            // redistribution mesh: 0: E mesh, 1: B mesh
} ctx;

/// This class implements the Boris algorithm as described in the article
/// `Why is Boris algorithm so good?` by H. Qin et al in Physics of Plasmas,
/// Volume 20 Issue 8, August 2013, https://doi.org/10.1063/1.4818428.
class Boris
{
public:
   /// Field indices
   /** Allows for convenient access to corresponding ParticleVector from
       ParticleSet. */
   enum Fields
   {
      MASS,   // vdim = 1
      CHARGE, // vdim = 1
      MOM,    // vdim = dim
      EFIELD, // vdim = dim
      BFIELD // vdim = dim
   };
protected:
   /// Pointers to E and B field GridFunctions
   GridFunction *E_gf = nullptr;
   GridFunction *B_gf = nullptr;

   /// FindPointsGSLIB objects for E and B field meshes
   FindPointsGSLIB E_finder;
   FindPointsGSLIB B_finder;

   /// ParticleSet of charged particles
   std::unique_ptr<ParticleSet> charged_particles;

   // Temporary vectors for particle computation
   mutable Vector pxB_, pm_, pp_;

   /// Single particle Boris step
   void ParticleStep(Particle &part, real_t &dt);
public:

   Boris(MPI_Comm comm, GridFunction *E_gf_, GridFunction *B_gf_,
         int nparticles, Ordering::Type pdata_ordering);

   /// Find Particles in mesh corresponding to E and B fields
   void FindParticles();

   /// Update E and B fields at particle locations. Must be called
   /// right after FindParticles has been called.
   void EvaluateFieldsAtParticles();

   /// Advance particles one time step using Boris algorithm
   void Step(real_t &t, real_t &dt);

   /// Remove lost particles and return their indices
   Array<int> RemoveLostParticles();

   /// Redistribute particles based on \p redist_mesh (0 - E field, 1 - B field)
   void Redistribute(int redist_mesh, Array<int> &removed_idxs);

   /// Get reference to the ParticleSet of charged particles
   ParticleSet& GetParticles() { return *charged_particles; }

   /// Get reference to the E field FindPointsGSLIB object
   FindPointsGSLIB& GetEFinder() { return E_finder; }
};

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

// Open the named VisItDataCollection and read the named field.
// Returns pointers to the two new objects.
int ReadGridFunction(std::string coll_name, std::string field_name,
                     int pad_digits_cycle, int pad_digits_rank, int cycle,
                     std::unique_ptr<VisItDataCollection> &dc,
                     ParGridFunction *&gf);

// Initialize particles from user input.
void InitializeChargedParticles(ParticleSet &particles, const Vector &pos_min,
                                const Vector &pos_max, const Vector &x_init,
                                const Vector &p_init, real_t m,
                                real_t q);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_ranks = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   if ( Mpi::Root() ) { display_banner(cout); }

   bool visualization = true;      // enable visualization
   int vis_tail_size = 5;          // particle trajectory tail size
   int vis_interval = 4;           // visualization interval

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.E.coll_name, "-er", "--e-root-file",
                  "Set the VisIt data collection E field root file prefix.");
   args.AddOption(&ctx.E.field_name, "-ef", "--e-field-name",
                  "Set the VisIt data collection E field name");
   args.AddOption(&ctx.E.cycle, "-ec", "--e-cycle",
                  "Set the E field cycle index to read.");
   args.AddOption(&ctx.E.pad_digits_cycle, "-epdc", "--e-pad-digits-cycle",
                  "Number of digits in E field cycle.");
   args.AddOption(&ctx.E.pad_digits_rank, "-epdr", "--e-pad-digits-rank",
                  "Number of digits in E field MPI rank.");
   args.AddOption(&ctx.B.coll_name, "-br", "--b-root-file",
                  "Set the VisIt data collection B field root file prefix.");
   args.AddOption(&ctx.B.field_name, "-bf", "--b-field-name",
                  "Set the VisIt data collection B field name");
   args.AddOption(&ctx.B.cycle, "-bc", "--b-cycle",
                  "Set the B field cycle index to read.");
   args.AddOption(&ctx.B.pad_digits_cycle, "-bpdc", "--b-pad-digits-cycle",
                  "Number of digits in B field cycle.");
   args.AddOption(&ctx.B.pad_digits_rank, "-bpdr", "--b-pad-digits-rank",
                  "Number of digits in B field MPI rank.");
   args.AddOption(&ctx.redist_interval, "-rdf", "--redist-interval",
                  "Redistribution after this many timesteps. 0 means "
                  "no redistribution.");
   args.AddOption(&ctx.redist_mesh, "-rdm", "--redistribution-mesh",
                  "Particle domain mesh for redistribution. 0 for E field mesh."
                  " 1 for B field mesh.");
   args.AddOption(&ctx.ordering, "-o", "--ordering",
                  "Ordering of particle data. 0 = byNODES, 1 = byVDIM.");
   args.AddOption(&ctx.npt, "-npt", "--num-particles",
                  "Total number of particles.");
   args.AddOption(&ctx.m, "-m", "--mass", "Particles' mass.");
   args.AddOption(&ctx.q, "-q", "--charge", "Particles' charge.");
   args.AddOption(&ctx.x_min, "-xmin", "--x-min",
                  "Minimum initial particle location.");
   args.AddOption(&ctx.x_max, "-xmax", "--x-max",
                  "Maximum initial particle location.");
   args.AddOption(&ctx.p_min, "-pmin", "--p-min",
                  "Minimum initial particle momentum.");
   args.AddOption(&ctx.p_max, "-pmax", "--p-max",
                  "Maximum initial particle momentum.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time Step.");
   args.AddOption(&ctx.nt, "-nt", "--num-timesteps", "Number of timesteps.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_tail_size, "-vt", "--vis-tail-size",
                  "GLVis visualization trajectory truncation tail size.");
   args.AddOption(&vis_interval, "-vf", "--vis-interval",
                  "GLVis visualization update after this many timesteps. "
                  "0 means no visualization.");

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

   std::unique_ptr<VisItDataCollection> E_dc, B_dc;
   ParGridFunction *E_gf = nullptr, *B_gf = nullptr;
   Vector bb_xmin, bb_xmax;

   // Read E field if provided
   if (ctx.E.coll_name != "")
   {
      if (ReadGridFunction(ctx.E.coll_name, ctx.E.field_name,
                           ctx.E.pad_digits_cycle, ctx.E.pad_digits_rank,
                           ctx.E.cycle, E_dc, E_gf))
      {
         mfem::err << "Error loading E field" << endl;
         return 1;
      }
      E_gf->ParFESpace()->GetParMesh()->GetBoundingBox(bb_xmin, bb_xmax, 2);
   }

   // Read B field if provided
   if (ctx.B.coll_name != "")
   {
      if (ReadGridFunction(ctx.B.coll_name, ctx.B.field_name,
                           ctx.B.pad_digits_cycle, ctx.B.pad_digits_rank,
                           ctx.B.cycle, B_dc, B_gf))
      {
         mfem::err << "Error loading B field" << endl;
         return 1;
      }
      Vector bb_xmint, bb_xmaxt;
      B_gf->ParFESpace()->GetParMesh()->GetBoundingBox(bb_xmint, bb_xmaxt, 2);
      if (ctx.E.coll_name != "")
      {
         // compute intersection of bounding boxes
         for (int d = 0; d < bb_xmin.Size(); d++)
         {
            bb_xmin[d] = std::max(bb_xmin[d], bb_xmint[d]);
            bb_xmax[d] = std::min(bb_xmax[d], bb_xmaxt[d]);
         }
      }
      else
      {
         bb_xmin = bb_xmint;
         bb_xmax = bb_xmaxt;
      }
   }

   Ordering::Type ordering_type = ctx.ordering == 0 ?
                                  Ordering::byNODES : Ordering::byVDIM;

   // Initialize particles
   int num_particles = ctx.npt/num_ranks +
                       (rank < (ctx.npt % num_ranks) ? 1 : 0);
   Boris boris(MPI_COMM_WORLD, E_gf, B_gf, num_particles, ordering_type);
   InitializeChargedParticles(boris.GetParticles(), ctx.x_min, ctx.x_max,
                              ctx.p_min, ctx.p_max, ctx.m, ctx.q);
   boris.FindParticles();
   boris.EvaluateFieldsAtParticles();

   real_t t = 0.0;
   real_t dt = ctx.dt;

   // Setup visualization
   char vishost[] = "localhost";
   socketstream pre_redist_sock, post_redist_sock;
   std::unique_ptr<ParticleTrajectories> traj_vis;
   if (visualization)
   {
      const char *keys = "baaa";
      traj_vis = std::make_unique<ParticleTrajectories>(boris.GetParticles(),
                                                        vis_tail_size,
                                                        vishost, 19916,
                                                        "Trajectories",
                                                        0, 0, 600, 600, keys);
      traj_vis->SetVisualizationBoundingBox(bb_xmin, bb_xmax);
   }

   for (int step = 1; step <= ctx.nt; step++)
   {
      // Step the Boris algorithm
      boris.Step(t, dt);
      if (Mpi::Root())
      {
         mfem::out << "Step: " << step << " | Time: " << t << endl;
      }

      // Visualize trajectories
      if (visualization && step % vis_interval == 0)
      {
         traj_vis->Visualize();
      }

      // Remove lost particles from particle set and output
      Array<int> removed_idxs = boris.RemoveLostParticles();

      // Redistribute
      if (ctx.redist_interval > 0 && step % ctx.redist_interval == 0 &&
          boris.GetParticles().GetGlobalNParticles() > 0)
      {
         // Redistribute particles - prior to redistribution, removed any lost
         // particles that were just removed from the set.
         boris.Redistribute(ctx.redist_mesh, removed_idxs);
      }
   }
}

void Boris::ParticleStep(Particle &part, real_t &dt)
{
   Vector &x = part.Coords();
   real_t m = part.FieldValue(MASS);
   real_t q = part.FieldValue(CHARGE);
   Vector &p = part.Field(MOM);
   Vector &e = part.Field(EFIELD);
   Vector &b = part.Field(BFIELD);

   // Compute half of the contribution from q E
   add(p, 0.5 * dt * q, e, pm_);

   // Compute the contribution from q p x B
   const real_t B2 = b * b;

   // ... along pm x B
   const real_t a1 = 4.0 * dt * q * m;
   pm_.cross3D(b, pxB_);
   pp_.Set(a1, pxB_);

   // ... along pm
   const real_t a2 = 4.0 * m * m -
                     dt * dt * q * q * B2;
   pp_.Add(a2, pm_);

   // ... along B
   const real_t a3 = 2.0 * dt * dt * q * q * (b * pm_);
   pp_.Add(a3, b);

   // scale by common denominator
   const real_t a4 = 4.0 * m * m +
                     dt * dt * q * q * B2;
   pp_ /= a4;

   // Update the momentum
   add(pp_, 0.5 * dt * q, e, p);

   // Update the position
   x.Add(dt / m, p);
}

Boris::Boris(MPI_Comm comm, GridFunction *E_gf_, GridFunction *B_gf_,
             int nparticles, Ordering::Type pdata_ordering)
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

   pxB_.SetSize(dim); pm_.SetSize(dim); pp_.SetSize(dim);

   /// Create particle set:
   /// 2 scalars of mass and charge,
   /// 3 vectors of size space dim for momentum, e field, and b field
   Array<int> field_vdims({1, 1, dim, dim, dim});

   charged_particles = std::make_unique<ParticleSet>
                       (comm, nparticles, dim, field_vdims, 0, pdata_ordering);
}

void Boris::FindParticles()
{
   ParticleVector &X = charged_particles->Coords();

   // Find particles in E and B field meshes
   if (E_gf)
   {
      E_finder.FindPoints(X); // X.GetOrdering() used internally
   }
   if (B_gf)
   {
      B_finder.FindPoints(X); // X.GetOrdering() used internally
   }
}

void Boris::EvaluateFieldsAtParticles()
{
   ParticleVector &E = charged_particles->Field(EFIELD);
   ParticleVector &B = charged_particles->Field(BFIELD);

   // Interpolate E-field + B-field onto particles
   if (E_gf)
   {
      E_finder.Interpolate(*E_gf, E, E.GetOrdering());
   }
   else
   {
      E = 0.0;
   }
   if (B_gf)
   {
      B_finder.Interpolate(*B_gf, B, B.GetOrdering());
   }
   else
   {
      B = 0.0;
   }
}

void Boris::Step(real_t &t, real_t &dt)
{
   // Interpolate E and B fields onto particles
   EvaluateFieldsAtParticles();

   // Individually step each particle. If all ParticleSet fields are ordered
   // byVDIM, we can use GetParticleRef for better performance.
   if (charged_particles->IsParticleRefValid())
   {
      for (int i = 0; i < charged_particles->GetNParticles(); i++)
      {
         Particle p = charged_particles->GetParticleRef(i);
         ParticleStep(p, dt);
      }
   }
   else
   {
      for (int i = 0; i < charged_particles->GetNParticles(); i++)
      {
         Particle p = charged_particles->GetParticle(i);
         ParticleStep(p, dt);
         charged_particles->SetParticle(i, p);
      }
   }

   // Find updated particle locations in E and B field meshes
   FindParticles();

   // Update time
   t += dt;
}

Array<int> Boris::RemoveLostParticles()
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
   return lost_idxs;
}

void Boris::Redistribute(int redist_mesh, Array<int> &removed_idxs)
{
   if (redist_mesh == 0 && E_gf)
   {
      Array<int> proc_list = E_finder.GetProc();
      proc_list.DeleteAt(removed_idxs);
      charged_particles->Redistribute(proc_list);
   }
   else
   {
      Array<int> proc_list = B_finder.GetProc();
      proc_list.DeleteAt(removed_idxs);
      charged_particles->Redistribute(proc_list);
   }

   // Find particles again since ParticleSet is not yet synced with
   // FindPointsGSLIB objects.
   FindParticles();
}

void display_banner(ostream & os)
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
      << endl << flush;
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
                                const Vector &x_min, const Vector &x_max, const Vector &p_min,
                                const Vector &p_max, real_t m, real_t q)
{
   int dim = charged_particles.Coords().GetVDim();
   int rank;
   MPI_Comm_rank(charged_particles.GetComm(), &rank);
   std::mt19937 gen(rank);

   // Set up uniform distribution for position
   std::uniform_real_distribution<real_t> real_dist_x(0_r,1_r);

   // Set up guassian distribution for momentum. Centered between p_min and
   // p_max with 3-sigma range covering the box.
   Vector p_center(dim);
   add(0.5, p_min, p_max, p_center);
   Vector dp = p_max; dp -= p_min; dp *= 1_r/6_r; // 3-sigma range
   std::vector<std::normal_distribution<real_t>> norm_dist_p;
   for (int d = 0; d < dim; d++)
   {
      norm_dist_p.emplace_back(p_center[d], dp[d] > 0_r ? dp[d] : 1_r);
   }

   ParticleVector &X = charged_particles.Coords();
   ParticleVector &P = charged_particles.Field(Boris::MOM);
   ParticleVector &M = charged_particles.Field(Boris::MASS);
   ParticleVector &Q = charged_particles.Field(Boris::CHARGE);

   for (int i = 0; i < charged_particles.GetNParticles(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         if (x_min[d] >= x_max[d]) { X(i,d) = x_min[d]; }
         else
         {
            X(i,d) = x_min[d] + real_dist_x(gen)*(x_max[d] - x_min[d]);
         }

         // Initialize momentum
         if (p_min[d] >= p_max[d]) { P(i,d) = p_min[d]; }
         else
         {
            real_t p_val = norm_dist_p[d](gen);
            while (p_val < p_min[d] || p_val > p_max[d])
            {
               p_val = norm_dist_p[d](gen);
            }
            P(i,d) = p_val;
         }
      }
      // Initialize mass + charge
      M(i) = m;
      Q(i) = q;
   }
}
