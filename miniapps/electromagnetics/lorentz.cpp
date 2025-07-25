// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
// This miniapp computes the trajectory of a single charged particle subject to
// Lorentz forces.
//
//                           dp/dt = q (E + v x B)
//
// The method used is the explicit Boris algortihm which conserves phase space
// volume for long term accuracy.
//
// The electric and magnetic fields are read from VisItDataCollection objects
// such as those produced by the Volta and Tesla miniapps. It is notable that
// these two fields do not need to be defined on the same mesh. Of course, the
// particle trajectory can only be computed on the intersection of the two
// domains. The starting point of the path must be chosen within in this
// intersection and the trajectory will terminate when it leaves the
// intersection or reaches a specified time duration.
//
// Note that the VisItDataCollection objects must have been stored using the
// parallel format e.g. visit_dc.SetFormat(DataCollection::PARALLEL_FORMAT);.
// Without this optional format specifier the vector field lookups will fail.
//
// Compile with: make lorentz
//
// Sample runs:
//
//   Free particle moving with constant velocity
//      mpirun -np 4 lorentz -p0 '1 1 1'
//
//   Particle accelerating in a constant electric field
//      mpirun -np 4 volta -m ../../data/inline-hex.mesh -dbcs '1 6' -dbcv '0 1'
//      mpirun -np 4 lorentz -er Volta-AMR-Parallel -x0 '0.5 0.5 0.9'
//                           -p0 '1 0 0'
//
//   Particle accelerating in a constant magnetic field
//      mpirun -np 4 tesla -m ../../data/inline-hex.mesh -ubbc '0 0 1'
//      mpirun -np 4 lorentz -br Tesla-AMR-Parallel -x0 '0.1 0.5 0.1'
//                           -p0 '0 0.4 0.1' -tf 9
//
//   Magnetic mirror effect near a charged sphere and a bar magnet
//      mpirun -np 4 volta -m ../../data/ball-nurbs.mesh -dbcs 1
//                         -cs '0 0 0 0.1 2e-11' -rs 2 -maxit 4
//      mpirun -np 4 tesla -m ../../data/fichera.mesh -maxit 4 -rs 3
//                         -bm '-0.1 -0.1 -0.1 0.1 0.1 0.1 0.1 -1e10'
//      mpirun -np 4 lorentz -er Volta-AMR-Parallel -ec 4
//                           -br Tesla-AMR-Parallel -bc 4 -x0 '0.8 0 0'
//                           -p0 '-8 -4 4' -q -10 -tf 0.2 -dt 1e-3 -rf 1e-6
//
// This miniapp demonstrates the use of the ParMesh::FindPoints functionality
// to evaluate field data from stored DataCollection objects.  While this
// miniapp is far from a full particle-in-cell (PIC) code it does demonstrate
// some of the building blocks that might be used to construct the particle
// mover portion of a PIC code.



// Joe test case:
// Adaptation of magnetic mirror effect:
// mpirun -np 4 ./volta -m ../../data/ball-nurbs.mesh -dbcs 1 -cs '0 0 0 0.1 2e-11' -rs 2 -maxit 4
// mpirun -np 4 ./tesla -m ../../data/fichera.mesh -maxit 4 -rs 3  -bm '-0.1 -0.1 -0.1 0.1 0.1 0.1 0.1 -1e10'
// mpirun -np 4 ./lorentz -er Volta-AMR-Parallel -ec 4 -br Tesla-AMR-Parallel -bc 4 -x0 '0.8 0 0' -p0 '-8 -4 4' -q -10 -dt 1e-3 -nt 1000 -npt 1000 -vt 3 -rf 1000 -pm 2 -vf 5

#include "mfem.hpp"
#include "../common/fem_extras.hpp"
#include "../common/particles_extras.hpp"
#include "../../general/text.hpp"

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

   int particle_mesh = -1;
   int ordering = 1;
   int npt = 1;
   real_t q = 1.0;
   real_t m = 1.0;
   Vector x_init;
   Vector p_init;
   real_t dt = 1e-2;
   real_t t0 = 0.0;
   int nt = 1000.0;
   int redist_freq = 10;
   int rm_lost_freq = 1;

   bool visualization = true;
   int visport = 19916;
   int vis_tail_size = 5;
   int vis_freq = 5;

   int csv_freq = 0;
} ctx;

enum ParticleFields
{
   MASS,   // vdim = 1
   CHARGE, // vdim = 1
   MOM     // vdim = dim
};

/// This class implements the Boris algorithm as described in the
/// article `Why is Boris algorithm so good?` by H. Qin et al in
/// Physics of Plasmas, Volume 20 Issue 8, August 2013,
/// https://doi.org/10.1063/1.4818428.
class BorisAlgorithm
{
protected:
   FindPointsGSLIB E_finder;
   FindPointsGSLIB B_finder;

   mutable Vector E_p_;
   mutable Vector B_p_;
   mutable Vector pxB_;
   mutable Vector pm_;
   mutable Vector pp_;

   static void GetValues(const ParticleVector &coords, FindPointsGSLIB &finder, GridFunction &gf, ParticleVector &pv);

public:

   BorisAlgorithm() {};

   BorisAlgorithm(MPI_Comm comm)
   : E_finder(comm), B_finder(comm) {}

   void Step(GridFunction *E_gf, GridFunction *B_gf, ParticleSet &charged_particles, real_t &t, real_t &dt);

   void ParticleStep(const Vector &e, const Vector &b, Particle &part, real_t &dt);

};

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

// Open the named VisItDataCollection and read the named field.
// Returns pointers to the two new objects.
int ReadGridFunction(std::string coll_name, std::string field_name,
                     int pad_digits_cycle, int pad_digits_rank, int cycle,
                     std::unique_ptr<VisItDataCollection> &dc, ParGridFunction *&gf);

// Initialize particles from user input.
void InitializeChargedParticles(ParticleSet &particles, ParMesh *pmesh, const Vector &x_init, const Vector &p_init, real_t q, real_t m);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   if ( Mpi::Root() ) { display_banner(cout); }

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
   args.AddOption(&ctx.particle_mesh, "-pm", "--particle-mesh", 
                  "Particle domain mesh for initialization, redistribution, + lost particle detection. -1 for unit cube. 0 for E field mesh. 1 for B field mesh.");
   args.AddOption(&ctx.redist_freq, "-rf", "--redist-freq", "Redistribution frequency.");
   args.AddOption(&ctx.rm_lost_freq, "-rm", "--remove-lost-freq", "Remove lost particles frequency.");
   args.AddOption(&ctx.ordering, "-o", "--ordering", "Ordering of particle data. 0 = byNODES, 1 = byVDIM.");
   args.AddOption(&ctx.npt, "-npt", "--num-part", "Total number of particles.");
   args.AddOption(&ctx.m, "-m", "--mass",
                  "Particles' mass.");
   args.AddOption(&ctx.q, "-q", "--charge",
                  "Particles' charge.");
   args.AddOption(&ctx.x_init, "-x0", "--initial-positions",
                  "Particle initial positions byVDIM (if # positions < # particles, remaining positions set randomly).");
   args.AddOption(&ctx.p_init, "-p0", "--initial-momenta",
                  "Particle initial momenta byVDIM (if # momenta < # particles, last momentum is used for remaining particles)");
   args.AddOption(&ctx.dt, "-dt", "--time-step",
                  "Time Step.");
   args.AddOption(&ctx.t0, "-t0", "--initial-time",
                  "Initial Time.");
   args.AddOption(&ctx.nt, "-nt", "--num-timesteps",
                  "Number of timesteps.");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.vis_tail_size, "-vt", "--vis-tail-size", "GLVis visualization trajectory truncation tail size.");
   args.AddOption(&ctx.vis_freq, "-vf", "--vis-freq", "GLVis visualization frequency.");
   args.AddOption(&ctx.visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&ctx.csv_freq, "-csv", "--csv-freq", "Frequency of particle CSV outputting.");

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

   if (ctx.E.coll_name != "")
   {
      if (ReadGridFunction(ctx.E.coll_name, ctx.E.field_name, ctx.E.pad_digits_cycle,
                           ctx.E.pad_digits_rank, ctx.E.cycle, E_dc, E_gf))
      {
         mfem::out << "Error loading E field" << endl;
         return 1;
      }
   }

   if (ctx.B.coll_name != "")
   {
      if (ReadGridFunction(ctx.B.coll_name, ctx.B.field_name, ctx.B.pad_digits_cycle,
                           ctx.B.pad_digits_rank, ctx.B.cycle, B_dc, B_gf))
      {
         mfem::out << "Error loading B field" << endl;
         return 1;
      }
   }

   if (B_gf && E_gf)
   {
      int E_dim = E_gf->ParFESpace()->GetParMesh()->SpaceDimension();
      int B_dim = E_gf->ParFESpace()->GetParMesh()->SpaceDimension();
      MFEM_VERIFY(E_dim == B_dim, "E field and B field must have the same spatial dimension.");
   }

   ParMesh *particle_mesh = nullptr;
   int dim = 3;
   if (ctx.particle_mesh != -1)
   {
      particle_mesh = ctx.particle_mesh == 0 ? static_cast<ParMesh*>(E_gf->ParFESpace()->GetParMesh()) : static_cast<ParMesh*>(B_gf->ParFESpace()->GetParMesh());
      dim = particle_mesh->SpaceDimension();
   }
   else
   {
      Mesh temp_m = Mesh::MakeCartesian3D(Mpi::WorldSize(), Mpi::WorldSize(), Mpi::WorldSize(), Element::Type::HEXAHEDRON);
      particle_mesh = new ParMesh(MPI_COMM_WORLD, temp_m);
   }

   Ordering::Type ordering_type = ctx.ordering == 0 ? Ordering::byNODES : Ordering::byVDIM;

   // Create particle set: 2 scalars of mass and charge, 1 vector of size space dim for momentum
   ParticleSet particles(MPI_COMM_WORLD, ctx.npt, dim, Array<int>({1,1,dim}), Array<const char*>({"Mass", "Charge", "Momentum"}), ordering_type);
   InitializeChargedParticles(particles, particle_mesh, ctx.x_init, ctx.p_init, ctx.q, ctx.m);
   
   // Create FindPointsGSLIB for redistributing + removing lost particles
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   particle_mesh->EnsureNodes();
   finder.Setup(*particle_mesh);

   // Initialize particle integrator
   BorisAlgorithm boris(MPI_COMM_WORLD);
   real_t t = ctx.t0;
   real_t dt = ctx.dt;

   // Setup visualization
   char vishost[] = "localhost";
   socketstream pre_redist_sock, post_redist_sock;
   std::unique_ptr<ParticleTrajectories> traj_vis;
   if (ctx.visualization)
   {
      traj_vis = std::make_unique<ParticleTrajectories>(particles, ctx.vis_tail_size, vishost, ctx.visport, "Particle Trajectories", 0, 0, 400, 400, "b");
   }

   for (int step = 1; step <= ctx.nt; step++)
   {
      // Step the Boris algorithm
      boris.Step(E_gf, B_gf, particles, t, dt);
      if (Mpi::Root())
      {
         mfem::out << "Step: " << step << " | Time: " << t << endl;
      }

      // Visualize trajectories
      if(ctx.visualization && step % ctx.vis_freq == 0)
      {
         traj_vis->Visualize();
      }

      // Remove lost particles
      if (step % ctx.rm_lost_freq == 0)
      {
         finder.FindPoints(particles.Coords(), particles.Coords().GetOrdering());
         particles.RemoveParticles(finder.GetPointsNotFoundIndices(), true);
      }

      // Redistribute
      if (step % ctx.redist_freq == 0 && particles.GetGlobalNP > 0)
      {  
         // Visualize particles pre-redistribute
         if (ctx.visualization)
         {
            Vector rank_vector(particles.GetNP());
            rank_vector = Mpi::WorldRank();
            VisualizeParticles(pre_redist_sock, vishost, ctx.visport, particles, rank_vector, 1e-2, "Particle Owning Rank (Pre-Redistribute)", 410, 0, 400, 400, "bca");
            char c;
            if (Mpi::Root())
            {
               cout << "Enter any key to redistribute: " << flush;
               cin >> c;
            }
            MPI_Barrier(MPI_COMM_WORLD);
         }
         
         // Redistribute
         finder.FindPoints(particles.Coords(), particles.Coords().GetOrdering());
         particles.Redistribute(finder.GetProc());

         // Visualize particles post-redistribute
         if (ctx.visualization)
         {
            Vector rank_vector(particles.GetNP());
            rank_vector = Mpi::WorldRank();
            VisualizeParticles(post_redist_sock, vishost, ctx.visport, particles, rank_vector, 1e-2, "Particle Owning Rank (Post-Redistribute)", 820, 0, 400, 400, "bca");
            char c;
            if (Mpi::Root())
            {
               cout << "Enter any key to continue: " << flush;
               cin >> c;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            pre_redist_sock << "keys q" << flush;
            post_redist_sock << "keys q" << flush;
            pre_redist_sock.close();
            post_redist_sock.close();
         }
      }

      // Write CSV
      if (step % ctx.csv_freq == 0)
      {
         std::string file_name = "Lorentz_Particles_" + mfem::to_padded_string(step, 9) + ".csv";
         particles.PrintCSV(file_name.c_str());
      }
   }

   // Delete particle_mesh if owned
   if (ctx.particle_mesh == -1)
   {
      delete particle_mesh;
   }

}


void BorisAlgorithm::GetValues(const ParticleVector &coords, FindPointsGSLIB &finder, GridFunction &gf, ParticleVector &pv)
{
   Mesh &mesh = *gf.FESpace()->GetMesh();
   mesh.EnsureNodes();
   finder.FindPoints(mesh, coords, coords.GetOrdering());
   finder.Interpolate(gf, pv);
   Ordering::Reorder(pv, pv.GetVDim(), gf.FESpace()->GetOrdering(), pv.GetOrdering());
}

void BorisAlgorithm::Step(GridFunction *E_gf, GridFunction *B_gf, ParticleSet &charged_particles, real_t &t, real_t &dt)
{
   ParticleVector &X = charged_particles.Coords();

   // TODO: What if we keep these interpolated-only ParticleVectors local here?
   // TODO: Inefficient to allocate every step. How can we improve in user-friendly way?
   ParticleVector E(charged_particles.GetNP(), X.GetVDim(), X.GetOrdering());
   ParticleVector B(charged_particles.GetNP(), X.GetVDim(), X.GetOrdering());

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

   pxB_.SetSize(X.GetVDim());
   pm_.SetSize(X.GetVDim());
   pp_.SetSize(X.GetVDim());

   // Individually step each particle:
   for (int i = 0; i < charged_particles.GetNP(); i++)
   {
      Particle p = charged_particles.GetParticle(i);
      
      E.GetParticleValues(i, E_p_);
      B.GetParticleValues(i, B_p_);

      ParticleStep(E_p_, B_p_, p, dt);

      charged_particles.SetParticle(i, p);
   }

   // Update time
   t += dt;
}

void BorisAlgorithm::ParticleStep(const Vector &e, const Vector &b, Particle &part, real_t &dt)
{
   Vector &x = part.Coords();
   real_t m = part.FieldValue(MASS);
   real_t q = part.FieldValue(CHARGE);
   Vector &p = part.Field(MOM);

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
   const real_t a3 = 2.0 * dt * dt * q * q * (b * p);
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
      mfem::out << "Error loading VisIt data collection: "
                << coll_name << endl;
      return 1;
   }

   if (dc->HasField(field_name))
   {
      gf = dc->GetParField(field_name);
   }

   return 0;
}

void InitializeChargedParticles(ParticleSet &charged_particles, ParMesh *pmesh, const Vector &x_init, const Vector &p_init, real_t q, real_t m)
{
   int rank, size;
   MPI_Comm_rank(charged_particles.GetComm(), &rank);
   MPI_Comm_size(charged_particles.GetComm(), &size);

   int dim = charged_particles.Coords().GetVDim();
   int npt = charged_particles.GetGlobalNP();

   // Initialize all x and p on rank 0, then split amongst ranks
   Array<int> sendcts(size);
   Array<int> displs(size);
   Vector x_all(npt*dim);
   Vector p_all(npt*dim);
   Vector pos_min, pos_max;
   if (pmesh)
   {
      pmesh->GetBoundingBox(pos_min, pos_max);
   }
   else
   {
      pos_min = Vector(dim);
      pos_max = Vector(dim);
      pos_min = 0.0;
      pos_max = 1.0;
   }
   if (rank == 0)
   {
      // Set x_all
      x_all.SetVector(x_init, 0);
      int provided_x = x_init.Size()/dim;

      Vector x_rem((npt - provided_x)*dim);
      x_rem.Randomize(17);

      for (int i = 0; i < x_rem.Size(); i++)
      {
         x_rem[i] = pos_min[i % dim] + x_rem[i]*(pos_max[i % dim] - pos_min[i % dim]);
      }
      x_all.SetVector(x_rem, provided_x*dim);


      // Set p_all
      Vector p_init_c = p_init;
      if (p_init_c.Size() == 0)
      {
         p_init_c.SetSize(dim);
         p_init_c = 0.0;
      }
      p_all.SetVector(p_init_c, 0);
      int provided_p = p_init_c.Size()/dim;

      Vector p_rem((npt - provided_p)*dim);
      for (int i = 0; i < p_rem.Size(); i++)
      {
         p_rem[i] = p_init_c[p_init_c.Size() - dim + i % dim];
      }
      p_all.SetVector(p_rem, provided_p*dim);
   }

   // Prepare args to MPI_Scatterv
   displs[0] = 0;
   for (int r = 0; r < size; r++)
   {
      sendcts[r] = charged_particles.GetNP()*dim;
      if (r > 0)
         displs[r] = sendcts[r-1] + displs[r-1];
   }

   Vector x_rank(sendcts[rank]);
   Vector p_rank(sendcts[rank]);

   // Scatter the large data buffer from Rank 0 to all other ranks
   MPI_Scatterv(x_all.GetData(), sendcts.GetData(), displs.GetData(), MPITypeMap<real_t>::mpi_type, x_rank.GetData(), x_rank.Size(), MPITypeMap<real_t>::mpi_type, 0, MPI_COMM_WORLD);
   MPI_Scatterv(p_all.GetData(), sendcts.GetData(), displs.GetData(), MPITypeMap<real_t>::mpi_type, p_rank.GetData(), p_rank.Size(), MPITypeMap<real_t>::mpi_type, 0, MPI_COMM_WORLD);


   // Set particles to set
   for (int i = 0; i < charged_particles.GetNP(); i++)
   {
      Particle p = charged_particles.GetParticle(i);

      // Set coords + momentum
      for (int d = 0; d < dim; d++)
      {
         p.Coords()[d] = x_rank[d+i*dim];
         p.Field(MOM)[d] = p_rank[d+i*dim];
      }
      // Set mass + charge
      p.Field(MASS) = m;
      p.Field(CHARGE) = q;

      // Set particle
      charged_particles.SetParticle(i, p);
   }
}
