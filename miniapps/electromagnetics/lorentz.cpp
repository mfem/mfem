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

// mpirun -np 4 ./tesla -m ../../data/inline-hex.mesh -ubbc '0 0 1'
// mpirun -np 4 ./lorentz -br Tesla-AMR-Parallel -x0 '0.1 0.5 0.1' -p0 '0 0.4 0.01 -0.2 -0.2 0.0' -tf 90 -n 1000 -vf 5 -vt 5 --redist-freq 1000

#include "mfem.hpp"
#include "../common/fem_extras.hpp"
#include "../common/pfem_extras.hpp"
#include "../common/particles_extras.hpp"
#include "electromagnetics.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::electromagnetics;

typedef DataCollection::FieldMapType fields_t;

enum Scalars
{
   MASS=0,
   CHARGE=1
};
enum Vectors
{
   MOM=0,
   EFIELD=1,
   BFIELD=2
};

/// This class implements the Boris algorithm as described in the
/// article `Why is Boris algorithm so good?` by H. Qin et al in
/// Physics of Plasmas, Volume 20 Issue 8, August 2013,
/// https://doi.org/10.1063/1.4818428.
class BorisAlgorithm
{
private:

   ParGridFunction *E_field;
   std::unique_ptr<FindPointsGSLIB> E_finder;

   ParGridFunction *B_field;
   std::unique_ptr<FindPointsGSLIB>  B_finder;

   mutable Vector pxB_;
   mutable Vector pm_;
   mutable Vector pp_;
   
   static void GetValues(const ParGridFunction &gf, const Vector &x, FindPointsGSLIB &finder, Vector &v, Ordering::Type ordering)
   {
      finder.FindPoints(x, ordering);
      finder.Interpolate(gf, v); // output ordering matches that of gf
      
      // Make sure ordering of v is correct
      Ordering::Reorder(v, gf.VectorDim(), gf.ParFESpace()->GetOrdering(), ordering);
   }

   static void GetCode2Indices(const FindPointsGSLIB &finder, Array<int> &lost_idxs)
   {
      const Array<unsigned int> &code = finder.GetCode();
      lost_idxs.SetSize(0);
      for (int i = 0; i < code.Size(); i++)
      {
         if (code[i] == 2)
            lost_idxs.Append(i);
      }
   }
   void GetLostIndices(Array<int> &idxs) const
   {
      idxs.SetSize(0);
      if (E_finder)
      {
         Array<int> E_lost_idxs;
         GetCode2Indices(*E_finder, E_lost_idxs);
         idxs.Append(E_lost_idxs);
      }
      if (B_finder)
      {
         Array<int> B_lost_idxs;
         GetCode2Indices(*B_finder, B_lost_idxs);
         idxs.Append(B_lost_idxs);
      }
      idxs.Unique();
   }

public:
   BorisAlgorithm(ParGridFunction *E_gf,
                  ParGridFunction *B_gf,
                  MPI_Comm comm)
      : E_field(E_gf),
        B_field(B_gf),
        E_finder(nullptr),
        B_finder(nullptr),
        pxB_(3), pm_(3), pp_(3)
   {
      if (E_field)
      {
         ParMesh &E_pmesh = *E_field->ParFESpace()->GetParMesh();
         E_finder = std::make_unique<FindPointsGSLIB>(comm);
         E_pmesh.EnsureNodes();
         E_finder->Setup(E_pmesh);
      }
      if (B_field)
      {
         ParMesh &B_pmesh = *B_field->ParFESpace()->GetParMesh();
         B_finder = std::make_unique<FindPointsGSLIB>(comm);
         B_pmesh.EnsureNodes();
         B_finder->Setup(B_pmesh);
      }
   }

   void Step(ParticleSet<Ordering::byVDIM> &particles, real_t &t, real_t &dt, bool redist)
   {
      // Update all particle E-fields + B-fields:
      const Vector &X = particles.GetSetCoords();
      Vector &E = particles.GetSetVector(EFIELD);
      Vector &B = particles.GetSetVector(BFIELD);

      if (E_field)
      {
         GetValues(*E_field, X, *E_finder, E, Ordering::byVDIM);
         redist = false;
      }
      else
      {
         E = 0.0;
      }

      if (B_field)
      {
         GetValues(*B_field, X, *B_finder, B, Ordering::byVDIM);
      }
      else
      {
         B = 0.0;
      }
      
      // Remove lost particles
      Array<int> lost_idxs;
      GetLostIndices(lost_idxs);
      particles.RemoveParticles(lost_idxs);

      if (redist)
      {
         // GetProc but remove elements associated with lost particles, as they were removed
         Array<unsigned int> pre_procs;
         if (B_finder) pre_procs = B_finder->GetProc();
         else if (E_finder) pre_procs = E_finder->GetProc();
         
         Array<unsigned int> post_procs;
         for (int i = 0; i < pre_procs.Size(); i++)
         {
            if (lost_idxs.Find(i) == -1)
            {
               post_procs.Append(pre_procs[i]);
            }
         }

         // int rank, size;
         // MPI_Comm_rank(particles.GetComm(), &rank);
         // MPI_Comm_size(particles.GetComm(), &size);
         // for (int r = 0; r < size; r++)
         // {
         //    if (r == rank)
         //    {
         //       cout << "\n---------------------------------------------------\nRANK " << r << endl;
         //       cout << "Removed Particle Idxs: "; lost_idxs.Print(cout, 100);
         //       cout << "Removed Particle Size: " << lost_idxs.Size() << endl;
         //       cout << "Proc Array before slice: "; pre_procs.Print(cout, 100);
         //       cout << "Proc Array size before slice: " << pre_procs.Size() << endl;
         //       cout << "Proc Array after slice: "; post_procs.Print(cout, 100);
         //       cout << "Proc Array size after slice: " << post_procs.Size() << endl;
         //    }
         //    MPI_Barrier(particles.GetComm());

         // }
         particles.Redistribute(post_procs);
         MPI_Barrier(particles.GetComm());
      }

      // Individually step each particle + update its position:
      for (int i = 0; i < particles.GetNP(); i++)
      {
         Particle part = particles.GetParticleRef(i);
         ParticleStep(part, dt);
      }

      // Update time
      t += dt;
   }

   void ParticleStep(Particle &part, real_t &dt)
   {
      Vector &x = part.GetCoords();
      Vector &p = part.GetVector(MOM);
      Vector &e = part.GetVector(EFIELD);
      Vector &b = part.GetVector(BFIELD);
      real_t m = part.GetScalar(MASS);
      real_t q = part.GetScalar(CHARGE);

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
};

// Open the named VisItDataCollection and read the named field.
// Returns pointers to the two new objects.
int ReadGridFunction(std::string coll_name, std::string field_name,
                     int pad_digits_cycle, int pad_digits_rank, int cycle,
                     std::unique_ptr<VisItDataCollection> &dc, ParGridFunction *&gf);

// Initialize particles from user input.
template<Ordering::Type VOrdering>
void InitializeParticles(ParticleSet<VOrdering> &particles, ParMesh *pmesh, const Vector &x_init, const Vector &p_init, real_t q, real_t m, int num_part);

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   if ( Mpi::Root() ) { display_banner(cout); }

   string E_coll_name = "";
   string E_field_name = "E";
   int E_cycle = 10;
   int E_pad_digits_cycle = 6;
   int E_pad_digits_rank = 6;

   string B_coll_name = "";
   string B_field_name = "B";
   int B_cycle = 10;
   int B_pad_digits_cycle = 6;
   int B_pad_digits_rank = 6;

   int num_part = 1;
   real_t q = 1.0;
   real_t m = 1.0;
   real_t dt = 1e-2;
   real_t t_init = 0.0;
   real_t t_final = 1.0;
   int redist_freq = 10;
   Vector x_init;
   Vector p_init;
   int visport = 19916;
   bool visualization = true;
   int vis_freq = 1;
   int vis_tail_size = 5;
   bool visit = true;

   OptionsParser args(argc, argv);
   args.AddOption(&E_coll_name, "-er", "--e-root-file",
                  "Set the VisIt data collection E field root file prefix.");
   args.AddOption(&E_field_name, "-ef", "--e-field-name",
                  "Set the VisIt data collection E field name");
   args.AddOption(&E_cycle, "-ec", "--e-cycle",
                  "Set the E field cycle index to read.");
   args.AddOption(&E_pad_digits_cycle, "-epdc", "--e-pad-digits-cycle",
                  "Number of digits in E field cycle.");
   args.AddOption(&E_pad_digits_rank, "-epdr", "--e-pad-digits-rank",
                  "Number of digits in E field MPI rank.");
   args.AddOption(&B_coll_name, "-br", "--b-root-file",
                  "Set the VisIt data collection B field root file prefix.");
   args.AddOption(&B_field_name, "-bf", "--b-field-name",
                  "Set the VisIt data collection B field name");
   args.AddOption(&B_cycle, "-bc", "--b-cycle",
                  "Set the B field cycle index to read.");
   args.AddOption(&B_pad_digits_cycle, "-bpdc", "--b-pad-digits-cycle",
                  "Number of digits in B field cycle.");
   args.AddOption(&B_pad_digits_rank, "-bpdr", "--b-pad-digits-rank",
                  "Number of digits in B field MPI rank.");
   args.AddOption(&num_part, "-n", "--num-part",
                  "Total number of particles.");
   args.AddOption(&x_init, "-x0", "--initial-positions",
                  "Initial positions byVDIM (if # positions < # particles, remaining positions set randomly within either E field mesh, B field mesh, or unit cube).");
   args.AddOption(&p_init, "-p0", "--initial-momenta",
                  "Initial momenta byVDIM (if # momenta < # particles, last momentum is used for remaining particles)");
   args.AddOption(&q, "-q", "--charge",
                  "Particles' charge.");
   args.AddOption(&m, "-m", "--mass",
                  "Particles' mass.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time Step.");
   args.AddOption(&t_init, "-ti", "--initial-time",
                  "Initial Time.");
   args.AddOption(&t_final, "-tf", "--final-time",
                  "Final Time.");
   args.AddOption(&redist_freq, "-rf", "--redist-freq", "Redistribution frequency.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_freq, "-vf", "--vis-freq", "GLVis visualization update frequency.");
   args.AddOption(&vis_tail_size, "-vt", "--vis-tail-size", "GLVis visualization trajectory tail size. 0 for infinite size.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
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

   if (E_coll_name != "")
   {
      if (ReadGridFunction(E_coll_name, E_field_name, E_pad_digits_cycle,
                           E_pad_digits_rank, E_cycle, E_dc, E_gf))
      {
         mfem::out << "Error loading E field" << endl;
         return 1;
      }
   }

   if (B_coll_name != "")
   {
      if (ReadGridFunction(B_coll_name, B_field_name, B_pad_digits_cycle,
                           B_pad_digits_rank, B_cycle, B_dc, B_gf))
      {
         mfem::out << "Error loading B field" << endl;
         return 1;
      }
   }

   // Initialize particles
   ParMesh *pmesh = E_gf ? static_cast<ParMesh*>(E_gf->ParFESpace()->GetParMesh()) : nullptr;
   if (!pmesh && B_gf) { pmesh = static_cast<ParMesh*>(B_gf->ParFESpace()->GetParMesh()); }

   // Create particle set: 2 scalars of mass and charge, 3 vectors of size 3 for momentum, electric field, and magnetic field
   ParticleSet<Ordering::byVDIM> particles(MPI_COMM_WORLD, 3, 2, {3,3,3});
   InitializeParticles(particles, pmesh, x_init, p_init, q, m, num_part);
   
   // Print all particles
   // for (int r = 0; r < Mpi::WorldSize(); r++)
   // {  
   //    if (r == Mpi::WorldRank())
   //    {
   //       mfem::out << "\nRank " << r << "\n";
   //       for (int i = 0; i < particles.GetNP(); i++)
   //       {
   //          particles.GetParticleData(i).Print();
   //       }
   //    }
   //    MPI_Barrier(MPI_COMM_WORLD);
   // }
   
   BorisAlgorithm boris(E_gf, B_gf, MPI_COMM_WORLD);

   ofstream ofs("Lorentz.dat");
   ofs.precision(14);

   int nsteps = 1 + (int)ceil((t_final - t_init) / dt);


   if (Mpi::Root())
   {
      mfem::out << "Number of steps: " << nsteps << endl;
   }

   real_t t = t_init;

   // Visualization stuff:
   std::unique_ptr<ParticleTrajectories> traj_vis;
   char vishost[] = "localhost";
   socketstream pre_redist_sock, post_redist_sock;
   if (visualization)
   {
      traj_vis = std::make_unique<ParticleTrajectories>(MPI_COMM_WORLD, vis_tail_size, vishost, visport, "Particle Trajectories", 0, 0, 400, 400, "b");
   }

   bool requires_update = true;
   for (int step = 1; step <= nsteps; step++)
   {
      bool redist = (step % redist_freq == 0);

      if (visualization)
      {  
         // Add start of trajectory segment (immediately after last segment is finished)
         if (requires_update)
         {
            traj_vis->AddSegmentStart(particles);
            requires_update = false;
         }
         // If redistributing this step, plot current particle owning ranks
         if (redist)
         {
            if (visualization)
            {
               Vector rank_vector(particles.GetNP());
               rank_vector = Mpi::WorldRank();
               VisualizeParticles(pre_redist_sock, vishost, visport, particles, rank_vector, 1e-2, "Particle Owning Rank (Pre-Redistribute)", 410, 0, 400, 400, "bca");
               char c;
               if (Mpi::Root())
               {
                  cout << "Enter (c) to redistribute + step | (q) to exit : " << flush;
                  cin >> c;
               }
               MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
               if (c == 'q')
               {
                  pre_redist_sock << "keys q" << flush;
                  pre_redist_sock.close();
                  traj_vis->GetSocketStream() << "keys q" << flush;
                  traj_vis->GetSocketStream().close();
                  return 0;
               }
            }
         }
      }


      boris.Step(particles, t, dt, redist);

      if (Mpi::Root())
      {
         mfem::out << "Step: " << step << " | Time: " << t << endl;
      }


      if(visualization)
      {
         // Plot end of segment + visualize trajectories
         if ((step-1) % vis_freq == 0)
         {
            traj_vis->SetSegmentEnd(particles);
            traj_vis->Visualize();
            requires_update = true;
         }

         // If redistributing, plot post-redistribute particle owning ranks
         if (redist)
         {
            Vector rank_vector(particles.GetNP());
            rank_vector = Mpi::WorldRank();
            socketstream post_redist_sock;
            VisualizeParticles(post_redist_sock, vishost, visport, particles, rank_vector, 1e-2, "Particle Owning Rank (Post-Redistribute)", 820, 0, 400, 400, "bca");
            char c;
            if (Mpi::Root())
            {
               cout << "Enter (c) to continue  | (q) to exit : " << flush;
               cin >> c;
            }
            pre_redist_sock << "keys q" << flush;
            post_redist_sock << "keys q" << flush;
            pre_redist_sock.close();
            post_redist_sock.close();

            MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
            if (c == 'q')
            {
               return 0;
            }

         }
      }

   }

}

// Print the Lorentz ascii logo to the given ostream
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

   if (dc->GetMesh()->Dimension() < 3)
   {
      mfem::out << "Field must be defined on a three dimensional mesh"
                << endl;
      return 1;
   }

   if (dc->HasField(field_name))
   {
      gf = dc->GetParField(field_name);
   }

   return 0;
}

template<Ordering::Type VOrdering>
void InitializeParticles(ParticleSet<VOrdering> &particles, ParMesh *pmesh, const Vector &x_init, const Vector &p_init, real_t q, real_t m, int num_part)
{
   int rank, size;
   MPI_Comm_rank(particles.GetComm(), &rank);
   MPI_Comm_size(particles.GetComm(), &size);

   int dim = particles.GetSpaceDim();

   // Initialize all x and p on rank 0, then split amongst ranks
   std::vector<int> sendcts(size);
   std::vector<int> displs(size);
   Vector x_all(num_part*dim);
   Vector p_all(num_part*dim);
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

      Vector x_rem((num_part - provided_x)*dim);
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

      Vector p_rem((num_part - provided_p)*dim);
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
      sendcts[r] = (num_part/size + (r < num_part % size ? 1 : 0))*dim;
      if (r > 0)
         displs[r] = sendcts[r-1] + displs[r-1];
   }

   Vector x_rank(sendcts[rank]);
   Vector p_rank(sendcts[rank]);

   // Scatter the large data buffer from Rank 0 to all other ranks
   MPI_Scatterv(x_all.GetData(), sendcts.data(), displs.data(), MPITypeMap<real_t>::mpi_type, x_rank.GetData(), x_rank.Size(), MPITypeMap<real_t>::mpi_type, 0, MPI_COMM_WORLD);
   MPI_Scatterv(p_all.GetData(), sendcts.data(), displs.data(), MPITypeMap<real_t>::mpi_type, p_rank.GetData(), p_rank.Size(), MPITypeMap<real_t>::mpi_type, 0, MPI_COMM_WORLD);


   // Add all particles to set
   for (int i = 0; i < sendcts[rank]/dim; i++)
   {
      Particle p(dim, particles.GetNumScalars(), particles.GetVectorVDims());

      // Set coords + momentum
      for (int d = 0; d < dim; d++)
      {
         p.GetCoords()[d] = x_rank[d+i*dim];
         p.GetVector(MOM)[d] = p_rank[d+i*dim];
      }
      // Set mass + charge
      p.GetScalar(MASS) = m;
      p.GetScalar(CHARGE) = q;

      // Add particle
      particles.AddParticle(p);
   }
}
