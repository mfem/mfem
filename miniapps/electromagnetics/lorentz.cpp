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

#include "mfem.hpp"
#include "../common/fem_extras.hpp"
#include "../common/pfem_extras.hpp"
#include "electromagnetics.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::electromagnetics;

typedef DataCollection::FieldMapType fields_t;

// 3D particle, 1 vector field of momentum, 2 scalar fields of charge and mass
using ChargedParticle = Particle<3,1,2>;

/// This class implements the Boris algorithm as described in the
/// article `Why is Boris algorithm so good?` by H. Qin et al in
/// Physics of Plasmas, Volume 20 Issue 8, August 2013,
/// https://doi.org/10.1063/1.4818428.
class BorisAlgorithm
{
private:

   FindPointsGSLIB  E_finder_;
   ParMesh         *E_pmesh_;
   ParGridFunction *E_field_;

   FindPointsGSLIB  B_finder_;
   ParMesh         *B_pmesh_;
   ParGridFunction *B_field_;

   mutable Vector E_;
   mutable Vector B_;
   mutable Vector pxB_;
   mutable Vector pm_;
   mutable Vector pp_;

public:
   BorisAlgorithm(ParGridFunction *E_gf,
                  ParGridFunction *B_gf,
                  MPI_Comm comm)
      : E_finder_(comm), B_finder_(comm),
        E_field_(E_gf), B_field_(B_gf),
        E_(3), B_(3), pxB_(3), pm_(3), pp_(3)
   {
      E_ = 0.0;
      E_pmesh_ = (E_field_) ? E_field_->ParFESpace()->GetParMesh() : NULL;
      if (E_pmesh_)
      {
         E_pmesh_->EnsureNodes();
         E_finder_.Setup(*E_pmesh_);
      }

      B_ = 0.0;
      B_pmesh_ = (B_field_) ? B_field_->ParFESpace()->GetParMesh() : NULL;
      if (B_pmesh_)
      {
         B_pmesh_->EnsureNodes();
         B_finder_.Setup(*B_pmesh_);
      }

   }

   bool Step(ParticleSet<ChargedParticle> &particles, real_t &t, real_t &dt)
   {
      E_.SetSize(particles.GetNP()*3);
      B_.SetSize(particles.GetNP()*3);

      if (E_field_)
      {
         E_finder_.FindPoints(particles.GetParticleCoords(), particles.GetOrdering());
         E_finder_.Interpolate(E_field_, E_); // Interpolate the E field onto the particles
         // Ordering of E_ output is the same as E_field_...

         // Fix the ordering if needed (So that it matches the particles)
         // (Note that Ordering::byNODES is easily fastest)
         if (E_field_->ParFESpace()->GetOrdering() != particles.)
      }
      else
      {
         // Otherwise set to 0
         E_ = 0.0;
      }
      if (B_field_)
      {
         return false;
      }

      // Compute updated position and momentum using the Boris algorithm (only on Rank 0)
      if (Mpi::Root())
      {
         // Compute half of the contribution from q E
         add(p, 0.5 * dt * charge_, E_, pm_);

         // Compute the contributiobn from q p x B
         const real_t B2 = B_ * B_;

         // ... along pm x B
         const real_t a1 = 4.0 * dt * charge_ * mass_;
         pm_.cross3D(B_, pxB_);
         pp_.Set(a1, pxB_);

         // ... along pm
         const real_t a2 = 4.0 * mass_ * mass_ -
                           dt * dt * charge_ * charge_ * B2;
         pp_.Add(a2, pm_);

         // ... along B
         const real_t a3 = 2.0 * dt * dt * charge_ * charge_ * (B_ * p);
         pp_.Add(a3, B_);

         // scale by common denominator
         const real_t a4 = 4.0 * mass_ * mass_ +
                           dt * dt * charge_ * charge_ * B2;
         pp_ /= a4;

         // Update the momentum
         add(pp_, 0.5 * dt * charge_, E_, p);

         // Update the position
         q.Add(dt / mass_, p);
      }

      // Update the time
      t += dt;

      // Broadcast the updated position
      MPI_Bcast(q.GetData(), 3, MPITypeMap<real_t>::mpi_type,
                0, MPI_COMM_WORLD);

      // Broadcast the updated momentum
      MPI_Bcast(p.GetData(), 3, MPITypeMap<real_t>::mpi_type,
                0, MPI_COMM_WORLD);

      return true;
   }
};

// Open the named VisItDataCollection and read the named field.
// Returns pointers to the two new objects.
int ReadGridFunction(const char * coll_name, const char * field_name,
                     int pad_digits_cycle, int pad_digits_rank, int cycle,
                     VisItDataCollection *&dc, ParGridFunction *& gf);

// By default the initial position will be the center of the intersection
// of the bounding boxes of the meshes containing the E and B fields.
void SetInitialPosition(VisItDataCollection *E_dc,
                        VisItDataCollection *B_dc,
                        Vector &x_init);

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   if ( Mpi::Root() ) { display_banner(cout); }

   const char *E_coll_name = "";
   const char *E_field_name = "E";
   int E_cycle = 10;
   int E_pad_digits_cycle = 6;
   int E_pad_digits_rank = 6;

   const char *B_coll_name = "";
   const char *B_field_name = "B";
   int B_cycle = 10;
   int B_pad_digits_cycle = 6;
   int B_pad_digits_rank = 6;

   bool aos = true;
   int dim_part = 3;
   int num_part = 1;
   real_t q = 1.0;
   real_t m = 1.0;
   real_t dt = 1e-2;
   real_t t_init = 0.0;
   real_t t_final = 1.0;
   Vector x_init;
   Vector p_init;
   int visport = 19916;
   bool visualization = true;
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
   args.AddOption(&aos, "-aos", "--array-of-structs", "-soa", "--struct-of-arrays", 
                  "Store particles in an Array-of-Structs (AoS) or Struct-of-Arrays (SoA).");
   args.AddOption(&dim_part, "-d", "--dim-part",
                  "Particle domain dimension.");
   args.AddOption(&num_part, "-n", "--num-part",
                  "Number of particles.");
   args.AddOption(&x_init, "-x0", "--initial-positions",
                  "Initial positions (if # positions < # particles, remaining positions set randomly within either E field mesh, B field mesh, or unit cube).");
   args.AddOption(&p_init, "-p0", "--initial-momenta",
                  "Initial momenta (if # momenta < # particles, last momentum is used for remaining particles)");
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
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   VisItDataCollection *E_dc = NULL;
   ParGridFunction     *E_gf = NULL;

   if (strcmp(E_coll_name, ""))
   {
      if (ReadGridFunction(E_coll_name, E_field_name, E_pad_digits_cycle,
                           E_pad_digits_rank, E_cycle, E_dc, E_gf))
      {
         mfem::out << "Error loading E field" << endl;
         return 1;
      }
   }

   VisItDataCollection *B_dc = NULL;
   ParGridFunction     *B_gf = NULL;

   if (strcmp(B_coll_name, ""))
   {
      if (ReadGridFunction(B_coll_name, B_field_name, B_pad_digits_cycle,
                           B_pad_digits_rank, B_cycle, B_dc, B_gf))
      {
         mfem::out << "Error loading B field" << endl;
         return 1;
      }
   }


   // Create the ParticleSet
   std::unique_ptr<ParticleSet<ChargedParticle>> particles;
   if (aos)
      particles.reset(new AoSParticleSet<ChargedParticle>(MPI_COMM_WORLD));
   else
      particles.reset(new SoAParticleSet<ChargedParticle>(MPI_COMM_WORLD));


   int provided_x = x_init.Size()/3;
   int provided_p = p_init.Size()/3;

   // Add all particles to set
   for (int i = 0; i < num_part; i++)
   {
      ChargedParticle p;

      // If a position was provided:
      if (i < provided_x)
      {
         for (int d = 0; d < 3; d++)
         {
            p.coords[d] = x_init[d + i*3];
         }
      }
      else // else set it randomly within E field mesh, B field mesh, or unit cube (in that order)
      {
         Mesh *mesh = nullptr;
         if (E_gf)
         {
            mesh = E_gf->ParFESpace()->GetParMesh();
         }
         else if (B_gf)
         {
            mesh = B_gf->ParFESpace()->GetParMesh();
         }

         Vector pos_min, pos_max;

         if (mesh)
         {
            mesh->GetBoundingBox(pos_min, pos_max);
         }
         else
         {
            pos_min = Vector({0.0,0.0,0.0});
            pos_max = Vector({1.0,1.0,1.0});
         }
         Vector r_pos(4); // 1st-number of each r_pos is too similar regardless of seed
         r_pos.Randomize(i);

         for (int d = 0; d < 3; d++)
         {
            p.coords[d] = pos_min[d] + r_pos[d+1]*(pos_max[d] - pos_min[d]);
         }
      }

      // If a momentum was provided:
      if (i < provided_p)
      {
         for (int d = 0; d < 3; d++)
         {
            p.vector_fields[0][d] = p_init[d + i*3];
         }
      }
      else // else use last one that was provided, or 0 if none.
      {
         for (int d = 0; d < 3; d++)
         {
            p.vector_fields[0][d] = p_init.Size() > 0 ? p_init[provided_p*3-3+d] : 0;
         }
      }

      // Set charge + mass
      p.scalar_fields[0] = q;
      p.scalar_fields[1] = m;

      // Add particle
      particles->AddParticle(p);
   }

   // Print particle locations + momentA
   for (int i = 0; i < particles->GetNP(); i++)
   {
      ChargedParticle p;
      particles->GetParticle(i, p);
      mfem::out << "Particle " << i << ": x=(" << p.coords[0] << "," << p.coords[1] << "," << p.coords[2] << ") ; "
                                    <<   "p=(" << p.vector_fields[0][0] << "," << p.vector_fields[0][1] << "," << p.vector_fields[0][2] << ")" << endl;
   }



   
   /*
   BorisAlgorithm boris(E_gf, B_gf, MPI_COMM_WORLD);
   Vector pos(x_init);
   Vector mom(p_init);

   ofstream ofs("Lorentz.dat");
   ofs.precision(14);

   int nsteps = 1 + (int)ceil((t_final - t_init) / dt);
   DenseMatrix pos_data(3, nsteps);
   DenseMatrix mom_data(3, nsteps + 1);
   mom_data.SetCol(0, particles.GetVectorField(0));

   if (Mpi::Root())
   {
      mfem::out << "Maximum number of steps: " << nsteps << endl;
   }

   int step = -1;
   real_t t = t_init;
   do
   {
      if (Mpi::Root())
      {
         ofs << t
             << '\t' << pos[0] << '\t' << pos[1] << '\t' << pos[2]
             << '\t' << mom[0] << '\t' << mom[1] << '\t' << mom[2]
             << '\n';
      }
      step++;

      pos_data.SetCol(step, pos);
      mom_data.SetCol(step + 1, mom);
   }
   while (boris.Step(pos, mom, t, dt) && step < nsteps - 1);

   if (Mpi::Root() && (visit || visualization))
   {
      Mesh trajectory(1, step, step-1, 0, 3);

      for (int i=0; i<=step; i++)
      {
         trajectory.AddVertex(pos_data(0,i), pos_data(1,i), pos_data(2,i));
         if (i > 0) { trajectory.AddSegment(i-1,i); }
      }

      trajectory.FinalizeMesh();

      L2_FECollection    fec_l2(0, 1);
      FiniteElementSpace fes_l2(&trajectory, &fec_l2);
      GridFunction traj_time(&fes_l2);
      for (int i=0; i<step; i++)
      {
         traj_time[i] = dt * i;
      }

      if (visit)
      {
         VisItDataCollection visit_dc("Lorentz", &trajectory);
         visit_dc.RegisterField("Time", &traj_time);
         visit_dc.SetCycle(step);
         visit_dc.SetTime(step * dt);
         visit_dc.Save();
      }

      if (visualization)
      {
         socketstream traj_sock;
         traj_sock.precision(8);

         char vishost[] = "localhost";

         int Wx = 0, Wy = 0; // window position
         int Ww = 350, Wh = 350; // window size

         VisualizeField(traj_sock, vishost, visport,
                        traj_time, "Trajectory", Wx, Wy, Ww, Wh);
      }
   }
   if (Mpi::Root())
   {
      mfem::out << "Number of steps taken: " << step << endl;
   }
   */
   // Clean up
   delete E_dc;
   delete B_dc;
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

int ReadGridFunction(const char * coll_name, const char * field_name,
                     int pad_digits_cycle, int pad_digits_rank, int cycle,
                     VisItDataCollection *&dc, ParGridFunction *& gf)
{
   dc = new VisItDataCollection(MPI_COMM_WORLD, coll_name);
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

void SetInitialPosition(VisItDataCollection *E_dc,
                        VisItDataCollection *B_dc,
                        Vector &x_init)
{
   x_init.SetSize(3); x_init = 0.0;

   if (E_dc != NULL || B_dc != NULL)
   {
      Vector E_p_min(3); E_p_min = std::numeric_limits<real_t>::lowest();
      Vector E_p_max(3); E_p_max = std::numeric_limits<real_t>::max();
      if (E_dc != NULL)
      {
         ParMesh * E_pmesh = dynamic_cast<ParMesh*>(E_dc->GetMesh());
         E_pmesh->GetBoundingBox(E_p_min, E_p_max);
      }

      Vector B_p_min(3); B_p_min = std::numeric_limits<real_t>::lowest();
      Vector B_p_max(3); B_p_max = std::numeric_limits<real_t>::max();
      if (B_dc != NULL)
      {
         ParMesh *B_pmesh = dynamic_cast<ParMesh*>(B_dc->GetMesh());
         B_pmesh->GetBoundingBox(B_p_min, B_p_max);
      }

      for (int d = 0; d<3; d++)
      {
         const real_t p_min = std::max(E_p_min[d], B_p_min[d]);
         const real_t p_max = std::min(E_p_max[d], B_p_max[d]);
         x_init[d] = 0.5 * (p_min + p_max);
      }
   }
}
