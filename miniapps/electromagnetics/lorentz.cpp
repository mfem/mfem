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

// 3D particle, 2 scalars of mass and charge, 3 vectors of size 3 for momentum, electric field, and magnetic field
using ChargedParticle = Particle<3,2,3,3,3>;
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

using ChargedParticleSet = ParticleSet<ChargedParticle, Ordering::byVDIM>;

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

public:
   BorisAlgorithm(ParGridFunction *E_gf,
                  ParGridFunction *B_gf,
                  MPI_Comm comm)
      : E_finder_(comm), B_finder_(comm),
        E_field_(E_gf), B_field_(B_gf),
        pxB_(3), pm_(3), pp_(3)
   {
      E_pmesh_ = (E_field_) ? E_field_->ParFESpace()->GetParMesh() : NULL;
      if (E_pmesh_)
      {
         E_pmesh_->EnsureNodes();
         E_finder_.Setup(*E_pmesh_);
      }
      B_pmesh_ = (B_field_) ? B_field_->ParFESpace()->GetParMesh() : NULL;
      if (B_pmesh_)
      {
         B_pmesh_->EnsureNodes();
         B_finder_.Setup(*B_pmesh_);
      }

   }

   void ParticleStep(ChargedParticle &part, real_t &dt)
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

   void Step(ChargedParticleSet &particles, real_t &t, real_t &dt)
   {
      // Update all particle E-fields + B-fields:
      const Vector &X = particles.GetSetCoords();
      Ordering::Type o = particles.GetOrdering();
      Vector &E = particles.GetSetVector(EFIELD);
      Vector &B = particles.GetSetVector(BFIELD);

      if (E_field_)
      {
         GetValues(*E_field_, X, E_finder_, E, o);
      }
      else
      {
         E = 0.0;
      }

      if (B_field_)
      {
         GetValues(*B_field_, X, B_finder_, B, o);
      }
      else
      {
         B = 0.0;
      }

      // Individually step each particle + update its position:
      for (int i = 0; i < particles.GetNP(); i++)
      {
         ChargedParticle part = particles.GetParticleRef(i);
         ParticleStep(part, dt);
      }

      // Update time
      t += dt;
   }
};

// Open the named VisItDataCollection and read the named field.
// Returns pointers to the two new objects.
int ReadGridFunction(const char * coll_name, const char * field_name,
                     int pad_digits_cycle, int pad_digits_rank, int cycle,
                     VisItDataCollection *&dc, ParGridFunction *& gf);

// Initialize particles from user input.
void InitializeParticles(ChargedParticleSet &particles, const Vector &x_init, const Vector &p_init, const real_t q, const real_t m, int num_part, Mesh *mesh);

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
   args.AddOption(&num_part, "-n", "--num-part",
                  "Number of particles.");
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
   ChargedParticleSet particles;

   // Initialize particles
   Mesh *mesh = E_gf ? E_gf->ParFESpace()->GetMesh() : nullptr;
   if (!mesh && B_gf) { mesh = B_gf->ParFESpace()->GetMesh(); }
   InitializeParticles(particles, x_init, p_init, q, m, num_part, mesh);

   // Print particle information   
   // for (int i = 0; i < particles.GetNP(); i++)
   // {
   //    particles.GetParticleData(i).Print();
   // }
   
   BorisAlgorithm boris(E_gf, B_gf, MPI_COMM_WORLD);

   ofstream ofs("Lorentz.dat");
   ofs.precision(14);

   int nsteps = 1 + (int)ceil((t_final - t_init) / dt);
   DenseMatrix pos_data(3*particles.GetNP(), nsteps+1); // +1 for IC
   DenseMatrix mom_data(3*particles.GetNP(), nsteps+1); // +1 for IC
   pos_data.SetCol(0, particles.GetSetCoords());
   mom_data.SetCol(0, particles.GetSetVector(MOM));

   if (Mpi::Root())
   {
      mfem::out << "Number of steps: " << nsteps << endl;
   }

   real_t t = t_init;
   for (int step = 1; step <= nsteps; step++)
   {
      boris.Step(particles, t, dt);
      pos_data.SetCol(step, particles.GetSetCoords());
      mom_data.SetCol(step, particles.GetSetVector(MOM));
   }

   if (visit || visualization)
   {
      socketstream traj_sock;
      traj_sock.precision(8);
      char vishost[] = "localhost";
      traj_sock.open(vishost, visport);

      L2_FECollection fec_l2(0,1);
      Mesh trajectories(1, (nsteps+1)*num_part, (nsteps)*num_part, 0, 3);

      for (int i = 0; i < num_part; i++)
      {
         for (int j = 0; j < nsteps+1; j++) // +1 for IC
         {
            trajectories.AddVertex(pos_data(i*3,j), pos_data(i*3+1,j), pos_data(i*3+2,j));
            if (j > 0) { trajectories.AddSegment((j-1)+i*(nsteps+1),j+i*(nsteps+1)); }
         }
      }

      trajectories.FinalizeMesh();
      
      mfem::out << "Number of particle pts: " << num_part*(nsteps+1) << std::endl;
      mfem::out << "NV: " << trajectories.GetNV() << std::endl;
      mfem::out << "Conforming: " << trajectories.Conforming() << std::endl;

      FiniteElementSpace fes_l2(&trajectories, &fec_l2);
      GridFunction traj_times(&fes_l2);

      for (int i = 0; i < num_part; i++)
      {
         for (int j = 0; j < nsteps; j++) // TODO: Why is this not <= nsteps??
         {
            traj_times[j+i*nsteps] = dt * j;
         }
      }
      

      if (visualization)
      {
         VisualizeField(traj_sock, vishost, visport, traj_times, "Trajectories");
         //VisualizeMesh(traj_sock, vishost, visport, trajectories, "Mesh");
      }
      if (visit)
      {
         VisItDataCollection visit_dc(MPI_COMM_WORLD, "Lorentz", &trajectories);
         visit_dc.RegisterField("Traj_Times", &traj_times);
         visit_dc.Save();
      }
   }
   
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

void InitializeParticles(ChargedParticleSet &particles, const Vector &x_init, const Vector &p_init, const real_t q, const real_t m, int num_part, Mesh *mesh)
{
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
            p.GetCoords()[d] = x_init[d + i*3];
         }
      }
      else // else set it randomly within mesh or unit cube (if no mesh provided)
      {
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
            p.GetCoords()[d] = pos_min[d] + r_pos[d+1]*(pos_max[d] - pos_min[d]);
         }
      }

      // If a momentum was provided:
      if (i < provided_p)
      {
         for (int d = 0; d < 3; d++)
         {
            p.GetVector(MOM)[d] = p_init[d + i*3];
         }
      }
      else // else use last one that was provided, or 0 if none.
      {
         for (int d = 0; d < 3; d++)
         {
            p.GetVector(MOM)[d] = p_init.Size() > 0 ? p_init[provided_p*3-3+d] : 0;
         }
      }

      // Set mass + charge
      p.GetScalar(MASS) = m;
      p.GetScalar(CHARGE) = q;

      // Add particle
      particles.AddParticle(p);
   }
}
