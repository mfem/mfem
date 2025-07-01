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

   FindPointsGSLIB E_finder;
   ParMesh *E_pmesh;
   ParGridFunction *E_field;

   FindPointsGSLIB  B_finder;
   ParMesh *B_pmesh;
   ParGridFunction *B_field;

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
      : E_finder(comm), B_finder(comm),
        E_field(E_gf), B_field(B_gf),
        pxB_(3), pm_(3), pp_(3)
   {
      E_pmesh = (E_field) ? E_field->ParFESpace()->GetParMesh() : nullptr;
      if (E_pmesh)
      {
         E_pmesh->EnsureNodes();
         E_finder.Setup(*E_pmesh);
      }
      B_pmesh = (B_field) ? B_field->ParFESpace()->GetParMesh() : nullptr;
      if (B_pmesh)
      {
         B_pmesh->EnsureNodes();
         B_finder.Setup(*B_pmesh);
      }

   }

   void Step(ParticleSet<Ordering::byVDIM> &particles, real_t &t, real_t &dt)
   {
      // Update all particle E-fields + B-fields:
      const Vector &X = particles.GetSetCoords();
      Vector &E = particles.GetSetVector(EFIELD);
      Vector &B = particles.GetSetVector(BFIELD);

      if (E_field)
      {
         GetValues(*E_field, X, E_finder, E, Ordering::byVDIM);
      }
      else
      {
         E = 0.0;
      }

      if (B_field)
      {
         GetValues(*B_field, X, B_finder, B, Ordering::byVDIM);
      }
      else
      {
         B = 0.0;
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
                     std::unique_ptr<VisItDataCollection> &dc, ParGridFunction *gf);

// Initialize particles from user input.
template<Ordering::Type VOrdering>
void InitializeParticles(ParticleSet<VOrdering> &particles, const ParMesh *pmesh, const Vector &x_init, const Vector &p_init, real_t q, real_t m, int num_part);

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
   args.AddOption(&N_redistribute, "-rf", "--redist-freq", "Redistribution frequency.");
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

   std::unique_ptr<VisItDataCollection> E_dc, B_dc;
   ParGridFunction *E_gf, *B_gf;

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
   ParMesh *pmesh = E_gf ? static_cast<ParMesh*>(E_gf->ParFESpace()->GetMesh()) : nullptr;
   if (!pmesh && B_gf) { pmesh = static_cast<ParMesh*>(B_gf->ParFESpace()->GetMesh()); }

   // Create particle set: 2 scalars of mass and charge, 3 vectors of size 3 for momentum, electric field, and magnetic field
   ParticleSet<Ordering::byVDIM> particles(MPI_COMM_WORLD, 3, 2, Array<int>{3,3,3});
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
                     std::unique_ptr<VisItDataCollection> &dc, ParGridFunction *gf)
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
void InitializeParticles(ParticleSet<VOrdering> &particles, const ParMesh *pmesh, const Vector &x_init, const Vector &p_init, real_t q, real_t m, int num_part)
{
   int rank, size;
   MPI_Comm_rank(particles.GetComm(), &rank);
   MPI_Comm_size(particles.GetComm(), &size);

   int dim = particles.GetSpaceDim();

   Mesh mesh;
   if (pmesh)
   {
      mesh = pmesh->GetSerialMesh(0);
   }
   // Initialize all x and p on rank 0, then split amongst ranks
   std::vector<int> sendcts(size);
   std::vector<int> displs(size);
   Vector x_all(num_part*dim);
   Vector p_all(num_part*dim);
   if (rank == 0)
   {
      // Set x_all
      x_all.SetVector(x_init, 0);
      int provided_x = x_init.Size()/dim;

      Vector x_rem((num_part - provided_x)*dim);
      x_rem.Randomize(17);
      Vector pos_min, pos_max;
      if (pmesh)
      {
         mesh.GetBoundingBox(pos_min, pos_max); // Get global bounding box
      }
      else
      {
         pos_min = Vector(dim);
         pos_max = Vector(dim);
         pos_min = 0.0;
         pos_max = 1.0;
      }
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
