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
//      mpirun -np 4 lorentz -er Volta-AMR-Parallel -x0 '0.5 0.5 0.9' -p0 '1 0 0'
//
//   Particle accelerating in a constant magnetic field
//      mpirun -np 4 tesla -m ../../data/inline-hex.mesh -ubbc '0 0 1'
//      mpirun -np 4 lorentz -br Tesla-AMR-Parallel -x0 '0.1 0.5 0.1' -p0 '0 0.4 0.1' -tf 9
//
//   Magnetic mirror effect near a charged sphere and a bar magnet
//      mpirun -np 4 volta -m ../../data/ball-nurbs.mesh -dbcs 1 -cs '0 0 0 0.1 2e-11' -rs 2 -maxit 4
//      mpirun -np 4 tesla -m ../../data/fichera.mesh -maxit 4 -rs 3 -bm '-0.1 -0.1 -0.1 0.1 0.1 0.1 0.1 -1e10'
//      mpirun -np 4 lorentz -er Volta-AMR-Parallel -ec 4 -br Tesla-AMR-Parallel -bc 4 -x0 '0.8 0 0' -p0 '-8 -4 4' -q -10 -tf 0.2 -dt 1e-3 -rf 1e-6
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

/// This class implements the Boris algorithm as described in the
/// article `Why is Boris algorithm so good?` by H. Qin et al in
/// Physics of Plasmas, Volume 20 Issue 8, August 2013,
/// https://doi.org/10.1063/1.4818428.
class BorisAlgorithm
{
private:
   real_t charge_;
   real_t mass_;

   ParMesh         *E_pmesh_;
   ParGridFunction *E_field_;

   ParMesh         *B_pmesh_;
   ParGridFunction *B_field_;

   mutable Array<int>              elem_id_;
   mutable Array<IntegrationPoint> ip_;

   mutable Vector E_;
   mutable Vector B_;
   mutable Vector pxB_;
   mutable Vector pm_;
   mutable Vector pp_;

   // Returns true if a usable V has been found. If @a pgf is NULL, V = 0 is
   // returned as a default value.
   bool GetValue(ParMesh *pmesh, ParGridFunction *pgf, Vector q, Vector &V)
   {
      DenseMatrix point(q.GetData(), 3, 1);

      int pt_found =
         (pmesh != NULL) ? pmesh->FindPoints(point, elem_id_, ip_, false) : -1;

      // We have a mesh but the point was not found. The path must be outside
      // the domain of interest.
      if (pmesh != NULL && pt_found <= 0) { return false; }

      int pt_root = -1;

      if (pt_found > 0 && elem_id_[0] >= 0 && pgf != NULL)
      {
         pt_root = pmesh->GetMyRank();

         pgf->GetVectorValue(elem_id_[0], ip_[0], V);
      }
      else
      {
         pt_root = 0;
         V = 0.0;
      }

      // Determine processor which found the field point
      int glb_pt_root = -1;
      MPI_Allreduce(&pt_root, &glb_pt_root, 1,
                    MPI_INT, MPI_MAX, MPI_COMM_WORLD);

      // Send the field value to the root processor
      if (pmesh != NULL && elem_id_[0] >= 0 && glb_pt_root != 0)
      {
         MPI_Send(V.GetData(), 3, MPITypeMap<real_t>::mpi_type,
                  0, 1030, MPI_COMM_WORLD);
      }

      // Receive the field value on the root processor
      if (Mpi::Root() && pmesh != NULL && glb_pt_root != 0)
      {
         MPI_Status status;
         MPI_Recv(V.GetData(), 3, MPITypeMap<real_t>::mpi_type,
                  glb_pt_root, 1030, MPI_COMM_WORLD, &status);
      }
      return true;
   }

public:
   BorisAlgorithm(ParGridFunction *E_gf,
                  ParGridFunction *B_gf,
                  real_t charge, real_t mass)
      : charge_(charge), mass_(mass),
        E_field_(E_gf),
        B_field_(B_gf),
        E_(3), B_(3), pxB_(3), pm_(3), pp_(3)
   {
      E_pmesh_ = (E_field_) ? E_field_->ParFESpace()->GetParMesh() : NULL;
      B_pmesh_ = (B_field_) ? B_field_->ParFESpace()->GetParMesh() : NULL;
   }

   bool Step(Vector &q, Vector &p, real_t &t, real_t &dt)
   {
      // Locate current point in each mesh, evaluate the fields, and collect
      // field values on the root processor.
      if (!GetValue(E_pmesh_, E_field_, q, E_)) { return false; }
      if (!GetValue(B_pmesh_, B_field_, q, B_)) { return false; }

      // Compute updated position and momentum using the Boris algorithm
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

// Build a quadrilateral mesh approximating the trajectory as a
// ribbon. One edge of the ribbon follows the trajectory of the
// particle. The opposite edge is offset by the acceleration vector
// (scaled by a constant called the r_factor).
Mesh MakeTrajectoryMesh(int step, real_t m, real_t dt, real_t r_factor,
                        const DenseMatrix &pos_data,
                        const DenseMatrix &mom_data);

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

   real_t q = 1.0;
   real_t m = 1.0;
   real_t dt = 1e-2;
   real_t t_init = 0.0;
   real_t t_final = 1.0;
   real_t r_factor = -1.0;
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
   args.AddOption(&q, "-q", "--charge",
                  "Particle charge.");
   args.AddOption(&m, "-m", "--mass",
                  "Particle mass.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time Step.");
   args.AddOption(&t_init, "-ti", "--initial-time",
                  "Initial Time.");
   args.AddOption(&t_final, "-tf", "--final-time",
                  "Final Time.");
   args.AddOption(&x_init, "-x0", "--initial-position",
                  "Initial position.");
   args.AddOption(&p_init, "-p0", "--initial-momentum",
                  "Initial momentum.");
   args.AddOption(&r_factor, "-rf", "--ribbon-factor",
                  "Scale factor for ribbon width (rf * (p1-p0) / (m * dt) "
                  "where p0 and p1 are computed momenta).");
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
   if (r_factor <= 0.0)
   {
      r_factor = dt;
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

   if (x_init.Size() < 3)
   {
      SetInitialPosition(E_dc, B_dc, x_init);
   }
   if (p_init.Size() < 3)
   {
      p_init.SetSize(3); p_init = 0.0;
   }
   if (Mpi::Root())
   {
      mfem::out << "Initial position: "; x_init.Print(mfem::out);
      mfem::out << "Initial momentum: "; p_init.Print(mfem::out);
   }

   BorisAlgorithm boris(E_gf, B_gf, q, m);
   Vector pos(x_init);
   Vector mom(p_init);

   ofstream ofs("Lorentz.dat");
   ofs.precision(14);

   int nsteps = 1 + (int)ceil((t_final - t_init) / dt);
   DenseMatrix pos_data(3, nsteps);
   DenseMatrix mom_data(3, nsteps + 1);
   mom_data.SetCol(0, p_init);

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
      Mesh trajectory = MakeTrajectoryMesh(step, m, dt, r_factor,
                                           pos_data, mom_data);

      L2_FECollection    fec_l2(0, 2);
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
      Vector E_p_min(3); E_p_min = -infinity();
      Vector E_p_max(3); E_p_max = infinity();
      if (E_dc != NULL)
      {
         ParMesh * E_pmesh = dynamic_cast<ParMesh*>(E_dc->GetMesh());
         E_pmesh->GetBoundingBox(E_p_min, E_p_max);
      }

      Vector B_p_min(3); B_p_min = -infinity();
      Vector B_p_max(3); B_p_max = infinity();
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

Mesh MakeTrajectoryMesh(int step, real_t m, real_t dt, real_t r_factor,
                        const DenseMatrix &pos_data,
                        const DenseMatrix &mom_data)
{
   Mesh trajectory(2, 2 * (step + 1), step, 0, 3);

   for (int i=0; i<=step; i++)
   {
      trajectory.AddVertex(pos_data(0,i), pos_data(1,i), pos_data(2,i));

      real_t dpx = (mom_data(0, i + 1) - mom_data(0, i)) / (m * dt);
      real_t dpy = (mom_data(1, i + 1) - mom_data(1, i)) / (m * dt);
      real_t dpz = (mom_data(2, i + 1) - mom_data(2, i)) / (m * dt);

      trajectory.AddVertex(pos_data(0,i) + r_factor * dpx,
                           pos_data(1,i) + r_factor * dpy,
                           pos_data(2,i) + r_factor * dpz);
   }

   int v[4];
   for (int i=0; i<step; i++)
   {
      v[0] = 2 * i;
      v[1] = 2 * (i + 1);
      v[2] = 2 * (i + 1) + 1;
      v[3] = 2 * i + 1;

      trajectory.AddQuad(v);
   }

   trajectory.FinalizeQuadMesh(1);

   return trajectory;
}
