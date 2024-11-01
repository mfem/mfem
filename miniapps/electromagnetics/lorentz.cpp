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
//          -------------------------------------------------------
//          LorentzPM Miniapp:  Simple Lorentz Force Particle Mover
//          -------------------------------------------------------
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
// Compile with: make lorentz_pm
//
// Sample runs:
//
// ./volta -m bcc_16.mesh -dbcs 1 -cs '0 0 0 0.1 2e-11' -rs 2
// ./tesla -m hex_prism_3.mesh -rs 2 -bm '0 0 -0.1 0 0 0.1 0.1 1e8'
// ./lorentz_pm -x0 '-0.5 0.1 0.0' -p0 '0 0 0' -q -10 -tf 8 -dt 1e-3 -rf 1e-6
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

// Background Electric field
static Vector e_dir({1.0, 0.0, 0.0});
static real_t e_mag = 1.0;
void background_e(const Vector &x, Vector &E)
{
   E = e_dir;
   E *= e_mag;
}

// Background Electric field
static Vector b_dir({0.0, 0.0, 1.0});
static real_t b_mag = 10.0;
void background_b(const Vector &x, Vector &B)
{
   B = b_dir;
   B *= b_mag;
}

class BorisAlgorithm
{
private:
   real_t charge_;
   real_t mass_;

   //ParMesh &pmesh_;
   //VectorCoefficient &ECoef_;
   //VectorCoefficient &BCoef_;
   /*
   VisItDataCollection &E_dc_;
   const char * E_field_;

   VisItDataCollection &B_dc_;
   const char * B_field_;
   */
   ParMesh *E_pmesh_;
   ParGridFunction *E_field_;

   ParMesh *B_pmesh_;
   ParGridFunction *B_field_;

   mutable Array<int> E_elem_id_;
   mutable Array<IntegrationPoint> E_ip_;

   mutable Array<int> B_elem_id_;
   mutable Array<IntegrationPoint> B_ip_;

   mutable Vector E_;
   mutable Vector B_;
   mutable Vector pxB_;
   mutable Vector pm_;
   mutable Vector pp_;

public:
   BorisAlgorithm(ParGridFunction *E_gf,
                  ParGridFunction *B_gf,
                  real_t charge, real_t mass)
      : charge_(charge), mass_(mass),
        E_field_(E_gf),
        B_field_(B_gf),
        E_(3), B_(3), pxB_(3), pm_(3), pp_(3)
   {
      E_pmesh_ = E_field_->ParFESpace()->GetParMesh();
      B_pmesh_ = B_field_->ParFESpace()->GetParMesh();
   }

   bool Step(Vector &q, Vector &p, real_t &t, real_t &dt)
   {
      DenseMatrix point(q.GetData(), 3, 1);

      int E_pt_found = (E_pmesh_ != NULL) ?
                       E_pmesh_->FindPoints(point, E_elem_id_, E_ip_, false) : -1;

      int B_pt_found = (B_pmesh_ != NULL) ?
                       B_pmesh_->FindPoints(point, B_elem_id_, B_ip_, false) : -1;

      if (E_pt_found <= 0 || B_pt_found <= 0) { return false; }

      int E_pt_root = -1;

      if (E_pt_found > 0 && E_elem_id_[0] >= 0 && E_field_ != NULL)
      {
         E_pt_root = E_pmesh_->GetMyRank();

         E_field_->GetVectorValue(E_elem_id_[0], E_ip_[0], E_);
      }
      else
      {
         E_pt_root = 0;
         E_ = 0.0;
      }

      // Determine processor which found the E field point
      int glb_E_pt_root = -1;
      MPI_Allreduce(&E_pt_root, &glb_E_pt_root, 1,
                    MPI_INT, MPI_MAX, MPI_COMM_WORLD);

      if (E_elem_id_[0] >= 0)
      {
         MPI_Send(E_.GetData(), 3, MPITypeMap<real_t>::mpi_type,
                  0, 1030, MPI_COMM_WORLD);
      }

      int B_pt_root = -1;

      if (B_pt_found > 0 && B_elem_id_[0] >= 0 && B_field_ != NULL)
      {
         B_pt_root = B_pmesh_->GetMyRank();

         B_field_->GetVectorValue(B_elem_id_[0], B_ip_[0], B_);
      }
      else
      {
         B_pt_root = 0;
         B_ = 0.0;
      }

      // Determine processor which found the B field point
      int glb_B_pt_root = -1;
      MPI_Allreduce(&B_pt_root, &glb_B_pt_root, 1,
                    MPI_INT, MPI_MAX, MPI_COMM_WORLD);

      if (B_elem_id_[0] >= 0)
      {
         MPI_Send(B_.GetData(), 3, MPITypeMap<real_t>::mpi_type,
                  0, 1031, MPI_COMM_WORLD);
      }

      if (Mpi::Root())
      {
         // Collect E and B from the processors which found them
         MPI_Status E_status, B_status;

         MPI_Recv(E_.GetData(), 3, MPITypeMap<real_t>::mpi_type,
                  glb_E_pt_root, 1030, MPI_COMM_WORLD, &E_status);

         MPI_Recv(B_.GetData(), 3, MPITypeMap<real_t>::mpi_type,
                  glb_B_pt_root, 1031, MPI_COMM_WORLD, &B_status);

         // Compute half of the contribution from q E
         add(p, 0.5 * dt * charge_, E_, pm_);

         // Compute the contributiobn from q p x B
         const real_t B2 = B_ * B_;

         // ... along pm x B
         const real_t a1 = 4.0 * dt * charge_ * mass_;
         pm_.cross3D(B_, pxB_);
         pp_.Set(a1, pxB_);

         // ... along pm
         const real_t a2 = 4.0 * mass_ * mass_ - dt * dt * charge_ * charge_ * B2;
         pp_.Add(a2, pm_);

         // ... along B
         const real_t a3 = 2.0 * dt * dt * charge_ * charge_ * (B_ * p);
         pp_.Add(a3, B_);

         // scale by common denominator
         const real_t a4 = 4.0 * mass_ * mass_ + dt * dt * charge_ * charge_ * B2;
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

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   if ( Mpi::Root() ) { display_banner(cout); }

   const char *E_coll_name = "Volta-AMR-Parallel";
   const char *E_field_name = "E";
   int E_cycle = 10;
   int E_pad_digits_cycle = 6;
   int E_pad_digits_rank = 6;

   const char *B_coll_name = "Tesla-AMR-Parallel";
   const char *B_field_name = "B";
   int B_cycle = 10;
   int B_pad_digits_cycle = 6;
   int B_pad_digits_rank = 6;

   real_t q = 1.0;
   real_t m = 1.0;
   real_t dt = 1e-3;
   real_t t_init = 0.0;
   real_t t_final = 1.0;
   real_t r_factor = 1.0;
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
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   VisItDataCollection E_dc(MPI_COMM_WORLD, E_coll_name);

   E_dc.SetPadDigitsCycle(E_pad_digits_cycle);
   E_dc.SetPadDigitsRank(E_pad_digits_rank);
   E_dc.Load(E_cycle);

   if (E_dc.Error() != DataCollection::No_Error)
   {
      mfem::out << "Error loading E field VisIt data collection: "
                << E_coll_name << endl;
      return 1;
   }

   ParGridFunction *E_gf = NULL;
   if (E_dc.HasField(E_field_name))
   {
      E_gf = E_dc.GetParField(E_field_name);
   }

   VisItDataCollection B_dc(MPI_COMM_WORLD, B_coll_name);

   B_dc.SetPadDigitsCycle(B_pad_digits_cycle);
   B_dc.SetPadDigitsRank(B_pad_digits_rank);
   B_dc.Load(B_cycle);

   if (B_dc.Error() != DataCollection::No_Error)
   {
      mfem::out << "Error loading B field VisIt data collection: "
                << B_coll_name << endl;
      return 1;
   }

   ParGridFunction *B_gf = NULL;
   if (B_dc.HasField(B_field_name))
   {
      B_gf = B_dc.GetParField(B_field_name);
   }

   BorisAlgorithm boris(E_gf, B_gf, q, m);
   Vector pos(x_init);
   Vector mom(p_init);

   ofstream ofs("LorentzPM.dat");
   ofs.precision(14);

   int nsteps = (int)ceil(t_final - t_init) / dt;
   DenseMatrix pos_data(3, nsteps);
   DenseMatrix mom_data(3, nsteps + 1);
   mom_data(0, 0) = p_init(0);
   mom_data(1, 0) = p_init(1);
   mom_data(2, 0) = p_init(2);

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
      for (int d=0; d<3; d++)
      {
         pos_data(d, step) = pos[d];
         mom_data(d, step + 1) = mom[d];
      }
   }
   while (boris.Step(pos, mom, t, dt) && t <= t_final);

   if (Mpi::Root() && (visit || visualization))
   {
      Mesh trajectory(2, 2 * (step + 1), step, 0, 3);

      for (int i=0; i<=step; i++)
      {
         trajectory.AddVertex(pos_data(0,i), pos_data(1,i), pos_data(2,i));

         real_t dpx = r_factor * (mom_data(0, i + 1) - mom_data(0, i)) / (m * dt);
         real_t dpy = r_factor * (mom_data(1, i + 1) - mom_data(1, i)) / (m * dt);
         real_t dpz = r_factor * (mom_data(2, i + 1) - mom_data(2, i)) / (m * dt);

         trajectory.AddVertex(pos_data(0,i) + dpx,
                              pos_data(1,i) + dpy,
                              pos_data(2,i) + dpz);
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

      L2_FECollection    fec_l2(0, 2);
      FiniteElementSpace fes_l2(&trajectory, &fec_l2);
      GridFunction traj_time(&fes_l2);
      for (int i=0; i<step; i++)
      {
         traj_time[i] = dt * i;
      }

      if (visit)
      {
         VisItDataCollection visit_dc("LorentzPM", &trajectory);
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
}

// Print the LorentzPM ascii logo to the given ostream
void display_banner(ostream & os)
{
   os << "   ____                                __         __________  _____   "
      << endl
      << "  |    |    ___________   ____   _____/  |________\\______   \\/     \\  "
      << endl
      << "  |    |   /  _ \\_  __ \\_/ __ \\ /    \\   __\\___   /|     ___/  \\ /  \\ "
      << endl
      << "  |    |__(  <_> )  | \\/\\  ___/|   |  \\  |  /    / |    |  /    Y    \\"
      << endl
      << "  |_______ \\____/|__|    \\___  >___|  /__| /_____ \\|____|  \\____|__  /"
      << endl
      << "          \\/                 \\/     \\/           \\/                \\/ "
      << endl << flush;
}
