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

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::electromagnetics;

/// @brief Return weighted harmonic function
class PsiGridFunctionCoefficient : public Coefficient
{
private:
   real_t Phi_0;
   real_t k_x;
   real_t k_y;

public:
   PsiGridFunctionCoefficient(real_t Phi_0, real_t nl_x, real_t nl_y, real_t xmax, real_t ymax)
       // k = 2 pi n / L  where L is the domain length
       : Coefficient(), Phi_0(Phi_0), k_x(2.0 * M_PI * nl_x / xmax), k_y(2.0 * M_PI * nl_y / ymax)
   {
   }
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0), z = x(1);
      return Phi_0 * cos(k_x * r) * cos(k_y * z);
   }
};
struct LorentzContext
{
   // mesh related parameters
   int order = 1;
   real_t nl_x = 1.0;
   real_t nl_y = 1.0;
   real_t Phi_0 = 1.0;
   int nx = 100;
   int ny = 100;
   real_t xmax = 1.0;
   real_t ymax = 1.0;

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
   Vector x_min{-1.0, -1.0, -1.0};
   Vector x_max{1.0, 1.0, 1.0};
   Vector p_min{-1.0, -1.0, -1.0};
   Vector p_max{1.0, 1.0, 1.0};
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
   void ParticleStep(Particle &part, real_t &dt);

public:
   Boris(MPI_Comm comm, GridFunction *E_gf_, GridFunction *B_gf_,
         int num_particles, Ordering::Type pdata_ordering);
   void InterpolateEB();
   void Step(real_t &t, real_t &dt);
   void RemoveLostParticles();
   void Redistribute(int redist_mesh); // 0 = E field, 1 = B field
   ParticleSet &GetParticles() { return *charged_particles; }
};

// Prints the program's logo to the given output stream
void display_banner(ostream &os);

// Open the named VisItDataCollection and read the named field.
// Returns pointers to the two new objects.
int ReadGridFunction(std::string coll_name, std::string field_name,
                     int pad_digits_cycle, int pad_digits_rank, int cycle,
                     std::unique_ptr<VisItDataCollection> &dc, ParGridFunction *&gf);

// Initialize particles from user input.
void InitializeChargedParticles(ParticleSet &particles, const Vector &pos_min,
                                const Vector &pos_max, const Vector &x_init, const Vector &p_init, real_t m,
                                real_t q);

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
   args.AddOption(&ctx.nl_x, "-nlx", "--nl-x", "Number of wavelengths in the x direction.");
   args.AddOption(&ctx.nl_y, "-nly", "--nl-y", "Number of wavelengths in the y direction.");
   args.AddOption(&ctx.Phi_0, "-phi0", "--phi-0", "Initial scalar potential.");

   // Mesh parameters (moved into ctx)
   args.AddOption(&ctx.nx, "-nx", "--num-x", "Number of elements in the x direction.");
   args.AddOption(&ctx.ny, "-ny", "--num-y", "Number of elements in the y direction.");
   args.AddOption(&ctx.xmax, "-mxmax", "--m-x-max", "Maximum x coordinate.");
   args.AddOption(&ctx.ymax, "-mymax", "--m-y-max", "Maximum y coordinate.");

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
   args.AddOption(&ctx.x_min, "-xmin", "--x-min",
                  "Minimum initial particle location.");
   args.AddOption(&ctx.x_max, "-xmax", "--x-max",
                  "Maximum initial particle location.");
   args.AddOption(&ctx.p_min, "-pmin", "--p-min",
                  "Minimum initial particle momentum.");
   args.AddOption(&ctx.p_max, "-pmax", "--p-max",
                  "Maximum initial particle momentum.");
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

   std::unique_ptr<VisItDataCollection> E_dc, B_dc;
   ParGridFunction *E_gf = nullptr, *B_gf = nullptr;

   // build up E_gf, B_gf can remain nullptr
   // 1. make a 2D Cartesian Mesh
   Mesh serial_mesh(Mesh::MakeCartesian2D(ctx.nx, ctx.ny, Element::QUADRILATERAL, false, ctx.xmax, ctx.ymax));
   // 2. parallelize the mesh
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear(); // the serial mesh is no longer needed
   // 3. Define a finite element space on the parallel mesh
   H1_FECollection sca_fec(ctx.order, dim);
   ParFiniteElementSpace sca_fespace(&mesh, &sca_fec);
   ND_FECollection vec_fec(ctx.order, dim);
   ParFiniteElementSpace vec_fespace(&mesh, &vec_fec);

   // 4. Define phi_gf as \phi(x, y) = \Phi_0 \cos(nl_x x)\cos(nl_y y)
   ParGridFunction phi_gf(&sca_fespace);
   E_gf = new ParGridFunction(&vec_fespace);

   PsiGridFunctionCoefficient phi_coeff(ctx.Phi_0, ctx.nl_x, ctx.nl_y, ctx.xmax, ctx.ymax);
   phi_gf.ProjectCoefficient(phi_coeff);
   if (ctx.visualization)
   {
      int num_procs = Mpi::WorldSize();
      int myid = Mpi::WorldRank();
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << phi_gf << flush;
   }

   // 5. Compute E_gf = - \grad \phi_gf
   ParGridFunction neg_phi_gf(&sca_fespace);
   neg_phi_gf = 0.0;
   neg_phi_gf -= phi_gf;
   GradientGridFunctionCoefficient E_coeff(&neg_phi_gf);
   E_gf->ProjectCoefficient(E_coeff);
   if (ctx.visualization)
   {
      int num_procs = Mpi::WorldSize();
      int myid = Mpi::WorldRank();
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << *E_gf << flush;
   }

   Ordering::Type ordering_type = ctx.ordering == 0 ? Ordering::byNODES : Ordering::byVDIM;

   // Initialize Boris
   int num_particles = ctx.npt / size + (rank < (ctx.npt % size) ? 1 : 0);
   Boris boris(MPI_COMM_WORLD, E_gf, B_gf, num_particles, ordering_type);
   InitializeChargedParticles(boris.GetParticles(), ctx.x_min, ctx.x_max,
                              ctx.p_min, ctx.p_max, ctx.m, ctx.q);
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
                                                        800, "ba");
   }

   // set up timer
   auto start_time = std::chrono::high_resolution_clock::now();
   for (int step = 1; step <= ctx.nt; step++)
   {
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
         Array<int> field_idx, tag_idx;
         std::string file_name = csv_prefix + mfem::to_padded_string(step, 6) + ".csv";
         boris.GetParticles().PrintCSV(file_name.c_str(), field_idx, tag_idx);
      }

      // Redistribute
      if (ctx.redist_freq > 0 && (step % ctx.redist_freq == 0 || step == 1) &&
          boris.GetParticles().GetGlobalNP() > 0)
      {
         // Visualize particles pre-redistribute
         if (ctx.visualization)
         {
            Vector rank_vector(boris.GetParticles().GetNP());
            rank_vector = Mpi::WorldRank();
            VisualizeParticles(pre_redist_sock, vishost, ctx.visport, boris.GetParticles(),
                               rank_vector, 1e-2, "Particle Owning Rank (Pre-Redistribute)", 410, 0, 400, 400,
                               "bca");
            if (Mpi::Root())
            {
               cout << "Enter any key to redistribute: " << flush;
               cin.ignore();
               cin.get();
            }
            MPI_Barrier(MPI_COMM_WORLD);
         }

         // Redistribute
         boris.Redistribute(ctx.redist_mesh);

         // Visualize particles post-redistribute
         // if (ctx.visualization)
         // {
         //    Vector rank_vector(boris.GetParticles().GetNP());
         //    rank_vector = Mpi::WorldRank();
         //    VisualizeParticles(post_redist_sock, vishost, ctx.visport, boris.GetParticles(),
         //                       rank_vector, 1e-2, "Particle Owning Rank (Post-Redistribute)", 820, 0, 400, 400,
         //                       "bca");
         //    if (Mpi::Root())
         //    {
         //       cout << "Enter any key to continue: " << flush;
         //       cin.ignore();
         //       cin.get();
         //    }
         //    MPI_Barrier(MPI_COMM_WORLD);
         //    pre_redist_sock << "keys q" << flush;
         //    post_redist_sock << "keys q" << flush;
         //    pre_redist_sock.close();
         //    post_redist_sock.close();
         // }
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

   // Update the position
   x.Add(dt / m, p);

   // periodic boundary: wrap around using ctx mesh extents
   while (x(0) < 0.0)
   {
      x(0) += ctx.xmax;
   }
   while (x(0) > ctx.xmax)
   {
      x(0) -= ctx.xmax;
   }
   while (x(1) < 0.0)
   {
      x(1) += ctx.ymax;
   }
   while (x(1) > ctx.ymax)
   {
      x(1) -= ctx.ymax;
   }
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
                                                     field_vdims, pdata_ordering);
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

void Boris::Step(real_t &t, real_t &dt)
{
   // Individually step each particle:
   if (charged_particles->ParticleRefValid())
   {
      for (int i = 0; i < charged_particles->GetNP(); i++)
      {
         Particle p = charged_particles->GetParticleRef(i);
         ParticleStep(p, dt);
      }
   }
   else
   {
      for (int i = 0; i < charged_particles->GetNP(); i++)
      {
         Particle p = charged_particles->GetParticle(i);
         ParticleStep(p, dt);
         charged_particles->SetParticle(i, p);
      }
   }

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
                                const Vector &x_min, const Vector &x_max, const Vector &p_min,
                                const Vector &p_max, real_t m, real_t q)
{
   int rank;
   MPI_Comm_rank(charged_particles.GetComm(), &rank);
   std::mt19937 gen(rank);
   std::uniform_real_distribution<> real_dist(0.0, 1.0);

   int dim = charged_particles.Coords().GetVDim();

   MultiVector &X = charged_particles.Coords();
   MultiVector &P = charged_particles.Field(Boris::MOM);
   MultiVector &M = charged_particles.Field(Boris::MASS);
   MultiVector &Q = charged_particles.Field(Boris::CHARGE);

   for (int i = 0; i < charged_particles.GetNP(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         // Initialize coords
         X(i, d) = x_min[d] + real_dist(gen) * (x_max[d] - x_min[d]);

         // Initialize momentum
         P(i, d) = p_min[d] + real_dist(gen) * (p_max[d] - p_min[d]);
      }
      // Initialize mass + charge
      M(i) = m;
      Q(i) = q;
   }
}
