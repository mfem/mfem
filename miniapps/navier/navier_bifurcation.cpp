// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// Sample run:
//   mpirun -np 4 navier_bifurcation -rs 2

#include "mfem.hpp"
#include "navier_solver.hpp"
#include "../common/particles_extras.hpp"
#include "../../general/text.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace navier;
using namespace mfem::common;

struct flow_context
{
   // common
   real_t dt = 1e-3;
   real_t t_final = 100;
   // fluid
   int order = 4;
   real_t kin_vis = 0.001;
   // particle
   int num_particles = 1000;
   int p_ordering = Ordering::byNODES;
   real_t kappa = 10.0;
   real_t gamma = 0.2; // should be 6
   real_t zeta = 0.19;
   int paraview_freq = 100;
   int print_csv_freq = 100;
} ctx;

// Particle State Variables:
enum State : int
{
   U_NP3, // 0
   U_NP2, // 1
   U_NP1, // 2
   U_N,   // 3

   V_NP3, // 4
   V_NP2, // 5
   V_NP1, // 6
   V_N,   // 7

   W_NP3, // 8
   W_NP2, // 9
   W_NP1, // 10
   W_N,   // 11

   X_NP3, // 12
   X_NP2, // 13
   X_NP1, // 14

   SIZE
};

// Dirichlet conditions for velocity
void vel_dbc(const Vector &x, real_t t, Vector &u);

class ParticleIntegrator
{
private:
   const int dim;
   const real_t kappa, gamma, zeta;
   FindPointsGSLIB finder;

   void ParticleStep2D(const real_t &dt, const Array<real_t> &beta, const Array<real_t> &alpha, Particle &p)
   {
      real_t w_n_ext = 0.0;

      // Extrapolate particle vorticity using EXTk (w_n is new vorticity at old particle loc)
      // w_n_ext = alpha1*w_np1 + alpha2*w_np2 + alpha3*w_np3
      for (int j = 0; j < 3; j++)
      {
         w_n_ext += alpha[j]*(p.GetStateVar(W_N-(j+1))[0]);
      }

      // Assemble the 2D matrix B
      DenseMatrix B({{beta[0]+dt*kappa, zeta*dt*w_n_ext},
                     {-zeta*dt*w_n_ext, beta[0]+dt*kappa}});

      // Assemble the RHS
      Vector r(2);
      r = 0.0;
      for (int j = 1; j <= 3; j++)
      {
         // Add particle velocity component
         add(r, -beta[j], p.GetStateVar(V_N-j), r);

         // Create C
         Vector C(p.GetStateVar(U_N-j));
         C *= kappa;
         add(C, -gamma, Vector({0.0, 1.0}), C);
         add(C, zeta*w_n_ext, Vector({ p.GetStateVar(U_N-j)[1],
                                      -p.GetStateVar(U_N-j)[0]}), C);

         // Add C
         add(r, dt*alpha[j-1], C, r);
      }

      // Solve for particle velocity
      DenseMatrixInverse B_inv(B);
      B_inv.Mult(r, p.GetStateVar(V_N));

      // Compute updated particle position
      p.GetCoords() = 0.0;
      for (int j = 1; j <= 3; j++)
      {
         add(p.GetCoords(), -beta[j], p.GetStateVar((X_NP1+1)-j), p.GetCoords());
      }
      add(p.GetCoords(), dt, p.GetStateVar(V_N), p.GetCoords());
      p.GetCoords() *= 1.0/beta[0];

   }

public:
   ParticleIntegrator(MPI_Comm comm, const int dim_, const real_t kappa_, const real_t gamma_, const real_t zeta_, Mesh &m)
   : dim(dim_),
     kappa(kappa_),
     gamma(gamma_),
     zeta(zeta_),
     finder(comm)
   {
      finder.Setup(m);
   }

   void Step(const real_t &dt, const Array<real_t> &beta, const Array<real_t> &alpha, const ParGridFunction &u_gf, const ParGridFunction &w_gf, ParticleSet &particles)
   {
      // Shift fluid velocity, fluid vorticity, particle velocity, and particle position
      for (int i = 0; i < particles.GetNP()*dim; i++)
      {
         for (int j = 3; j > 0; j--)
         {
            particles.GetAllStateVar(U_N-j)[i] = particles.GetAllStateVar(U_N-j+1)[i];
            particles.GetAllStateVar(V_N-j)[i] = particles.GetAllStateVar(V_N-j+1)[i];
            particles.GetAllStateVar(W_N-j)[i] = particles.GetAllStateVar(W_N-j+1)[i];
            if (j > 1)
            {
               particles.GetAllStateVar((X_NP1+1)-j)[i] = particles.GetAllStateVar((X_NP1+1)-(j-1))[i];
            }
            else
            {
               particles.GetAllStateVar(X_NP1)[i] = particles.GetAllCoords()[i];
            }
         }
      }

      if (dim == 2)
      {
         for (int i = 0; i < particles.GetNP(); i++)
         {
            Particle p(particles.GetMeta());
            particles.GetParticle(i, p);
            ParticleStep2D(dt, beta, alpha, p); // Compute particle velocity + new position
            particles.SetParticle(i, p);
         }
      }

      // Re-interpolate fluid velocity + vorticity onto particles' new location
      finder.FindPoints(particles.GetAllCoords());
      Vector &p_wn = particles.GetAllStateVar(W_N);
      finder.Interpolate(w_gf, p_wn);
      Ordering::Reorder(p_wn, dim, w_gf.ParFESpace()->GetOrdering(), particles.GetOrdering());

      Vector &p_un = particles.GetAllStateVar(U_N);
      finder.Interpolate(u_gf, p_un);
      Ordering::Reorder(p_un, dim, u_gf.ParFESpace()->GetOrdering(), particles.GetOrdering());
   }
};

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    bool vec = false);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   int rs_levels                 = 0;
   int visport               = 19916;
   bool visualization        = true;

   OptionsParser args(argc, argv);
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (rank == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh("../../data/channel2.mesh");

   int dim = mesh.Dimension();
   mesh.SetCurvature(1);

   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Setup ParMesh based on the communicator for each mesh
   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kin_vis);
   flowsolver.EnablePA(true);
   // flowsolver.EnableNI(ctx.ni);

   // Initialize two particle sets - one numerical, one analytical
   Array<int> vdims(State::SIZE);
   vdims = 2;
   ParticleMeta pmeta(2, 2, vdims);
   ParticleSet particles(MPI_COMM_WORLD, pmeta, ctx.p_ordering == 0 ? Ordering::byNODES : Ordering::byVDIM);

   // Initialize both particle sets the same way
   Vector pos_min(2), pos_max(2);
   pos_min[0] = 0.0;
   pos_max[0] = 4.0;
   pos_min[1] = 0.05;
   pos_max[1] = 0.95;
   int seed = rank;

   for (int p = 0; p < ctx.num_particles; p++)
   {
      Particle part(pmeta);

      InitializeRandom(part, seed, pos_min, pos_max);
      particles.AddParticle(part);

      seed += size;
   }

   // Setup pointer for FESpaces, GridFunctions, and Solvers
   ParGridFunction *u_gf             = NULL; // Velocity solution
   ParGridFunction *w_gf             = NULL; // Vorticity solution

   real_t t       = 0,
          dt      = ctx.dt,
          t_final = ctx.t_final;
   bool last_step = false;

   {
      u_gf = flowsolver.GetCurrentVelocity();
      w_gf = flowsolver.GetCurrentVorticity();
      VectorFunctionCoefficient u_excoeff(dim, vel_dbc);
      u_gf->ProjectCoefficient(u_excoeff);

      // Dirichlet boundary conditions for fluid
      Array<int> attr(pmesh->bdr_attributes.Max());
      attr = 0;
      // Inlet is attribute 1.
      attr[0] = 1;
      // Walls is attribute 2.
      attr[1] = 1;
      flowsolver.AddVelDirichletBC(vel_dbc, attr);

      flowsolver.Setup(dt);
      u_gf->ProjectCoefficient(u_excoeff);
   }
   // Visualize the solution.
   char vishost[] = "localhost";
   socketstream vis_sol;
   int Ww = 350, Wh = 350; // window size
   int Wx = 10, Wy = 0; // window position
   if (visualization)
   {
      VisualizeField(vis_sol, vishost, visport, *u_gf,
                        "Velocity", Wx, Wy, Ww, Wh);
   }

   // Initialize particle fluid-dependent IC
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(*pmesh);
   finder.FindPoints(particles.GetAllCoords());
   finder.Interpolate(*u_gf, particles.GetAllStateVar(U_N));
   finder.Interpolate(*w_gf, particles.GetAllStateVar(W_N));

   // Initialize particle integrator
   ParticleIntegrator pint(MPI_COMM_WORLD, 2, ctx.kappa, ctx.gamma, ctx.zeta, *pmesh);

   // Initialize ParaView DC (if freq != 0)
   std::unique_ptr<ParaViewDataCollection> pvdc;
   if (ctx.paraview_freq > 0)
   {
      pvdc = std::make_unique<ParaViewDataCollection>("Bifurcation", pmesh);
      pvdc->SetPrefixPath("ParaView");
      pvdc->SetLevelsOfDetail(ctx.order);
      pvdc->SetDataFormat(VTKFormat::BINARY);
      pvdc->SetHighOrderOutput(true);
      pvdc->RegisterField("Velocity",flowsolver.GetCurrentVelocity());
      pvdc->RegisterField("Vorticity",flowsolver.GetCurrentVorticity());
      pvdc->SetTime(t);
      pvdc->SetCycle(0);
      pvdc->Save();
   }

   std::string csv_prefix = "Navier_Bifurcation_";
   if (ctx.print_csv_freq > 0)
   {
      std::string file_name = csv_prefix + mfem::to_padded_string(0, 9) + ".csv";
      particles.PrintCSV(file_name.c_str());
   }
   Array<real_t> beta, alpha; // EXTk/BDF coefficients

   int vis_step = 0;
   int pstep = 0;
   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      real_t cfl;
      flowsolver.Step(t, dt, step);
      cfl = flowsolver.ComputeCFL(*u_gf, dt);

      // Get the time-integration coefficients
      flowsolver.GetTimeIntegrationCoefficients(beta, alpha);

      // Step particles
      if (t > 5)
      {
         if (pstep == 0)
         {
            beta = 0.0;
            beta[0] = 1.0;
            beta[1] = -1.0;
            alpha = 0.0;
            alpha[0] = 1.0;
            alpha[1] = -1.0;
            pstep++;
         }
         else if (pstep == 1)
         {
            beta = 0.0;
            beta[0] = 1.5;
            beta[1] = -2.0;
            beta[2] = 0.5;
            alpha = 0.0;
            alpha[0] = 2.0;
            alpha[1] = -1.0;
            pstep++;
         }
         pint.Step(ctx.dt, beta, alpha, *u_gf, *w_gf, particles);
      }
      if (ctx.print_csv_freq > 0 && step % ctx.print_csv_freq == 0)
      {
         // Output the particles
         std::string file_name = csv_prefix + mfem::to_padded_string(vis_step, 9) + ".csv";
         particles.PrintCSV(file_name.c_str());
      }
      if (ctx.paraview_freq > 0 && step % ctx.paraview_freq == 0)
      {
         pvdc->SetCycle(vis_step++);
         pvdc->SetTime(t);
         pvdc->Save();
      }
      if (visualization)
      {
         VisualizeField(vis_sol, vishost, visport, *u_gf,
                        "Velocity", Wx, Wy, Ww, Wh);
      }
      if (rank == 0)
      {
         printf("%11s %11s %11s\n", "Time", "dt", "CFL");
         printf("%.5E %.5E %.5E\n", t, dt,cfl);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
{
   gf.HostRead();
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, rank;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &rank);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (rank == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (rank == 0 && newly_opened)
      {
         const char* keys = (gf.FESpace()->GetMesh()->Dimension() == 2)
                            ? "mAcRjlmm]]]]]]]]]" : "mmaaAcl";

         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys " << keys;
         if ( vec ) { sock << "vvv"; }
         sock << std::endl;
      }

      if (rank == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

/// Fluid data
// Dirichlet conditions for velocity
void vel_dbc(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);
   real_t height = 1.0;

   u(0) = 0.;
   u(1) = 0.;
   if (std::fabs(xi)<1.e-5) { u(0) = 6.0*yi*(height-yi)/(height*height); }
}