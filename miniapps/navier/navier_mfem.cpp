#include "navier_solver.hpp"
#include "navier_particles.hpp"
#include "../common/pfem_extras.hpp"
#include "../common/particles_extras.hpp"
#include "../../general/text.hpp"
#include <fstream>
#include <iostream>
// make navier_mfem && mpirun -np 10 ./navier_mfem -rs 3 -npt 10 -pt 0 -nt 4e5 -csv 10 -pv 10 -dt 1e-2 -kmin 0 -kmax 0
using namespace std;
using namespace mfem;
using namespace navier;
using namespace mfem::common;

struct flow_context
{
   // common
   real_t dt = 1e-3;
   real_t nt = 10000;

   // fluid
   int rs_levels = 3;
   int order = 3;
   real_t inlet_speed = 1.0;
   real_t Re = 1000;
   int paraview_freq = 10;
   bool visualization = true;
   int visport = 19916;
   int vis_tail_size = 5;
   int vis_freq = 5;

   // particle
   int pnt_0 = round(2.0/dt);
   int add_particles_freq = 100;
   int num_add_particles = 10;
   real_t kappa_min = 1.0;
   real_t kappa_max = 10.0;
   real_t gamma = 0.0;
   real_t zeta = 0.19;
   int print_csv_freq = 10;
} ctx;

void SetInjectedParticles(NavierParticles &particle_solver, const Array<int> &p_idxs, real_t kappa_min, real_t kappa_max, int kappa_seed, real_t zeta, real_t gamma, int step);

void inlet_dbc(const Vector &x, real_t t, Vector &u)
{
   u[0] = 0.0;
   u[1] = ctx.inlet_speed;
}

void no_slip_dbc(const Vector &x, real_t t, Vector &u)
{
   u = 0.0;
}

bool normalOffsetFromCenter(Vector &xy0,
                            Vector &xy1,
                            Vector &newpt)
{
    // Midpoint of the segment (x0,y0) -> (x1,y1)
    const double cx = 0.5 * (xy0[0] + xy1[0]);
    const double cy = 0.5 * (xy0[1] + xy1[1]);

    const double dx = xy1[0] - xy0[0];
    const double dy = xy1[1] - xy0[1];
    const double len = sqrt(dx*dx + dy*dy);

    // Unit left-hand normal to the segment direction (dx,dy)
    const double nx = -dy / len;
    const double ny =  dx / len;

    // offset from the midpoint along the normal.
    newpt[0] = cx + 0.001*nx;
    newpt[1] = cy + 0.001*ny;
    return true;
}

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   // Parse command line arguments
   OptionsParser args(argc, argv);
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.nt, "-nt", "--num-timesteps", "Number of time steps.");
   args.AddOption(&ctx.rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&ctx.order, "-o", "--order", "Order (degree) of the finite elements.");
   args.AddOption(&ctx.inlet_speed, "-is", "--inlet-speed", "Inlet speed.");
   args.AddOption(&ctx.Re, "-Re", "--reynolds-number", "Reynolds number.");
   args.AddOption(&ctx.paraview_freq, "-pv", "--paraview-freq", "ParaView data collection write frequency. 0 to disable.");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&ctx.vis_tail_size, "-vt", "--vis-tail-size", "GLVis visualization trajectory truncation tail size.");
   args.AddOption(&ctx.vis_freq, "-vf", "--vis-freq", "GLVis visualization frequency.");
   args.AddOption(&ctx.pnt_0, "-pt", "--particle-timestep", "Timestep to begin integrating particles.");
   args.AddOption(&ctx.add_particles_freq, "-ipf", "--inject-particles-freq", "Frequency of particle injection at domain inlet.");
   args.AddOption(&ctx.num_add_particles, "-npt", "--num-particles-inject", "Number of particles to add each injection.");
   args.AddOption(&ctx.kappa_min, "-kmin", "--kappa-min", "Kappa constant minimum.");
   args.AddOption(&ctx.kappa_max, "-kmax", "--kappa-max", "Kappa constant maximum.");
   args.AddOption(&ctx.gamma, "-g", "--gamma", "Gamma constant.");
   args.AddOption(&ctx.zeta, "-z", "--zeta", "Zeta constant.");
   args.AddOption(&ctx.print_csv_freq, "-csv", "--csv-freq", "Frequency of particle CSV outputting. 0 to disable.");

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

   // Load mesh + complete any serial refinements
   Mesh mesh("../../data/mfem.mesh");
   int dim = mesh.Dimension();
   mesh.SetCurvature(1);

   // Process boundary conditions before doing anything

   int nbdr = mesh.GetNBE();
   int nbattr = mesh.bdr_attributes.Max();
   Array<Array<int> *> bdr_verts(nbattr);
   for (int i = 0; i < nbattr; i++)
   {
      bdr_verts[i] = new Array<int>;
   }
   for (int i = 0; i < nbdr; i++)
   {
      int attr = mesh.GetBdrElement(i)->GetAttribute();
      int nv = mesh.GetBdrElement(i)->GetNVertices();
      int *elem_vert =  mesh.GetBdrElement(i)->GetVertices();
      for (int j = 0; j < nv; j++)
      {
         if (bdr_verts[attr-1]->Find(elem_vert[j]) == -1)
         {
            bdr_verts[attr-1]->Append(elem_vert[j]);
         }
         else
         {
            bdr_verts[attr-1]->DeleteFirst(elem_vert[j]);
         }
      }
   }
   for (int i = 0; i < nbattr; i++)
   {
      MFEM_VERIFY(bdr_verts[i]->Size()  == 2, "Each boundary edge should have exactly 2 vertices.");
   }
   Vector xstart(nbattr*2);
   Vector xend(nbattr*2);
   Array<bool> invert_normal(nbattr);
   // now check if normal would point inward
   {
      FindPointsGSLIB finder;
      finder.Setup(mesh);
      Vector xyzlist(nbattr*2);
      for (int i = 0; i < nbattr; i++)
      {
         Vector midpt(2);
         midpt = 0.0;
         for (int j = 0; j < bdr_verts[i]->Size(); j++)
         {
            double *coord = mesh.GetVertex((*bdr_verts[i])[j]);
            if (j == 0)
            {
               xstart(2*i) = coord[0];
               xstart(2*i+1) = coord[1];
            }
            else
            {
               xend(2*i) = coord[0];
               xend(2*i+1) = coord[1];
            }
            Vector newpt(xyzlist.GetData() + 2*i, 2);
            Vector xs(xstart.GetData() + 2*i, 2);
            Vector xe(xend.GetData() + 2*i, 2);
            normalOffsetFromCenter(xs, xe, newpt);
         }

         finder.FindPoints(xyzlist, 1); // ordered by vdim
         const Array<int> &code = finder.GetCode();
         for (int j = 0; j < code.Size(); j++)
         {
            invert_normal[j] = code[j];
         }
      }
   }

   for (int lev = 0; lev < ctx.rs_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Parallel decompose mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   pmesh.EnsureNodes();
   mesh.Clear();

   // Create the flow solver
   NavierSolver flow_solver(&pmesh, ctx.order, 1.0/ctx.Re);

   // Create the particle solver
   NavierParticles particle_solver(MPI_COMM_WORLD, 0, pmesh);
   particle_solver.GetParticles().Reserve( ((ctx.nt - ctx.pnt_0)/ctx.add_particles_freq ) * ctx.num_add_particles / size);

   real_t time = 0.0;

   // Initialize fluid IC
   ParGridFunction &u_gf = *flow_solver.GetCurrentVelocity();
   ParGridFunction &w_gf = *flow_solver.GetCurrentVorticity();
   u_gf = 0.0;
   w_gf = 0.0;

   // Set fluid BCs
   Array<int> wall_attr(pmesh.bdr_attributes.Max());
   wall_attr = 1;
   wall_attr[0] = 0;
   wall_attr[1] = 0;
   flow_solver.AddVelDirichletBC(no_slip_dbc, wall_attr);

   Array<int> inlet_attr(pmesh.bdr_attributes.Max());
   inlet_attr = 0;
   inlet_attr[0] = 1;
   flow_solver.AddVelDirichletBC(inlet_dbc, inlet_attr);


   for (int i = 0; i < nbattr; i++)
   {
      Vector xs(xstart.GetData() + 2*i, 2);
      Vector xe(xend.GetData() + 2*i, 2);
      particle_solver.Add2DReflectionBC(xs, xe, 1.0, invert_normal[i]);
   }


   // Set up solution visualization
   char vishost[] = "localhost";
   socketstream vis_sol;
   int Ww = 350, Wh = 350; // window size
   int Wx = 10, Wy = 0; // window position
   char keys[] = "mAcRjlmm]]]]]]]]]";
   std::unique_ptr<ParticleTrajectories> traj_vis;
   // if (ctx.visualization)
   // {
   //    //VisualizeField(vis_sol, vishost, ctx.visport, u_gf, "Velocity", Wx, Wy, Ww, Wh, keys);
   //    traj_vis = std::make_unique<ParticleTrajectories>(particle_solver.GetParticles(), ctx.vis_tail_size, vishost, ctx.visport, "Particle Trajectories", Wx, Wy, Ww, Wh, keys);
   // }

   // Initialize ParaView DC (if freq != 0)
   std::unique_ptr<ParaViewDataCollection> pvdc;
   if (ctx.paraview_freq > 0)
   {
      pvdc = std::make_unique<ParaViewDataCollection>("MFEM", &pmesh);
      pvdc->SetPrefixPath("ParaView");
      pvdc->SetLevelsOfDetail(ctx.order);
      pvdc->SetDataFormat(VTKFormat::BINARY);
      pvdc->SetHighOrderOutput(true);
      pvdc->RegisterField("Velocity",flow_solver.GetCurrentVelocity());
      pvdc->RegisterField("Vorticity",flow_solver.GetCurrentVorticity());
      pvdc->SetTime(time);
      pvdc->SetCycle(0);
      pvdc->Save();
   }

   std::string csv_prefix = "Navier_MFEM_";
   if (ctx.print_csv_freq > 0)
   {
      std::string file_name = csv_prefix + mfem::to_padded_string(0, 6) + ".csv";
      particle_solver.GetParticles().PrintCSV(file_name.c_str());
   }
   int vis_count = -1;

   flow_solver.Setup(ctx.dt);
   particle_solver.Setup(ctx.dt);

   int pstep = 0;
   Array<int> add_particle_idxs;
   bool flow = false;
   if (!flow)
   {
      u_gf = 0.0;
      w_gf = 0.0;
   }
   for (int step = 1; step <= ctx.nt; step++)
   {
      real_t cfl;
      if (flow)
      {
         flow_solver.Step(time, ctx.dt, step-1);
      }
      else {
         step++;
         time += ctx.dt;
      }

      // Step particles after pnt_0
      if (step >= ctx.pnt_0)
      {
         // Inject particles
         if (step % ctx.add_particles_freq == 0)
         {
            int rank_num_particles = ctx.num_add_particles/size + (rank < (ctx.num_add_particles % size) ? 1 : 0);
            particle_solver.GetParticles().AddParticles(rank_num_particles, &add_particle_idxs);
            SetInjectedParticles(particle_solver, add_particle_idxs, ctx.kappa_min, ctx.kappa_max, (rank+1)*step, ctx.zeta, ctx.gamma,step);
         }
         if (!traj_vis)
         {
            traj_vis = std::make_unique<ParticleTrajectories>(particle_solver.GetParticles(), ctx.vis_tail_size, vishost, ctx.visport, "Particle Trajectories", Wx, Wy, Ww, Wh, keys);
         }
         particle_solver.Step(ctx.dt, u_gf, w_gf);
         pstep++;
      }

      if(ctx.visualization && step % ctx.vis_freq == 0 && step >= ctx.pnt_0)
      {
         traj_vis->Visualize();
      }

      if (ctx.print_csv_freq > 0 && step % ctx.print_csv_freq == 0)
      {
         vis_count++;
         // Output the particles
         std::string file_name = csv_prefix + mfem::to_padded_string(vis_count, 6) + ".csv";
         particle_solver.GetParticles().PrintCSV(file_name.c_str());
      }

      if (ctx.paraview_freq > 0 && step % ctx.paraview_freq == 0)
      {
         pvdc->SetTime(vis_count);
         pvdc->SetCycle(step);
         pvdc->Save();
      }

      cfl = flow_solver.ComputeCFL(u_gf, ctx.dt);
      int global_np = particle_solver.GetParticles().GetGlobalNP();
      int inactive_global_np = particle_solver.GetInactiveParticles().GetGlobalNP();
      if (rank == 0)
      {
         printf("\n%-11s %-11s %-11s %-11s\n", "Step", "Time", "dt", "CFL");
         printf("%-11i %-11.5E %-11.5E %-11.5E\n", step, time, ctx.dt, cfl);
         printf("\n%16s: %-9i\n", "Active Particles", global_np);
         printf("%16s: %-9i\n", "Lost Particles", inactive_global_np);
         printf("-----------------------------------------------\n");
         fflush(stdout);
      }
   }

   flow_solver.PrintTimingData();

   return 0;
}

void SetInjectedParticles(NavierParticles &particle_solver, const Array<int> &p_idxs, real_t kappa_min, real_t kappa_max, int kappa_seed, real_t zeta, real_t gamma, int step)
{
   // Inject uniformly-distributed along inlet
   real_t width = 1.0;

   MPI_Comm comm = particle_solver.GetParticles().GetComm();
   int rank;
   MPI_Comm_rank(comm, &rank);

   int rank_num_add = p_idxs.Size();
   int global_num_particles = 0;
   MPI_Allreduce(&rank_num_add, &global_num_particles, 1, MPI_INT, MPI_SUM, comm);

   real_t spacing = width/(global_num_particles + 1);

   // Determine this rank's offset
   int offset = 0;
   MPI_Scan(&rank_num_add, &offset, 1, MPI_INT, MPI_SUM, comm);
   offset -= rank_num_add;

   for (int i = 0; i < p_idxs.Size(); i++)
   {
      int idx = p_idxs[i];

      std::uniform_real_distribution<> real_dist(0.0,1.0);
      std::mt19937 gen(kappa_seed+rank+step+idx);

      for (int j = 0; j < 4; j++)
      {
         if (j == 0)
         {
            // Set position
            double xinit = real_dist(gen);
            particle_solver.X().SetVectorValues(idx, Vector({xinit, 0.0}));
         }
         else
         {
            // Zero-out position history
            particle_solver.X(j).SetVectorValues(idx, Vector({0.0,0.0}));
         }
         // Zero-out particle velocities, fluid velocities, and fluid vorticities
         // particle_solver.V(j).SetVectorValues(idx, Vector({0.0,0.0}));
         real_t vx = -2 + 4*real_dist(gen);
         real_t vy = real_dist(gen)*2.0;
         particle_solver.V(j).SetVectorValues(idx, Vector({vx,vy}));
         particle_solver.U(j).SetVectorValues(idx, Vector({0.0,0.0}));
         particle_solver.W(j).SetVectorValues(idx, Vector({0.0,0.0}));

         // Set Kappa, Zeta, Gamma
         particle_solver.Kappa()[idx] = kappa_min + real_dist(gen)*(kappa_max - kappa_min);
         particle_solver.Zeta()[idx] = zeta;
         particle_solver.Gamma()[idx] = gamma;
      }

      // Set order to 0
      particle_solver.Order()[idx] = 0;
   }
}
