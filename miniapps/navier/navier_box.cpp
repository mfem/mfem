#include "navier_solver.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
    int order = 4;

    double t_final = 0.05;
    double dt = 1e-3;

    double kin_vis = 0.01;
    double reynolds = 1.0 / kin_vis;
    double lam = 0.5 * reynolds
                - sqrt(0.25 * reynolds * reynolds + 4.0 * M_PI * M_PI);

    double uref = 11.4; 
    double ustar = uref * 0.41*0.5;
    double z0 = 0.02;

    bool visualization = true;
} ctx;

void vel(const Vector &x, double t, Vector &u)
{
    double xi = x(0);
    double yi = x(1);
    double zi = x(2);

    double U = 2.25;

    if (xi <= 1e-8)
    {
        // u(0) = 1.0 - cos(2.0 * M_PI * yi);
        // u(1) = ctx.lam / (2.0 * M_PI) * exp(ctx.lam * xi) * sin(2.0 * M_PI * yi);
        // 16.0 * U * yi * zi *
        // u(0) = 0.01*sin(2.0 * M_PI * yi); // * (0.41 - yi) * (0.41 - zi) / pow(0.41, 4.0);
        // u(0) = 0
        u(0) = ctx.ustar / 0.41 * log(zi / ctx.z0 + 1.); // logarithmic profile 
    }
    else
    {
        u(0) = 0.0;
        u(1) = 0.0;
    }
    u(1) = 0.0;
    u(2) = 0.0;
}


int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();

    OptionsParser args(argc, argv);

    args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
    // args.AddOption(&ctx.visualization,
                //   "-vis",
                //   "--visualization",
                //   "-no-vis",
                //   "--no-visualization",
                //   "Enable or disable GLVis visualization.");


    if (!args.Good())
    {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
    }

   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }

    int serial_refinements = 0;

    Mesh *mesh = new Mesh("transfinite_cubical.msh");

    for (int i = 0; i < serial_refinements; ++i)
    {
        mesh->UniformRefinement();
    }

    if (Mpi::Root())
    {
        std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
    }

    auto *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

    delete mesh;        

    // Create the flow solver.
    NavierSolver flowsolver(pmesh, ctx.order, ctx.kin_vis);
    flowsolver.EnablePA(true);

    // Set the initial condition.
    ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();

    VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel);
    
    u_ic->ProjectCoefficient(u_excoeff);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr.Print(std::cout, 6);

    // 2 1 "Wall_back"
    // 2 2 "Wall_front"
    // 2 3 "Wall_bottom"
    // 2 4 "Outflow"
    // 2 5 "Wall_top"
    // 2 6 "Inflow"
    // Outflow is attribute 4, at index 3. 

    // Inlet is attribute 6.
    attr[5] = 1;
    // Walls are attributes 1,2,3,5.
    attr[0] = 1; // this is the bottom of the box 
    
    // attr[1] = 1; 
    // attr[2] = 1;  
    // attr[4] = 1; 

    flowsolver.AddVelDirichletBC(vel, attr);

    double t = 0.0;
    double dt = ctx.dt;
    double t_final = ctx.t_final;
    bool last_step = false;

    flowsolver.Setup(dt);

    ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
    ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

    ParaViewDataCollection pvdc("wind_tunnel", pmesh);
    pvdc.SetDataFormat(VTKFormat::BINARY32);
    pvdc.SetHighOrderOutput(true);
    pvdc.SetLevelsOfDetail(ctx.order);
    pvdc.SetCycle(0);
    pvdc.SetTime(t);
    pvdc.RegisterField("velocity", u_gf);
    pvdc.RegisterField("pressure", p_gf);
    pvdc.Save();

    for (int step = 0; !last_step; ++step)
    {
        if (t + dt >= t_final - dt / 2)
        {
            last_step = true;
        }

        flowsolver.Step(t, dt, step);

        if (step % 2 == 0) // was every 10
        {
            pvdc.SetCycle(step);
            pvdc.SetTime(t);
            pvdc.Save();
        }

        if (Mpi::Root())
        {
            printf("%11s %11s\n", "Time", "dt");
            printf("%.5E %.5E\n", t, dt);
            fflush(stdout);
        }
    }


    if (ctx.visualization) { 
        char vishost[] = "localhost";
        int visport = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "parallel " << Mpi::WorldSize() << " "
                << Mpi::WorldRank() << "\n";
        sol_sock << "solution\n" << *pmesh << *u_ic << std::flush;
    }

    flowsolver.PrintTimingData();

    delete pmesh;

    return 0;
}