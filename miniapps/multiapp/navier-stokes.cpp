#include "mfem.hpp"
#include "multiapp.hpp"
#include "apps/navier_stokes.hpp"

#include <string>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace mfem;

// mpirun -np 6 ./navier-stokes -vs 5 -dt 1e-2 -tf 5 -o 2 -rs 2 -ode 21

int main(int argc, char *argv[])
{
    Mpi::Init();
    Hypre::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();

    int order = 2;
    real_t t_final = .005;
    real_t dt = 1.0e-5;
    bool visualization = true;
    int visport = 19916;
    int vis_steps = 10;
    int ode_solver_type = 21;  // (34) SDIRK34Solver, (21) BackwardEulerSolver, (23) SDIRK33Solver, (3) RK3SSPSolver
    int ser_ref = 0;


    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                    "Finite element order (polynomial degree).");
    args.AddOption(&t_final, "-tf", "--t-final",
                    "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
                    "Time step.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                    "--no-visualization",
                    "Enable or disable GLVis visualization.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                    "Visualize every n-th timestep.");
    args.AddOption(&ode_solver_type, "-ode", "--ode-solver-type",
                    "ODESolver id.");
    args.AddOption(&ser_ref, "-rs", "--serial-refine",
                    "Number of times to refine the mesh in serial.");                  
    args.ParseCheck();

    Mesh *serial_mesh = new Mesh("channel-cylinder.msh");
    for (int i = 0; i < ser_ref; ++i) { serial_mesh->UniformRefinement(); }   
    serial_mesh->EnsureNCMesh();   
    ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
    delete serial_mesh;
    int dim = parent_mesh.Dimension();

    H1_FECollection pfec(order-1, dim);
    H1_FECollection ufec(order, dim);

    ParFiniteElementSpace p_fes(&parent_mesh, &pfec);
    ParFiniteElementSpace u_fes(&parent_mesh, &ufec, dim, Ordering::byNODES);

    Array<int> u_ess_attr, p_ess_attr;
        
    if (parent_mesh.bdr_attributes.Size() > 0)
    {
        u_ess_attr.SetSize(parent_mesh.bdr_attributes.Max());
        p_ess_attr.SetSize(parent_mesh.bdr_attributes.Max());
        
        u_ess_attr = 1;
        u_ess_attr[1] = 0; // outlet
        u_ess_attr[4] = 0; // beam wall
        p_ess_attr = 0;
        p_ess_attr[1] = 1;      
    }


    // Inlet velocity boundary condition
    real_t U0 = 1.5;
    auto velocity_profile = [&U0](const Vector &x, double t, Vector &u) mutable
    {
        double xi = x(0), xc=0.2, r=0.051;
        double yi = x(1), yc=0.2;
        double d  = pow(xi-xc,2)+pow(yi-yc,2);
        double U = U0;
        u = 0.0;
        // if (xi == 0.0) 
        if (d > (r*r))
        {
            u(0) = 4.0 * U * yi * (0.41 - yi) / (pow(0.41, 2.0));
            u(1) = 0.0;
        }
    };

    VectorFunctionCoefficient u_coeff(dim, velocity_profile);
    ConstantCoefficient p_coeff(0.0);

    real_t nu = 1.0e-3; // Kinematic viscosity
    real_t rho = 1.0; // Density
    real_t compressibility = 1e-4; // Artificial Compressibility for Navier-Stokes

    NavierStokes nse(u_fes, p_fes, u_ess_attr, p_ess_attr, rho, nu, compressibility);

    ParGridFunction &p_gf = nse.p_gf;   p_gf = 0.0;
    ParGridFunction &u_gf = nse.u_gf;   u_gf = 0.0;
    ParGridFunction &tau_gf = nse.stress_gf; tau_gf = 0.0;

    u_gf.ProjectCoefficient(u_coeff);
    p_gf.ProjectCoefficient(p_coeff);
    tau_gf.ProjectBdrCoefficient(nse.stress_coeff,u_ess_attr);
   
    Array<int> offsets({0,u_fes.GetTrueVSize(), p_fes.GetTrueVSize()});
    offsets.PartialSum();   
    BlockVector up(offsets);

    u_gf.GetTrueDofs(up.GetBlock(0));
    p_gf.GetTrueDofs(up.GetBlock(1));


    std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);   
    ode_solver->Init(nse);


    ParaViewDataCollection fluid_pv("navier-"+std::to_string(ode_solver_type), &parent_mesh);

    fluid_pv.SetLevelsOfDetail(order);
    fluid_pv.SetDataFormat(VTKFormat::BINARY);
    fluid_pv.SetHighOrderOutput(true);
    fluid_pv.RegisterField("pressure",&p_gf);
    fluid_pv.RegisterField("velocity",&u_gf);
    fluid_pv.RegisterField("stress",&tau_gf);
    

    auto save_callback = [&](int cycle, double t)
    {
        fluid_pv.SetCycle(cycle);
        fluid_pv.SetTime(t);
        fluid_pv.Save();
    };


    StopWatch timer;
    timer.Start();

    real_t t = 0.0;
    bool last_step = false;
    real_t t_trans = 0.5;
    save_callback(0, t);

    //    U0 = 1.5; // Adjusted for simplicity
    //    u_gf.SetFromTrueDofs(up.GetBlock(0));
    //    u_gf.ProjectBdrCoefficient(u_coeff, u_ess_attr);
    //    u_gf.GetTrueDofs(up.GetBlock(0));

    for (int ti = 1; !last_step; ti++)
    {
        if (t + dt >= t_final - dt/2){ last_step = true; }
        ode_solver->Step(up,t,dt);   
        u_gf.SetFromTrueDofs(up.GetBlock(0));      
        p_gf.SetFromTrueDofs(up.GetBlock(1));

        if (last_step || (ti % vis_steps) == 0){
            if (myid == 0) { out << "step " << ti << ", t = " << t << std::endl;}
            
            tau_gf.ProjectBdrCoefficient(nse.stress_coeff,u_ess_attr);
            save_callback(ti, t);
        }
    }

    timer.Stop();
    if (myid == 0){
        out << "Total time: " << timer.RealTime() << " seconds." << std::endl;
    }

    return 0;
}
