#include "mfem.hpp"
#include "multiapp.hpp"
#include "apps/navier_stokes.hpp"
#include "apps/elasticity.hpp"

#include <string>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace mfem;



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
    real_t F0 = -1.0e3;


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
    args.AddOption(&F0, "-F", "--force",
                    "Applied force.");                  

    args.ParseCheck();

    Mesh *serial_mesh = new Mesh("channel-cylinder.msh");
    int dim = serial_mesh->Dimension();

    for (int i = 0; i < ser_ref; ++i) { serial_mesh->UniformRefinement(); }   
    serial_mesh->SetCurvature(order, false, dim, Ordering::byNODES);    
    serial_mesh->EnsureNCMesh();
    ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
    delete serial_mesh;


    Array<int> domain_attributes(1);
    domain_attributes[0] = 2;
    auto mesh = ParSubMesh::CreateFromDomain(parent_mesh, domain_attributes);
    mesh.SetAttributes();
    mesh.EnsureNodes();


    H1_FECollection xfec(order, dim);
    ParFiniteElementSpace x_fes(&mesh, &xfec, dim, Ordering::byNODES);

    Array<int> ess_attr, nat_attr;
        
    if (mesh.bdr_attributes.Size() > 0)
    {
        ess_attr.SetSize(mesh.bdr_attributes.Max());
        nat_attr.SetSize(mesh.bdr_attributes.Max());
        ess_attr = 0; nat_attr = 0;
        nat_attr[4] = 1;
        ess_attr[6] = 1;
    }


    Vector vzero(dim); vzero = 0.0;
    VectorConstantCoefficient zerovec(vzero);
   
    real_t E = 5.6e6; 
    real_t nu = 4.0e-2;
    real_t rho = 1.0e3;
    real_t mu = E / (2.0 * (1.0 + nu));
    real_t lambda = 2*mu*nu/(1-2*nu);

    Elasticity elasticity(x_fes, ess_attr, nat_attr, mu, rho, lambda);
    
    ParGridFunction &x_gf = elasticity.x_gf;   x_gf = 0.0;
    ParGridFunction &u_gf = elasticity.u_gf;   u_gf = 0.0;
    ParGridFunction &sigma_gf = elasticity.sigma_gf;   sigma_gf = 0.0;

    x_gf.ProjectCoefficient(zerovec);
    u_gf.ProjectCoefficient(zerovec);
    sigma_gf.ProjectCoefficient(zerovec);


    Vector vg(dim); vg = 0.0;
    vg(1) = -9.81; // Gravity in the y-direction
    VectorConstantCoefficient gravity(vg);
    auto force_profile = [&F0](const Vector &x, double t, Vector &u) mutable
    {
        double xi = x(0), xc=0.2;
        double yi = x(1), yc=0.2;
        u = 0.0;
        if (xi >= 0.5 && xi < 0.55 && yi > 0.2) 
        {
            u(0) = 0.0;
            u(1) = F0;
        }
    };

    VectorFunctionCoefficient f_coeff(dim, force_profile);

    sigma_gf.ProjectBdrCoefficient(f_coeff, nat_attr);
    elasticity.Update();

    Array<int> offsets({0,x_fes.GetTrueVSize(), x_fes.GetTrueVSize()});
    offsets.PartialSum();   
    BlockVector up(offsets);
    
    x_gf.GetTrueDofs(up.GetBlock(0));
    u_gf.GetTrueDofs(up.GetBlock(1));    


    std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);   
    ode_solver->Init(elasticity);


    ParaViewDataCollection solid_pv("elasticity-"+std::to_string(ode_solver_type), &mesh);

    solid_pv.SetLevelsOfDetail(order);
    solid_pv.SetDataFormat(VTKFormat::BINARY);
    solid_pv.SetHighOrderOutput(true);
    solid_pv.RegisterField("displacement",&x_gf);
    solid_pv.RegisterField("velocity",&u_gf);
    solid_pv.RegisterField("force",&sigma_gf);
    

    auto save_callback = [&](int cycle, double t)
    {
        solid_pv.SetCycle(cycle);
        solid_pv.SetTime(t);
        solid_pv.Save();
    };

    StopWatch timer;
    timer.Start();

    real_t t = 0.0;
    bool last_step = false;
    save_callback(0, t);

    for (int ti = 1; !last_step; ti++)
    {
        if (t + dt >= t_final - dt/2){ last_step = true; }
        ode_solver->Step(up,t,dt);

        x_gf.SetFromTrueDofs(up.GetBlock(0));
        u_gf.SetFromTrueDofs(up.GetBlock(1));      

        if (last_step || (ti % vis_steps) == 0){
            if (myid == 0) { out << "step " << ti << ", t = " << t << std::endl;}
            GridFunction *nodes = mesh.GetNodes();
            *nodes += x_gf;
            mesh.DeleteGeometricFactors();
            
            save_callback(ti, t);
        }
    }

    timer.Stop();
    if (myid == 0){
        out << "Total time: " << timer.RealTime() << " seconds." << std::endl;
    }

    return 0;
}
