#include "mfem.hpp"
#include "multiapp.hpp"
#include "apps/navier_stokes.hpp"
#include "apps/elasticity.hpp"
#include "apps/mesh_morpher.hpp"


#include <string>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace mfem;

//  mpirun -np 6 ./fsi -vs 5 -dt 1e-2 -tf 5 -o 2 -rs 2 -F 0.0e5 -ode 21 -couple_type 0

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
    int couple_type = 0; // Use flow map for mesh morphing

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
    args.AddOption(&F0, "-F", "--force", "Applied force.");
    args.AddOption(&couple_type, "-couple_type", "--couple_type",
                    "Coupling type.");    
    args.ParseCheck();

    Mesh *serial_mesh = new Mesh("channel-cylinder.msh");
    int dim = serial_mesh->Dimension();

    for (int i = 0; i < ser_ref; ++i) { serial_mesh->UniformRefinement(); }
    serial_mesh->SetCurvature(order, false, dim, Ordering::byNODES);    
    serial_mesh->EnsureNCMesh();   
    ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
    delete serial_mesh;
    
    /// Create meshes for solid and fluid regions
    Array<int> domain_attributes(1);

    // Create submesh for solid
    domain_attributes[0] = 2;
    auto solid_mesh = ParSubMesh::CreateFromDomain(parent_mesh, domain_attributes);
    solid_mesh.SetAttributes();
    solid_mesh.EnsureNodes();

    // Create submesh for fluid    
    domain_attributes[0] = 1;
    auto fluid_mesh = ParSubMesh::CreateFromDomain(parent_mesh, domain_attributes);
    fluid_mesh.SetAttributes();
    fluid_mesh.EnsureNodes();


    // Finite element spaces for solid and fluid domains
    H1_FECollection xfec(order, dim); // Displacement field (solid and fluid domains)
    H1_FECollection ufec(order, dim); // Velocity field (fluid domain)
    H1_FECollection pfec(order-1, dim); // Pressure field (fluid domain)

    ParFiniteElementSpace xs_fes(&solid_mesh, &xfec, dim, Ordering::byNODES);
    ParFiniteElementSpace xf_fes(&fluid_mesh, &xfec, dim, Ordering::byNODES);    
    ParFiniteElementSpace u_fes(&fluid_mesh, &ufec, dim, Ordering::byNODES);    
    ParFiniteElementSpace p_fes(&fluid_mesh, &pfec);
    
    // Set essential and natural boundary conditions attributes
    Array<int> u_ess_attr, p_ess_attr, noslip_attr;
    Array<int> xs_ess_attr, xs_nat_attr;
    Array<int> xf_ess_attr;
    Array<int> empty;

    if (solid_mesh.bdr_attributes.Size() > 0)
    {
        xs_ess_attr.SetSize(solid_mesh.bdr_attributes.Max());
        xs_nat_attr.SetSize(solid_mesh.bdr_attributes.Max());
        xs_ess_attr = 0; xs_nat_attr = 0;
        xs_nat_attr[4] = 1;
        xs_ess_attr[6] = 1;
    }
    
    if (fluid_mesh.bdr_attributes.Size() > 0)
    {
        u_ess_attr.SetSize(fluid_mesh.bdr_attributes.Max());
        p_ess_attr.SetSize(fluid_mesh.bdr_attributes.Max());
        noslip_attr.SetSize(fluid_mesh.bdr_attributes.Max());
        xf_ess_attr.SetSize(fluid_mesh.bdr_attributes.Max());

        xf_ess_attr = 1;
        // xf_ess_attr[4] = 0;
        
        u_ess_attr = 1;
        u_ess_attr[1] = 0;

        p_ess_attr = 0;
        p_ess_attr[1] = 1;
        
        noslip_attr = 1;
        noslip_attr[0] = 0;
        noslip_attr[1] = 0;
    }


    Vector vzero(dim); vzero = 0.0;
    Vector vone(dim); vone = 1.0;
    VectorConstantCoefficient zerovec(vzero);


    // Inlet velocity boundary condition
    real_t U0 = 1.5;
    auto velocity_profile = [&U0](const Vector &x, double t, Vector &u) mutable
    {
        double xi = x(0), xc=0.2, r=0.051;
        double yi = x(1), yc=0.2;
        double d  = pow(xi-xc,2)+pow(yi-yc,2);
        double U = U0;
        u = 0.0;
        if (d > (r*r))
        {
            u(0) = 4.0 * U * yi * (0.41 - yi) / (pow(0.41, 2.0));
            u(1) = 0.0;
        }
    };

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


    // Set material properties
    real_t E = 3.6e3; // Young's modulus
    real_t nu = 4.0e-1; // Poisson's ratio
    real_t solid_density = 1.0e3;
    real_t viscosity = 1.0e-3;
    real_t fluid_density = 1.0;
    real_t mesh_diffusion = 1.0e6;
    real_t mesh_wave_speed = 1.0e1;
    real_t lame_mu = E / (2.0 * (1.0 + nu));
    real_t lame_lambda = 2.0*lame_mu*nu/(1.0-2.0*nu);

    if(myid == 0)
    {
        std::cout << "Young's Modulus: " << E << std::endl;
        std::cout << "Poisson's Ratio: " << nu << std::endl;
        std::cout << "Lame's Mu: " << lame_mu << std::endl;
        std::cout << "Lame's Lambda: " << lame_lambda << std::endl;
        std::cout << "Solid Density: " << solid_density << std::endl;
        std::cout << "Fluid Density: " << fluid_density << std::endl;
        std::cout << "Viscosity: " << viscosity << std::endl;
        std::cout << "Mesh Diffusion: " << mesh_diffusion << std::endl;
        std::cout << "Mesh Wave Speed: " << mesh_wave_speed << std::endl;
    }

    Elasticity elasticity(xs_fes, xs_ess_attr, xs_nat_attr, lame_mu, solid_density, lame_lambda);
    NavierStokes nse(u_fes, p_fes, u_ess_attr, p_ess_attr, fluid_density, viscosity);
    MeshDiffusion morpher(xf_fes, xf_ess_attr, mesh_diffusion);
    // MeshWave morpher(xf_fes, xf_ess_attr, mesh_wave_speed);    
    // Elasticity morpher(xf_fes, xf_ess_attr, empty, lame_mu, solid_density, lame_lambda);    

    // Elasticity grid functions
    ParGridFunction &xs_gf = elasticity.x_gf;   xs_gf = 0.0;
    ParGridFunction &us_gf = elasticity.u_gf;   us_gf = 0.0;
    ParGridFunction &stress_gf = elasticity.stress_gf;   stress_gf = 0.0;

    // Navier-Stokes grid functions
    ParGridFunction &p_gf = nse.p_gf;   p_gf = 0.0;
    ParGridFunction &uf_gf = nse.u_gf;   uf_gf = 0.0;
    ParGridFunction &tau_gf = nse.stress_gf; tau_gf = 0.0;

    // Mesh morpher grid function
    ParGridFunction &xf_gf = morpher.x_gf; xf_gf = 0.0;
    ParGridFunction &dxf_gf = morpher.u_gf; dxf_gf = 0.0;


    // Link the required fields
    LinkedFields xs_to_xf_lf(xs_gf, xf_gf); // Elasticity/solid displacement to fluid mesh displacement
    LinkedFields us_to_dxf_lf(us_gf, dxf_gf); // Elasticity/solid velocity to fluid mesh velocity
    LinkedFields us_to_uf_lf(us_gf, uf_gf); // Elasticity/solid velocity to fluid velocity (navier-stokes)
    LinkedFields strsf_to_strss_lf(tau_gf, stress_gf); // Navier-Stokes stress to elasticity stress

    // elasticity.AddLinkedFields(&xs_to_xf_lf);
    // elasticity.AddLinkedFields(&us_to_dxf_lf);
    // elasticity.AddLinkedFields(&us_to_uf_lf);
    // nse.AddLinkedFields(&strsf_to_strss_lf);


    ConstantCoefficient p_coeff(0.0);
    VectorConstantCoefficient x_coeff(vzero);
    VectorFunctionCoefficient u_coeff(dim, velocity_profile);
    VectorFunctionCoefficient f_coeff(dim, force_profile);


    // Set initial conditions in solid
    xs_gf.ProjectCoefficient(zerovec);
    us_gf.ProjectCoefficient(zerovec);
    stress_gf.ProjectCoefficient(zerovec);
    stress_gf.ProjectBdrCoefficient(f_coeff, xs_nat_attr);
    elasticity.Update();
    
    // Set initial conditions in fluid    
    p_gf.ProjectCoefficient(p_coeff);
    uf_gf.ProjectCoefficient(u_coeff);
    uf_gf.ProjectBdrCoefficient(zerovec,noslip_attr);
    tau_gf.ProjectBdrCoefficient(nse.stress_coeff,u_ess_attr);
    
    xf_gf.ProjectBdrCoefficient( x_coeff, xf_ess_attr);
    dxf_gf.ProjectCoefficient( x_coeff, xf_ess_attr);


    /// Fluid solution
    Array<int> nse_offsets({0,u_fes.GetTrueVSize(), p_fes.GetTrueVSize()});
    nse_offsets.PartialSum();   
    BlockVector up(nse_offsets);

    uf_gf.GetTrueDofs(up.GetBlock(0));
    p_gf.GetTrueDofs(up.GetBlock(1));

    /// Solid solution
    Array<int> solid_offsets({0,xs_fes.GetTrueVSize(), xs_fes.GetTrueVSize()});
    solid_offsets.PartialSum();   
    BlockVector xu(solid_offsets); // Solid displacement and velocity
    
    xs_gf.GetTrueDofs(xu.GetBlock(0));
    us_gf.GetTrueDofs(xu.GetBlock(1));

    /// Morphing solution
    Array<int> mesh_offsets({0,xf_fes.GetTrueVSize(), xf_fes.GetTrueVSize()});
    mesh_offsets.PartialSum();
    BlockVector xuf(mesh_offsets); // Fluid displacement and velocity
    xf_gf.GetTrueDofs(xuf.GetBlock(0));
    dxf_gf.GetTrueDofs(xuf.GetBlock(1));


    // Create ODE solvers for each application
    std::unique_ptr<ODESolver> elasticity_solver = ODESolver::Select(ode_solver_type);   
    std::unique_ptr<ODESolver> nse_solver = ODESolver::Select(ode_solver_type);
    std::unique_ptr<ODESolver> morpher_solver = ODESolver::Select(ode_solver_type);

    elasticity_solver->Init(elasticity); 
    nse_solver->Init(nse);
    morpher_solver->Init(morpher);
    

    ParaViewDataCollection fluid_pv("fsi-fluid-"+std::to_string(ode_solver_type), &fluid_mesh);
    ParaViewDataCollection solid_pv("fsi-solid-"+std::to_string(ode_solver_type), &solid_mesh);

    fluid_pv.SetLevelsOfDetail(order);
    fluid_pv.SetDataFormat(VTKFormat::BINARY);
    fluid_pv.SetHighOrderOutput(true);
    fluid_pv.RegisterField("displacement",&xf_gf);
    fluid_pv.RegisterField("dxdt",&dxf_gf);
    fluid_pv.RegisterField("pressure",&p_gf);
    fluid_pv.RegisterField("velocity",&uf_gf);
    fluid_pv.RegisterField("stress",&tau_gf);
    
    solid_pv.SetLevelsOfDetail(order);
    solid_pv.SetDataFormat(VTKFormat::BINARY);
    solid_pv.SetHighOrderOutput(true);
    solid_pv.RegisterField("displacement",&xs_gf);
    solid_pv.RegisterField("velocity",&us_gf);
    solid_pv.RegisterField("force",&stress_gf);
    


    // Set up the coupled multiapp
    CoupledApplication multiapp(3); // A total of 3 coupled applications: navier, elasticity, and mesh morpher

    Application* nse_app = nullptr;
    Application* elasticity_app = nullptr;
    Application* morpher_app = nullptr;

    if(couple_type == 0)
    {
        nse_app = multiapp.AddOperator(nse_solver.get()); // Navier-Stokes integrator
        elasticity_app = multiapp.AddOperator(elasticity_solver.get()); // Elasticity integrator
        morpher_app = multiapp.AddOperator(morpher_solver.get()); // Mesh morpher integrator
    }
    else
    {
        nse_app = multiapp.AddOperator(&nse); // Navier-Stokes application
        elasticity_app = multiapp.AddOperator(&elasticity); // Elasticity application
        morpher_app = multiapp.AddOperator(&morpher); // Mesh morpher application
    }

    elasticity_app->AddLinkedFields(&xs_to_xf_lf);
    elasticity_app->AddLinkedFields(&us_to_dxf_lf);
    elasticity_app->AddLinkedFields(&us_to_uf_lf);
    nse_app->AddLinkedFields(&strsf_to_strss_lf);


    // multiapp.SetCouplingScheme(CoupledApplication::Scheme::ALTERNATING_SCHWARZ); // Select the coupling scheme
    multiapp.SetCouplingScheme(CoupledApplication::Scheme::ADDITIVE_SCHWARZ); // Select the coupling scheme
   
    // Select solver for partitioned (Schwarz) coupling   
    FPISolver fp_solver(MPI_COMM_WORLD);
    AitkenRelaxation fp_relax;
    fp_solver.iterative_mode = true;
    fp_solver.SetRelTol(1e-8);
    fp_solver.SetAbsTol(0.0);
    fp_solver.SetMaxIter(100);
    fp_solver.SetPrintLevel(1);
    fp_solver.SetRelaxation(1e-6, &fp_relax);

    multiapp.SetSolver(&fp_solver); // Set the solver for the multiapp
    multiapp.Assemble(); // Assemble the multiapp (build CouplingOperator)
    multiapp.Finalize(); // Finalize the multiapp (perform checks)

    // ODE solver for the coupled operator
    std::unique_ptr<ODESolver> coupled_solver = ODESolver::Select(ode_solver_type);

    // Set up the block vector for the coupled application in the correct order
    BlockVector xb(multiapp.GetOffsets());
    xb.GetBlock(0) = up; // Fluid velocity and pressure
    xb.GetBlock(1) = xu; // Solid displacement and velocity
    xb.GetBlock(2) = xuf.GetBlock(0); // Solid displacement (velocity is not used in diffusion-based mesh morphing)


    auto nse_preprocess = [&nse](Vector &x) mutable
    {
        nse.PreProcess(x); // Preprocess the Navier-Stokes application
    };

    auto nse_postprocess = [&nse](Vector &x) mutable
    {
        nse.PostProcess(x); // Postprocess the Navier-Stokes application
    };

    auto morph_preprocess = [&xf_gf](Vector &x) mutable
    {
        xf_gf.GetTrueDofs(x);
    };

    auto morph_postprocess = [&xf_gf](Vector &x) mutable
    {
        xf_gf.SetFromTrueDofs(x);
    };
    
    auto elasticity_preprocess = [&elasticity](Vector &x) mutable
    {
        elasticity.PreProcess(x); // Preprocess the Elasticity application
    };

    auto elasticity_postprocess = [&elasticity](Vector &x) mutable
    {
        elasticity.PostProcess(x); // Postprocess the Elasticity application
    };

    if(couple_type == 0)
    {
        nse_app->SetPreProcessFunction(nse_preprocess);
        nse_app->SetPostProcessFunction(nse_postprocess);
        morpher_app->SetPreProcessFunction(morph_preprocess);
        morpher_app->SetPostProcessFunction(morph_postprocess);
        elasticity_app->SetPreProcessFunction(elasticity_preprocess);
        elasticity_app->SetPostProcessFunction(elasticity_postprocess);
    }
    else
    {
        coupled_solver->Init(multiapp);
    }

    auto save_callback = [&](int cycle, double t)
    {
        fluid_pv.SetCycle(cycle);
        fluid_pv.SetTime(t);
        fluid_pv.Save();

        solid_pv.SetCycle(cycle);
        solid_pv.SetTime(t);
        solid_pv.Save();        
    };


    StopWatch timer;
    timer.Start();

    real_t t = 0.0;
    bool last_step = false;
    real_t t_trans = 0.05;
    save_callback(0, t);

    if (myid == 0) { out << "Starting time integration..." << std::endl; }

    // Solve the Navier-Stokes equations to fully developed flow
    for (int ti = 1; !last_step; ti++)
    {
        if (t + dt >= t_trans - dt/2){ last_step = true; }
        nse.PreProcess(up);
        nse_solver->Step(up,t,dt);
        nse.PostProcess(up);
        nse.Transfer(up);
        if (last_step || (ti % vis_steps) == 0){
            if (myid == 0) { out << "step " << ti << ", t = " << t << std::endl;}
            tau_gf.ProjectBdrCoefficient(nse.stress_coeff,u_ess_attr);
            save_callback(ti, t);
        }
    }
    if (myid == 0) { out << "Flow is fully developed." << std::endl; }

    if (myid == 0) { out << "Reducing timestep." << std::endl; }
    dt = 1e-3;
    last_step = false;
    multiapp.Transfer(xb);

    for (int ti = 1; !last_step; ti++)
    {
        if (t + dt >= t_final - dt/2){ last_step = true; }

        if (couple_type == 0)
        {   // Couple flow maps
            multiapp.Step(xb, t, dt);
        }
        else
        {   // Couple TimeDependentOperators
            multiapp.PreProcess(xb,true);
            coupled_solver->Step(xb, t, dt);
            multiapp.PostProcess(xb,true);
            multiapp.Transfer(xb);                
        }
        

        GridFunction *solid_nodes = solid_mesh.GetNodes();
        *solid_nodes += xs_gf;
        solid_mesh.DeleteGeometricFactors();

        GridFunction *fluid_nodes = fluid_mesh.GetNodes();
        *fluid_nodes += xf_gf;
        fluid_mesh.DeleteGeometricFactors();

        xs_fes.Update();
        xf_fes.Update();
        u_fes.Update();
        p_fes.Update();
        xf_gf.Update();
        dxf_gf.Update();
        us_gf.Update();
        xs_gf.Update();
        stress_gf.Update();
        // elasticity.Update();
        // morpher.Update();

        if (last_step || (ti % vis_steps) == 0){
            if (myid == 0) { out << "step " << ti << ", t = " << t << std::endl;}
            // tau_gf.ProjectBdrCoefficient(nse.stress_coeff,u_ess_attr);
            save_callback(ti, t);
        }
    }

    timer.Stop();
    if (myid == 0){
        out << "Total time: " << timer.RealTime() << " seconds." << std::endl;
    }

    return 0;
}
