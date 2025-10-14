/**
 * Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
 * at the Lawrence Livermore National Laboratory. All Rights reserved. See files
 * LICENSE and NOTICE for details. LLNL-CODE-806117.
 * 
 * This file is part of the MFEM library. For more information and source code
 * availability visit https://mfem.org.
 * 
 * MFEM is free software; you can redistribute it and/or modify it under the
 * terms of the BSD-3 license. We welcome feedback and contributions, see file
 * CONTRIBUTING.md for details.
 * 
 *            --------------------------------------------------
 *                    Fluid-Structure Interaction miniapp
 *            --------------------------------------------------
 * 
 * This miniapp simulates fluid-structure interaction (FSI) problems described in
 * the paper:
 *   
 * with the incompressible Navier-Stokes equations, in Arbitrary Lagrangian-Eulerian (ALE)
 * formulation, coupled with linear elasticity equations in the solid domain. 
 * The coupling is done with a partitioned approach using the alternating or additive
 * Schwarz method. The fluid mesh motion is handled with a mesh displacement diffusion
 * approach. The geometry is a channel with a cylinder and a flexible beam attached to
 * the cylinder downstream. 
 * 
 * The following boundary conditions are applied:
 *  1) Channel inlet (attribute 1): parabolic velocity profile
 *  2) Channel outlet (attribute 2): zero-pressure
 *  3) Channel walls: no-slip
 * 
 * with the following interface conditions at the fluid-structure interface:
 * 1) Continuity of velocity: u_f = u_s
 * 2) Continuity of traction: -*pI + mu(grad(u_f)+grad(u_f)^T))•n = sigma_s•n
 * 
 * The velocity continuity is imposed using the stage-slope in the implicit multistage method
 * ku_f = du_f/dt = du_s/dt = ku_s, where ku_f and ku_s are the fluid and solid stage-slopes
 * 
 * The mesh morphing is modeled as a displacement diffusion equation, 
 * dx/dt = κΔx, with the ALE mesh velocity, w = dx/dt.
 * 
 * Sample run:
 *    mpirun -np 6 ./fsi -vs 5 -dt 5e-3 -tf 10 -o 3 -rs 2 -ode 21 -U 1.0 -cs 1 -idir fsi-turek
 */

#include "mfem.hpp"
#include "multiapp.hpp"
#include "apps/navier_stokes.hpp"
#include "apps/elasticity.hpp"
#include "apps/mesh_morpher.hpp"

#include <filesystem>

using namespace mfem;
using namespace std;

// mpirun -np 6 ./fsi -vs 5 -dt 1e-2 -tf 5 -o 2 -rs 2 -ode 21
// mpirun -np 6 ./fsi -vs 5 -dt 5e-3 -tf 10 -o 3 -rs 2 -ode 21 -U 1.0 -cs 0 -init

void SetSolverParameters(IterativeSolver *solver, real_t rtol, real_t atol , int max_it, 
                         int print_level, bool iterative_mode);

int main(int argc, char *argv[])
{
    Mpi::Init();
    Hypre::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();

    int order = 2;
    int ser_ref = 0;    
    int ode_solver_type = 21; // (21) BackwardEulerSolver
                              // (22) SDIRK23Solver
                              // (23) SDIRK33Solver
                              // (34) SDIRK34Solver
    real_t Uavg = 2.0;
    real_t t_dev = 2.0;
    real_t t_final = 1.0;
    real_t dt = 1.0e-3;
    int couple_scheme = 1;
    bool init = false;
    bool lsave = false;
    bool visualization = true;
    int vis_steps = 10;
    std::string init_dir= "";


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
    args.AddOption(&Uavg, "-U", "--velocity", "Mean velocity.");
    args.AddOption(&t_dev, "-t_dev", "--transistion", "Developed flow transition time.");
    args.AddOption(&couple_scheme, "-cs", "--coupling-scheme", 
                    "Coupling scheme: -1 = Monolithic; 0 = Add. Schw.; >0 = Alt. Schw.");
    args.AddOption(&init_dir, "-idir", "--init-directory", 
                    "Directory containing intialization files. If doesn't exist, used to write init files.");
    args.AddOption(&lsave, "-save", "--save-init", "-no-save",
                    "--no-save", "Enable or disable saving initialization files.");
    args.ParseCheck();


    init = !init_dir.empty(); // if init_dir is provided, then init = true
    
    std::string mesh_file = "channel-cylinder.msh";
    Mesh *serial_mesh = new Mesh(mesh_file);
    int dim = serial_mesh->Dimension();

    for (int i = 0; i < ser_ref; ++i) { serial_mesh->UniformRefinement(); }
    serial_mesh->SetCurvature(order, false, dim, Ordering::byNODES);
    serial_mesh->EnsureNCMesh();
    ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
    delete serial_mesh;

    // Create mesh (sub)domains
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
        xs_nat_attr[4] = 1; // beam wall
        xs_ess_attr[6] = 1; // beam cylinder curve
    }

    if (fluid_mesh.bdr_attributes.Size() > 0)
    {
        u_ess_attr.SetSize(fluid_mesh.bdr_attributes.Max());
        p_ess_attr.SetSize(fluid_mesh.bdr_attributes.Max());
        noslip_attr.SetSize(fluid_mesh.bdr_attributes.Max());
        xf_ess_attr.SetSize(fluid_mesh.bdr_attributes.Max());

        xf_ess_attr = 1;

        u_ess_attr = 1;
        u_ess_attr[1] = 0; // outlet

        p_ess_attr = 0;
        p_ess_attr[1] = 1; // outlet

        noslip_attr = 1;
        noslip_attr[0] = 0; // inlet
        noslip_attr[1] = 0; // outlet
    }


    // Finite element spaces for solid and fluid domains
    H1_FECollection xfec(order, dim); // Displacement field (solid and fluid domains)
    H1_FECollection ufec(order, dim); // Velocity field (fluid domain)
    H1_FECollection pfec(order-1, dim); // Pressure field (fluid domain)

    ParFiniteElementSpace xs_fes(&solid_mesh, &xfec, dim, Ordering::byNODES);
    ParFiniteElementSpace xf_fes(&fluid_mesh, &xfec, dim, Ordering::byNODES);
    ParFiniteElementSpace u_fes(&fluid_mesh, &ufec, dim, Ordering::byNODES);
    ParFiniteElementSpace p_fes(&fluid_mesh, &pfec);


    Vector vzero(dim); vzero = 0.0;
    VectorConstantCoefficient zerovec(vzero);

    // Inlet velocity boundary condition
    auto velocity_profile = [&Uavg, &t_dev](const Vector &x, double t, Vector &u) mutable
    {
        double xi = x(0), yi = x(1);
        double U = Uavg;
        double ramp_time = 2.0;
        u = 0.0;
        if (xi == 0.0)
        {
            u(0) = 1.5 * 4.0 * U * yi * (0.41 - yi) / (pow(0.41, 2.0));
            u(1) = 0.0;
        }
        if(t < ramp_time) u(0) *= 0.5*(1.0 - cos(0.5*M_PI*t));
    };


    // Set material properties
    // real_t E = 5.6e6; // Young's modulus
    real_t E = 1.4e6; // Young's modulus
    real_t nu = 4.0e-1; // Poisson's ratio
    real_t solid_density = 1.0e4;
    real_t viscosity = 1.0e-3;
    real_t fluid_density = 1.0e3;
    real_t mesh_diffusion = 1.0e0;
    real_t lame_mu = E / (2.0 * (1.0 + nu));
    real_t lame_lambda = 2.0*lame_mu*nu/(1.0-2.0*nu);
    real_t compressibility = 0e-4; // Artificial Compressibility
    bool scaled_pressure = true;

    if(Mpi::Root())
    {
        std::cout << "Young's Modulus: " << E << std::endl;
        std::cout << "Poisson's Ratio: " << nu << std::endl;
        std::cout << "Lame's Mu: " << lame_mu << std::endl;
        std::cout << "Lame's Lambda: " << lame_lambda << std::endl;
        std::cout << "Solid Density: " << solid_density << std::endl;
        std::cout << "Fluid Density: " << fluid_density << std::endl;
        std::cout << "Viscosity: " << viscosity << std::endl;
        std::cout << "Mesh Diffusion: " << mesh_diffusion << std::endl;
        std::cout << "Artificial Compressibility: " << compressibility << std::endl;
        std::cout << "Mean Velocity: " << Uavg << std::endl;
        std::cout << "Reynolds Number: " << (Uavg*0.1)/viscosity << std::endl;
        std::cout << "AE Number: " << (E)/(Uavg*Uavg*fluid_density) << std::endl;
        std::cout << "Beta Number: " << solid_density/fluid_density << std::endl;
    }

    // Build individual applications (morpher, elasticity, navier-stokes)

    // Mesh Morpher
    MeshDiffusion morpher(xf_fes, xf_ess_attr, mesh_diffusion);

    // Morphing grid functions
    ParGridFunction &xf_gf  = *morpher.Fields().GetField("Displacement");
    ParGridFunction &dxf_gf = *morpher.Fields().GetField("dxdt");
    ParGridFunction &xf_gf_bc = *morpher.Fields().GetField("Displacement_BC");

    /// Morphing solution
    Vector xf(xf_fes.GetTrueVSize());
    xf_gf.GetTrueDofs(xf);

    ParGridFunction mesh_disp(&xf_fes), mesh_vel(&xf_fes);
    mesh_disp = 0.0; mesh_vel = 0.0;


    // Elasticity
    Elasticity elasticity(xs_fes, xs_ess_attr, xs_nat_attr, solid_density, lame_mu, lame_lambda);

    // Elasticity grid functions
    ParGridFunction &xs_gf = *elasticity.Fields().GetField("Displacement");
    ParGridFunction &us_gf = *elasticity.Fields().GetField("Velocity");
    ParGridFunction &stress_gf = *elasticity.Fields().GetField("Traction");

    /// Solid solution
    Array<int> solid_offsets({0,xs_fes.GetTrueVSize(), xs_fes.GetTrueVSize()});
    solid_offsets.PartialSum();
    BlockVector xu(solid_offsets); // Solid displacement and velocity
    us_gf.GetTrueDofs(xu.GetBlock(0));
    xs_gf.GetTrueDofs(xu.GetBlock(1));


    // Navier-Stokes
    VectorGridFunctionCoefficient ale_uf(&dxf_gf);
    ScalarVectorProductCoefficient neg_ale(-1.0, ale_uf);
    VectorCoefficient *ale_velocity = &neg_ale;

    NavierStokes nse(u_fes, p_fes, u_ess_attr, p_ess_attr,
                     fluid_density, viscosity, compressibility,
                     scaled_pressure, ale_velocity);

    std::unique_ptr<ODESolver> nse_solver = ODESolver::Select(ode_solver_type);
    nse_solver->Init(nse);

    // Navier-Stokes grid functions
    ParGridFunction &p_gf = *nse.Fields().GetField("Pressure");
    ParGridFunction &uf_gf = *nse.Fields().GetField("Velocity");
    ParGridFunction &tau_gf = *nse.Fields().GetField("Stress");
    ParGridFunction &uf_gf_bc = *nse.Fields().GetField("Velocity_BC");

    ConstantCoefficient p_coeff(0.0);
    VectorFunctionCoefficient u_coeff(dim, velocity_profile);
    
    // Set initial conditions in fluid
    p_gf.ProjectCoefficient(p_coeff);
    uf_gf.ProjectCoefficient(u_coeff);
    uf_gf.ProjectBdrCoefficient(zerovec,noslip_attr);

    if(init) // Initialize from file
    {
        std::string mpirank = std::to_string(myid);
        bool file_error = false;
        std::string pfilename = init_dir+"/p-init.gf."+mpirank.insert(0,6-mpirank.length(),'0');
        std::string ufilename = init_dir+"/u-init.gf."+mpirank.insert(0,6-mpirank.length(),'0');
        if (!std::filesystem::exists(pfilename) ||
            !std::filesystem::exists(ufilename))
        {
            file_error = true;
            init = false;
        }
        
        if(file_error) MPI_Bcast(&file_error, 1, MPI_CXX_BOOL, myid, MPI_COMM_WORLD);

        if(!file_error)
        {
            istream *pfile, *ufile;
            pfile = new ifstream(pfilename);
            ufile = new ifstream(ufilename);

            nse.p_gf = ParGridFunction(&fluid_mesh,*pfile);
            nse.u_gf = ParGridFunction(&fluid_mesh,*ufile);
            delete pfile;
            delete ufile;

            u_coeff.SetTime(t_dev+2.0); // time beyond ramp-up
            uf_gf.ProjectBdrCoefficient(u_coeff,u_ess_attr);
        }
    }
    tau_gf.ProjectBdrCoefficient(nse.stress_coeff,u_ess_attr);

    /// Fluid solution
    Array<int> nse_offsets({0,u_fes.GetTrueVSize(), p_fes.GetTrueVSize()});
    nse_offsets.PartialSum();
    BlockVector up(nse_offsets);

    uf_gf.GetTrueDofs(up.GetBlock(0));
    p_gf.GetTrueDofs(up.GetBlock(1));


    // Three coupled applications: navier, elasticity, and morpher
    CoupledOperator multiapp(3);
    std::unique_ptr<ODESolver> coupled_solver = ODESolver::Select(ode_solver_type);

    Application* nse_app = multiapp.AddOperator(&nse);
    Application* elasticity_app = multiapp.AddOperator(&elasticity);
    Application* morpher_app = multiapp.AddOperator(&morpher);


    // Set up field transfer
    // NativeTransfer strsf_to_strss_map(u_fes, xs_fes); // default map if none provided
    // GSLibTransfer strsf_to_strss_map(u_fes, xs_fes, xs_nat_attr);
    nse_app->Fields().AddTargetField("Stress", &stress_gf);
    elasticity_app->Fields().AddTargetField("Velocity_BC", &uf_gf_bc);
    elasticity_app->Fields().AddTargetField("Displacement_BC", &xf_gf_bc);


    // Set up coupling parameters (schemes and solvers)
    FPISolver fp_solver(MPI_COMM_WORLD); // For partitioned solves
    AitkenRelaxation fp_relax;
    SetSolverParameters(&fp_solver, 0.0, 5e-4, 100, 1, false);
    fp_relax.SetBounds(-1.0e1,1.0e1);
    fp_solver.SetRelaxation(1e0, nullptr); // Use default relaxation method
    // fp_solver.SetRelaxation(1e0, &fp_relax);

    NewtonSolver newton_solver(MPI_COMM_WORLD); // For fully coupled
    GMRESSolver gmres_solver(MPI_COMM_WORLD); // For fully coupled

    if(couple_scheme == -1)
    {
        MFEM_ABORT("Monolithic coupling not supported for FSI.")
        multiapp.SetCouplingScheme(CoupledOperator::Scheme::MONOLITHIC);

        SetSolverParameters(&gmres_solver, 1e-3, 1e-3, 500, 1, false);
        gmres_solver.SetKDim(300);

        SetSolverParameters(&newton_solver, 0.0, 1e-4, 30, 1, false);
        newton_solver.SetSolver(gmres_solver);

        multiapp.SetSolver(&newton_solver); // Set the solver for the multiapp
    }
    else if(couple_scheme == 0)
    {
        multiapp.SetCouplingScheme(CoupledOperator::Scheme::ADDITIVE_SCHWARZ);
        multiapp.SetSolver(&fp_solver); // Set the solver for the multiapp
    }
    else
    {
        multiapp.SetCouplingScheme(CoupledOperator::Scheme::ALTERNATING_SCHWARZ);
        multiapp.SetSolver(&fp_solver); // Set the solver for the multiapp
    }

    multiapp.Assemble(false); // Assemble the multiapp (build OperatorCoupler)
    multiapp.Finalize(false); // Finalize the multiapp (perform checks)

    coupled_solver->Init(multiapp);

    // Set up the initial conditions in block vector for the
    // coupled application in the correct order
    int fl_id = nse_app->GetOperatorIndex();
    int el_id = elasticity_app->GetOperatorIndex();
    int morph_id = morpher_app->GetOperatorIndex();

    BlockVector xb(multiapp.GetBlockOffsets());
    xb.GetBlock(el_id) = xu; // Solid displacement and velocity
    xb.GetBlock(fl_id) = up; // Fluid velocity and pressure
    xb.GetBlock(morph_id) = xf; // Fluid displacement



    auto update_grid_functions  = [&](Vector &x) mutable
    {
        BlockVector xb(x.GetData(), multiapp.GetBlockOffsets());
        BlockVector elas_x(xb.GetBlock(el_id).GetData(), solid_offsets);
        BlockVector nse_x(xb.GetBlock(fl_id).GetData(), nse_offsets);
        Vector morph_x(xb.GetBlock(morph_id).GetData(), xb.BlockSize(morph_id));

        xf_gf.SetFromTrueDofs(morph_x);
        us_gf.SetFromTrueDofs(elas_x.GetBlock(0));
        xs_gf.SetFromTrueDofs(elas_x.GetBlock(1));
        uf_gf.SetFromTrueDofs(nse_x.GetBlock(0));
        p_gf.SetFromTrueDofs(nse_x.GetBlock(1));
        tau_gf.ProjectBdrCoefficient(nse.stress_coeff,u_ess_attr);
    };

    auto update_nse_grid_functions  = [&](Vector &x) mutable
    {
        BlockVector nse_x(x.GetData(), nse_offsets);
        uf_gf.SetFromTrueDofs(nse_x.GetBlock(0));
        p_gf.SetFromTrueDofs(nse_x.GetBlock(1));
        tau_gf.ProjectBdrCoefficient(nse.stress_coeff,u_ess_attr);
    };    

    // Set up visualization
    ParaViewDataCollection *fluid_pv = nullptr;
    ParaViewDataCollection *solid_pv = nullptr;

    if(visualization)
    {
        fluid_pv = new ParaViewDataCollection("fsi-fluid", &fluid_mesh);
        solid_pv = new ParaViewDataCollection("fsi-solid", &solid_mesh);

        fluid_pv->SetLevelsOfDetail(order);
        fluid_pv->SetDataFormat(VTKFormat::BINARY);
        fluid_pv->SetHighOrderOutput(true);
        fluid_pv->RegisterField("displacement",&mesh_disp);
        fluid_pv->RegisterField("dxdt",&mesh_vel);

        fluid_pv->RegisterField("pressure",&p_gf);
        fluid_pv->RegisterField("velocity",&uf_gf);
        fluid_pv->RegisterField("stress",&tau_gf);
        fluid_pv->RegisterField("ale_velocity",&dxf_gf);

        solid_pv->SetLevelsOfDetail(order);
        solid_pv->SetDataFormat(VTKFormat::BINARY);
        solid_pv->SetHighOrderOutput(true);
        solid_pv->RegisterField("displacement",&xs_gf);
        solid_pv->RegisterField("velocity",&us_gf);
        solid_pv->RegisterField("stress",&stress_gf);
    }

    auto save_callback = [&](int cycle, double t)
    {
        if(fluid_pv)
        {
            fluid_pv->SetCycle(cycle);
            fluid_pv->SetTime(t);
            fluid_pv->Save();
        }

        if(solid_pv)
        {
            solid_pv->SetCycle(cycle);
            solid_pv->SetTime(t);
            solid_pv->Save();
        }
    };


    if (Mpi::Root()) out << "Starting time integration..." << std::endl;

    StopWatch timer;
    timer.Start();

    real_t t = 0.0;
    bool last_step = false;
    int tindex = 1;
    save_callback(0, t);


    // Solve the Navier-Stokes equations to fully developed flow time, t_dev
    if(t_dev > 0.0 && !init)
    {
        nse_app->SetOperationID(Application::OperationID::STEP);
        for (; !last_step; tindex++)
        {
            if (t + dt >= t_dev - dt/2){ last_step = true; }

            u_coeff.SetTime(t); // Slowly ramp-up inlet velocity
            uf_gf.ProjectBdrCoefficient(u_coeff,u_ess_attr);
            nse_solver->Step(up,t,dt);

            if (last_step || (tindex % vis_steps) == 0){
                if (Mpi::Root()) { out << "step " << tindex << ", t = " << t << std::endl;}
                update_nse_grid_functions(up);
                save_callback(tindex, t);
            }
        }

        bool dir_created = true;
        if ((myid==0) && lsave)
        {
            std::filesystem::path dir_path = init_dir;
            dir_created = std::filesystem::create_directory(dir_path);
            if(!dir_created) MPI_Bcast(&dir_created, 1, MPI_CXX_BOOL, myid, MPI_COMM_WORLD);
        }
        if(lsave && dir_created){
            p_gf.Save((init_dir+"/p-init.gf").c_str());
            uf_gf.Save((init_dir+"/u-init.gf").c_str());
        }

        xb.GetBlock(fl_id) = up; // Update nse block
    }

    vis_steps = 1;
    last_step = false;

    nse_app->SetCoupled(true);
    morpher_app->SetCoupled(true);
    elasticity_app->SetCoupled(true);
    multiapp.Transfer(xb);

    // Store original fluid nodes; morphing is done w.r.t. original configuration
    GridFunction fluid_nodes_orig = *(fluid_mesh.GetNodes());

    for (; !last_step; tindex++)
    {
        if (t + dt >= t_final - dt/2){ last_step = true; }

        coupled_solver->Step(xb, t, dt);
        multiapp.Transfer(xb);
        update_grid_functions(xb);


        // Morph fluid mesh
        mesh_disp.SetFromTrueDofs(xb.GetBlock(morph_id));
        mesh_vel = dxf_gf;

        GridFunction *fluid_nodes = fluid_mesh.GetNodes();
        *fluid_nodes = fluid_nodes_orig;
        *fluid_nodes += mesh_disp;
        fluid_mesh.DeleteGeometricFactors();

        // Update FE spaces, grid functions and forms
        // after fluid mesh update
        nse_app->Update();
        morpher_app->Update();

        if (last_step || (tindex % vis_steps) == 0){
            if (Mpi::Root()){ out << "step " << tindex << ", t = " << t << std::endl;}
            save_callback(tindex, t);
        }
    }

    timer.Stop();
    if (Mpi::Root()){
        out << "Total time: " << timer.RealTime() << " seconds." << std::endl;
    }

    return 0;
}

void SetSolverParameters(IterativeSolver *solver, real_t rtol, real_t atol , int max_it, 
                         int print_level, bool iterative_mode)
{
    solver->SetRelTol(rtol);
    solver->SetAbsTol(atol);
    solver->SetMaxIter(max_it);
    solver->SetPrintLevel(print_level);
    solver->iterative_mode = iterative_mode;
}
