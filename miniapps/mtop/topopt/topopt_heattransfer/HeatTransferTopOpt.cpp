#include "mfem.hpp"
#include "HeatTransferTopOpt.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>

using namespace std;
using namespace mfem;

void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v(0) = 0.0;
   v(1) = 0.0;
}

real_t q0_function(const Vector &x)
{
   // int dim = x.Size();
   // real_t cx = 0.01; 
   // real_t cy1 = 0.8;
   // real_t cy2 = 0.5;
   // real_t cy3 = 0.1;
   // real_t rx = 0.1, ry = 0.1, w = 10.;
   // real_t e1 = std::erfc(w*(x(0)-cx-rx))*std::erfc(-w*(x(0)-cx+rx)) * std::erfc(w*(x(1)-cy1-ry))*std::erfc(-w*(x(1)-cy1+ry));
   // real_t e2 = std::erfc(w*(x(0)-cx-rx))*std::erfc(-w*(x(0)-cx+rx)) * std::erfc(w*(x(1)-cy2-ry))*std::erfc(-w*(x(1)-cy2+ry));
   // real_t e3 = std::erfc(w*(x(0)-cx-rx))*std::erfc(-w*(x(0)-cx+rx)) * std::erfc(w*(x(1)-cy3-ry))*std::erfc(-w*(x(1)-cy3+ry));
   // return e1+e2+e3;
   return 1.0;
}

real_t inflow_function(const Vector &x)
{
   return 1.0;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   const MPI_Comm comm = MPI_COMM_WORLD; 
   int myid = Mpi::WorldRank(); 
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../../../data/inline-quad.mesh"; 
   int ser_ref_levels = 1;
   int par_ref_levels = 2;
   int order = 1;
   bool vis_forward = true;
   bool vis_adjoint = true;
   int ode_solver_type = 61; // 61 - Forward Backward Euler
   // 62 - IMEXRK2(2,2,2)
   // 63 - IMEXRK2(2,3,2)
   // 64 - IMEXRK3(3,4,3)
   real_t t_final = 1.0;        
   real_t dt = 0.001;
   real_t diffusion_term = 0.0;
   int vis_steps = 10;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                    "Number of times to refine the mesh uniformly in serial,"
                    " -1 for auto.");   
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                    "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                    "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&vis_forward, "-visf", "--visualization-forward", "-no-visf",
                    "--no-visualization-forward",
                    "Enable or disable Paraview Visualization for Forward Problem");
   args.AddOption(&vis_forward, "-visa", "--visualization-adjoint", "-no-visa",
                    "--no-visualization-adjoint",
                    "Enable or disable Paraview Visualization for Forward Problem");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                    ODESolver::IMEXTypes.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                    "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                    "Time step.");
   args.AddOption(&diffusion_term, "-dc", "--diffusion-coeff",
                    "Diffusion coefficient in the PDE.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
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

   // 3. Read the meshfile 
   Mesh *mesh = new Mesh(mesh_file); 
   const int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh->UniformRefinement(); }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }


   // 5. Define the parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   FiniteElementCollection *fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);;
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << global_vSize << endl;        
   }

   // 7. Boundary Conditions
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1; 
   pmesh->MarkExternalBoundaries(ess_bdr);
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 8. Define the Coefficients 
   VectorFunctionCoefficient velocity(dim, velocity_function);
   ConstantCoefficient diff_coeff(diffusion_term);
   ConstantCoefficient dt_diff_coeff(dt*diffusion_term);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient q0(q0_function);

   // 9. Construct the Solver
   IMEXAdvectionDiffusionSolver for_solve(*fes, velocity, dt_diff_coeff, diff_coeff, inflow, ess_tdof_list, ess_bdr, q0, dt, comm);

   // 10. Set up the visualization
   ParGridFunction &q_gf = for_solve.Getq();
   HypreParVector *q_vec = q_gf.GetTrueDofs();
   std::cout << "main: obtained q_vec = " << q_vec << std::endl;
   ParaViewDataCollection *pd = NULL;
   if (vis_forward)
   {
      pd = new ParaViewDataCollection("forward", pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("solution", &q_gf);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }
   std::cout << "main: ParaView setup done" << std::endl;

   // 11. Initialize the Forward time-stepping
   unique_ptr<ODESolver> ode_solver = ODESolver::SelectIMEX(ode_solver_type);
   real_t t = 0.0;
   for_solve.SetTime(t);
   ode_solver->Init(for_solve);
   std::cout << "main: ode_solver init done" << std::endl;
   int n_steps = (int)ceil(t_final / dt);

   // 12. Perform Integration
   bool done = false;
   for (int ti = 0; !done; )
   {
      real_t dt_real = min(dt, t_final - t);  
      ode_solver->Step(*q_vec, t, dt);
      ti++;

      done = (t >= t_final - 1e-8*dt); 
      
      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;  
         }
         q_gf = *q_vec;
         if (vis_forward)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }
      }
   }

   // 11. Set up and compute the cost function at the terminal time
   ConstantCoefficient zero(0.0);
   RectangularIndicator indicator(0, 1, 0, 1);
   TerminalL2Objective obj_func(fes, indicator, comm); 
   obj_func.ComputeObjective(q_gf); 
   real_t loss = obj_func.GetObjective();
   std::cout << "Cost at terminal time is: " << loss << std::endl;
   std::cout << "l2 norm at terminal time is " << q_gf.ComputeL2Error(zero) << std::endl;
   ParLinearForm grad_form(fes);
   obj_func.ComputeObjectiveGradient(q_gf, grad_form);
   std::unique_ptr<HypreParVector> grad_vec(grad_form.ParallelAssemble());
   std::cout << "l2 norm of gradient at terminal time is " << grad_vec->Norml2() << std::endl;
   std::cout << "gradient size " << grad_vec->Size() << std::endl; 
   *q_vec -= *grad_vec;
   std::cout << "l2 norm of diff at terminal time is " << q_vec->Norml2() << std::endl;


   // Free the used memory.
   delete pd;
   delete fes;  
   delete pmesh;
   delete fec; 
  
   return 0; 
}

// int main(int argc, char *argv[])
// {
//     // 1. Initialize MPI and HYPRE.
//     Mpi::Init();
//     int num_procs = Mpi::WorldSize();
//     int myid = Mpi::WorldRank();
//     Hypre::Init();

//     // 2. Parse command-line options.
//     const char *mesh_file = "../../../../data/periodic-square.mesh";
//     int ser_ref_levels = 2;
//     int par_ref_levels = 2;
//     int order = 1;
//     real_t kappa_0 = -1.0;
//     bool vis_forward = true;
//     bool vis_adjoint = true;
//     int ode_solver_type = 64; // 61 - Forward Backward Euler
//     // 62 - IMEXRK2(2,2,2)
//     // 63 - IMEXRK2(2,3,2)
//     // 64 - IMEXRK3(3,4,3)
//     real_t t_final = 5.0;
//     real_t dt = 0.01;
//     real_t diffusion_term = 0.01;
//     int vis_steps = 10;
//     const char *device_config = "cpu";

//     OptionsParser args(argc, argv);
//     args.AddOption(&mesh_file, "-m", "--mesh",
//                     "Mesh file to use.");
//     args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
//                     "Number of times to refine the mesh uniformly in serial,"
//                     " -1 for auto.");
//     args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
//                     "Number of times to refine the mesh uniformly in parallel.");     
//     args.AddOption(&order, "-o", "--order",
//                     "Finite element order (polynomial degree) >= 0.");
//     args.AddOption(&kappa_0, "-k", "--kappa",
//                     "DG IP Penalty, should be positive. Negative values are replaced with (order+1)^2.");
//     args.AddOption(&vis_forward, "-visf", "--visualization-forward", "-no-visf",
//                     "--no-visualization-forward",
//                     "Enable or disable Paraview Visualization for Forward Problem");
//         args.AddOption(&vis_forward, "-visa", "--visualization-adjoint", "-no-visa",
//                     "--no-visualization-adjoint",
//                     "Enable or disable Paraview Visualization for Forward Problem");
//         args.AddOption(&ode_solver_type, "-s", "--ode-solver",
//                     ODESolver::IMEXTypes.c_str());
//     args.AddOption(&t_final, "-tf", "--t-final",
//                     "Final time; start time is 0.");
//     args.AddOption(&dt, "-dt", "--time-step",
//                     "Time step.");
//     args.AddOption(&diffusion_term, "-dc", "--diffusion-coeff",
//                     "Diffusion coefficient in the PDE.");
//     args.AddOption(&vis_steps, "-vs", "--visualization-steps",
//                   "Visualize every n-th timestep.");
//     args.AddOption(&device_config, "-d", "--device",
//                     "Device configuration string, see Device::Configure().");
//     args.Parse();
//     if (!args.Good())
//     {
//         if (Mpi::Root())
//         {
//             args.PrintUsage(cout);
//         }
//         return 1;
//     }
//     if (Mpi::Root())
//     {
//         args.PrintOptions(cout);
//     }
//     if (kappa_0 < 0)
//     {
//         kappa_0 = (order+1)*(order+1);
//     }

//     // 3. Read the mesh from the given mesh file. We can handle geometrically
//     //    periodic meshes in this code.
//     Mesh *mesh = new Mesh(mesh_file);
//     const int dim = mesh->Dimension();

//     // 4. Define the IMEX (Split) ODE solver used for time integration. The IMEX
//     // solvers currently available are: 55 - Forward Backward Euler,
//     // 56 - IMEXRK2(2,2,2), 57 - IMEXRK2(2,3,2), and
//     unique_ptr<ODESolver> ode_solver = ODESolver::SelectIMEX(ode_solver_type);
//     unique_ptr<ODESolver> ode_solver_adj = ODESolver::SelectIMEX(ode_solver_type);

//     // 5. Refine the mesh to increase the resolution. In this example we do
//     //    'ref_levels' of uniform refinement, where 'ref_levels' is a
//     //    command-line parameter.
//     for (int lev = 0; lev < ser_ref_levels; lev++) { mesh->UniformRefinement(); }
//     if (mesh->NURBSext)
//     {
//         mesh->SetCurvature(max(order, 1));
//     }
//     mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));


//     // 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
//     //    this mesh further in parallel to increase the resolution. Once the
//     //    parallel mesh is defined, the serial mesh can be deleted.
//     ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
//     delete mesh;
//     for (int lev = 0; lev < par_ref_levels; lev++)
//     {
//         pmesh->UniformRefinement();
//     }

//     // 7. Define the discontinuous DG finite element space of the given
//     //    polynomial order on the refined mesh.
//     FiniteElementCollection *fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);;
//     ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, fec);
//     HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();
//     if (Mpi::Root())
//     {
//         cout << "Number of unknowns: " << global_vSize << endl;
//     }

//     // 8. Set up and assemble the bilinear and linear forms corresponding to the
//     //    DG discretization. The DGTraceIntegrator involves integrals over mesh
//     //    interior faces.
//     VectorFunctionCoefficient velocity(dim, velocity_function);
//     ConstantCoefficient diff_coeff(diffusion_term);
//     ConstantCoefficient dt_diff_coeff(dt*diffusion_term);
//     ParBilinearForm *m = new ParBilinearForm(fes);
//     ParBilinearForm *k = new ParBilinearForm(fes);
//     ParBilinearForm *s = new ParBilinearForm(fes);
//     m->AddDomainIntegrator(new MassIntegrator());
//     constexpr real_t alpha = -1.0;
//     k->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
//     s->AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
//     // For the preconditioner - create billinear form corresponding to
//     // operator (M + dt S)
//     const real_t sigma = -1.0;
//     ParBilinearForm *a = new ParBilinearForm(fes);
//     a->AddDomainIntegrator(new MassIntegrator);
//     a->AddDomainIntegrator(new DiffusionIntegrator(dt_diff_coeff));
//     k->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity,
//                                                                             alpha));
                                                                        
//     k->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, alpha));
//     s->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma,
//                                                              kappa_0));
//     s->AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma, kappa_0));
//     a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dt_diff_coeff, sigma,
//                                                              kappa_0));
//     a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(dt_diff_coeff, sigma, kappa_0));

//    int skip_zeros = 0;
//    m->Assemble(skip_zeros);
//    k->Assemble(skip_zeros);
//    s->Assemble(skip_zeros);
//    a->Assemble();

//    m->Finalize(skip_zeros);
//    k->Finalize(skip_zeros);
//    s->Finalize(skip_zeros);
//    a->Finalize(skip_zeros);
//    HypreParVector b(fes);
//    b = 0.0;

//    // 9. Define the initial conditions. Set up visualization (if desired).
//    FunctionCoefficient theta0(theta0_function);
//    ParGridFunction *theta = new ParGridFunction(fes);
//    theta->ProjectCoefficient(theta0);
//    HypreParVector *theta_vec = theta->GetTrueDofs();

//    DataCollection *dc = NULL;
//    ParaViewDataCollection *pd = NULL;
//    if (vis_forward)
//    {
//       pd = new ParaViewDataCollection("forward", pmesh);
//       pd->SetPrefixPath("ParaView");
//       pd->RegisterField("solution", theta);
//       pd->SetLevelsOfDetail(order);
//       pd->SetDataFormat(VTKFormat::BINARY);
//       pd->SetHighOrderOutput(true);
//       pd->SetCycle(0);
//       pd->SetTime(0.0);
//       pd->Save();
//    }



//    // 10. Define the time-dependent evolution operator describing the
//    //     ODE right-hand side, and perform time-integration (looping
//    //     over the time iterations, ti, with a time-step dt).
//    IMEX_Evolution adv(*m, *k, *s, b, *a);

//    real_t t = 0.0;
//    adv.SetTime(t);
//    ode_solver->Init(adv);

//    int n_steps = (int)ceil(t_final / dt);

//    bool done = false;
//    for (int ti = 0; !done; )
//    {
//       real_t dt_real = min(dt, t_final - t);
//       ode_solver->Step(*theta_vec, t, dt_real);
//       ti++;

//       done = (t >= t_final - 1e-8*dt);

//       if (done || ti % vis_steps == 0)
//       {
//          if (Mpi::Root())
//          {
//             cout << "time step: " << ti << ", time: " << t << endl;
//          }
//          *theta = *theta_vec;
//          if (vis_forward)
//          {
//             pd->SetCycle(ti);
//             pd->SetTime(t);
//             pd->Save();
//          }
//       }
//    }

//    // ******Backward Advection-Diffusion solve
//    // 17. Define the DG finite element space on the
//    //    refined mesh of the given polynomial order.
//    DG_FECollection *fec_adjoint = new DG_FECollection(order, dim);
//    ParFiniteElementSpace *fes_adjoint = new ParFiniteElementSpace(pmesh, fec_adjoint);
//    int num_dofs = fes->GetNDofs();

//    // 18. Set up and assemble the parallel bilinear and linear forms (and the
//    //    parallel hypre matrices) corresponding to the DG discretization. The
//    //    DGTraceIntegrator involves integrals over mesh interior faces.
//    ConstantCoefficient zero(0.0);
//    // GridFunctionCoefficient theta_coeff(&(theta_gf_vector[n_steps-1]));
//    // FunctionCoefficient inflow_adj(inflow_function); //zero for now
//    ConstantCoefficient diff_coef_adj(diffusion_term);
//    ConstantCoefficient dt_diff_coef_adj(dt*diffusion_term);
//    VectorFunctionCoefficient neg_velocity(dim, neg_velocity_function);

//    // FunctionCoefficient theta_exact_coeff(theta_exact);
//    ParBilinearForm *a_adj = new ParBilinearForm(fes_adjoint);
//    a_adj->AddDomainIntegrator(new MassIntegrator);
//    a_adj->AddDomainIntegrator(new DiffusionIntegrator(dt_diff_coef_adj));
//    ParBilinearForm *m_adj = new ParBilinearForm(fes_adjoint);
//    m_adj->AddDomainIntegrator(new MassIntegrator);

//    ParBilinearForm *k_adj = new ParBilinearForm(fes_adjoint);
//    k_adj->AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(velocity, alpha)));
//    k_adj->AddInteriorFaceIntegrator(new TransposeIntegrator(new NonconservativeDGTraceIntegrator(velocity, alpha)));
//    k_adj->AddBdrFaceIntegrator(new TransposeIntegrator(new NonconservativeDGTraceIntegrator(velocity, alpha)));

//    ParBilinearForm *s_adj = new ParBilinearForm(fes_adjoint);
//    s_adj->AddDomainIntegrator(new DiffusionIntegrator(diff_coef_adj));
//    s_adj->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coef_adj, sigma,
//                                                              kappa_0));
//    s_adj->AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coef_adj, sigma,
//                                                         kappa_0));
//    a_adj->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dt_diff_coef_adj, sigma,
//                                                              kappa_0));
//    a_adj->AddBdrFaceIntegrator(new DGDiffusionIntegrator(dt_diff_coef_adj, sigma, kappa_0));

//    // ParLinearForm *b_adj(fes_adjoint);
//    // b_adj->AddDomainIntegrator(new DomainLFIntegrator(zero));
//    //b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(zero, diff_coef, sigma, kappa));
//    HypreParVector b_adj(fes_adjoint);
//    b_adj = 0.0;

//    //int skip_zeros = 0;
//    a_adj->Assemble();
//    a_adj->Finalize();
//    m_adj->Assemble(skip_zeros);
//    m_adj->Finalize(skip_zeros);
//    k_adj->Assemble(skip_zeros);
//    k_adj->Finalize(skip_zeros);
//    s_adj->Assemble(skip_zeros);
//    s_adj->Finalize(skip_zeros);
//    // b_adj->Assemble();

//    // 19. Define the initial conditions, save the corresponding grid function to
//    //    a file and (optionally) save data in the VisIt format and initialize
//    //    GLVis visualization.
//    GridFunctionCoefficient theta_coeff(theta);
//    FunctionCoefficient lam0(lam0_function);
//    ParGridFunction *lam = new ParGridFunction(fes_adjoint);
//    lam->ProjectCoefficient(theta_coeff);
//    HypreParVector *lam_vec = lam->GetTrueDofs();
//    ParaViewDataCollection *pd_backward = NULL;
//    if (vis_adjoint)
//    {
//       pd_backward = new ParaViewDataCollection("adjoint", pmesh);
//       pd_backward->SetPrefixPath("ParaView");
//       pd_backward->RegisterField("solution-backward", lam);
//       pd_backward->SetLevelsOfDetail(order);
//       pd_backward->SetDataFormat(VTKFormat::BINARY);
//       pd_backward->SetHighOrderOutput(true);
//       pd_backward->SetCycle(0);
//       pd_backward->SetTime(t_final);
//       pd_backward->Save();
//    }

//    // 20. Define the time-dependent evolution operator describing the ODE
//    //    right-hand side, and perform time-integration (looping over the time
//    //    iterations, ti, with a time-step dt).
//    IMEX_Evolution adv_adj(*m_adj, *k_adj, *s_adj, b_adj, *a_adj);

//    real_t t_adj = 0.0;
//    adv_adj.SetTime(t_adj);
//    ode_solver_adj->Init(adv_adj);

//    // int n_steps = (int)ceil(t_final / dt);
//    double dt_real_adj = dt;
//    std::cout << "dt back = " << dt_real_adj << std::endl;
//    //Vector err_vec(n_steps-1);

//    for (int ti = 0; ti < n_steps; ti++)
//    {
//       ode_solver_adj->Step(*lam_vec, t_adj, dt_real_adj);
//       // Vector lam_vals(num_dofs);
//       // Vector theta_values(num_dofs);
//       // const GridFunction* theta_gf = theta_coeff.GetGridFunction();
//       // theta_gf->GetTrueDofs(theta_values);
//       // lam.GetTrueDofs(lam_vals);
//       // theta_coeff = *(new GridFunctionCoefficient(&(theta_gf_vector[n_steps - ti -
//       //                                                                       1])));
//       // b_adj = *(new LinearForm(&fes_adjoint));
//       // b_adj.AddDomainIntegrator(new DomainLFIntegrator(theta_coeff));
//       // b_adj.Assemble();
//       if (ti % vis_steps == 0 || ti == n_steps - 1)
//       {
//          if (Mpi::Root())
//          {
//             cout << "time step: " << ti << ", time: " << t_adj << endl;
//          }
//          *lam = *lam_vec;
//          if (vis_adjoint)
//          {
//             pd_backward->SetCycle(ti);
//             pd_backward->SetTime(t_adj);
//             pd_backward->Save();
//          }
//       }
//    }

//    // 11. Free the used memory.
//    delete pd;
//    delete pd_backward;
//    delete theta_vec;
//    delete lam_vec;
//    delete lam;
//    delete theta;
//    delete a;
//    delete s;
//    delete k;
//    delete m;
//    delete a_adj;
//    delete s_adj;
//    delete k_adj;
//    delete m_adj;
//    delete fes;
//    delete fes_adjoint;
//    delete pmesh;
//    delete dc;
//    delete fec;
//    delete fec_adjoint;

//    return 0;
// }