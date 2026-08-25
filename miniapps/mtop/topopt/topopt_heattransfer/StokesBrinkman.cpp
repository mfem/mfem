#include "mfem.hpp"
#include "TopOptDesignSolvers.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>


using namespace std;
using namespace mfem;

real_t f_natural(const Vector & x);

real_t simple_init_design(const Vector &x)    
{    
   //return 0.5;  
//    return 0.5 + 0.4 * std::sin(M_PI*x(0)) * std::cos(M_PI*x(1));  
   if (x(0) > 0.4 && x(0) < 0.6 && x(1) > 0.4 && x(1) < 0.6)  
   {    
      return 0.0;   
   }
   else
   { 
      return 1.0;
   }  
}
  
bool InitializeDesign(ParGridFunction &rho, real_t x_max, real_t y_max)       
{
   // GaussianDesignCoefficient gaussian(x_max/2.0, y_max/2.0,  
   //                                       0.25*x_max, 0.25*y_max,    
   //                                       0.10, 1.0); 
   FunctionCoefficient one(simple_init_design);  
   rho.ProjectCoefficient(one);   
   return true;
}


int main(int argc, char *argv[]) 
{
    // 1. Initialize MPI and HYPRE.
    Mpi::Init();  
    int num_procs = Mpi::WorldSize();   
    const MPI_Comm comm = MPI_COMM_WORLD;     
    int myid = Mpi::WorldRank();                  
    Hypre::Init();  

    const char *mesh_file = "../../../../data/inline-quad.mesh";   
    int ser_ref_levels = 1;
    int par_ref_levels = 1;    
    int order = 2; 
    real_t dynamic_viscosity = 1.0;
    bool visualization = true;
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
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",  
                        "--no-visualization", 
                        "Enable or disable Visualization"); 
    args.AddOption(&dynamic_viscosity, "-dv", "--dynamic-viscosity",
                        "Dynamic Viscosity of the Fluid.");  
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

    Device device(device_config);
    if (myid == 0) { device.Print(); }

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

    // 6. FE Collections for pressure and velocity spaces, using taylor hood elements for now
    FiniteElementCollection *v_coll(new H1_FECollection(order, dim));
    FiniteElementCollection *p_coll(new H1_FECollection(order-1, dim));

    
    ParFiniteElementSpace *V_space = new ParFiniteElementSpace(pmesh, v_coll, 2);
    ParFiniteElementSpace *P_space = new ParFiniteElementSpace(pmesh, p_coll);

    HYPRE_BigInt dimV = V_space->GlobalTrueVSize();
    HYPRE_BigInt dimP = P_space->GlobalTrueVSize();

    if(Mpi::Root)
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(V) = " << dimV << "\n";
        std::cout << "dim(P) = " << dimP << "\n";
        std::cout << "dim(V+P) = " << dimV + dimP << "\n";
        std::cout << "***********************************************************\n";
    }

    H1_FECollection filter_fec(order, dim); 
    L2_FECollection control_fec(order-1, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace filter_fes(pmesh, &filter_fec);  
    ParFiniteElementSpace control_fes(pmesh, &control_fec);  
    
    ParGridFunction rho(&control_fes);  
    ParGridFunction rho_tilde(&filter_fes); 
    if (!InitializeDesign(rho, 1.0, 1.0))  
    { 
        if (myid == 0)
        {
            cerr << "Error: unknown -init value. Use uniform, solid, void, or gaussian.\n";        
        }
        return 1;
    } 

    toopt::PDEFilterOptions filter_opts;
    filter_opts.print_level = 0; 
    // filter_opts.solver_rtol = 1e-12;
    filter_opts.filter_radius = 0.05; 
    toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);     
    filter.Assemble();   
    filter.Mult(rho, rho_tilde);    
    rho_tilde.ExchangeFaceNbrData();

    BrinkmanCoefficient brink_coeff(&rho_tilde, 0.5);  
    ProductCoefficient b_coeff(10000.0, brink_coeff);


    
    // 7. Define the two BlockStructure of the problem.
    Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = V_space->GetVSize();
    block_offsets[2] = P_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(3); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = V_space->TrueVSize();
    block_trueOffsets[2] = P_space->TrueVSize();
    block_trueOffsets.PartialSum();

    // 8. coeffs
    ConstantCoefficient visc_coeff(dynamic_viscosity);
    ConstantCoefficient zero(0.0);
    ConstantCoefficient one(1.0);
    Vector vec_one(V_space->GetVSize());
    vec_one = 1.0;
    VectorConstantCoefficient vone(vec_one);

    FunctionCoefficient fnatcoeff(f_natural);


    Vector vec_zero(V_space->GetVSize());
    vec_zero = 0.0;
    VectorConstantCoefficient vzero(vec_zero);

    // 9. Block operator
    ParBilinearForm *a(new ParBilinearForm(V_space)); 
    ParMixedBilinearForm *b(new ParMixedBilinearForm(V_space, P_space));

    HypreParMatrix *A = NULL;
    HypreParMatrix *B = NULL;

    a->AddDomainIntegrator(new VectorMassIntegrator(b_coeff));
    a->AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
    a->Assemble();
    a->Finalize();


    b->AddDomainIntegrator(new VectorDivergenceIntegrator);
    b->Assemble();
    b->Finalize();


    BlockOperator *BrinkStokesOp = new BlockOperator(block_trueOffsets);

    Array<int> empty_tdof_list;  // empty
    OperatorPtr opA, opB;


    TransposeOperator *Bt = NULL;

    A = a->ParallelAssemble();
    B = b->ParallelAssemble();
    (*B) *= -1;
    Bt = new TransposeOperator(B);
    BrinkStokesOp->SetBlock(0,0, A);
    BrinkStokesOp->SetBlock(0,1, Bt);
    BrinkStokesOp->SetBlock(1,0, B);

    // 10. RHS
    MemoryType mt = device.GetMemoryType();
    BlockVector x(block_offsets, mt), rhs(block_offsets, mt);
    BlockVector trueX(block_trueOffsets, mt), trueRhs(block_trueOffsets, mt);

    ParLinearForm *fform(new ParLinearForm);
    fform->Update(V_space, rhs.GetBlock(0), 0);
    fform->AddDomainIntegrator(new VectorDomainLFIntegrator(vzero));
    fform->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(fnatcoeff));
    fform->Assemble();
    fform->SyncAliasMemory(rhs);
    fform->ParallelAssemble(trueRhs.GetBlock(0));
    trueRhs.GetBlock(0).SyncAliasMemory(trueRhs);

    ParLinearForm *gform(new ParLinearForm);
    gform->Update(P_space, rhs.GetBlock(1), 0);
    gform->AddDomainIntegrator(new DomainLFIntegrator(zero));
    gform->Assemble();
    gform->SyncAliasMemory(rhs);
    gform->ParallelAssemble(trueRhs.GetBlock(1));
    trueRhs.GetBlock(1).SyncAliasMemory(trueRhs);

    // 11. Construct the operators for preconditioner
    // HypreParMatrix *AinvBt = NULL;
    // HypreParVector *Ad = NULL;
    // HypreParMatrix *S = NULL; 
    // Solver *invA, *invS;

    // Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(),
    //                         A->GetRowStarts());
    // A->GetDiag(*Ad);
    // AinvBt = B->Transpose();
    // AinvBt->InvScaleRows(*Ad);
    // S = ParMult(B, AinvBt);
    // invA = new HypreDiagScale(*A);
    // invS = new HypreBoomerAMG(*S);

    // invA->iterative_mode = false;
    // invS->iterative_mode = false;

    // BlockDiagonalPreconditioner *StokesBrinkmanPr = new BlockDiagonalPreconditioner(
    //     block_trueOffsets);
    // StokesBrinkmanPr->SetDiagonalBlock(0, invA);
    // StokesBrinkmanPr->SetDiagonalBlock(1, invS);

    // 12. Solve the linear system with MINRES.
    //     Check the norm of the unpreconditioned residual.
    int maxIter(2000);
    real_t rtol(1.e-6);
    real_t atol(1.e-10);
    MINRESSolver solver(MPI_COMM_WORLD);
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(maxIter);
    solver.SetOperator(*BrinkStokesOp); 
    //solver.SetPreconditioner(*StokesBrinkmanPr);
    solver.SetPrintLevel(1);
    trueX = 0.0;
    solver.Mult(trueRhs, trueX);
    if (device.IsEnabled()) { trueX.HostRead(); }

    if(Mpi::Root())
    {
        if (solver.GetConverged())
        {
            std::cout << "MINRES converged in " << solver.GetNumIterations() << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
        }
        else
        {
            std::cout << "MINRES did not converge in " << solver.GetNumIterations() << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
        }
    }

    ParGridFunction *u(new ParGridFunction);
    ParGridFunction *p(new ParGridFunction);
    u->MakeRef(V_space, x.GetBlock(0), 0);
    p->MakeRef(P_space, x.GetBlock(1), 0);
    u->Distribute(&(trueX.GetBlock(0)));
    p->Distribute(&(trueX.GetBlock(1)));


    
   // 19. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *pmesh << *u << "window_title 'Velocity'"
             << endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << num_procs << " " << myid << "\n";
      p_sock.precision(8);
      p_sock << "solution\n" << *pmesh << *p << "window_title 'Pressure'"
             << endl;
   }



    // Free the used memory.  
    // delete pd;
    delete fform;
    delete gform;
    delete u;
    delete p;
    delete BrinkStokesOp;
    // delete StokesBrinkmanPr;
    // delete invA;
    // delete invS;
    // delete S;
    // delete Ad;
    // delete AinvBt;
    delete Bt;
    delete B;
    delete A;
    delete a;
    delete b;
    delete P_space;
    delete V_space;
    delete p_coll;
    delete v_coll;
    delete pmesh;
    
    return 0; 
}


real_t f_natural(const Vector & x)
{
   return -exp(x(0))*sin(x(1));
}