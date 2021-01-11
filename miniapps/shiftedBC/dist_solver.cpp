#include "nldist.hpp"

double Gyroid(const mfem::Vector &xx)
{
    double rez;
    double pp=4*M_PI;

    mfem::Vector lvec(3);
    lvec=0.0;
    for(int i=0;i<xx.Size();i++)
    {
        lvec[i]=xx[i]*pp;
    }
    rez=sin(lvec[0])*cos(lvec[1])+sin(lvec[1])*cos(lvec[2])+sin(lvec[2])*cos(lvec[0]);
    return rez;
}


double Sph(const mfem::Vector &xx)
{
    double R=0.4;
    mfem::Vector lvec(3);
    lvec=0.0;
    for(int i=0;i<xx.Size();i++)
    {
        lvec[i]=xx[i];
    }

    return lvec[0]*lvec[0]+lvec[1]*lvec[1]+lvec[2]*lvec[2]-R*R;
}

void DGyroid(const mfem::Vector &xx, mfem::Vector &vals)
{
    vals.SetSize(xx.Size());
    vals=0.0;

    double pp=4*M_PI;

    mfem::Vector lvec(3);
    lvec=0.0;
    for(int i=0;i<xx.Size();i++)
    {
        lvec[i]=xx[i]*pp;
    }

    vals[0]=cos(lvec[0])*cos(lvec[1])-sin(lvec[2])*sin(lvec[0]);
    vals[1]=-sin(lvec[0])*sin(lvec[1])+cos(lvec[1])*cos(lvec[2]);
    if(xx.Size()>2)
    {
        vals[2]=-sin(lvec[1])*sin(lvec[2])+cos(lvec[2])*cos(lvec[0]);
    }

    vals*=pp;
}

int main(int argc, char *argv[])
{
    // 1. Initialize MPI
    int num_procs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    std::cout<<"nproc="<<num_procs<<" myrank="<<myrank<<std::endl<<std::flush;

    // 2. Parse command-line options
    const char *mesh_file = "../../data/beam-tet.mesh";
    int ser_ref_levels = 1;
    int par_ref_levels = 1;
    int order = 2;
    bool visualization = true;
    double newton_rel_tol = 1e-4;
    double newton_abs_tol = 1e-6;
    int newton_iter = 10;
    int print_level = 0;


    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&ser_ref_levels,
                      "-rs",
                      "--refine-serial",
                      "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels,
                      "-rp",
                      "--refine-parallel",
                      "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&order,
                      "-o",
                      "--order",
                      "Order (degree) of the finite elements.");
    args.AddOption(&visualization,
                      "-vis",
                      "--visualization",
                      "-no-vis",
                      "--no-visualization",
                      "Enable or disable GLVis visualization.");
    args.AddOption(&newton_rel_tol,
                      "-rel",
                      "--relative-tolerance",
                      "Relative tolerance for the Newton solve.");
    args.AddOption(&newton_abs_tol,
                      "-abs",
                      "--absolute-tolerance",
                      "Absolute tolerance for the Newton solve.");
    args.AddOption(&newton_iter,
                      "-it",
                      "--newton-iterations",
                      "Maximum iterations for the Newton solve.");


    args.Parse();
    if (!args.Good())
    {
        if (myrank == 0)
        {
            args.PrintUsage(std::cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myrank == 0)
    {
        args.PrintOptions(std::cout);
    }

    // 3. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    //    with the same code.
    mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 4. Refine the mesh in serial to increase the resolution. In this example
    //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    //    a command-line parameter.
    for (int lev = 0; lev < ser_ref_levels; lev++)
    {
        mesh->UniformRefinement();
    }


    // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    mfem::ParMesh *pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    for (int lev = 0; lev < par_ref_levels; lev++)
    {
        pmesh->UniformRefinement();
    }

    // 7. Define the finite element spaces for the solution
    mfem::H1_FECollection fecp(order, dim);
    mfem::L2_FECollection fecv(order-1, dim);
    mfem::ParFiniteElementSpace fespacep(pmesh, &fecp, 1, mfem::Ordering::byVDIM);
    mfem::ParFiniteElementSpace fespacev(pmesh, &fecv, dim, mfem::Ordering::byVDIM);
    HYPRE_Int glob_size = fespacep.GlobalTrueVSize();
    if (myrank == 0)
    {
        std::cout << "Number of finite element unknowns: " << glob_size
                  << std::endl;
    }

    // 8. Define the solution vector x as a parallel finite element grid function
    //     corresponding to fespace. Initialize x with initial guess of zero,
    //     which satisfies the boundary conditions.

    mfem::ParGridFunction w(&fespacep);
    mfem::ParGridFunction x(&fespacep);
    x = 1.0;
    mfem::HypreParVector *sv = x.GetTrueDofs();

    mfem::ParNonlinearForm* nf=new mfem::ParNonlinearForm(&fespacep);


    //define the function coefficeints
    mfem::FunctionCoefficient ffc(Gyroid); //signed function
    //mfem::FunctionCoefficient ffc(Sph); //signed function
    //mfem::VectorFunctionCoefficient dfc(dim,DGyroid); //gradient
    //mfem::UGradCoefficient ugc(ffc,dfc);

    //add the integrator
    //nf->AddDomainIntegrator(new mfem::PUMPLaplacian(&uuc,&guc,false));
    nf->AddDomainIntegrator(new mfem::ScreenedPoisson(ffc,0.05));



    //define the solvers
    mfem::HypreBoomerAMG* prec=new mfem::HypreBoomerAMG();
    prec->SetPrintLevel(print_level);

    mfem::GMRESSolver *gmres;
    gmres = new mfem::GMRESSolver(MPI_COMM_WORLD);
    gmres->SetAbsTol(newton_abs_tol/10);
    gmres->SetRelTol(newton_rel_tol/10);
    gmres->SetMaxIter(100);
    gmres->SetPrintLevel(print_level);
    gmres->SetPreconditioner(*prec);

    mfem::NewtonSolver *ns;
    ns = new mfem::NewtonSolver(MPI_COMM_WORLD);

    ns->iterative_mode = true;
    ns->SetSolver(*gmres);
    ns->SetOperator(*nf);
    ns->SetPrintLevel(print_level);
    ns->SetRelTol(newton_rel_tol);
    ns->SetAbsTol(newton_abs_tol);
    ns->SetMaxIter(newton_iter);


    mfem::Vector b; //RHS is zero
    ns->Mult(b, *sv);
    w.SetFromTrueDofs(*sv);

    mfem::PDEFilter* filt= new mfem::PDEFilter(*pmesh, 0.05);
    filt->Filter(ffc,w);


    //w.ProjectCoefficient(ffc);
    mfem::GridFunctionCoefficient wgf(&w);
    mfem::GradientGridFunctionCoefficient gwf(&w);


    //Now we can construct the shape functions
    mfem::Array< mfem::NonlinearFormIntegrator * >* dnfi= nf->GetDNFI();
    delete (*dnfi)[0];
    mfem::PUMPLaplacian* pint=new mfem::PUMPLaplacian(&wgf,&gwf,false);
    (*dnfi)[0]=pint;
    pint->SetPower(2);
    ns->Mult(b, *sv);
    /*
    pint->SetPower(3);
    ns->Mult(b, *sv);
    pint->SetPower(4);
    ns->Mult(b, *sv);
    pint->SetPower(5);
    ns->Mult(b, *sv);
    pint->SetPower(6);
    ns->Mult(b, *sv);
    pint->SetPower(7);
    ns->Mult(b, *sv);
    pint->SetPower(8);
    ns->Mult(b, *sv);
    pint->SetPower(9);
    ns->Mult(b, *sv);
    pint->SetPower(10);
    ns->Mult(b, *sv);
    pint->SetPower(11);
    ns->Mult(b, *sv);
    pint->SetPower(12);
    ns->Mult(b, *sv);
    pint->SetPower(13);
    ns->Mult(b, *sv);
    pint->SetPower(14);
    ns->Mult(b, *sv);
    pint->SetPower(15);
    ns->Mult(b, *sv);
    pint->SetPower(16);
    ns->Mult(b, *sv);
    */

    x.SetFromTrueDofs(*sv);


    mfem::GridFunctionCoefficient gfx(&x);
    mfem::PProductCoefficient tsol(wgf,gfx);
    mfem::ParGridFunction o(&fespacep);
    //o.ProjectCoefficient(tsol);


    const int p = 10;
    mfem::PLapDistanceSolver dsol(p,1,50);
    dsol.DistanceField(ffc,o);


    // 9. Define ParaView DataCollection
    mfem::ParaViewDataCollection *dacol = new mfem::ParaViewDataCollection("Example71",
                                                                  pmesh);

    dacol->SetLevelsOfDetail(order);
    dacol->RegisterField("nsol", &x);
    dacol->RegisterField("tsol", &w);
    dacol->RegisterField("fsol", &o);


    //x.SetFromTrueDofs(*sv);
    dacol->SetTime(1.0);
    dacol->SetCycle(1);
    dacol->Save();


    delete dacol;

    delete filt;
    delete ns;
    delete gmres;
    delete prec;

    delete nf;
    delete sv;

    delete pmesh;

    MPI_Finalize();
}
