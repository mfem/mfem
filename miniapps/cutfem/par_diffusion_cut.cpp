#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "my_integrators.hpp"

using namespace std;
using namespace mfem;

real_t f_rhs(const Vector &x);
real_t u_ex(const Vector &x);
real_t bcf(const Vector &x);
real_t g_neumann(const Vector &x);
void u_grad_exact(const Vector &x, Vector &u);
real_t circle_func(const Vector &x);
real_t ellipsoide_func(const Vector &x);


int main(int argc, char *argv[])
{
    // 1. Initialize MPI and HYPRE.
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 2. Parse command-line options.
    const char *mesh_file = "../data/star.mesh";
    int order = 1;
    bool static_cond = false;
    bool pa = false;
    bool fa = false;
    const char *device_config = "cpu";
    bool visualization = true;
    bool algebraic_ceed = false;
    int ser_ref_levels = 1;
    int aorder = 6; // Algoim integration points
    real_t g = 0.1;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                   "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                   "--no-full-assembly", "Enable Full Assembly.");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");

    args.AddOption(&aorder, "-aorder", "--aorder",
                  "fix.");
    args.AddOption(&g, "-g", "--g",
                  "fix.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }
    //Mesh mesh(mesh_file, 1, 1);

    Mesh mesh = Mesh::MakeCartesian2D( 2*2, 2 , mfem::Element::Type::QUADRILATERAL, true, 1, 0.5);

    int dim = mesh.Dimension();
    {
        // int ref_levels =
        //     (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
        for (int l = 0; l < ser_ref_levels; l++)
        {
            mesh.UniformRefinement();
        }
    }

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();
    {
        int par_ref_levels = 1;
        for (int l = 0; l < par_ref_levels; l++)
        {
            pmesh.UniformRefinement();
        }
    }

    std::cout<<"id="<<myid<<" "<<pmesh.GetNE()<<" "<<pmesh.GetNumFaces()
            <<" "<<pmesh.GetNSharedFaces()<<std::endl; std::cout.flush();

    double h_min, h_max, kappa_min, kappa_max;
    pmesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

    FiniteElementCollection *fec;
    fec = new H1_FECollection(order, dim);
    ParFiniteElementSpace fespace(&pmesh, fec);
    HYPRE_BigInt size = fespace.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of finite element unknowns: " << size << endl;
    }

    ConstantCoefficient one(1.0);

    ParGridFunction x(&fespace);
    FunctionCoefficient bc (bcf);
    FunctionCoefficient f (f_rhs);
    x.ProjectCoefficient(bc);
    FunctionCoefficient neumann(g_neumann);

    // level set function
    ParGridFunction cgf(&fespace);
    FunctionCoefficient circle(ellipsoide_func);
    cgf.ProjectCoefficient(circle);

    // mark elements and outside DOFs
    Array<int> boundary_dofs;
    fespace.GetBoundaryTrueDofs(boundary_dofs);
    Array<int> outside_dofs;
    Array<int> marks;
    Array<int> face_marks;
    {
        ParElementMarker* elmark=new ParElementMarker(pmesh,true,true);
        elmark->SetLevelSetFunction(cgf);
        elmark->MarkElements(marks);
        elmark->MarkGhostPenaltyFaces(face_marks);
        elmark->ListEssentialTDofs(marks,fespace,outside_dofs);
        delete elmark;
    }

    outside_dofs.Append(boundary_dofs);
    outside_dofs.Sort();
    outside_dofs.Unique();


    std::cout<<"myid="<<myid<<" marks_size="<<marks.Size()<<std::endl;
    std::cout.flush();

    int otherorder = 2;
    AlgoimIntegrationRules* air=new AlgoimIntegrationRules(aorder,circle,otherorder);
    real_t gp = g/(h_min*h_min);

    ParLinearForm b(&fespace);

    b.AddDomainIntegrator(new CutDomainLFIntegrator(f,&marks,air));
    b.AddDomainIntegrator(new CutUnfittedBoundaryLFIntegrator(neumann,&marks,air));
    b.Assemble();

    // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
    ParBilinearForm a(&fespace);
    a.AddDomainIntegrator(new CutDiffusionIntegrator(one,&marks,air,false));
    // something is wrong with the CutGhostPenaltyIntegrator
    a.AddInteriorFaceIntegrator(new CutGhostPenaltyIntegrator(gp,&face_marks));
    a.Assemble();

    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(outside_dofs, x, b, A, X, B);

    Solver *prec = nullptr;
    prec = new HypreBoomerAMG;
    CGSolver cg(MPI_COMM_WORLD);
    cg.SetRelTol(1e-16);
    cg.SetMaxIter(6000);
    cg.SetPrintLevel(1);
    if (prec) { cg.SetPreconditioner(*prec); }
    cg.SetOperator(*A);
    cg.Mult(B, X);
    delete prec;

    a.RecoverFEMSolution(X, b, x);

    FunctionCoefficient uex (u_ex);
   ParGridFunction exact_sol(&fespace);
   exact_sol.ProjectCoefficient(uex);

    // to visualize level set and markings
    L2_FECollection* l2fec= new L2_FECollection(0,pmesh.Dimension());
    ParFiniteElementSpace* l2fes= new ParFiniteElementSpace(&pmesh,l2fec,1);
    ParGridFunction mgf(l2fes); mgf=0.0;
    ParGridFunction par(l2fes); par=0.0;
    for(int i=0;i<marks.Size();i++){
        mgf[i]=marks[i];
        par[i]=myid;
    }


   {
       ParNonlinearForm* nf=new ParNonlinearForm(&fespace);
       nf->AddDomainIntegrator(new CutScalarErrorIntegrator(uex,&marks,air));
        real_t error_squared = nf->GetEnergy(x.GetTrueVector());
       cout << "\n|| u_h - u ||_{L^2} = " << sqrt(error_squared)<< std::endl;

       delete nf;
   }
    cout << "h: " << h_min<< std::endl;



   ParGridFunction error(&fespace);
   error = x;
   error -= exact_sol;

    ParaViewDataCollection paraview_dc("diffusion_cut", &pmesh);
    paraview_dc.SetPrefixPath("ParaView");
    paraview_dc.SetLevelsOfDetail(order);
    paraview_dc.SetCycle(0);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    paraview_dc.SetHighOrderOutput(true);
    paraview_dc.SetTime(0.0); // set the time
    paraview_dc.RegisterField("solution",&x);
    paraview_dc.RegisterField("marks", &mgf);
    paraview_dc.RegisterField("parts", &par);
           paraview_dc.RegisterField("level_set",&cgf);
   paraview_dc.RegisterField("error",&error);
    paraview_dc.Save();


    delete l2fes;
    delete l2fec;
    delete air;
    delete fec;

    return 0;
}

real_t f_rhs(const Vector &x)
{

    return  2*M_PI*M_PI*(sin(M_PI*x(0))*cos((M_PI*x(1))));
}


real_t u_ex(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.5;
    return sin(M_PI*x(0))*cos((M_PI*x(1)));
}

real_t bcf(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.5;
    return 0;//sin(M_PI*x(0))*cos((M_PI*x(1)));
}

void u_grad_exact(const Vector &x, Vector &u)
{
    u(0) =M_PI*cos(M_PI*x(0)) *cos(M_PI* x(1));
    u(1) =  -M_PI*sin(M_PI*x(0)) * sin(M_PI* x(1));
}


real_t g_neumann(const Vector &x)
{
    real_t a = 1.5;
    real_t b = 0.5;
    real_t x0 = 0.5;
    real_t y0 = 0.25;
    real_t xx = x(0)-x0;
    real_t y = x(1)-y0;
    real_t normalize = sqrt((xx*xx)/(a*a*a*a) + y*y/(b*b*b*b));
    return M_PI*cos(M_PI*x(0)) *cos(M_PI* x(1))*xx/(a*a*normalize) -M_PI*sin(M_PI*x(0)) * sin(M_PI* x(1))*y/(b*b*normalize);
}

real_t circle_func(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.5;
    real_t r = 0.4;
    return -(x(0)-x0)*(x(0)-x0) - (x(1)-y0)*(x(1)-y0) + r*r;
}

real_t ellipsoide_func(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.25;
    real_t r = 0.351;
    real_t xx = x(0)-x0;
    real_t y = x(1)-y0;
    return -(xx)*(xx)/(1.5*1.5) - (y)*(y)/(0.5*0.5)+ r*r; // + 0.25*cos(atan2(x(1)-y0,x(0)-x0))*cos(atan2(x(1)-y0,x(0)-x0));
}
