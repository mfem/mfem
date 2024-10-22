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
real_t g_neumann3d(const Vector &x);
real_t circle_func(const Vector &x);
real_t ellipsoide_func(const Vector &x);
real_t sphere_func(const Vector &x);
real_t new_func(const Vector &xx);
real_t new_func3d(const Vector &xx);
real_t new_func2d(const Vector &xx);
real_t koeff(const Vector &x);


// solves the diffusion problem Delta u = f, with either Dirichlet conditions weakly imposed, 
// or neumann + Dirichlet conditions 
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
    bool visualization = false;
    bool visualization_paraview = true;
    bool algebraic_ceed = false;
    int ser_ref_levels = 1;
    int aorder = 2; // Algoim integration points
    real_t g = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc", 
                   "--no-static-condensation", "Enable static condensation.");  //these three not valid options now
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                   "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                   "--no-full-assembly", "Enable Full Assembly.");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&visualization_paraview, "-vispv", "--visualizationpv", "-no-vispv",
                   "--no-visualizationpv",
                   "Enable or disable ParaView visualization.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&aorder, "-ao", "--aorder",
                  "Set order for alogim integration");
    args.AddOption(&g, "-g", "--ghost penalty constant",
                  "Ghost penalty constant");
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

    //Mesh mesh = Mesh::MakeCartesian2D(2, 2, mfem::Element::Type::QUADRILATERAL, true, 1, 1);
    Mesh mesh = Mesh::MakeCartesian3D( 4, 4 ,4, mfem::Element::Type::HEXAHEDRON, 2.2, 2.2,2.2 );


    int dim = mesh.Dimension();
    {
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
    // FunctionCoefficient coeff(koeff);
    ParGridFunction x(&fespace);
    FunctionCoefficient bc (u_ex);
    FunctionCoefficient f (f_rhs);
    x.ProjectCoefficient(bc);
    FunctionCoefficient neumann(g_neumann3d);

    // level set function
    ParGridFunction lsgf(&fespace);
    FunctionCoefficient level_set(sphere_func);
    lsgf.ProjectCoefficient(level_set);

    // mark elements and outside DOFs
    Array<int> boundary_dofs;
    // fespace.GetBoundaryTrueDofs(boundary_dofs);
    Array<int> outside_dofs;
    Array<int> marks;
    Array<int> face_marks;
    {
        ParElementMarker* elmark=new ParElementMarker(pmesh,true,true);
        elmark->SetLevelSetFunction(lsgf);
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
    AlgoimIntegrationRules* air=new AlgoimIntegrationRules(aorder,level_set,otherorder);
    real_t gp = g/(h_min*h_min);
    real_t lambda = 10/h_min;

    ParLinearForm b(&fespace);

    b.AddDomainIntegrator(new CutDomainLFIntegrator(f,&marks,air));
    // b.AddDomainIntegrator(new CutUnfittedBoundaryLFIntegrator(neumann,&marks,air)); //when neumann condition 
    b.AddDomainIntegrator(new  CutUnfittedNitscheLFIntegrator(bc,one,lambda,&marks,air));

    b.Assemble();

    ParBilinearForm a(&fespace);
    a.AddDomainIntegrator(new CutDiffusionIntegrator(one,&marks,air,false));
    a.AddInteriorFaceIntegrator(new CutGhostPenaltyIntegrator(gp,&face_marks));
    a.AddDomainIntegrator(new CutNitscheIntegrator(one,lambda,&marks,air));

    a.Assemble();

    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(outside_dofs, x, b, A, X, B);

    Solver *prec = nullptr;
    prec = new HypreBoomerAMG;
    CGSolver cg(MPI_COMM_WORLD);
    cg.SetRelTol(1e-20);
    cg.SetMaxIter(1500);
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
        if (myid==0)
        {
       cout << "\n|| u_h - u ||_{L^2} = " << sqrt(error_squared)<< std::endl;
        cout << "h: " << h_min<< std::endl;
        }
       delete nf;
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }



    if (visualization_paraview)
    {

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
        paraview_dc.RegisterField("u",&x);
        paraview_dc.RegisterField("marks", &mgf);
        paraview_dc.RegisterField("parts", &par);
        paraview_dc.RegisterField("level_set",&lsgf);
        paraview_dc.RegisterField("error",&error);
        paraview_dc.RegisterField("u_ex",&exact_sol);
        paraview_dc.Save();
    }

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
    return cos(M_PI*x(0))*cos((M_PI*x(1)))+ sin(M_PI*x(0))*sin((M_PI*x(1)));;//sin(M_PI*x(0))*cos((M_PI*x(1)));
}


real_t g_neumann(const Vector &xx)
{    
    real_t x = xx(0);
    real_t y = xx(1);

    real_t dldx =- M_PI*(4*sin(x*2* M_PI)*cos(x*2* M_PI) +cos(x*M_PI/2)/2 );
    real_t dldy = 1 ;

    real_t dudx = M_PI*(-sin(M_PI*xx(0)) * cos(M_PI* xx(1)) + cos(M_PI*xx(0)) * sin(M_PI* xx(1)));
    real_t dudy = M_PI*( - cos(M_PI*xx(0)) * sin(M_PI* xx(1)) + sin(M_PI*xx(0)) * cos(M_PI* xx(1)));

    real_t normalize = sqrt(dldx*dldx + dldy*dldy);
   return dudx *dldx/normalize + dudy *dldy/normalize;
}



real_t g_neumann3d(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0;
    real_t z0 = 0.5;
    real_t xx = x(0)-x0;
    real_t y = x(1)-y0;
    real_t z = x(2)-z0;
    real_t normalize = sqrt(xx*xx + y*y + z*z);
    return M_PI*cos(M_PI*x(0)) *cos(M_PI* x(1))*cos((x(2)))*xx/(normalize) -M_PI*sin(M_PI*x(0)) * sin(M_PI* x(1))*cos((x(2)))*y/(normalize)- sin(M_PI*x(0)) * cos(M_PI* x(1))*sin((x(2)))*z/(normalize);
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


real_t sphere_func(const Vector &x)
{
    real_t x0 = 0.5;
    real_t y0 = 0.5;
    real_t z0 = 0.5;
    real_t r = 0.4;
    return -(x(0)-x0)*(x(0)-x0) - (x(1)-y0)*(x(1)-y0)  - (x(2)-z0)*(x(2)-z0) + r*r;
}


real_t new_func(const Vector &xx)
{

    real_t x = xx(0);
    real_t y = xx(1);

    return sin(x*2* M_PI)*sin(x*2* M_PI)+sin(x*M_PI/2)-y;

}

real_t new_func3d(const Vector &xx)
{

    real_t x = xx(0)-1;
    real_t y = xx(1)-1;
    real_t z = xx(2)-1;
    real_t r = 0.5;
    real_t r0 = 3.5;

    return -(sqrt(x*x + y*y + z*z) - r + r/r0*cos(5*atan2(y,x))*cos(M_PI*z));

}


real_t new_func2d(const Vector &xx)
{

    real_t x = xx(0)-1.1;
    real_t y = xx(1)-1.1;
    real_t r = 0.6;
    real_t r0 = 0.2;
// - r0*cos(atan2(y,x)))
    return -(sqrt(x*x + y*y) - r- r0*cos(5*atan2(y,x)));

}

// real_t f_rhs(const Vector &x) //koeff
// {
//    return sin(x(0)) * sin( x(1)) + (2*x(0)+4)*cos(x(0))*sin(x(1));
// }


real_t koeff(const Vector &x)
{
   return x(0) + 2; 
}

// real_t u_ex(const Vector &x)
// {
//     return cos(x(0))*sin(x(1));
// }
