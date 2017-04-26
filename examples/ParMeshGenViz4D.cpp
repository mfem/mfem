/*
 * A place holder for any temporary tests in parelag setup
 */

//                                MFEM(with 4D elements) CFOSLS for 3D/4D hyperbolic equation
//                                  with mesh generator and visualization
//
// Compile with: make
//
// Sample runs:  ./HybridHdivL2 -dim 3 or ./HybridHdivL2 -dim 4
//
// Description:  This example code solves a simple 3D/4D hyperbolic problem over [0,1]^3(4)
//               corresponding to the saddle point system
//                                  sigma_1 = u * b
//							 		sigma_2 - u        = 0
//                                  div_(x,t) sigma    = f
//                       with b = vector function (~velocity),
//						 NO boundary conditions (which work only in case when b * n = 0 pointwise at the domain space boundary)
//						 and initial condition:
//                                  u(x,0)            = 0
//               Here, we use a given exact solution
//                                  u(xt) = uFun_ex(xt)
//               and compute the corresponding r.h.s.
//               We discretize with Raviart-Thomas finite elements (sigma), continuous H1 elements (u) and
//					  discontinuous polynomials (mu) for the lagrange multiplier.
//
//				 If you want to run your own solution, be sure to change uFun_ex, as well as fFun_ex and check
//				 that the bFun_ex satisfies the condition b * n = 0 (see above).
// Solver: MINRES preconditioned by boomerAMG or ADS

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

void f_Func(const Vector &p, Vector &f);
void u_Func(const Vector &p, Vector &u);
void Hdivtest_fun(const Vector& xt, Vector& out );

int main(int argc, char *argv[])
{
    int num_procs, myid;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);
    bool solve_problem = 0; // if true, solves a model problem
    bool visualization = 0; // if true, created VTK output for paraview
    bool convert_to_mesh = 0; // if true, converts the pmesh to a serial mesh and prints it out

    if (verbose)
    {
        std::cerr << "Started example for parallel mesh generator" << std::endl;
    }

    int nDimensions     = 4;

    int ser_ref_levels  = 0;
    int par_ref_levels  = 1;
    int Nsteps          = 2;   // number of time slabs (e.g. Nsteps = 2 corresponds to 3 levels: t = 0, t = tau, t = 2 * tau
    double tau          = 0.5; // time step


    int generate_frombase   = 1; // if 0, read mesh from file; if 1, read base mesh from file and extend it to (d+1)-mesh
    int generate_parallel   = generate_frombase * 1; // 0 for serial mesh extension, 1 for parallel
    int whichparallel       = generate_parallel * 2; // default value is 2 (doesn't us qhull)
    int bnd_method          = 0; // default value is 0
    int local_method        = 2; // default value is 2

    //const char *mesh_file = "../build3/meshes/cube_3d_moderate.mesh";
    //const char *mesh_file = "../build3/meshes/square_2d_moderate.mesh";

    //const char *mesh_file = "../build3/meshes/cube4d_low.MFEM";
    //const char *mesh_file = "./data/cube4d.MFEM";
    const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../build3/mesh_par1_id0_np_1.mesh";
    //const char *mesh_file = "../build3/mesh_par1_id0_np_2.mesh";
    //const char *mesh_file = "../build3/meshes/tempmesh_frompmesh.mesh";
    //const char *mesh_file = "../build3/meshes/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../build3/meshes/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../build3/meshes/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../build3/meshes/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../build3/meshes/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../build3/meshes/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../build3/meshes/beam-tet.mesh";
    //const char * meshbase_file = "../build3/meshes/escher-p3.mesh";
    //const char * meshbase_file = "./data/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "./data/orthotope3D_fine.mesh";
    const char * meshbase_file = "../data/cube_3d_moderate.mesh";
    //const char * meshbase_file = "./data/cube_3d_moderate.mesh";
    //const char * meshbase_file = "./data/square_2d_moderate.mesh";
    //const char * meshbase_file = "../build3/meshes/square_2d_fine.mesh";
    //const char * meshbase_file = "../build3/meshes/square-disc.mesh";
    //const char *meshbase_file = "dsadsad";
    //const char * meshbase_file = "../build3/meshes/circle_fine_0.1.mfem";
    //const char * meshbase_file = "../build3/meshes/circle_moderate_0.2.mfem";

    int feorder         = 0; // in 4D cannot use feorder > 0

    if (verbose)
        std::cerr << "Parsing input options" << std::endl;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&meshbase_file, "-mbase", "--meshbase",
                   "Mesh base file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements 4d mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    args.AddOption(&Nsteps, "-nstps", "--nsteps",
                   "Number of time steps.");
    args.AddOption(&tau, "-tau", "--tau",
                   "Time step.");
    args.AddOption(&generate_frombase, "-gbase", "--genfrombase",
                   "Generating mesh from the base mesh.");
    args.AddOption(&generate_parallel, "-gp", "--genpar",
                   "Generating mesh in parallel.");
    args.AddOption(&whichparallel, "-pv", "--parver",
                   "Version of parallel algorithm.");
    args.AddOption(&bnd_method, "-bnd", "--bndmeth",
                   "Method for generating boundary elements.");
    args.AddOption(&local_method, "-loc", "--locmeth",
                   "Method for local mesh procedure.");
    args.Parse();
    if (!args.Good())
    {
       if (verbose)
       {
          args.PrintUsage(std::cerr);
       }
       if (verbose)
           std::cerr << "Bad input arguments" << std:: endl;
       MPI_Finalize();
       return 1;
    }
    if (verbose)
    {
       args.PrintOptions(std::cerr);
    }

    if (verbose)
        std::cerr << "Number of mpi processes: " << num_procs << endl << flush;

#ifndef WITH_QHULL
    if (verbose)
    {
        std::cerr << "WITH_QHULL flag is not set -> local method = 2 must be used" << endl;
    }
    if (local_method !=2 )
    {
        if (verbose)
            std::cerr << "Wrong local_method is provided." << endl;
        MPI_Finalize();
        return 0;
    }

#endif

    StopWatch chrono;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    if (nDimensions == 3 || nDimensions == 4)
    {
        if ( generate_frombase == 1 )
        {
            if ( verbose )
                std::cerr << "Creating a " << nDimensions << "d mesh from a " <<
                        nDimensions - 1 << "d mesh from the file " << meshbase_file << endl;

            Mesh * meshbase;
            ifstream imesh(meshbase_file);
            if (!imesh)
            {
                 cerr << "\nCan not open mesh file for base mesh: " <<
                                                    meshbase_file << endl << flush;
                 MPI_Finalize();
                 return -2;
            }
            meshbase = new Mesh(imesh, 1, 1);
            imesh.close();

            meshbase->MeshCheck(verbose);

            for (int l = 0; l < ser_ref_levels; l++)
                meshbase->UniformRefinement();

            /*
            if ( verbose )
            {
                std::stringstream fname;
                fname << "mesh_" << nDimensions - 1 << "dbase.mesh";
                std::ofstream ofid(fname.str().c_str());
                ofid.precision(8);
                meshbase->Print(ofid);

            }
            */

            if (verbose)
                meshbase->PrintInfo();


            if (generate_parallel == 1) //parallel version
            {
                ParMesh * pmeshbase = new ParMesh(comm, *meshbase);

                /*
                std::stringstream fname;
                fname << "pmesh_"<< nDimensions - 1 << "dbase_" << myid << ".mesh";
                std::ofstream ofid(fname.str().c_str());
                ofid.precision(8);
                pmesh3dbase->Print(ofid);
                */

                chrono.Clear();
                chrono.Start();

                if ( whichparallel == 1 )
                {
                    if ( nDimensions == 3)
                    {
                        if  (myid == 0)
                            std::cerr << "Not implemented for 2D->3D. Use parallel version2"
                                    " instead" << endl << flush;
                        MPI_Finalize();
                        return 0;
                    }
                    else // nDimensions == 4
                    {
                        mesh = new Mesh( comm, *pmeshbase, tau, Nsteps, bnd_method, local_method);
                        if ( myid == 0)
                            std::cerr << "Success: ParMesh is created by deprecated method"
                                 << endl << flush;

                        std::stringstream fname;
                        fname << "mesh_par1_id" << myid << "_np_" << num_procs << ".mesh";
                        std::ofstream ofid(fname.str().c_str());
                        ofid.precision(8);
                        mesh->Print(ofid);

                        MPI_Barrier(comm);
                    }
                }
                else
                {
                    if (myid == 0)
                        std::cerr << "Starting parallel \"" << nDimensions-1 << "D->"
                             << nDimensions <<"D\" mesh generator" << endl;

                    pmesh = make_shared<ParMesh>( comm, *pmeshbase, tau, Nsteps,
                                                  bnd_method, local_method);

                    if ( myid == 0)
                        std::cerr << "Success: ParMesh created" << endl << flush;
                    MPI_Barrier(comm);
                }

                chrono.Stop();
                if (myid == 0 && whichparallel == 2)
                    std::cerr << "Timing: Space-time mesh extension done in parallel in "
                              << chrono.RealTime() << " seconds.\n" << endl << flush;
                delete pmeshbase;
            }
            else // serial version
            {
                if (myid == 0)
                    std::cerr << "Starting serial \"" << nDimensions-1 << "D->"
                         << nDimensions <<"D\" mesh generator" << endl;
                mesh = new Mesh( *meshbase, tau, Nsteps, bnd_method, local_method);
                if ( myid == 0)
                    std::cerr << "Timing: Space-time mesh extension done in serial in "
                              << chrono.RealTime() << " seconds.\n" << endl << flush;
            }

            delete meshbase;

        }
        else // not generating from a lower dimensional mesh
        {
            std::cerr << "Reading a " << nDimensions << "d mesh from the file " << mesh_file << endl;
            ifstream imesh(mesh_file);
            if (!imesh)
            {
                 std::cerr << "\nCan not open mesh file: " << mesh_file << '\n' << std::endl;
                 MPI_Finalize();
                 return -2;
            }
            else
            {
                mesh = new Mesh(imesh, 1, 1);
                imesh.close();
            }

        }

    }
    else //if nDimensions is no 3 or 4
    {
        if (myid == 0)
            cerr << "Case nDimensions = " << nDimensions << " is not supported"
                 << endl << flush;
        MPI_Finalize();
        return -1;

    }

    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        // Checking that mesh is legal = domain and boundary volume + checking boundary elements and faces consistency
        if (myid == 0)
            std::cerr << "Checking the mesh" << endl << flush;
        mesh->MeshCheck(verbose);

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if ( verbose )
            std::cerr << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    for (int l = 0; l < par_ref_levels; l++)
    {
       pmesh->UniformRefinement();
    }

    // if true, converts a pmesh to a mesh (so a global mesh will be produced on each process)
    // which can be printed in a file (as a whole)
    // can be useful for testing purposes
    if (convert_to_mesh)
    {
        int * partitioning = new int [pmesh->GetNE()];
        Mesh * convertedpmesh = new Mesh (*pmesh.get(), &partitioning);
        if (verbose)
        {
            std::stringstream fname;
            fname << "converted_pmesh.mesh";
            std::ofstream ofid(fname.str().c_str());
            ofid.precision(8);
            convertedpmesh->Print(ofid);
        }
    }


    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cerr); if(verbose) std::cerr << endl;

    if (verbose)
        std::cerr << "Mesh generator was called successfully" << endl;

    // solving a model problem in Hdiv if solve_problem = true
    if (solve_problem)
    {
        int dim = nDimensions;
        int order = feorder;
        // taken from ex4D_RT

        FiniteElementCollection *fec;
        if(dim==4) fec = new RT0_4DFECollection;
        else fec = new RT_FECollection(order,dim);
        ParFiniteElementSpace fespace(pmesh.get(), fec);

        int dofs = fespace.GlobalTrueVSize();
        if(verbose) std::cerr << "dofs: " << dofs << endl;

        chrono.Clear(); chrono.Start();

        VectorFunctionCoefficient f(dim, f_Func);
        ParLinearForm b(&fespace);
        b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
        b.Assemble();

        ParGridFunction x(&fespace);
        VectorFunctionCoefficient u_exact(dim, u_Func);
        x = 0.0;


        ParBilinearForm a(&fespace);
        a.AddDomainIntegrator(new DivDivIntegrator);
        a.AddDomainIntegrator(new VectorFEMassIntegrator);
        a.Assemble();
        if(pmesh->bdr_attributes.Size())
        {
           Array<int> ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr = 1;
           x.ProjectCoefficient(u_exact);
           a.EliminateEssentialBC(ess_bdr, x, b);
        }
        a.Finalize();

        chrono.Stop();
        if(verbose) std::cerr << "Assembling took " << chrono.UserTime() << "s." << endl;

        HypreParMatrix *A = a.ParallelAssemble();
        HypreParVector *B = b.ParallelAssemble();
        HypreParVector *X = x.ParallelAverage();
        *X = 0.0;


        chrono.Clear(); chrono.Start();

        HypreSolver *prec = NULL;
        if (dim == 2) prec = new HypreAMS(*A, &fespace);
        else if(dim==3)  prec = new HypreADS(*A, &fespace);


        //  HypreGMRES *pcg = new HypreGMRES(*A);
        HyprePCG *pcg = new HyprePCG(*A);
        pcg->SetTol(1e-10);
        pcg->SetMaxIter(50000);
        //pcg->SetPrintLevel(2);
        pcg->SetPrintLevel(0);
        if(prec!=NULL) pcg->SetPreconditioner(*prec);
        pcg->Mult(*B, *X);

        chrono.Stop();
        if(verbose) std::cerr << "Solving took " << chrono.UserTime() << "s." << endl;

        x = *X;

        chrono.Clear(); chrono.Start();
        int intOrder = 8;
        const IntegrationRule *irs[Geometry::NumGeom]; for (int i=0; i < Geometry::NumGeom; ++i) irs[i] = &(IntRules.Get(i, intOrder));
        double norm = x.ComputeL2Error(u_exact, irs);
        if(verbose) std::cerr << "L2 norm: " << norm << endl;
        if(verbose) std::cerr << "Computing error took " << chrono.UserTime() << "s." << endl;

        x = 0.0; x.ProjectCoefficient(u_exact);
        double projection_error = x.ComputeL2Error(u_exact, irs);
        if(verbose) std::cerr << "L2 norm of projection error: " << projection_error << endl;

        // 15. Free the used memory.
        delete pcg;
        if(prec != NULL) delete prec;
        delete X;
        delete B;
        delete A;

        delete fec;
    }

    if (verbose)
        std::cerr << "Test problem was solved successfully" << endl;

    if (visualization && nDimensions > 2)
    {
        int dim = nDimensions;

        FiniteElementCollection *hdiv_coll;
        if ( dim == 4 )
        {
            hdiv_coll = new RT0_4DFECollection;
            std::cerr << "RT: order 0 for 4D" << endl;
        }
        else
        {
            hdiv_coll = new RT_FECollection(feorder, dim);
            std::cerr << "RT: order " << feorder << " for 3D" << endl;
        }

        ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);

        // creating Hdiv grid-function slices (and printing them in VTK format in a file for paraview)
        ParGridFunction *pgridfuntest = new ParGridFunction(R_space);
        VectorFunctionCoefficient Hdivtest_fun_coeff(nDimensions, Hdivtest_fun);
        pgridfuntest->ProjectCoefficient(Hdivtest_fun_coeff);
        pgridfuntest->ComputeSlices ( 0.1, 2, 0.3, myid, false);

        // creating mesh slices (and printing them in VTK format in a file for paraview)
        pmesh->ComputeSlices ( 0.1, 2, 0.3, myid);
    }

    if (verbose)
        std::cerr << "Test Hdiv function was sliced successfully" << endl;

    MPI_Finalize();

    return 0;
}

void videofun(const Vector& xt, Vector& vecvalue )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());
    vecvalue(0) = 3 * x * ( 1 + 0.4 * sin (M_PI * (t + 1.0))) + 2.0 * (y * (y - 0.5) - z) * exp(-0.5*t) + exp(-100.0*(x*x + y * y + (z-0.5)*(z-0.5)));
    vecvalue(1) = 0.0;
    vecvalue(2) = 0.0;
    vecvalue(3) = 0.0;
    //return 3 * x * ( 1 + 0.2 * sin (M_PI * 0.5 * t/(t + 1))) + 2.0 * (y * (y - 0.5) - z) * exp(-0.5*t);
}

void f_Func(const Vector &p, Vector &f)
{
   int dim = p.Size();

   f(0) = sin(M_PI*p(0));
   f(1) = sin(M_PI*p(1));
   if (dim >= 3) f(2) = sin(M_PI*p(2));
   if (dim == 4) f(3) = sin(M_PI*p(3));

   f *= (1.0+M_PI*M_PI);
}

void u_Func(const Vector &p, Vector &u)
{
   int dim = p.Size();

   u(0) = sin(M_PI*p(0));
   u(1) = sin(M_PI*p(1));
   if (dim >= 3) u(2) = sin(M_PI*p(2));
   if (dim == 4) u(3) = sin(M_PI*p(3));
}

void Hdivtest_fun(const Vector& xt, Vector& out )
{
    out.SetSize(xt.Size());

    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    out(0) = x;
    out(1) = 0.0;
    out(2) = 0.0;
    out(xt.Size()-1) = 0.;

}
