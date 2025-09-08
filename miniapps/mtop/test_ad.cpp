
#include "mfem.hpp"
#include "mms_coefficients.hpp"
#include <iostream>



using namespace mfem;

int main(int argc, char *argv[])
{
    // 1. Initialize MPI and HYPRE.
    mfem::Mpi::Init(argc, argv);
    int myrank = mfem::Mpi::WorldRank();
    mfem::Hypre::Init();

    // Parse command-line options.
    const char *mesh_file = "./mini_flow2d_ball.msh";
    int order = 2;
    bool static_cond = false;
    int ser_ref_levels = 1;
    int par_ref_levels = 1;
    real_t newton_rel_tol = 1e-7;
    real_t newton_abs_tol = 1e-12;
    int newton_iter = 10;
    int print_level = 1;
    bool visualization = false;

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&ser_ref_levels,
                   "-rs",
                   "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels,
                   "-rp",
                   "--refine-parallel",
                   "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&visualization,
                   "-vis",
                   "--visualization",
                   "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
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
       return 1;
    }

    if (myrank == 0)
    {
       args.PrintOptions(std::cout);
    }

    // Read the (serial) mesh from the given mesh file on all processors.  We
    // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    // and volume meshes with the same code.
    mfem::Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    // Refine the serial mesh on all processors to increase the resolution. In
    // this example we do 'ref_levels' of uniform refinement. We choose
    // 'ref_levels' to be the largest number that gives a final mesh with no
    // more than 10,000 elements.
    {
       int ref_levels =
          (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
       for (int l = 0; l < ref_levels; l++)
       {
          mesh.UniformRefinement();
       }
    }

    // Define a parallel mesh by a partitioning of the serial mesh. Refine
    // this mesh further in parallel to increase the resolution. Once the
    // parallel mesh is defined, the serial mesh can be deleted.
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();
    {
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh.UniformRefinement();
       }
    }

    std::cout<<"My rank="<<pmesh.GetMyRank()<<std::endl;

    H1_FECollection* vfec=new H1_FECollection(order, pmesh.Dimension());
    H1_FECollection* pfec=new H1_FECollection(order-1);
    ParFiniteElementSpace* vfes=new ParFiniteElementSpace(&pmesh, vfec, pmesh.Dimension());
    ParFiniteElementSpace* pfes=new ParFiniteElementSpace(&pmesh, pfec);

    VectorCoefficient* vel;
    VectorCoefficient* vlaplacian;
    Coefficient* presc;
    VectorCoefficient* gradp;
    Coefficient* lapp;
    if(2==pmesh.Dimension())
    {
        ADDivFree2DVelocity<ExPotential>* v=new ADDivFree2DVelocity<ExPotential>(); vel=v;
        vlaplacian=v->VectorLaplacian();

        ADScalar2DCoeff<ExPotential>* c=new ADScalar2DCoeff<ExPotential>(); presc=c;
        gradp=c->GetGradient();
        lapp=c->GetLaplacian();

    }else{
        ADDivFree3DVelocity<ExPotentialX,ExPotentialY,ExPotentialZ>* v=new ADDivFree3DVelocity(); vel=v;
        vlaplacian=v->VectorLaplacian();

        ADScalar3DCoeff<ExPotentialX>* c=new ADScalar3DCoeff<ExPotentialX>(); presc=c;
        gradp=c->GetGradient();
        lapp=c->GetLaplacian();
    }

    vel->SetTime(1.0);
    vlaplacian->SetTime(1.0);
    presc->SetTime(1.0);
    gradp->SetTime(1.0);
    lapp->SetTime(1.0);


    ParGridFunction pgvel(vfes); pgvel.ProjectCoefficient(*vel);
    ParGridFunction pglap(vfes); pglap.ProjectCoefficient(*vlaplacian);

    ParGridFunction pgpre(pfes); pgpre.ProjectCoefficient(*presc);
    ParGridFunction pggra(vfes); pggra.ProjectCoefficient(*gradp);
    ParGridFunction pglapp(pfes); pglapp.ProjectCoefficient(*lapp);


    {
        ParaViewDataCollection paraview_dc("mms", &pmesh);
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.SetCycle(0);
        paraview_dc.SetTime(0.0);
        paraview_dc.RegisterField("velo",&pgvel);
        paraview_dc.RegisterField("vlap",&pglap);

        paraview_dc.RegisterField("press",&pgpre);
        paraview_dc.RegisterField("pgrad",&pggra);
        paraview_dc.RegisterField("plapl",&pglapp);
        paraview_dc.Save();
    }

    delete vel;
    delete presc;
    delete vfes;
    delete pfes;
    delete pfec;
    delete vfec;

    future::dual<real_t,real_t> bla; bla=1.0;
    std::cout<<"bla="<<bla<<std::endl;


    return 0;
}
