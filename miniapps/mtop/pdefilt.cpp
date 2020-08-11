#include<mfem.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include "pdenssolver.hpp"


double DensFunc(const mfem::Vector& a){
    double sca=(4.0*M_PI);
    double rez=(std::sin(sca*a[0])*std::sin(sca*a[1])*std::sin(sca*a[2]));
    if(rez>0.0){ rez=1.0;} else {rez=0.0;}
    return rez;
}


int main(int argc, char *argv[])
{
    // 1. Initialize MPI.
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // 2. Parse command-line options.
    const char *mesh_file = "";
    int  element_order = 1;
    int  input_order = 2;
    bool static_cond = false;
    bool visualization = true;
    double len_scale=0.1;

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                          "Mesh file to use.");
    args.AddOption(&element_order, "-o", "--order",
                          "Finite element order (filtered field - polynomial degree) or -1 for"
                          " isoparametric space.");
    args.AddOption(&input_order, "-io", "--iorder",
                          "Finite element order (input field - polynomial degree) or -1 for"
                          " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                          "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&len_scale, "-ls","--lscale",
                          "Length scale for the PDE filter.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                          "--no-visualization",
                          "Enable or disable GLVis visualization.");
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

    //generate parallel mesh
    mfem::ParMesh *pmesh;
    if(strlen(mesh_file)==0)
    {
        //generate the mesh
        int nx=10;
        int ny=10;
        int nz=10;

        double sx=1.0;
        double sy=1.0;
        double sz=1.0;
        //alternative
        //mfem::Element::Type::HEXAHEDRON
        mfem::Mesh *mesh=new mfem::Mesh(nx,ny,nz, mfem::Element::Type::TETRAHEDRON,sx,sy,sz);
        int dim = mesh->Dimension();
        {
            int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
            for (int l = 0; l < ref_levels; l++)
            {
                mesh->UniformRefinement();
            }
        }
        //create the parallel mesh
        pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
        delete mesh;
    }
    else{
        int generate_edges=0;
        int refine=1;
        bool fix_orientation=true;
        mfem::Mesh *mesh = new mfem::Mesh(mesh_file, generate_edges, refine, fix_orientation);
        int dim = mesh->Dimension();
        {
            int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
            for (int l = 0; l < ref_levels; l++)
            {
                mesh->UniformRefinement();
            }
        }
        //create the parallel mesh
        pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
        delete mesh;
    }

    mfem::FunctionCoefficient fco(DensFunc);


    int dim = pmesh->Dimension();
    mfem::FiniteElementCollection* ffec=new mfem::H1_FECollection(element_order ,dim);
    mfem::FiniteElementCollection* ifec=new mfem::L2_FECollection(input_order, dim,mfem::BasisType::Positive);
    mfem::ParFiniteElementSpace* ffs= new mfem::ParFiniteElementSpace(pmesh,ffec,1,mfem::Ordering::byNODES);
    mfem::ParFiniteElementSpace* ifs= new mfem::ParFiniteElementSpace(pmesh,ifec,1,mfem::Ordering::byNODES);

    mfem::ParaViewDataCollection *dacol=new mfem::ParaViewDataCollection("filt",pmesh);
    dacol->SetLevelsOfDetail(2);

    mfem::PDEFilter* filt=new mfem::PDEFilter(pmesh,ifs,ffs);
    filt->SetLenScale(len_scale);

    //define the grid functions
    mfem::ParGridFunction* gfin=new mfem::ParGridFunction(ifs); //input field
    mfem::ParGridFunction* gfft=new mfem::ParGridFunction(ffs); //filtered filed
    //true-dof vectors
    mfem::HypreParVector* vin=gfin->GetTrueDofs();
    mfem::HypreParVector* vft=gfft->GetTrueDofs();
    *vft=0.0;

    gfin->ProjectCoefficient(fco);
    filt->FFilter(fco,*vft);
    gfft->SetFromTrueDofs(*vft);
    filt->FFilter(fco,*vft);
    gfft->SetFromTrueDofs(*vft);

    dacol->RegisterField("inp",gfin);
    dacol->RegisterField("flt",gfft);

    dacol->SetTime(0.0);
    dacol->SetCycle(0);
    dacol->Save();

    gfin->GetTrueDofs(*vin);
    filt->FFilter(*vin,*vft);
    gfft->SetFromTrueDofs(*vft);
    dacol->SetTime(1.0);
    dacol->SetCycle(1);
    dacol->Save();

    delete dacol;

    delete vft;
    delete vin;
    delete gfft;
    delete gfin;
    delete filt;
    delete ifs;
    delete ffs;
    delete ifec;
    delete ffec;
    delete pmesh;

    MPI_Finalize();
    return 0;
}
