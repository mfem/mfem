#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "stokes.hpp"

// example runs
// mpirun -np 4 ./stokes -m ./ball2D.msh -petscopts ./stokes_fieldsplit
// mpirun -np 4 ./stokes -m ./ball2D.msh -petscopts ./stokes_fieldsplit_01

double inlet_vel(const mfem::Vector &x)
{
    double d=(x[1]-1.4);
    if(fabs(d)>0.2){return 0.0;}
    return 1.5*(1-d*d/(0.2*0.2));
}

double charfunc(const mfem::Vector &x)
{
    double nx=(x[0]-1.5)*(x[0]-1.5)+(x[1]-0.5)*(x[1]-0.5);
    if(x.Size()==3){
        nx=nx+(x[2]-0.5)*(x[2]-0.5);
    }

    nx=std::sqrt(nx);
    double r=0.25;

    double rez=1.0;
    if(nx<=1.5*r){
        double a=-26.0/27.0;;
        double b=62.0/27.0;
        double c=-5.0/6.0;
        nx=nx/r;
        rez=a*nx*nx*nx+b*nx*nx+c*nx;
        if(rez<0.0){rez=0.0;}
    }
    return rez;
}

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);



   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   int tot_iter = 100;
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
   args.AddOption(&rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&tot_iter,
                  "-it",
                  "--linear-iterations",
                  "Maximum iterations for the linear solve.");
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

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   mfem::Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   mesh.EnsureNCMesh(true);

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(100./mesh.GetNE())/log(2.)/dim);
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

   // Define design space
   //mfem::FiniteElementCollection* dfec=new mfem::L2_FECollection(1,dim);
   mfem::FiniteElementCollection* dfec=new mfem::H1_FECollection(1,dim);
   mfem::ParFiniteElementSpace* dfes=new mfem::ParFiniteElementSpace(&pmesh,dfec);
   mfem::ParGridFunction design; design.SetSpace(dfes); design=0.0;
   mfem::Vector tdesign(dfes->GetTrueVSize());
   //project the characteristic function onto the design space
   {
        mfem::FunctionCoefficient fco(charfunc);
        design.ProjectCoefficient(fco);
   }
   design.GetTrueDofs(tdesign);

   mfem::FiniteElementCollection* vfec=new mfem::H1_FECollection(2,dim);
   mfem::ParFiniteElementSpace* vfes=new mfem::ParFiniteElementSpace(&pmesh,vfec,dim, mfem::Ordering::byVDIM);

   mfem::Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr=0;
   ess_bdr[0]=1;
   mfem::Array<int> ess_tdof_list;
   vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,0);


   delete vfes;
   delete vfec;

   delete dfes;
   delete dfec;

   MPI_Finalize();
   return 0;


}

