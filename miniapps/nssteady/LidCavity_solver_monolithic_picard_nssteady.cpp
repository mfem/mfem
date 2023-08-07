// Include mfem and I/O
#include "mfem.hpp"
#include <fstream>
#include <iostream>

// Include for defining exact solution
#include <math.h>

// Include for std::bind and std::function
#include <functional>

// Include steady ns miniapp
#include "snavier_picard_monolithic.hpp"


#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979
#endif

using namespace mfem;

// Forward declarations of functions
double ComputeLift(ParGridFunction &p);

void   V_inflow(const Vector &X, Vector &v);
void   noSlip(const Vector &X, Vector &v);

// Test
int main(int argc, char *argv[])
{
   //
   /// 1. Initialize MPI and HYPRE.
   //
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   Hypre::Init();


   //
   /// 2. Parse command-line options. 
   //
   int porder = 1;            // fe
   int vorder = 2;

   int n = 10;                // mesh
   int dim = 2;
   int elem = 0;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;

   double re = 100;          // Reynolds number

   double  rel_tol = 1e-7;   // solvers
   double  abs_tol = 1e-6;
   int    tot_iter = 1000;
   int print_level = 0;

   bool stokes = false;       // if true solves stokes problem
   double alpha = 1.;         // steady-state scheme
   double gamma = 1.;         // relaxation

   bool paraview = false;     // postprocessing (paraview)
   bool visit    = false;     // postprocessing (VISit)
   
   bool verbose = false;
   const char *outFolder = "./";

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&dim,
                     "-d",
                     "--dimension",
                     "Dimension of the problem (2 = 2d, 3 = 3d)");
   args.AddOption(&elem,
                     "-e",
                     "--element-type",
                     "Type of elements used (0: Quad/Hex, 1: Tri/Tet)");
   args.AddOption(&n,
                     "-n",
                     "--num-elements",
                     "Number of elements in uniform mesh.");
   args.AddOption(&ser_ref_levels,
                     "-rs",
                     "--refine-serial",
                     "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                     "-rp",
                     "--refine-parallel",
                     "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&re, "-re", "--reynolds",
                  "Reynolds number");   
   args.AddOption(&vorder, "-ov", "--order_vel",
                     "Finite element order for velocity (polynomial degree) or -1 for"
                     " isoparametric space.");
   args.AddOption(&porder, "-op", "--order_pres",
                     "Finite element order for pressure(polynomial degree) or -1 for"
                     " isoparametric space.");
   args.AddOption(&rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Outer solve.");
   args.AddOption(&tot_iter,
                  "-it",
                  "--linear-iterations",
                  "Maximum iterations for the linear solve.");
   args.AddOption(&outFolder,
                  "-o",
                  "--output-folder",
                  "Output folder.");
   args.AddOption(&paraview, "-p", "--paraview", "-no-p",
                  "--no-paraview",
                  "Enable Paraview output.");
   args.AddOption(&visit, "-v", "--visit", "-no-v",
                  "--no-visit",
                  "Enable VisIt output.");    
   args.AddOption(&verbose, "-vb", "--verbose", "-no-verb",
                  "--no-verbose",
                  "Verbosity level of code");    
   args.AddOption(&print_level,
                  "-pl",
                  "--print-level",
                  "Print level.");     
   args.AddOption(&gamma,
                     "-g",
                     "--gamma",
                     "Relaxation parameter");
   args.AddOption(&alpha,
                     "-a",
                     "--alpha",
                     "Parameter controlling linearization");


   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(mfem::out);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0)
   {
       args.PrintOptions(mfem::out);
   }



   //
   /// 3. Read the (serial) mesh from the given mesh file on all processors.
   //
   Element::Type type;
   switch (elem)
   {
      case 0: // quad
         type = (dim == 2) ? Element::QUADRILATERAL: Element::HEXAHEDRON;
         break;
      case 1: // tri
         type = (dim == 2) ? Element::TRIANGLE: Element::TETRAHEDRON;
         break;
   }

   Mesh mesh;
   switch (dim)
   {
      case 2: // 2d
         mesh = Mesh::MakeCartesian2D(n,n,type,true);	
         break;
      case 3: // 3d
         mesh = Mesh::MakeCartesian3D(n,n,n,type,true);	
         break;
   }


   for (int l = 0; l < ser_ref_levels; l++)
   {
       mesh.UniformRefinement();
   }



   //
   /// 4. Define a parallel mesh by a partitioning of the serial mesh. 
   // Refine this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   //
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh->UniformRefinement();
       }
   }


   //
   /// 5. Create solver
   // 
   double kin_vis = 1/re;
   SNavierPicardMonolithicSolver* NSSolver = new SNavierPicardMonolithicSolver(pmesh,vorder,porder,kin_vis,verbose);



   //
   /// 6. Set parameters of the Fixed Point Solver
   // 
   SolverParams sFP = {rel_tol, abs_tol, tot_iter, print_level}; // rtol, atol, maxIter, print level
   NSSolver->SetOuterSolver(sFP);

   NSSolver->SetAlpha(alpha, AlphaType::CONSTANT);
   NSSolver->SetGamma(gamma);


   //
   /// 7. Add boundary conditions (Velocity-Dirichlet, Traction) and forcing term/s
   //
   // Essential velocity bcs
   int inflow_attr = 3;
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 1;
   ess_attr[inflow_attr - 1] = 0;

   // Inflow
   NSSolver->AddVelDirichletBC(V_inflow,inflow_attr);
         
   // No Slip
   NSSolver->AddVelDirichletBC(noSlip,ess_attr);



   //
   /// 8. Finalize Setup of solver
   //
   NSSolver->Setup();
   //DataCollection::Format forma = DataCollection::PARALLEL_FORMAT
   NSSolver->SetupOutput( outFolder, visit, paraview );



   //
   /// 9. Solve the forward problem
   //
   NSSolver->FSolve(); 


   // Free memory
   delete pmesh;
   delete NSSolver;

   // Finalize Hypre and MPI
   HYPRE_Finalize();
   MPI_Finalize();

   return 0;
}


// Functions
void V_inflow(const Vector &X, Vector &v)
{
   const int dim = X.Size();

   v = 0.0;
   v(0) = 1.0;
   v(1) = 0.0;

   if( dim == 3) { v(2) = 0; }
}

void noSlip(const Vector &X, Vector &v)
{
   v = 0.0;
}



