// Include mfem and I/O
#include "mfem.hpp"
#include <fstream>
#include <iostream>

// Include for defining exact solution
#include <math.h>

// Include steady ns miniapp
#include "snavier_cg.hpp"

// Include for mkdir
#include <sys/stat.h>

using namespace mfem;

// Forward declarations
void   VectorFun(const Vector &X, Vector &v);
double ScalarFun(const Vector &X);
double pZero(const Vector &X);
void   vZero(const Vector &X, Vector &v);
std::function<void(const Vector &, Vector &)> FunctionWithParam(const double &param);

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
   const char *mesh_file = "../../data/star.mesh";
   int porder = 1;            // fe
   int vorder = 2;

   int ser_ref_levels = 0;    // mesh
   int par_ref_levels = 0;

   double kin_vis = 0.01;     // kinematic viscosity

   double rel_tol = 1e-7;     // solvers
   double abs_tol = 1e-15;
   int tot_iter = 100;
   int print_level = 1;

   bool visualization = false; // postprocessing
   bool verbose = false;
   bool par_format = false;
   const char *outFolder = "./";

   // TODO: check parsing and assign variables
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
   args.AddOption(&vorder, "-ov", "--order_vel",
                     "Finite element order for velocity (polynomial degree) or -1 for"
                     " isoparametric space.");
   args.AddOption(&porder, "-op", "--order_pres",
                     "Finite element order for pressure(polynomial degree) or -1 for"
                     " isoparametric space.");
   args.AddOption(&kin_vis,
                     "-kv",
                     "--kin-viscosity",
                     "Kinematic viscosity");
   args.AddOption(&visualization,
                     "-vis",
                     "--visualization",
                     "-no-vis",
                     "--no-visualization",
                     "Enable or disable GLVis visualization.");
   args.AddOption(&par_format, "-pf", "--parallel-format", "-sf",
                  "--serial-format",
                  "Format to use when saving the results for VisIt.");
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
   args.AddOption(&outFolder,
                  "-o",
                  "--output-folder",
                  "Output folder.");

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


   //
   /// 3. Read the (serial) mesh from the given mesh file on all processors.
   //
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
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
   /// 5. Define the coefficients (e.g. parameters, analytical solution/s).
   //
   FunctionCoefficient P_ex(ScalarFun);
   VectorFunctionCoefficient V_ex(dim,VectorFun);
   VectorFunctionCoefficient f_coeff(dim,FunctionWithParam(kin_vis));


   //
   /// 6. Create solver
   // 
   SNavierPicardCGSolver* NSSolver = new SNavierPicardCGSolver(pmesh,vorder,porder,kin_vis,verbose);


   //
   /// 7. Set parameters of the Fixed Point Solver
   // 
   SolverParams sFP = {1e-6, 1e-10, 1000, 1};   // rtol, atol, maxIter, print level
   NSSolver->SetFixedPointSolver(sFP);

   double alpha = 0.1;
   NSSolver->SetAlpha(alpha, AlphaType::CONSTANT);


   //
   /// 8. Set parameters of the Linear Solvers
   //
   SolverParams s1 = {1e-6, 1e-10, 1000, 0}; 
   SolverParams s2 = {1e-6, 1e-10, 1000, 0}; 
   SolverParams s3 = {1e-6, 1e-10, 1000, 0}; 
   NSSolver->SetLinearSolvers(s1,s2,s3);


   //
   /// 9. Add boundary conditions (Velocity-Dirichlet, Traction) and forcing term/s
   //
   // Acceleration term
   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1;
   NSSolver->AddAccelTerm(&f_coeff,domain_attr);

   // Essential velocity bcs
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 1;
   //NSSolver->AddVelDirichletBC(&V_ex, ess_attr);

   // Traction (neumann) bcs
   //Array<int> trac_attr(pmesh->bdr_attributes.Max());
   //NSSolver->AddTractionBC(coeff,attr);


   //
   /// 10. Set initial condition
   //
   VectorFunctionCoefficient v_in(dim, VectorFun);
   FunctionCoefficient p_in(ScalarFun);
   NSSolver->SetInitialConditionVel(v_in);
   NSSolver->SetInitialConditionPres(p_in);


   //
   /// 11. Finalize Setup of solver
   //
   NSSolver->Setup();


   //
   /// 12. Solve the forward problem
   //
   NSSolver->FSolve(); 


   //
   /// 13. Return forward problem solution and output results
   //
   ParGridFunction* velocityPtr = &(NSSolver->GetVelocity());
   ParGridFunction* pressurePtr = &(NSSolver->GetPressure());

   //
   /// 13.1 Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol_*".

   // Creating output directory if not existent
   if (mkdir(outFolder, 0777) == -1)
      std::cerr << "Error :  " << strerror(errno) << std::endl;
   else
      out << "Directory created";

   /*std::ostringstream mesh_name, v_name, p_name;
   mesh_name << outFolder << "/mesh." << std::setfill('0') << std::setw(6) << myrank; 

   v_name << outFolder << "/sol_v." << std::setfill('0') << std::setw(6) << myrank;
   p_name << outFolder << "/sol_p." << std::setfill('0') << std::setw(6) << myrank;

   std::ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   pmesh->Print(mesh_ofs);

   std::ostringstream omesh_file, omesh_file_bdr;
   omesh_file << outFolder << "/mesh.vtk";
   omesh_file_bdr << outFolder << "/mesh_bdr.vtu";
   std::ofstream omesh(omesh_file.str().c_str());
   omesh.precision(14);
   pmesh->PrintVTK(omesh);
   pmesh->PrintBdrVTU(omesh_file_bdr.str());

   std::ofstream v_ofs(v_name.str().c_str());
   v_ofs.precision(8);
   velocityPtr->Save(v_ofs);

   std::ofstream p_ofs(p_name.str().c_str());
   p_ofs.precision(8);
   pressurePtr->Save(p_ofs);*/


   //
   /// 13.2 Setup output in the solver
   bool visit;
   bool paraview;
   //DataCollection::Format forma = DataCollection::PARALLEL_FORMAT
   NSSolver->SetupOutput( outFolder, visit, paraview );

   //
   /// 13.3 Save exact solution in the Paraview format.
   //   
   ParGridFunction* velocityExactPtr = new ParGridFunction(NSSolver->GetVFes());
   ParGridFunction* pressureExactPtr = new ParGridFunction(NSSolver->GetPFes());
   ParGridFunction*           rhsPtr = new ParGridFunction(NSSolver->GetVFes());
   velocityExactPtr->ProjectCoefficient(V_ex);
   pressureExactPtr->ProjectCoefficient(P_ex);
   rhsPtr->ProjectCoefficient(f_coeff);

   ParaViewDataCollection paraview_dc("Results-Paraview-Exact", pmesh);
   paraview_dc.SetPrefixPath(outFolder);
   paraview_dc.SetLevelsOfDetail(vorder);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("velocity_exact",velocityExactPtr);
   paraview_dc.RegisterField("pressure_exact",pressureExactPtr);
   paraview_dc.RegisterField("rhs",rhsPtr);
   paraview_dc.Save();

   //
   // 13.4 Send the solution by socket to a GLVis server.
   //
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream v_sock(vishost, visport);
      v_sock << "parallel " << nprocs << " " << myrank << "\n";
      v_sock.precision(8);
      v_sock << "solution\n" << *pmesh << *velocityPtr << "window_title 'Velocity'"
             << std::endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << nprocs << " " << myrank << "\n";
      p_sock.precision(8);
      p_sock << "solution\n" << *pmesh << *pressurePtr << "window_title 'Pressure'"
             << std::endl;
   }


   // Finalize Hypre and MPI
   HYPRE_Finalize();
   MPI_Finalize();

   return 0;
}




void VectorFun(const Vector &X, Vector &v)
{
   const int dim = X.Size();

   double x = X[0];
   double y = X[1];
   if( dim == 3) {
      double z = X[2];
   }

   v = 0.0;

   v(0) = -cos(y)*sin(x);
   v(1) = cos(x)*sin(y);
   if( dim == 3) { v(2) = 0; }
}

double ScalarFun(const Vector &X)
{
   const int dim = X.Size();

   double x = X[0];
   double y = X[1];

   if( dim == 3) {
      double z = X[2];
   }
   
   double p = sin(x) + sin(y);

   return p;
}


std::function<void(const Vector &, Vector &)> FunctionWithParam(const double &param)
{
   return [param](const Vector &X, Vector &v)
   {
      const int dim = X.Size();

      double x = X[0];
      double y = X[1];
      
      if( dim == 3) {
         double z = X[2];
      }
      
      v = 0.0;

      v(0) = cos(x) + cos(x)*sin(x) - 2*param*cos(y)*sin(x);
      v(1) = cos(y) + cos(y)*sin(y) + 2*param*cos(x)*sin(y);
      if( dim == 3) {
         v(2) = 0;
      }
   };
}

void vZero(const Vector &X, Vector &v)
{
   v = 0.0;
}



double pZero(const Vector &X)
{
   return 0;
}