//                                MFEM MarkingLS Example
//
// Compile with: make markinglst
//
// Sample runs:  mpirun -np 4 markinglst -m ../../../../data/ref-square.mesh -r 5 -g 0
//
// Description: This example code demonstrates marking of arbitrary number of level sets.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../../marking.hpp"

using namespace std;
using namespace mfem;

double ti = 0.0;

double phi0(std::vector<double> x){
  double x0 = 0.5;
  double y0 = 1.0 - ti;

  double radius = 0.23;
  
  return pow((x[0]-x0),2)+pow((x[1]-y0),2) - pow(radius,2);
}

double phi1(std::vector<double> x){
  return x[1] - 0.51;
}

double phi2(std::vector<double> x){
  return x[0] - 0.51;
}


int main(int argc, char *argv[])
{

   // 0. Initialize MPI and HYPRE                                                                    
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();


  // 1. Parse command line options.
   const char *mesh_file = "star.mesh";
   int order = 1;
   int rf = 0;
   int glv = 1;
   
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&rf, "-r", "--refine-serial",
		  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&glv, "-g", "--glvis", "use glvis");
   args.ParseCheck();

   args.Parse();
   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh serial_mesh(mesh_file);

   for (int i = 0; i < rf; i++){serial_mesh.UniformRefinement();}

   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear(); // the serial mesh is no longer needed 
   
   int dim = mesh.Dimension();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fespace(&mesh, &fec);
   HYPRE_BigInt total_num_dofs = fespace.GlobalTrueVSize();

   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << total_num_dofs << endl;
   }
   
   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   Vector vxyz;
   
   ParFiniteElementSpace fespace_mesh(&mesh, &fec, dim);
 
   mesh.SetNodalFESpace(&fespace_mesh);
   ParGridFunction x_mesh(&fespace_mesh);
   mesh.SetNodalGridFunction(&x_mesh);
   vxyz = *mesh.GetNodes();
   int nodes_cnt = vxyz.Size()/dim;

   cout << "Number of nodes per proc: " << nodes_cnt << endl;

   
   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;


   ParGridFunction sdf0(&fespace);
   ParGridFunction sdf1(&fespace);
   ParGridFunction sdf2(&fespace);
   ParGridFunction vof0(&fespace);
   ParGridFunction vof1(&fespace);
   ParGridFunction vof2(&fespace);
   ParGridFunction alpha3(&fespace);
   ParGridFunction materials(&fespace);

   ParGridFunction alpha0(&fespace);
   ParGridFunction alpha1(&fespace);
   ParGridFunction alpha2(&fespace);

   std::vector<ParGridFunction*> VOFs;
   std::vector<ParGridFunction*> alphas;

   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("Example1P", &mesh);
   pd->SetPrefixPath("ParaView");
   pd->RegisterField("solution", &x);
   pd->RegisterField("phi0", &sdf0);
   pd->RegisterField("phi1", &sdf1);
   pd->RegisterField("phi2", &sdf2);
   pd->RegisterField("vof0", &vof0);
   pd->RegisterField("vof1", &vof1);
   pd->RegisterField("vof2", &vof2);
   pd->RegisterField("alpha0", &alpha0);
   pd->RegisterField("alpha1", &alpha1);
   pd->RegisterField("alpha2", &alpha2);
   pd->RegisterField("alpha3", &alpha3);
   pd->RegisterField("materials", &materials);
   pd->SetLevelsOfDetail(order);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);

   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();

   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sout;
   if (glv==1)
   {
      sout.open(vishost, visport);
      if (!sout)
      {
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
      }
      sout.precision(8);
   }

   MarkingLS LSM;

   for (int ts = 0; ts < 100; ts++){


     ti = 1.0*ts/100.0;


     LSM.createSDF(&mesh,sdf0, &phi0);
     LSM.createSDF(&mesh,sdf1, &phi1);
     LSM.createSDF(&mesh,sdf2, &phi2);
     
     LSM.mapVOFToNodes(sdf0, vof0);
     LSM.mapVOFToNodes(sdf1, vof1);
     LSM.mapVOFToNodes(sdf2, vof2);

     VOFs= {&vof0, &vof1, &vof2};
     alphas={&alpha0,&alpha1,&alpha2,&alpha3};

     LSM.orderedAlpha(VOFs,alphas);
 
     LSM.markMaterials(materials,alphas);
     
     LSM.tagCells(&mesh, materials,3);

     // 6. Set up the linear form b(.) corresponding to the right-hand side.
     ConstantCoefficient one(1.0);
     ParLinearForm b(&fespace);
     b.AddDomainIntegrator(new DomainLFIntegrator(one));
     b.Assemble();

     // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
     ParBilinearForm a(&fespace);
     a.AddDomainIntegrator(new DiffusionIntegrator);
     a.Assemble();

     // 8. Form the linear system A X = B. This includes eliminating boundary
     //    conditions, applying AMR constraints, and other transformations.
     HypreParMatrix A;
     Vector B, X;
     a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

     // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
 
     HypreBoomerAMG M(A);
     CGSolver cg(MPI_COMM_WORLD);
     cg.SetRelTol(1e-12);
     cg.SetMaxIter(2000);
     cg.SetPrintLevel(1);
     cg.SetPreconditioner(M);
     cg.SetOperator(A);
     cg.Mult(B, X);
     // 10. Recover the solution x as a grid function and save to file. The output
     //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
     a.RecoverFEMSolution(X, b, x);

     if (glv==1){
       sout << "parallel " << num_procs << " " << myid << "\n";
       sout << "solution\n" << mesh << materials << flush;

     }else{

     pd->SetCycle(ts);
     pd->SetTime(ts);
     pd->Save();
     }
   }
   
   return 0;
}
