//                       A 2D static magnetic levitation miniapp
//
// Compile with: make maglev
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh

//
// Description:  This example code demonstrates the use of MFEM to define a

#include "maglev.hpp"
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   int order = 1;
   int num_refine = 2;
   double domain_length = 3.0;
   double domain_height = 2.0;   
   double ha_len = 2.0;
   double ha_thick = 0.1;
   int ha_num_magnets = 20;
   double ha_yoff = 0.05;
   double ha_vx = 0.0;         
   double ha_mu = 1.32e-6;
   double ha_sigma = 6.67e5;
   double ha_remanence = 1.0;
   double conductor_thick = 0.025;   
   double conductor_yoff = 1.0;
   double conductor_vx = -100.0;
   double conductor_mu = 1.257e-6;
   double conductor_sigma = 3.5e7;   
   double air_mu = 1.257e-6;
   double air_sigma = 3e-15;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&num_refine, "-nr", "--num_refine",
                  "Number of times to refine the mesh after initial setup.");
   args.AddOption(&domain_length, "-dl", "--domain_length",
                  "Length (x) of the simulation domain (m).");
   args.AddOption(&domain_height, "-dh", "--domain_height",
                  "Height (y) of the simulation domain (m).");   
   args.AddOption(&ha_len, "-hl", "--ha_len",
                  "Physical length of the Halbach array (m).");
   args.AddOption(&ha_thick, "-ht", "--ha_thick",
                  "Physical thickness of the Halbach array (m).");
   args.AddOption(&ha_num_magnets, "-hn", "--ha_num_magnets",
                  "Number of magnets in the Halbach array.");
   args.AddOption(&ha_yoff, "-hy", "--ha_yoff",
                  "Distance between the Halbach array and the conductor (m).");
   args.AddOption(&ha_vx, "-hv", "--ha_vx",
                  "Velocity of the Halbach array in the x direction.");
   args.AddOption(&ha_mu, "-hm", "--ha_mu",
                  "Permeability of the Halbach array (H/m).");
   args.AddOption(&ha_sigma, "-hs", "--ha_sigma",
                  "Conductivity of the halbach array (S/m).");
   args.AddOption(&ha_remanence, "-hr", "--ha_sigma",
                  "Remanence of the halbach array (T).");      
   args.AddOption(&conductor_thick, "-ct", "--conductor_thick",
                  "Physical thickness of the conductor (m).");
   args.AddOption(&conductor_yoff, "-cy", "--conductor_yoff",
                  "Distance from the bottom of the domain to the top of the conductor (m).");
   args.AddOption(&conductor_vx, "-cv", "--conductor_vx",
                  "Velocity of the conductor in the x direction (m/s).");   
   args.AddOption(&conductor_mu, "-cm", "--conductor_mu",
                  "Permeability of the conductor (H/m).");
   args.AddOption(&conductor_sigma, "-cs", "--conductor_sigma",
                  "Conductivity of the conductor (S/m).");   
   args.AddOption(&air_mu, "-am", "--air_mu",
                  "Permeability of the air (H/m).");
   args.AddOption(&air_sigma, "-as", "--air_sigma",
                  "Conductivity of the air (S/m).");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Generate the mesh from the given inputs and set up the 
   // the object that holds on to all the problem definitions.
   double dx = ha_len / double(ha_num_magnets);
   double dy = min(min(ha_thick, conductor_thick), ha_yoff);
   int nx = round(domain_length / dx);
   int ny = round(domain_height / dy);
   Mesh *mesh = new Mesh(nx, ny, Element::QUADRILATERAL, 1,
                         domain_length, domain_height);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int l = 0; l < num_refine; l++)
   {
      pmesh->UniformRefinement();
   }
   MaglevProblemGeometry *problem = new MaglevProblemGeometry(
                           conductor_yoff - conductor_thick, conductor_yoff, conductor_vx,
                           conductor_mu, conductor_sigma,
                           domain_length/2.0 - ha_len/2.0, domain_length/2.0 + ha_len/2.0, 
                           conductor_yoff+ha_yoff, conductor_yoff+ha_yoff+ha_thick,
                           ha_num_magnets, ha_vx, ha_mu, ha_sigma, ha_remanence/ha_mu,
                           air_mu, air_sigma);

   // Define a parallel finite element space on the parallel mesh. Here we
   // use continuous Lagrange finite elements of the specified order. 
   FiniteElementCollection *fec = new H1_FECollection(order, 2);
   FiniteElementCollection *viz_fec = new ND_FECollection(order, 2);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   ParFiniteElementSpace *viz_fespace = new ParFiniteElementSpace(pmesh, viz_fec);   
   HYPRE_Int ndof = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << ndof << endl;
   }


   // Set up the parallel bilinear form a(.,.) on the finite element space
   // corresponding to the Laplacian operator Delta Az - mu sigma v dot grad Az.
   ConstantCoefficient one(-1.0);
   ConvectionCoeff conv_coeff(problem);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddDomainIntegrator(new ConvectionIntegrator(conv_coeff));
   a->Assemble();

   // Determine the list of true (i.e. parallel conforming) essential
   // boundary dofs. In this example, the boundary conditions are defined
   // by marking all the boundary attributes from the mesh as essential
   // (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list(ndof);
   ess_tdof_list = 0;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // Set up the parallel linear form b(.) which corresponds to the
   // right-hand side of the FEM linear system, which in this case is
   // (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   MagnetizationCoeff mag_coeff(problem);
   b->AddDomainIntegrator(new DomainGradLFIntegrator(mag_coeff));
   b->Assemble();

   // Define the solution vector x as a parallel finite element grid function
   // corresponding to fespace. Initialize x with initial guess of zero,
   // which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;


   // Assemble the parallel bilinear form and the corresponding linear
   // system, applying any necessary transformations such as
   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   // Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   // preconditioner from hypre.
   HypreParaSails *sails = new HypreParaSails(A);
   HypreGMRES *gmres = new HypreGMRES(A);
   gmres->SetTol(1e-10);
   gmres->SetKDim(250);
   gmres->SetMaxIter(10000);
   gmres->SetPrintLevel(2);
   gmres->SetPreconditioner(*sails);
   gmres->Mult(B, X);

   // Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   //Project the coefficient onto some vector grid functions so we can vizualize them
   ParGridFunction conv(viz_fespace);
   conv.ProjectCoefficient(conv_coeff);
   ParGridFunction mag(viz_fespace);
   mag.ProjectCoefficient(mag_coeff);



   VisItDataCollection visit_dc("maglev", pmesh);
   visit_dc.RegisterField("Az", &x);
   visit_dc.RegisterField("convection_coeff", &conv);
   visit_dc.RegisterField("mag_coeff", &mag);
   visit_dc.Save();

   // Save the refined mesh and the solution in parallel. This output can
   // be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   /*{
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // Send the solution by socket to a GLVis server.
   if (false)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }*/

   // Free the used memory.
   delete gmres;
   delete sails;
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;

   MPI_Finalize();

   return 0;
}

