//                       A 2D static magnetic levitation miniapp
//
// Compile with: make maglev
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh

//
// Description:  This example code demonstrates the use of MFEM to define a

#include "maglev.hpp"
#include "../../common/pfem_extras.hpp"
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;
using namespace mfem;

void solveForAz(MaglevProblemGeometry &problem, 
                ParFiniteElementSpace &fes_h1,
                ParGridFunction &x, 
                ParGridFunction &conv, 
                ParGridFunction &mag);

void solveForJz(MaglevProblemGeometry &problem, 
                ParFiniteElementSpace &fes_h1,
                ParGridFunction &az,
                ParGridFunction &jz);


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   int order = 1;
   int num_refine = 4;
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
   args.AddOption(&ha_remanence, "-hr", "--ha_remanence",
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
   MaglevProblemGeometry problem(conductor_yoff - conductor_thick, conductor_yoff, conductor_vx,
                                 conductor_mu, conductor_sigma,
                                 domain_length/2.0 - ha_len/2.0, domain_length/2.0 + ha_len/2.0,
                                 conductor_yoff+ha_yoff, conductor_yoff+ha_yoff+ha_thick,
                                 ha_num_magnets, ha_vx, ha_mu, ha_sigma, ha_remanence/ha_mu,
                                 air_mu, air_sigma);

   // Define a parallel finite element space on the parallel mesh. Here we
   // use continuous Lagrange finite elements of the specified order.
   FiniteElementCollection *fec_h1 = new H1_FECollection(order, 2);
   FiniteElementCollection *fec_rt = new RT_FECollection(order, 2);   
   FiniteElementCollection *fec_l2 = new L2_FECollection(0, 2);
   ParFiniteElementSpace *fes_h1 = new ParFiniteElementSpace(pmesh, fec_h1);
   ParFiniteElementSpace *fes_rt = new ParFiniteElementSpace(pmesh, fec_rt);
   ParFiniteElementSpace *fes_l2 = new ParFiniteElementSpace(pmesh, fec_l2);
   ParFiniteElementSpace *fes_vl2 = new ParFiniteElementSpace(pmesh, fec_l2, 2);
   HYPRE_Int ndof = fes_h1->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << ndof << endl;
   }

   ParGridFunction az(fes_h1);
   ParGridFunction conv(fes_vl2);
   ParGridFunction mag(fes_vl2);
   solveForAz(problem, *fes_h1, az, conv, mag);

   //Compute Bfield = curl A so we can visualize it
   ParGridFunction bfield(fes_rt);
   miniapps::ParDiscreteCurlOperator curl(fes_h1,fes_rt);
   curl.Assemble();
   curl.Finalize();
   curl.Mult(az, bfield);

   ParGridFunction jz(fes_h1);
   solveForJz(problem, *fes_h1, az, jz);

   //Make grid functions out of these coefficients for visualization
   ParGridFunction vx(fes_l2);
   ParGridFunction sigma(fes_l2);
   ParGridFunction mu(fes_l2);
   VxCoeff vx_coeff(problem);
   SigmaCoeff sigma_coeff(problem);
   MuCoeff mu_coeff(problem);
   vx.ProjectCoefficient(vx_coeff);
   sigma.ProjectCoefficient(sigma_coeff);
   mu.ProjectCoefficient(mu_coeff);

   VisItDataCollection visit_dc("maglev", pmesh);
   visit_dc.RegisterField("Az", &az);
   visit_dc.RegisterField("Jz", &jz);
   visit_dc.RegisterField("B", &bfield);
   visit_dc.RegisterField("convection_coeff", &conv);
   visit_dc.RegisterField("mag_coeff", &mag);
   visit_dc.RegisterField("vx", &vx);
   visit_dc.RegisterField("sigma", &sigma);
   visit_dc.RegisterField("mu", &mu);
   visit_dc.Save();

   MPI_Finalize();

   return 0;
}


//Solve Delta Az - mu sigma v dot grad Az = curl M
//x:     The solution Az
//conv:  The convection coefficient (for vizualization)
//mag:   The magnitization coefficient (for vizualization)
void solveForAz(MaglevProblemGeometry &problem, 
                ParFiniteElementSpace &fes_h1,
                ParGridFunction &az,
                ParGridFunction &conv,
                ParGridFunction &mag)
{
   // Set up the parallel bilinear form dc(.,.) on the finite element space
   // corresponding to the Laplacian operator Delta Az - mu sigma v dot grad Az.
   ConstantCoefficient one(-1.0);
   ConvectionCoeff conv_coeff(problem);
   ParBilinearForm dc(&fes_h1);
   dc.AddDomainIntegrator(new DiffusionIntegrator(one));
   dc.AddDomainIntegrator(new ConvectionIntegrator(conv_coeff));
   dc.Assemble();

   // Determine the list of true (i.e. parallel conforming) essential
   // boundary dofs. In this example, the boundary conditions are defined
   // by marking all the boundary attributes from the mesh as essential
   // (Dirichlet) and converting them to a list of true dofs.
   ParMesh *pmesh = fes_h1.GetParMesh();
   Array<int> ess_tdof_list(fes_h1.GlobalTrueVSize());
   ess_tdof_list = 0;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fes_h1.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // Set up the parallel linear form b(.) which corresponds to the
   // right-hand side of the FEM linear system, which in this case is
   // (1,phi_i) where phi_i are the basis functions in fes_h1.
   ParLinearForm b(&fes_h1);
   MagnetizationCoeff mag_coeff(problem);
   b.AddDomainIntegrator(new DomainGradLFIntegrator(mag_coeff));
   b.Assemble();

   // Assemble the parallel bilinear form and the corresponding linear
   // system, applying any necessary transformations such as
   HypreParMatrix DC;
   Vector B, AZ;
   dc.FormLinearSystem(ess_tdof_list, az, b, DC, AZ, B);

   // Define and apply a parallel GMRES solver for AX=B with the BoomerAMG
   // preconditioner from hypre.
   HypreParaSails sails(DC);
   HypreGMRES gmres(DC);
   gmres.SetTol(1e-7);
   gmres.SetKDim(250);
   gmres.SetMaxIter(10000);
   gmres.SetPrintLevel(2);
   gmres.SetPreconditioner(sails);
   gmres.Mult(B, AZ);

   // Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   dc.RecoverFEMSolution(AZ, b, az);

   //Project these coefficients onto corresponding grid functions for vizualization
   conv.ProjectCoefficient(conv_coeff);
   mag.ProjectCoefficient(mag_coeff);
}


//Solve Jz = -Delta 1/mu Az
void solveForJz(MaglevProblemGeometry &problem, 
                ParFiniteElementSpace &fes_h1,
                ParGridFunction &az,
                ParGridFunction &jz)
{
   Array<int> ess_tdof_list(fes_h1.GlobalTrueVSize());
   ess_tdof_list = 0;

   //Set up M matrix for the left hand side
   ConstantCoefficient one(-1.0);
   ParBilinearForm m(&fes_h1);
   m.AddDomainIntegrator(new MassIntegrator(one));
   m.Assemble();

   //Set up the biliinear for the K matrix for -Delta 1/mu on the right hand side
   MuInvCoeff muinv_coeff(problem);
   ParBilinearForm k(&fes_h1);
   k.AddDomainIntegrator(new DiffusionIntegrator(muinv_coeff));
   k.Assemble();

   //Compute B = K AZ
   ParGridFunction b(&fes_h1);
   HypreParMatrix K;
   HypreParVector *AZ = az.GetTrueDofs();
   HypreParVector B(*AZ);
   k.FormSystemMatrix(ess_tdof_list, K);
   K.Mult(*AZ, B);

   //Now Set up the linear system for M JZ = B and get jz
   jz = 0.0;
   HypreParVector *JZ = jz.GetTrueDofs();
   HypreParMatrix M;
   m.FormSystemMatrix(ess_tdof_list, M);
   HypreBoomerAMG amg(M);
   HypreGMRES gmres(M);
   gmres.SetTol(1e-12);
   gmres.SetKDim(250);
   gmres.SetMaxIter(10000);
   gmres.SetPrintLevel(2);
   gmres.SetPreconditioner(amg);
   gmres.Mult(B, *JZ);
   jz = *JZ;

   delete AZ;
}