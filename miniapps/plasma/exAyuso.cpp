//                       MFEM Example 14 - Parallel Version
//
// Compile with: make ex14p
//
// Sample runs:  mpirun -np 4 ex14p -m ../data/inline-quad.mesh -o 0
//               mpirun -np 4 ex14p -m ../data/star.mesh -o 2
//               mpirun -np 4 ex14p -m ../data/star-mixed.mesh -o 2
//               mpirun -np 4 ex14p -m ../data/escher.mesh -s 1
//               mpirun -np 4 ex14p -m ../data/fichera.mesh -s 1 -k 1
//               mpirun -np 4 ex14p -m ../data/fichera-mixed.mesh -s 1 -k 1
//               mpirun -np 4 ex14p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex14p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex14p -m ../data/square-disc-nurbs.mesh -o 1
//               mpirun -np 4 ex14p -m ../data/disc-nurbs.mesh -rs 4 -o 2 -s 1 -k 0
//               mpirun -np 4 ex14p -m ../data/pipe-nurbs.mesh -o 1
//               mpirun -np 4 ex14p -m ../data/inline-segment.mesh -rs 5
//               mpirun -np 4 ex14p -m ../data/amr-quad.mesh -rs 3
//               mpirun -np 4 ex14p -m ../data/amr-hex.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               discontinuous Galerkin (DG) finite element discretization of
//               the Laplace problem -Delta u = 1 with homogeneous Dirichlet
//               boundary conditions. Finite element spaces of any order,
//               including zero on regular grids, are supported. The example
//               highlights the use of discontinuous spaces and DG-specific face
//               integrators.
//
//               We recommend viewing examples 1 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#include "transport_solver.hpp"

using namespace mfem::common;
using namespace mfem::plasma;
using namespace mfem::plasma::transport;

class SolutionCoefficient : public Coefficient
{
private:
   int prob_;

public:
   SolutionCoefficient(int p) : prob_(p) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      double x[2];
      Vector transip(x, 2);

      T.Transform(ip, transip);

      switch (prob_)
      {
         case 1:
            return sin(2.0 * M_PI * x[0]) * sin(2.0 * M_PI * x[1]);
         case 3:
            if (x[1] <= 0.2 + 1.7320508076 * x[0])
            {
               return 1.0;
            }
            else
            {
               return 0.0;
            }

      }
      return 0.0;
   }
};

class BoundaryCoefficient : public Coefficient
{
private:
   int prob_;

public:
   BoundaryCoefficient(int p) : prob_(p) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      double x[2];
      Vector transip(x, 2);

      T.Transform(ip, transip);

      switch (prob_)
      {
         case 0:
            return 0.0;
         case 1:
            return sin(2.0 * M_PI * x[0]) * sin(2.0 * M_PI * x[1]);
         case 3:
            if (x[1] <= 0.2 + 1.7320508076 * x[0])
            {
               return 1.0;
            }
            else
            {
               return 0.0;
            }
            break;
         case 5:
            return 2.0 * x[0] * (1.0 - x[1]) + 3.0 * x[1] * (1.0 - x[0]);
      }
      return 0.0;
   }
};

class SourceCoefficient : public Coefficient
{
private:
   int prob_;
   double epsilon_para;
   double epsilon_perp;
   double gamma;
   Vector beta;
   VectorCoefficient &betaCoef;

public:
   SourceCoefficient(int p, double eps_para, double eps_perp,
                     double gam, VectorCoefficient &bet)
      : prob_(p),
        epsilon_para((eps_para >= 0.0) ? eps_para : eps_perp),
        epsilon_perp(eps_perp), gamma(gam), betaCoef(bet) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      double x[2];
      Vector transip(x, 2);

      T.Transform(ip, transip);

      betaCoef.Eval(beta, T, ip);

      switch (prob_)
      {
         case 0:
            return 1.0;
         case 1:
         {
            double cx = cos(2.0 * M_PI * x[0]);
            double cy = cos(2.0 * M_PI * x[1]);
            double sx = sin(2.0 * M_PI * x[0]);
            double sy = sin(2.0 * M_PI * x[1]);
            double b2 = beta * beta;
            double epss = epsilon_para + epsilon_perp;
            double epsd = epsilon_para - epsilon_perp;
            return 2.0 * M_PI * beta[0] * cx *
                   (sy - 4.0 * M_PI * epsd * cy / b2) +
                   sx * (2.0 * M_PI * beta[1] * cy +
                         (gamma + 4.0 * M_PI * M_PI * epss) * sy);
         }
         case 3:
            return 0.0;
         case 5:
            if (x[0] <= 0.5)
            {
               if (x[1] <= 1.0/3.0)
               {
                  return 0.0;
               }
               else // ???
               {
                  return 0.0;
               }
            }
            else if (x[1] > 1.0/3.0)
            {
               return -1.0;
            }
            else // ???
            {
               return 0.0;
            }
      }

      return 0.0;
   }
};

class AdvectionCoefficient : public VectorCoefficient
{
private:
   int prob_;
   Vector beta;

public:
   AdvectionCoefficient(int p)
      : VectorCoefficient(2), prob_(p), beta(2) {}

   void Eval(Vector &b, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      double x[2];
      Vector transip(x, 2);

      T.Transform(ip, transip);

      b.SetSize(2);

      switch (prob_)
      {
         case 1:
            b[0] = 1.0;
            b[1] = 1.0;
            break;
         case 3:
            b[0] = 0.5;
            b[1] = 0.5 * sqrt(3.0);;
            // b[0] = 5.0/sqrt(41);
            // b[1] = 4.0/sqrt(41);
            break;
         case 5:
            b[0] = x[0] * x[0] * x[1] + 1.0;
            b[1] = x[0] * x[1] * x[1] + 1.0;
            break;
      }

      return;
   }
};

Mesh * BuildSol1DMesh(const Vector &sol1d);

void GMRES_AMG_Solve(int logging, HypreParMatrix &A, HypreParVector &B,
                     HypreParVector &X);
void GMRES_AIR_Solve(int logging, ParFiniteElementSpace &fes,
                     HypreParMatrix &A, HypreParVector &B, HypreParVector &X);

void ComputeL2Error(const string &header,
                    ParGridFunction &x, Coefficient &coef);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   // Parse command-line options.
   const char *mesh_file = "ellipse_origin_h0pt0625_o3.mesh";
   Vector mesh_sol1d;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 1;

   int prob = 1;
   double gamma =  0.0;
   double epsilon_perp = 1.0;
   double epsilon_para = -1.0;
   double lambda = 1.0;
   double sigma = -1.0;
   double kappa1 = -1.0;
   double kappa2 = 0.0;
   int logging = 0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_sol1d, "-m-sol1d", "--mesh-sol1d",
                  "Build a mesh using the algortihm from sol1d."
                  " Needs three parameters: number of elements, "
                  " width of mesh, and beta");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial,"
                  " -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&prob, "-p", "--problem",
                  "Example problem: 1...");
   args.AddOption(&gamma, "-g", "--gamma",
                  "Reaction coefficient");
   args.AddOption(&epsilon_perp, "-e", "--epsilon-perp",
                  "Perpendicular Diffusion coefficient");
   args.AddOption(&epsilon_para, "-ep", "--epsilon-para",
                  "Parallel Diffusion coefficient");
   args.AddOption(&lambda, "-dg-l", "--dg-lambda",
                  "DG lambda parameter (not used).");
   args.AddOption(&sigma, "-dg-s", "--dg-sigma",
                  "DG sigma parameter.");
   args.AddOption(&kappa1, "-dg-k1", "--dg-kappa1",
                  "DG kappa parameter (diffusion).");
   args.AddOption(&kappa2, "-dg-k2", "--dg-kappa2",
                  "DG kappa parameter (advection).");
   args.AddOption(&logging, "-l", "--log-level",
                  "Logging level for iterative solvers.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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
   if (kappa1 < 0)
   {
      kappa1 = order*(order+1);
   }
   if (kappa2 < 0)
   {
      kappa2 = 0.0;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors. We
   // can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   // with the same code. NURBS meshes are projected to second order meshes.
   Mesh *mesh = (mesh_sol1d.Size() == 0) ? new Mesh(mesh_file, 1, 1) :
                BuildSol1DMesh(mesh_sol1d);
   int dim = mesh->Dimension();

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ser_ref_levels' of uniform refinement. By default,
   // or if ser_ref_levels < 0, we choose it to be the largest number that
   // gives a final mesh with no more than 50,000 elements.
   {
      if (ser_ref_levels < 0)
      {
         ser_ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ser_ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   Array<int> dbc_marker(pmesh->bdr_attributes.Max());
   dbc_marker = 0;
   if (prob == 0)
   {
      dbc_marker[1] = 1;
   }
   if (prob == 1)
   {
      dbc_marker = 1;
   }
   else if (prob == 3)
   {
      dbc_marker[0] = 1;
      dbc_marker[3] = 1;
      // dbc_marker = 1;
   }
   else if (prob == 5)
   {
      dbc_marker[0] = 1;
      dbc_marker[3] = 1;
   }

   // Define a parallel finite element space on the parallel mesh. Here we
   // use discontinuous finite elements of the specified order >= 0.
   DG_FECollection       dg_fec(order, dim, BasisType::GaussLobatto);
   H1_FECollection       cg_fec(order, dim, BasisType::GaussLobatto);

   ParFiniteElementSpace dg_fes(pmesh, &dg_fec);
   ParFiniteElementSpace cg_fes(pmesh, &cg_fec);

   int n_elems = pmesh->GetGlobalNE();
   HYPRE_Int dg_size = dg_fes.GlobalTrueVSize();
   HYPRE_Int cg_size = cg_fes.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of mesh elements: " << n_elems << endl;
      cout << "Number of unknowns (DG): " << dg_size << endl;
      cout << "Number of unknowns (H1): " << cg_size << endl;
   }

   Array<int> cg_ess_tdof_list;
   cg_fes.GetEssentialTrueDofs(dbc_marker, cg_ess_tdof_list);

   ConstantCoefficient epsilonPerpCoef(epsilon_perp);
   ConstantCoefficient epsilonParaCoef(epsilon_para);
   ConstantCoefficient gammaCoef(gamma);

   AdvectionCoefficient betaCoef(prob);
   SolutionCoefficient solCoef(prob);
   BoundaryCoefficient bdrCoef(prob);
   SourceCoefficient srcCoef(prob, epsilon_para, epsilon_perp, gamma, betaCoef);

   Aniso2DDiffusionCoef epsilonMatCoef(&epsilonParaCoef, &epsilonPerpCoef,
                                       betaCoef);

   // Set up the parallel linear form dg_lf(.) which corresponds to the
   // right-hand side of the FEM linear system.
   ParLinearForm dg_lf(&dg_fes);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   dg_lf.AddDomainIntegrator(new DomainLFIntegrator(srcCoef));
   if (epsilon_para < 0.0)
   {
      dg_lf.AddBdrFaceIntegrator(
         new DGAdvDiffDirichletLFIntegrator(bdrCoef,
                                            epsilonPerpCoef,
                                            betaCoef,
                                            lambda,
                                            sigma,
                                            kappa1,
                                            kappa2),
         dbc_marker);
   }
   else
   {
      dg_lf.AddBdrFaceIntegrator(
         new DGAdvDiffDirichletLFIntegrator(bdrCoef,
                                            epsilonMatCoef,
                                            betaCoef,
                                            &epsilonParaCoef,
                                            &epsilonPerpCoef,
                                            lambda,
                                            sigma,
                                            kappa1,
                                            kappa2),
         dbc_marker);
   }
   dg_lf.Assemble();

   // Define the solution vector x as a parallel finite element grid function
   // corresponding to fespace. Initialize x with initial guess of zero.
   ParGridFunction x_gmres_amg(&dg_fes);
   ParGridFunction x_gmres_air(&dg_fes);
   x_gmres_amg = 0.0;
   x_gmres_air = 0.0;

   // Set up the bilinear form a(.,.) on the finite element space
   // corresponding to the Laplacian operator -Delta, by adding the Diffusion
   // domain integrator and the interior and boundary DG face integrators.
   // Note that boundary conditions are imposed weakly in the form, so there
   // is no need for dof elimination. After serial and parallel assembly we
   // extract the corresponding parallel matrix A.
   ParBilinearForm dg_blf(&dg_fes);
   ParBilinearForm cg_blf(&cg_fes);

   dg_blf.AddDomainIntegrator(new MassIntegrator(gammaCoef));
   cg_blf.AddDomainIntegrator(new MassIntegrator(gammaCoef));

   dg_blf.AddDomainIntegrator(new ConservativeConvectionIntegrator(betaCoef));
   cg_blf.AddDomainIntegrator(new ConservativeConvectionIntegrator(betaCoef));

   if (epsilon_para < 0.0)
   {
      dg_blf.AddDomainIntegrator(new DiffusionIntegrator(epsilonPerpCoef));
      cg_blf.AddDomainIntegrator(new DiffusionIntegrator(epsilonPerpCoef));

      dg_blf.AddInteriorFaceIntegrator(new DGAdvDiffIntegrator(epsilonPerpCoef,
                                                               betaCoef,
                                                               lambda,
                                                               sigma,
                                                               kappa1,
                                                               kappa2));

      dg_blf.AddBdrFaceIntegrator(new DGAdvDiffBdrIntegrator(epsilonPerpCoef,
                                                             betaCoef,
                                                             lambda,
                                                             sigma,
                                                             kappa1,
                                                             kappa2),
                                  dbc_marker);
   }
   else
   {
      dg_blf.AddDomainIntegrator(new DiffusionIntegrator(epsilonMatCoef));
      cg_blf.AddDomainIntegrator(new DiffusionIntegrator(epsilonMatCoef));

      dg_blf.AddInteriorFaceIntegrator(new DGAdvDiffIntegrator(epsilonMatCoef,
                                                               betaCoef,
                                                               &epsilonParaCoef,
                                                               &epsilonPerpCoef,
                                                               lambda,
                                                               sigma,
                                                               kappa1,
                                                               kappa2));
      dg_blf.AddBdrFaceIntegrator(new DGAdvDiffBdrIntegrator(epsilonMatCoef,
                                                             betaCoef,
                                                             &epsilonParaCoef,
                                                             &epsilonPerpCoef,
                                                             lambda,
                                                             sigma,
                                                             kappa1,
                                                             kappa2),
                                  dbc_marker);
   }

   dg_blf.AddBdrFaceIntegrator(new DGTraceIntegrator(betaCoef, 1.0, 0.5));

   dg_blf.Assemble();
   cg_blf.Assemble();
   dg_blf.Finalize();
   cg_blf.Finalize();

   // Define the parallel (hypre) matrix and vectors representing a(.,.),
   // b(.) and the finite element approximation.
   OperatorPtr op_dg_A(Operator::Type::Hypre_ParCSR);
   OperatorPtr op_cg_A(Operator::Type::Hypre_ParCSR);

   dg_blf.ParallelAssemble(op_dg_A);
   HypreParMatrix *dg_A = op_dg_A.As<HypreParMatrix>();

   cg_blf.FormSystemMatrix(cg_ess_tdof_list, op_cg_A);
   HypreParMatrix *cg_A = op_cg_A.As<HypreParMatrix>();

   HypreParVector B(*dg_A);
   dg_lf.ParallelAssemble(B);

   HypreParVector X(*dg_A);
   X = 0.0; GMRES_AMG_Solve(logging, *dg_A, B, X); x_gmres_amg.Distribute(X);
   X = 0.0; GMRES_AIR_Solve(logging, dg_fes, *dg_A, B, X);
   x_gmres_air.Distribute(X);

   // Compute and print the L^2 norm of the error.
   ComputeL2Error("GMRES AMG", x_gmres_amg, solCoef);
   ComputeL2Error("GMRES AIR", x_gmres_air, solCoef);

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      int Wx = 0, Wy = 0, Ww = 275, Wh = 250;

      socketstream sol_sock_0, sol_sock_1;
      VisualizeField(sol_sock_0, vishost, visport, x_gmres_amg, "GMRES AMG",
                     Wx, Wy, Ww, Wh);
      Wx += Ww;
      VisualizeField(sol_sock_1, vishost, visport, x_gmres_air, "GMRES AIR",
                     Wx, Wy, Ww, Wh);
   }

   // Free the used memory.
   delete pmesh;

   return 0;
}

Mesh * BuildSol1DMesh(const Vector &sol1d)
{
   MFEM_VERIFY(sol1d.Size() == 3, "Must have 3 sol1d mesh parameters.");
   int NUMC = (int)sol1d(0);
   double L0 = sol1d(1);
   double beta = sol1d(2);

   Mesh mesh = Mesh::MakeCartesian2D(NUMC, 1, Element::QUADRILATERAL);

   double delx;

   double * xn = new double[NUMC+1];

   if (beta<0.0)
   {
      xn[0] = 0.0;
      delx = L0/((double)(NUMC));
      for (int i=1; i<=NUMC; i++) { xn[i]=xn[i-1]+delx; }
      xn[NUMC+1]=xn[NUMC];
   }
   else
   {
      double ksi;

      double * xtemp = new double[NUMC+1];
      xtemp[0] = 0.0;
      for (int i=1; i<=NUMC; i++)
      {
         ksi=(double)(i)/(double)(NUMC);
         xtemp[i] =L0*( (beta+1.0)-(beta-1.0)*pow((beta+1.0)/(beta-1.0),1.0-ksi) )
                   /( 1.0 + pow((beta+1.0)/(beta-1.0),1.0-ksi) );
      }

      for (int i=0; i<=NUMC; i++) { xn[i]=L0-xtemp[NUMC-i]; }
      xn[NUMC+1]=xn[NUMC];

      delete [] xtemp;
   }

   Vector vert_coord(4*(NUMC+1));

   for (int i=0; i<=NUMC; i++)
   {
      vert_coord(0 * (NUMC + 1) + i) = xn[i];
      vert_coord(1 * (NUMC + 1) + i) = xn[i];
      vert_coord(2 * (NUMC + 1) + i) = 0.0;
      vert_coord(3 * (NUMC + 1) + i) = 0.125 * L0;
   }

   delete [] xn;

   mesh.SetVertices(vert_coord);

   return new Mesh(mesh);
}

void GMRES_AMG_Solve(int logging, HypreParMatrix &A, HypreParVector &B,
                     HypreParVector &X)
{
   int myid = Mpi::WorldRank();

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetAbsTol(0.0);
   gmres.SetRelTol(1e-12);
   gmres.SetMaxIter(500);
   gmres.SetKDim(10);
   gmres.SetPrintLevel(logging);

   HypreBoomerAMG amg;
   amg.SetPrintLevel(0);
   gmres.SetPreconditioner(amg);
   gmres.SetOperator(A);

   gmres.Mult(B, X);
   int its = gmres.GetNumIterations();
   if (myid == 0)
   {
      cout << "GMRES AMG Final Its: " << its << endl;
   }

}

void GMRES_AIR_Solve(int logging, ParFiniteElementSpace &fes,
                     HypreParMatrix &A, HypreParVector &B, HypreParVector &X)
{
   int myid = Mpi::WorldRank();

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetAbsTol(0.0);
   gmres.SetRelTol(1e-12);
   gmres.SetMaxIter(500);
   gmres.SetKDim(10);
   gmres.SetPrintLevel(logging);
#if MFEM_HYPRE_VERSION >= 21800
   // gmres.SetPreconditioner(*amg);
   AIR_prec air(fes.GetFE(0)->GetDof());
   gmres.SetPreconditioner(air);
#else
   HypreEuclid ilu(*dg_A);
   gmres.SetPreconditioner(ilu);
#endif
   gmres.SetOperator(A);
   // gmres.SetMonitor(monitor);
   gmres.Mult(B, X);
   int its = gmres.GetNumIterations();
   if (myid == 0)
   {
      cout << "GMRES AIR Final Its: " << its << endl;
   }

}

void ComputeL2Error(const string &header, ParGridFunction &x, Coefficient &coef)
{
   double sol_err = x.ComputeL2Error(coef);
   if (Mpi::WorldRank() == 0)
   {
      cout << header << ": || u_h - u ||_{L^2} = " << sol_err << endl;
   }
}
