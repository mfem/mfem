//                       MFEM Example 14 - Parallel Version
//
// Compile with: make ex14p
//
// Sample runs:  mpirun -np 4 ex14p -m ../data/inline-quad.mesh -o 0
//               mpirun -np 4 ex14p -m ../data/star.mesh -o 2
//               mpirun -np 4 ex14p -m ../data/star-mixed.mesh -o 2
//               mpirun -np 4 ex14p -m ../data/star-mixed.mesh -o 2 -k 0 -e 1
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
#include "transport_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::plasma::transport;

class CustomSolverMonitor : public IterativeSolverMonitor
{
public:
   CustomSolverMonitor(const ParMesh *m,
                       ParGridFunction *f) :
      pmesh(m),
      pgf(f) {}

   void MonitorSolution(int i, double norm, const Vector &x, bool final)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      int  num_procs, myid;

      MPI_Comm_size(pmesh->GetComm(),&num_procs);
      MPI_Comm_rank(pmesh->GetComm(),&myid);

      pgf->SetFromTrueDofs(x);

      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << *pgf
               << "window_title 'Iteration no " << i << "'"
               << "keys rRjlc\n" << flush;
   }

private:
   const ParMesh *pmesh;
   ParGridFunction *pgf;
};

class DBCCoefficient : public Coefficient
{
private:
   double Lx_, Ly_, hx_, hy_;
   mutable Vector x_;

public:
   DBCCoefficient(double Lx, double Ly, int nx, int ny)
      : Lx_(Lx), Ly_(Ly), hx_(Lx/nx), hy_(Ly/ny), x_(2)
   {}

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);

      if (x_[0] < hx_)
      {
         if (x_[1] < hy_)
         {
            return 0.5 + 0.5 * x_[0] / hx_ - 0.5 * x_[1] / hy_;
         }
         else
         {
            return 0.0;
         }
      }
      else if (x_[0] > Lx_ - hx_)
      {
         if (x_[1] > Ly_ - hy_)
         {
            return 0.5 - 0.5 * (Lx_ - x_[0]) / hx_ + 0.5 * (Ly_ - x_[1]) / hy_;
         }
         else
         {
            return 1.0;
         }
      }
      else if (x_[1] < hy_)
      {
         return 1.0;
      }
      else if (x_[1] > Ly_ - hy_)
      {
         return 0.0;
      }
      else
      {
         return 0.5;
      }
   }
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   // const char *mesh_file = "./umansky_10x10.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   int order = 1;
   int dim = 2;
   int nx = 10;
   int ny = 10;
   double Lx = 10.0;
   double Ly = 10.0;
   double K_eta = 1.0;
   double K_xi = 1.0;
   double A_K = -1.0;
   double theta_m = 0.0;
   double sigma = -1.0;
   double kappa = -1.0;
   bool triangles = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   // args.AddOption(&mesh_file, "-m", "--mesh",
   //             "Mesh file to use.");
   args.AddOption(&nx, "-nx", "--nx",
                  "Number of elements in the x-direction.");
   args.AddOption(&ny, "-ny", "--ny",
                  "Number of elements in the y-direction.");
   args.AddOption(&Lx, "-Lx", "--Lx",
                  "Width of the domain in the x-direction.");
   args.AddOption(&Ly, "-Ly", "--Ly",
                  "Width of the domain in the y-direction.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&K_eta, "-K-eta", "--K-eta",
                  "Perpendicular diffusion coefficient.");
   args.AddOption(&K_xi, "-K-xi", "--K-xi",
                  "Parallel diffusion coefficient.");
   args.AddOption(&A_K, "-A-K", "--A-K",
                  "Anisotropy ratio.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the three DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&triangles, "-tri", "--triangles", "-quad",
                  "--quadrilaterals",
                  "Triangle or quadrilateral elements.");
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
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   if (A_K < 0.0)
   {
      A_K = K_xi / K_eta;
   }
   else
   {
      K_eta = 1.0;
      K_xi = A_K;
   }
   {
      theta_m = atan2(Ly, Lx);
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code. NURBS meshes are projected to second order meshes.
   ParMesh pmesh;
   {
      Mesh mesh = Mesh::MakeCartesian2D(nx, ny,
                                        triangles ?
                                        Element::TRIANGLE :
                                        Element::QUADRILATERAL,
                                        false, Lx, Ly);

      // 4. Refine the serial mesh on all processors.
      for (int l = 0; l < ser_ref_levels; l++)
      {
         mesh.UniformRefinement();
      }

      // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
      //    this mesh further in parallel to increase the resolution. Once the
      //    parallel mesh is defined, the serial mesh can be deleted.
      pmesh = ParMesh(MPI_COMM_WORLD, mesh);
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use discontinuous finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(&pmesh, fec);
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   // 7. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system.
   if (myid == 0)
   {
      cout << "Angle theta_m (radians): " << theta_m << endl;
      cout << "Angle theta_m (degrees): " << 180.0 * theta_m / M_PI << endl;
   }
   ConstantCoefficient one_Coef(1.0);
   ConstantCoefficient zero_Coef(0.0);

   Vector Zero_vec(2); Zero_vec = 0.0;
   VectorConstantCoefficient Zero_Coef(Zero_vec);

   ConstantCoefficient K_xi_Coef(K_xi);
   ConstantCoefficient K_eta_Coef(K_eta);

   /*
   Vector DBC_vals(pmesh.bdr_attributes.Max());
   DBC_vals = 0.0; DBC_vals[0] = 1.0; DBC_vals[1] = 1.0;
   PWConstCoefficient DBC_Coef(DBC_vals);
   */
   int nf = (int)pow(2, ser_ref_levels + par_ref_levels);
   DBCCoefficient DBC_Coef(Lx, Ly, nx * nf, ny * nf);

   Vector B_vec(2); B_vec[0] = cos(theta_m); B_vec[1] = sin(theta_m);
   VectorConstantCoefficient B_Coef(B_vec);

   DenseMatrix K_mat(2);
   K_mat(0,0) = K_xi * B_vec[0] * B_vec[0] + K_eta * B_vec[1] * B_vec[1];
   K_mat(0,1) = K_xi * B_vec[0] * B_vec[1] - K_eta * B_vec[0] * B_vec[1];
   K_mat(1,0) = K_mat(0,1);
   K_mat(1,1) = K_xi * B_vec[1] * B_vec[1] + K_eta * B_vec[0] * B_vec[0];
   MatrixConstantCoefficient K_Coef(K_mat);

   ParLinearForm *b = new ParLinearForm(fespace);
   // b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->AddBdrFaceIntegrator(new DGAdvDiffDirichletLFIntegrator(DBC_Coef,
                                                              K_Coef,
                                                              Zero_Coef,
                                                              &K_xi_Coef,
                                                              &K_eta_Coef,
                                                              1.0,
                                                              sigma, kappa,
                                                              0.0));
   b->Assemble();

   // 8. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   ParGridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and the interior and boundary DG face integrators.
   //    Note that boundary conditions are imposed weakly in the form, so there
   //    is no need for dof elimination. After serial and parallel assembly we
   //    extract the corresponding parallel matrix A.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one_Coef));
   a->AddInteriorFaceIntegrator(new DGAdvDiffIntegrator(K_Coef,
                                                        Zero_Coef,
                                                        &K_xi_Coef,
                                                        &K_eta_Coef,
                                                        1.0,
                                                        sigma, kappa,
                                                        0.0));
   a->AddBdrFaceIntegrator(new DGAdvDiffBdrIntegrator(K_Coef,
                                                      Zero_Coef,
                                                      &K_xi_Coef,
                                                      &K_eta_Coef,
                                                      1.0,
                                                      sigma, kappa,
                                                      0.0));

   a->Assemble();
   a->Finalize();

   // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelProject();

   {
      HypreParVector y(*X);
      HypreParVector Ay(*X);
      HypreParVector ATy(*X);

      y.Randomize(123);
      A->Mult(y, Ay);
      A->MultTranspose(y, ATy);

      double yAy = InnerProduct(y, Ay);
      double yATy = InnerProduct(y, ATy);

      cout << "yTAy = " << yAy << endl;
      cout << "y^T(A-A^T)y " << (yAy - yATy) << endl;
      cout << "y^T(A-A^T)y/yTAy " << (yAy - yATy) / yAy << endl;
   }

   delete a;
   delete b;

   // 11. Depending on the symmetry of A, define and apply a parallel PCG or
   //     GMRES solver for AX=B using the BoomerAMG preconditioner from hypre.
   HypreSolver *amg = new HypreBoomerAMG(*A);
   if (sigma == -1.0 && false)
   {
      HyprePCG pcg(*A);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(5000);
      pcg.SetPrintLevel(1);
      pcg.SetPreconditioner(*amg);
      pcg.Mult(*B, *X);
   }
   else
   {
      // CustomSolverMonitor monitor(&pmesh, &x);
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetAbsTol(0.0);
      gmres.SetRelTol(1e-12);
      gmres.SetMaxIter(5000);
      gmres.SetKDim(10);
      gmres.SetPrintLevel(1);
      gmres.SetOperator(*A);
      gmres.SetPreconditioner(*amg);
      // gmres.SetMonitor(monitor);
      gmres.Mult(*B, *X);
   }
   delete amg;

   // 12. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor.
   x = *X;

   // 13. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   // 15. Free the used memory.
   delete X;
   delete B;
   delete A;

   delete fespace;
   delete fec;

   return 0;
}
