//                       MFEM Example 31 - Parallel Version
//
// Compile with: make ex31p
//
// Sample runs:  mpirun -np 4 ex31p -m ../data/hexagon.mesh -o 2
//               mpirun -np 4 ex31p -m ../data/star.mesh
//               mpirun -np 4 ex31p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex31p -m ../data/fichera.mesh -o 3 -rs 1 -rp 0
//               mpirun -np 4 ex31p -m ../data/square-disc-nurbs.mesh -o 3
//               mpirun -np 4 ex31p -m ../data/amr-quad.mesh -o 2 -rs 1
//               mpirun -np 4 ex31p -m ../data/amr-hex.mesh -rs 1
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + sigma E = f with boundary condition
//               E x n = <given tangential field>. In this example sigma is an
//               anisotropic 3x3 tensor. Here, we use a given exact solution E
//               and compute the corresponding r.h.s. f.  We discretize with
//               Nedelec finite elements in 1D, 2D, or 3D.
//
//               The example demonstrates the use of restricted H(curl) finite
//               element spaces with the curl-curl and the (vector finite
//               element) mass bilinear form, as well as the computation of
//               discretization error when the exact solution is known. These
//               restricted spaces allow the solution of 1D or 2D
//               electromagnetic problems which involve 3D field vectors.  Such
//               problems arise in plasma physics and crystallography.
//
//               We recommend viewing example 3 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void CurlE_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
real_t freq = 1.0, kappa;
int dim;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 1;
   bool use_ams = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&use_ams, "-ams", "--hypre-ams", "-slu",
                  "--superlu", "Use AMS or SuperLU solver.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   kappa = freq * M_PI;

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement (2 by default, or
   //    specified on the command line with -rs).
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution (1 time by
   //    default, or specified on the command line with -rp). Once the parallel
   //    mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = NULL;
   if (dim == 1)
   {
      fec = new ND_R1D_FECollection(order, dim);
   }
   else if (dim == 2)
   {
      fec = new ND_R2D_FECollection(order, dim);
   }
   else
   {
      fec = new ND_FECollection(order, dim);
   }
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   if (Mpi::Root()) { cout << "Number of H(Curl) unknowns: " << size << endl; }

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(3, f_exact);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b.Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction sol(&fespace);
   VectorFunctionCoefficient E(3, E_exact);
   VectorFunctionCoefficient CurlE(3, CurlE_exact);
   sol.ProjectCoefficient(E);

   // 10. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   DenseMatrix sigmaMat(3);
   sigmaMat(0,0) = 2.0; sigmaMat(1,1) = 2.0; sigmaMat(2,2) = 2.0;
   sigmaMat(0,2) = 0.0; sigmaMat(2,0) = 0.0;
   sigmaMat(0,1) = M_SQRT1_2; sigmaMat(1,0) = M_SQRT1_2; // 1/sqrt(2) in cmath
   sigmaMat(1,2) = M_SQRT1_2; sigmaMat(2,1) = M_SQRT1_2;

   ConstantCoefficient muinv(1.0);
   MatrixConstantCoefficient sigma(sigmaMat);
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));

   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, etc.
   a.Assemble();

   OperatorPtr A;
   Vector B, X;

   a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B);

   // 12. Solve the system AX=B using PCG with the AMS preconditioner from hypre
   if (use_ams)
   {
      if (Mpi::Root())
      {
         cout << "Size of linear system: "
              << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;
      }

      HypreAMS ams(*A.As<HypreParMatrix>(), &fespace);

      HyprePCG pcg(*A.As<HypreParMatrix>());
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(1000);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(ams);
      pcg.Mult(B, X);
   }
   else
#ifdef MFEM_USE_SUPERLU
   {
      if (Mpi::Root())
      {
         cout << "Size of linear system: "
              << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;
      }

      SuperLURowLocMatrix A_SuperLU(*A.As<HypreParMatrix>());
      SuperLUSolver AInv(MPI_COMM_WORLD);
      AInv.SetOperator(A_SuperLU);
      AInv.Mult(B,X);
   }
#else
   {
      if (Mpi::Root()) { cout << "No solvers available." << endl; }
      return 1;
   }
#endif

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, sol);

   // 14. Compute and print the H(Curl) norm of the error.
   {
      real_t error = sol.ComputeHCurlError(&E, &CurlE);
      if (Mpi::Root())
      {
         cout << "\n|| E_h - E ||_{H(Curl)} = " << error << '\n' << endl;
      }
   }


   // 15. Save the refined mesh and the solution in parallel. This output can
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
      sol.Save(sol_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      VectorGridFunctionCoefficient solCoef(&sol);
      CurlGridFunctionCoefficient dsolCoef(&sol);

      if (dim ==1)
      {
         socketstream x_sock(vishost, visport);
         socketstream y_sock(vishost, visport);
         socketstream z_sock(vishost, visport);
         socketstream dy_sock(vishost, visport);
         socketstream dz_sock(vishost, visport);
         x_sock.precision(8);
         y_sock.precision(8);
         z_sock.precision(8);
         dy_sock.precision(8);
         dz_sock.precision(8);

         Vector xVec(3); xVec = 0.0; xVec(0) = 1;
         Vector yVec(3); yVec = 0.0; yVec(1) = 1;
         Vector zVec(3); zVec = 0.0; zVec(2) = 1;
         VectorConstantCoefficient xVecCoef(xVec);
         VectorConstantCoefficient yVecCoef(yVec);
         VectorConstantCoefficient zVecCoef(zVec);

         H1_FECollection fec_h1(order, dim);
         L2_FECollection fec_l2(order-1, dim);

         ParFiniteElementSpace fes_h1(&pmesh, &fec_h1);
         ParFiniteElementSpace fes_l2(&pmesh, &fec_l2);

         ParGridFunction xComp(&fes_l2);
         ParGridFunction yComp(&fes_h1);
         ParGridFunction zComp(&fes_h1);

         ParGridFunction dyComp(&fes_l2);
         ParGridFunction dzComp(&fes_l2);

         InnerProductCoefficient xCoef(xVecCoef, solCoef);
         InnerProductCoefficient yCoef(yVecCoef, solCoef);
         InnerProductCoefficient zCoef(zVecCoef, solCoef);

         xComp.ProjectCoefficient(xCoef);
         yComp.ProjectCoefficient(yCoef);
         zComp.ProjectCoefficient(zCoef);

         x_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << xComp << flush
                << "window_title 'X component'" << endl;
         y_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << yComp << flush
                << "window_geometry 403 0 400 350 "
                << "window_title 'Y component'" << endl;
         z_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << zComp << flush
                << "window_geometry 806 0 400 350 "
                << "window_title 'Z component'" << endl;

         InnerProductCoefficient dyCoef(yVecCoef, dsolCoef);
         InnerProductCoefficient dzCoef(zVecCoef, dsolCoef);

         dyComp.ProjectCoefficient(dyCoef);
         dzComp.ProjectCoefficient(dzCoef);

         dy_sock << "parallel " << num_procs << " " << myid << "\n"
                 << "solution\n" << pmesh << dyComp << flush
                 << "window_geometry 403 375 400 350 "
                 << "window_title 'Y component of Curl'" << endl;
         dz_sock << "parallel " << num_procs << " " << myid << "\n"
                 << "solution\n" << pmesh << dzComp << flush
                 << "window_geometry 806 375 400 350 "
                 << "window_title 'Z component of Curl'" << endl;
      }
      else if (dim == 2)
      {
         socketstream xy_sock(vishost, visport);
         socketstream z_sock(vishost, visport);
         socketstream dxy_sock(vishost, visport);
         socketstream dz_sock(vishost, visport);

         DenseMatrix xyMat(2,3); xyMat = 0.0;
         xyMat(0,0) = 1.0; xyMat(1,1) = 1.0;
         MatrixConstantCoefficient xyMatCoef(xyMat);
         Vector zVec(3); zVec = 0.0; zVec(2) = 1;
         VectorConstantCoefficient zVecCoef(zVec);

         MatrixVectorProductCoefficient xyCoef(xyMatCoef, solCoef);
         InnerProductCoefficient zCoef(zVecCoef, solCoef);

         H1_FECollection fec_h1(order, dim);
         ND_FECollection fec_nd(order, dim);
         RT_FECollection fec_rt(order-1, dim);
         L2_FECollection fec_l2(order-1, dim);

         ParFiniteElementSpace fes_h1(&pmesh, &fec_h1);
         ParFiniteElementSpace fes_nd(&pmesh, &fec_nd);
         ParFiniteElementSpace fes_rt(&pmesh, &fec_rt);
         ParFiniteElementSpace fes_l2(&pmesh, &fec_l2);

         ParGridFunction xyComp(&fes_nd);
         ParGridFunction zComp(&fes_h1);

         ParGridFunction dxyComp(&fes_rt);
         ParGridFunction dzComp(&fes_l2);

         xyComp.ProjectCoefficient(xyCoef);
         zComp.ProjectCoefficient(zCoef);

         xy_sock << "parallel " << num_procs << " " << myid << "\n";
         xy_sock.precision(8);
         xy_sock << "solution\n" << pmesh << xyComp
                 << "window_title 'XY components'\n" << flush;
         z_sock << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << zComp << flush
                << "window_geometry 403 0 400 350 "
                << "window_title 'Z component'" << endl;

         MatrixVectorProductCoefficient dxyCoef(xyMatCoef, dsolCoef);
         InnerProductCoefficient dzCoef(zVecCoef, dsolCoef);

         dxyComp.ProjectCoefficient(dxyCoef);
         dzComp.ProjectCoefficient(dzCoef);

         dxy_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << pmesh << dxyComp << flush
                  << "window_geometry 0 375 400 350 "
                  << "window_title 'XY components of Curl'" << endl;
         dz_sock << "parallel " << num_procs << " " << myid << "\n"
                 << "solution\n" << pmesh << dzComp << flush
                 << "window_geometry 403 375 400 350 "
                 << "window_title 'Z component of Curl'" << endl;
      }
      else
      {
         socketstream sol_sock(vishost, visport);
         socketstream dsol_sock(vishost, visport);

         RT_FECollection fec_rt(order-1, dim);

         ParFiniteElementSpace fes_rt(&pmesh, &fec_rt);

         ParGridFunction dsol(&fes_rt);

         dsol.ProjectCoefficient(dsolCoef);

         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << pmesh << sol
                  << "window_title 'Solution'" << flush << endl;
         dsol_sock << "parallel " << num_procs << " " << myid << "\n"
                   << "solution\n" << pmesh << dsol << flush
                   << "window_geometry 0 375 400 350 "
                   << "window_title 'Curl of solution'" << endl;
      }
   }

   // 17. Free the used memory.
   delete fec;

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 1)
   {
      E(0) = 1.1 * sin(kappa * x(0) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * x(0) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * x(0) + 0.9 * M_PI);
   }
   else if (dim == 2)
   {
      E(0) = 1.1 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
   }
   else
   {
      E(0) = 1.1 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      E *= cos(kappa * x(2));
   }
}

void CurlE_exact(const Vector &x, Vector &dE)
{
   if (dim == 1)
   {
      real_t c4 = cos(kappa * x(0) + 0.4 * M_PI);
      real_t c9 = cos(kappa * x(0) + 0.9 * M_PI);

      dE(0) =  0.0;
      dE(1) = -1.3 * c9;
      dE(2) =  1.2 * c4;
      dE *= kappa;
   }
   else if (dim == 2)
   {
      real_t c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);

      dE(0) =  1.3 * c9;
      dE(1) = -1.3 * c9;
      dE(2) =  1.2 * c4 - 1.1 * c0;
      dE *= kappa * M_SQRT1_2;
   }
   else
   {
      real_t s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      real_t sk = sin(kappa * x(2));
      real_t ck = cos(kappa * x(2));

      dE(0) =  1.2 * s4 * sk + 1.3 * M_SQRT1_2 * c9 * ck;
      dE(1) = -1.1 * s0 * sk - 1.3 * M_SQRT1_2 * c9 * ck;
      dE(2) = -M_SQRT1_2 * (1.1 * c0 - 1.2 * c4) * ck;
      dE *= kappa;
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 1)
   {
      real_t s0 = sin(kappa * x(0) + 0.0 * M_PI);
      real_t s4 = sin(kappa * x(0) + 0.4 * M_PI);
      real_t s9 = sin(kappa * x(0) + 0.9 * M_PI);

      f(0) = 2.2 * s0 + 1.2 * M_SQRT1_2 * s4;
      f(1) = 1.2 * (2.0 + kappa * kappa) * s4 +
             M_SQRT1_2 * (1.1 * s0 + 1.3 * s9);
      f(2) = 1.3 * (2.0 + kappa * kappa) * s9 + 1.2 * M_SQRT1_2 * s4;
   }
   else if (dim == 2)
   {
      real_t s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t s9 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);

      f(0) = 0.55 * (4.0 + kappa * kappa) * s0 +
             0.6 * (M_SQRT2 - kappa * kappa) * s4;
      f(1) = 0.55 * (M_SQRT2 - kappa * kappa) * s0 +
             0.6 * (4.0 + kappa * kappa) * s4 +
             0.65 * M_SQRT2 * s9;
      f(2) = 0.6 * M_SQRT2 * s4 + 1.3 * (2.0 + kappa * kappa) * s9;
   }
   else
   {
      real_t s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t s9 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      real_t c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      real_t sk = sin(kappa * x(2));
      real_t ck = cos(kappa * x(2));

      f(0) = 0.55 * (4.0 + 3.0 * kappa * kappa) * s0 * ck +
             0.6 * (M_SQRT2 - kappa * kappa) * s4 * ck -
             0.65 * M_SQRT2 * kappa * kappa * c9 * sk;

      f(1) = 0.55 * (M_SQRT2 - kappa * kappa) * s0 * ck +
             0.6 * (4.0 + 3.0 * kappa * kappa) * s4 * ck +
             0.65 * M_SQRT2 * s9 * ck -
             0.65 * M_SQRT2 * kappa * kappa * c9 * sk;

      f(2) = 0.6 * M_SQRT2 * s4 * ck -
             M_SQRT2 * kappa * kappa * (0.55 * c0 + 0.6 * c4) * sk
             + 1.3 * (2.0 + kappa * kappa) * s9 * ck;
   }
}
