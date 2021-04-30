
//                       MFEM Example 31 - Parallel Version
//
// Compile with: make ex31p
//
// Sample runs:  mpirun -np 4 ex31p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex31p -m ../data/inline-segment.mesh -o 2
//               mpirun -np 4 ex31p -m ../data/star.mesh
//               mpirun -np 4 ex31p -m ../data/square-disc.mesh -o 2 -n 4 -rs 1
//               mpirun -np 4 ex31p -m ../data/square-disc-nurbs.mesh
//               mpirun -np 4 ex31p -m ../data/amr-quad.mesh -o 2 -rs 1
//               mpirun -np 4 ex31p -m ../data/amr-hex.mesh -rs 1
//               mpirun -np 4 ex31p -m ../data/fichera.mesh -rs 1
//
// Description:  This example code solves the Maxwell (electromagnetic)
//               eigenvalue problem curl curl E = lambda epsilon E with
//               an anisotropic dielectric tensor, epsilon, and homogeneous
//               Dirichlet boundary conditions E x n = 0.
//
//               We compute a number of the lowest nonzero eigenmodes by
//               discretizing the curl curl operator using a Nedelec FE space of
//               the specified order in 1D, 2D, or 3D.
//
//               The example highlights the use of restricted H(curl) finite
//               element spaces with the AME subspace eigenvalue solver from
//               HYPRE, which uses LOBPCG and AMS internally. Reusing a single
//               GLVis visualization window for multiple eigenfunctions is also
//               illustrated.
//
//               We recommend viewing examples 30 and 13 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 1;
   int nev = 5;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&nev, "-n", "--num-eigs",
                  "Number of desired eigenmodes.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   // 3. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

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
   pmesh.ReorientTetMesh();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec_nd = NULL;
   FiniteElementCollection *fec_rt = NULL;
   if (dim == 1)
   {
      fec_nd = new ND_R1D_FECollection(order, dim);
      fec_rt = new RT_R1D_FECollection(order-1, dim);
   }
   else if (dim == 2)
   {
      fec_nd = new ND_R2D_FECollection(order, dim);
      fec_rt = new RT_R2D_FECollection(order-1, dim);
   }
   else
   {
      fec_nd = new ND_FECollection(order, dim);
      fec_rt = new RT_FECollection(order-1, dim);
   }
   ParFiniteElementSpace fespace_nd(&pmesh, fec_nd);
   ParFiniteElementSpace fespace_rt(&pmesh, fec_rt);
   HYPRE_Int size_nd = fespace_nd.GlobalTrueVSize();
   HYPRE_Int size_rt = fespace_rt.GlobalTrueVSize();
   mfem::out << "Number of H(Curl) unknowns: " << size_nd << endl;
   mfem::out << "Number of H(Div) unknowns: " << size_rt << endl;

   // 7. Set up the parallel bilinear forms a(.,.) and m(.,.) on the finite
   //    element space. The first corresponds to the curl curl, while the second
   //    is a simple mass matrix needed on the right hand side of the
   //    generalized eigenvalue problem below. The boundary conditions are
   //    implemented by marking all the boundary attributes from the mesh as
   //    essential. The corresponding degrees of freedom are eliminated with
   //    special values on the diagonal to shift the Dirichlet eigenvalues out
   //    of the computational range. After serial and parallel assembly we
   //    extract the corresponding parallel matrices A and M.
   HypreParMatrix *A = NULL;
   HypreParMatrix *M = NULL;
   double shift = 0.0;
   {
      DenseMatrix epsilonMat(3);
      epsilonMat(0,0) = 2.0; epsilonMat(1,1) = 2.0; epsilonMat(2,2) = 2.0;
      epsilonMat(0,2) = 0.0; epsilonMat(2,0) = 0.0;
      epsilonMat(0,1) = M_SQRT1_2; epsilonMat(1,0) = M_SQRT1_2;
      epsilonMat(1,2) = M_SQRT1_2; epsilonMat(2,1) = M_SQRT1_2;
      MatrixConstantCoefficient epsilon(epsilonMat);

      ConstantCoefficient one(1.0);
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
      }

      ParBilinearForm a(&fespace_nd);
      a.AddDomainIntegrator(new CurlCurlIntegrator(one));
      if (pmesh.bdr_attributes.Size() == 0 || dim == 1)
      {
         // Add a mass term if the mesh has no boundary, e.g. periodic mesh or
         // closed surface.
         a.AddDomainIntegrator(new VectorFEMassIntegrator(epsilon));
         shift = 1.0;
         mfem::out << "Computing eigenvalues shifted by " << shift << endl;
      }
      a.Assemble();
      a.EliminateEssentialBCDiag(ess_bdr, 1.0);
      a.Finalize();

      ParBilinearForm m(&fespace_nd);
      m.AddDomainIntegrator(new VectorFEMassIntegrator(epsilon));
      m.Assemble();
      // shift the eigenvalue corresponding to eliminated dofs to a large value
      m.EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
      m.Finalize();

      A = a.ParallelAssemble();
      M = m.ParallelAssemble();
   }

   // 8. Define and configure the AME eigensolver and the AMS preconditioner for
   //    A to be used within the solver. Set the matrices which define the
   //    generalized eigenproblem A x = lambda M x.
   HypreAMS *ams = new HypreAMS(*A,&fespace_nd);
   ams->SetPrintLevel(0);
   ams->SetSingularProblem();

   HypreAME *ame = new HypreAME(MPI_COMM_WORLD);
   ame->SetNumModes(nev);
   ame->SetPreconditioner(*ams);
   ame->SetMaxIter(100);
   ame->SetTol(1e-8);
   ame->SetPrintLevel(1);
   ame->SetMassMatrix(*M);
   ame->SetOperator(*A);

   // 9. Compute the eigenmodes and extract the array of eigenvalues. Define
   //    parallel grid functions to represent each of the eigenmodes returned by
   //    the solver and their derivatives.
   Array<double> eigenvalues;
   ame->Solve();
   ame->GetEigenvalues(eigenvalues);
   ParGridFunction x(&fespace_nd);
   ParGridFunction dx(&fespace_rt);

   ParDiscreteLinearOperator curl(&fespace_nd, &fespace_rt);
   curl.AddDomainInterpolator(new CurlInterpolator);
   curl.Assemble();
   curl.Finalize();

   // 10. Save the refined mesh and the modes in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g mode".
   {
      ostringstream mesh_name, mode_name, mode_deriv_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << mpi.WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      for (int i=0; i<nev; i++)
      {
         // convert eigenvector from HypreParVector to ParGridFunction
         x = ame->GetEigenvector(i);
         curl.Mult(x, dx);

         mode_name << "mode_" << setfill('0') << setw(2) << i << "."
                   << setfill('0') << setw(6) << mpi.WorldRank();
         mode_deriv_name << "mode_deriv_" << setfill('0') << setw(2) << i << "."
                         << setfill('0') << setw(6) << mpi.WorldRank();

         ofstream mode_ofs(mode_name.str().c_str());
         mode_ofs.precision(8);
         x.Save(mode_ofs);
         mode_name.str("");

         ofstream mode_deriv_ofs(mode_deriv_name.str().c_str());
         mode_deriv_ofs.precision(8);
         dx.Save(mode_deriv_ofs);
         mode_deriv_name.str("");
      }
   }

   // 11. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      if (dim == 1)
      {
         socketstream mode_x_sock(vishost, visport);
         socketstream mode_y_sock(vishost, visport);
         socketstream mode_z_sock(vishost, visport);
         socketstream mode_dy_sock(vishost, visport);
         socketstream mode_dz_sock(vishost, visport);
         mode_x_sock.precision(8);
         mode_y_sock.precision(8);
         mode_z_sock.precision(8);
         mode_dy_sock.precision(8);
         mode_dz_sock.precision(8);

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

         for (int i=0; i<nev; i++)
         {
            mfem::out << "Eigenmode " << i+1 << '/' << nev
                      << ", Lambda = " << eigenvalues[i] - shift << endl;

            // convert eigenvector from HypreParVector to ParGridFunction
            x = ame->GetEigenvector(i);
            curl.Mult(x, dx);

            {
               VectorGridFunctionCoefficient modeCoef(&x);
               InnerProductCoefficient xCoef(xVecCoef, modeCoef);
               InnerProductCoefficient yCoef(yVecCoef, modeCoef);
               InnerProductCoefficient zCoef(zVecCoef, modeCoef);

               xComp.ProjectCoefficient(xCoef);
               yComp.ProjectCoefficient(yCoef);
               zComp.ProjectCoefficient(zCoef);

               mode_x_sock << "parallel " << mpi.WorldSize() << " "
                           << mpi.WorldRank() << "\n"
                           << "solution\n" << pmesh << xComp << flush
                           << "window_title 'Eigenmode " << i+1 << '/' << nev
                           << " X, Lambda = " << eigenvalues[i] - shift
                           << "'" << endl;
               mode_y_sock << "parallel " << mpi.WorldSize() << " "
                           << mpi.WorldRank() << "\n"
                           << "solution\n" << pmesh << yComp << flush
                           << "window_geometry 403 0 400 350 "
                           << "window_title 'Eigenmode " << i+1 << '/' << nev
                           << " Y, Lambda = " << eigenvalues[i] - shift
                           << "'" << endl;
               mode_z_sock << "parallel " << mpi.WorldSize() << " "
                           << mpi.WorldRank() << "\n"
                           << "solution\n" << pmesh << zComp << flush
                           << "window_geometry 806 0 400 350 "
                           << "window_title 'Eigenmode " << i+1 << '/' << nev
                           << " Z, Lambda = " << eigenvalues[i] - shift
                           << "'" << endl;

               VectorGridFunctionCoefficient dmodeCoef(&dx);
               InnerProductCoefficient dyCoef(yVecCoef, dmodeCoef);
               InnerProductCoefficient dzCoef(zVecCoef, dmodeCoef);

               dyComp.ProjectCoefficient(dyCoef);
               dzComp.ProjectCoefficient(dzCoef);

               mode_dy_sock << "parallel " << mpi.WorldSize() << " "
                            << mpi.WorldRank() << "\n"
                            << "solution\n" << pmesh << dyComp << flush
                            << "window_geometry 0 375 400 350 "
                            << "window_title 'Curl Eigenmode "
                            << i+1 << '/' << nev
                            << " Y, Lambda = " << eigenvalues[i] - shift
                            << "'" << endl;
               mode_dz_sock << "parallel " << mpi.WorldSize() << " "
                            << mpi.WorldRank() << "\n"
                            << "solution\n" << pmesh << dzComp << flush
                            << "window_geometry 403 375 400 350 "
                            << "window_title 'Curl Eigenmode "
                            << i+1 << '/' << nev
                            << " Z, Lambda = " << eigenvalues[i] - shift
                            << "'" << endl;
            }
            char c;
            mfem::out << "press (q)uit or (c)ontinue --> " << flush;
            if (mpi.Root())
            {
               cin >> c;
            }
            MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

            if (c != 'c')
            {
               break;
            }
         }
         mode_x_sock.close();
         mode_y_sock.close();
         mode_z_sock.close();
         mode_dy_sock.close();
         mode_dz_sock.close();
      }
      else if (dim == 2)
      {
         socketstream mode_xy_sock(vishost, visport);
         socketstream mode_z_sock(vishost, visport);
         socketstream mode_dxy_sock(vishost, visport);
         socketstream mode_dz_sock(vishost, visport);
         mode_xy_sock.precision(8);
         mode_z_sock.precision(8);
         mode_dxy_sock.precision(8);
         mode_dz_sock.precision(8);

         DenseMatrix xyMat(2,3); xyMat = 0.0;
         xyMat(0,0) = 1.0; xyMat(1,1) = 1.0;
         MatrixConstantCoefficient xyMatCoef(xyMat);
         Vector zVec(3); zVec = 0.0; zVec(2) = 1;
         VectorConstantCoefficient zVecCoef(zVec);

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

         for (int i=0; i<nev; i++)
         {
            mfem::out << "Eigenmode " << i+1 << '/' << nev
                      << ", Lambda = " << eigenvalues[i] - shift << endl;

            // convert eigenvector from HypreParVector to ParGridFunction
            x = ame->GetEigenvector(i);
            curl.Mult(x, dx);

            {
               VectorGridFunctionCoefficient modeCoef(&x);
               MatrixVectorProductCoefficient xyCoef(xyMatCoef, modeCoef);
               InnerProductCoefficient zCoef(zVecCoef, modeCoef);

               xyComp.ProjectCoefficient(xyCoef);
               zComp.ProjectCoefficient(zCoef);

               mode_xy_sock << "parallel " << mpi.WorldSize() << " "
                            << mpi.WorldRank() << "\n"
                            << "solution\n" << pmesh << xyComp << flush
                            << "keys vvv "
                            << "window_title 'Eigenmode " << i+1 << '/' << nev
                            << " XY, Lambda = " << eigenvalues[i] - shift
                            << "'" << endl;
               mode_z_sock << "parallel " << mpi.WorldSize() << " "
                           << mpi.WorldRank() << "\n"
                           << "solution\n" << pmesh << zComp << flush
                           << "window_geometry 403 0 400 350 "
                           << "window_title 'Eigenmode " << i+1 << '/' << nev
                           << " Z, Lambda = " << eigenvalues[i] - shift
                           << "'" << endl;

               VectorGridFunctionCoefficient dmodeCoef(&dx);
               MatrixVectorProductCoefficient dxyCoef(xyMatCoef, dmodeCoef);
               InnerProductCoefficient dzCoef(zVecCoef, dmodeCoef);

               dxyComp.ProjectCoefficient(dxyCoef);
               dzComp.ProjectCoefficient(dzCoef);

               mode_dxy_sock << "parallel " << mpi.WorldSize() << " "
                             << mpi.WorldRank() << "\n"
                             << "solution\n" << pmesh << dxyComp << flush
                             << "keys vvv "
                             << "window_geometry 0 375 400 350 "
                             << "window_title 'Curl Eigenmode "
                             << i+1 << '/' << nev
                             << " XY, Lambda = " << eigenvalues[i] - shift
                             << "'" << endl;
               mode_dz_sock << "parallel " << mpi.WorldSize() << " "
                            << mpi.WorldRank() << "\n"
                            << "solution\n" << pmesh << dzComp << flush
                            << "window_geometry 403 375 400 350 "
                            << "window_title 'Curl Eigenmode "
                            << i+1 << '/' << nev
                            << " Z, Lambda = " << eigenvalues[i] - shift
                            << "'" << endl;
            }
            char c;
            mfem::out << "press (q)uit or (c)ontinue --> " << flush;
            if (mpi.Root())
            {
               cin >> c;
            }
            MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

            if (c != 'c')
            {
               break;
            }
         }
         mode_xy_sock.close();
         mode_z_sock.close();
         mode_dxy_sock.close();
         mode_dz_sock.close();
      }
      else
      {
         socketstream mode_sock(vishost, visport);
         socketstream mode_deriv_sock(vishost, visport);
         mode_sock.precision(8);
         mode_deriv_sock.precision(8);

         for (int i=0; i<nev; i++)
         {
            if (mpi.Root())
            {
               cout << "Eigenmode " << i+1 << '/' << nev
                    << ", Lambda = " << eigenvalues[i] - shift << endl;
            }

            // convert eigenvector from HypreParVector to ParGridFunction
            x = ame->GetEigenvector(i);
            curl.Mult(x, dx);

            mode_sock << "parallel " << mpi.WorldSize() << " "
                      << mpi.WorldRank() << "\n"
                      << "solution\n" << pmesh << x << flush
                      << "window_title 'Eigenmode " << i+1 << '/' << nev
                      << ", Lambda = " << eigenvalues[i] - shift
                      << "'" << endl;
            mode_deriv_sock << "parallel " << mpi.WorldSize() << " "
                            << mpi.WorldRank() << "\n"
                            << "solution\n" << pmesh << dx << flush
                            << "window_geometry 0 375 400 350 "
                            << "window_title 'Curl Eigenmode "
                            << i+1 << '/' << nev
                            << ", Lambda = " << eigenvalues[i] - shift
                            << "'" << endl;

            char c;
            if (mpi.Root())
            {
               cout << "press (q)uit or (c)ontinue --> " << flush;
               cin >> c;
            }
            MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

            if (c != 'c')
            {
               break;
            }
         }
         mode_sock.close();
      }
   }

   // 12. Free the used memory.
   delete ame;
   delete ams;
   delete M;
   delete A;

   delete fec_nd;
   delete fec_rt;

   return 0;
}
