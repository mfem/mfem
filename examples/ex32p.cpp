//                       MFEM Example 32 - Parallel Version
//
// Compile with: make ex32p
//
// Sample runs:  mpirun -np 4 ex32p -m ../data/hexagon.mesh -o 2
//               mpirun -np 4 ex32p -m ../data/star.mesh
//               mpirun -np 4 ex32p -m ../data/square-disc.mesh -o 2 -n 4 -rs 1
//               mpirun -np 4 ex32p -m ../data/square-disc-nurbs.mesh -rs 3 -o 3
//               mpirun -np 4 ex32p -m ../data/amr-quad.mesh -o 2 -rs 1
//               mpirun -np 4 ex32p -m ../data/amr-hex.mesh -rs 1
//               mpirun -np 4 ex32p -m ../data/fichera.mesh -rs 1
//
// Description:  This example code solves the Maxwell (electromagnetic)
//               eigenvalue problem curl curl E = lambda epsilon E with an
//               anisotropic dielectric tensor, epsilon, and homogeneous
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
//               We recommend viewing examples 31 and 13 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t GetVectorMax(int vdim, const ParGridFunction &x);
real_t GetScalarMax(const ParGridFunction &x);

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
   args.ParseCheck();

   // 3. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   Vector bbMin(dim);
   Vector bbMax(dim);
   mesh->GetBoundingBox(bbMin, bbMax);

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
   if (Mpi::Root())
   {
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      cout << "Number of H(Div) unknowns: " << size_rt << endl;
   }

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
   real_t shift = 0.0;
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
         if (Mpi::Root())
         {
            cout << "Computing eigenvalues shifted by " << shift << endl;
         }
      }
      a.Assemble();
      a.EliminateEssentialBCDiag(ess_bdr, 1.0);
      a.Finalize();

      ParBilinearForm m(&fespace_nd);
      m.AddDomainIntegrator(new VectorFEMassIntegrator(epsilon));
      m.Assemble();
      // shift the eigenvalue corresponding to eliminated dofs to a large value
      m.EliminateEssentialBCDiag(ess_bdr, numeric_limits<real_t>::min());
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
   Array<real_t> eigenvalues;
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
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      for (int i=0; i<nev; i++)
      {
         // convert eigenvector from HypreParVector to ParGridFunction
         x = ame->GetEigenvector(i);
         curl.Mult(x, dx);

         mode_name << "mode_" << setfill('0') << setw(2) << i << "."
                   << setfill('0') << setw(6) << myid;
         mode_deriv_name << "mode_deriv_" << setfill('0') << setw(2) << i << "."
                         << setfill('0') << setw(6) << myid;

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
            if (Mpi::Root())
            {
               cout << "Eigenmode " << i+1 << '/' << nev
                    << ", Lambda = " << eigenvalues[i] - shift << endl;
            }

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

               real_t max_x = GetScalarMax(xComp);
               real_t max_y = GetScalarMax(yComp);
               real_t max_z = GetScalarMax(zComp);
               real_t max_r = std::max(max_x, std::max(max_y, max_z));

               ostringstream x_cmd;
               x_cmd << " window_title 'Eigenmode " << i+1 << '/' << nev
                     << " X, Lambda = " << eigenvalues[i] - shift << "'"
                     << " valuerange -"<< max_r << ' ' << max_r;
               if (i == 0)
               {
                  x_cmd << " keys aa"
                        << " window_geometry 0 0 400 350";
               }

               mode_x_sock << "parallel " << num_procs << " " << myid << "\n"
                           << "solution\n" << pmesh << xComp << flush
                           << x_cmd.str() << endl << flush;

               MPI_Barrier(MPI_COMM_WORLD);

               ostringstream y_cmd;
               y_cmd << " window_title 'Eigenmode " << i+1 << '/' << nev
                     << " Y, Lambda = " << eigenvalues[i] - shift << "'"
                     << " valuerange -"<< max_r << ' ' << max_r;
               if (i == 0)
               {
                  y_cmd << " keys aa "
                        << " window_geometry 403 0 400 350";
               }

               mode_y_sock << "parallel " << num_procs << " " << myid << "\n"
                           << "solution\n" << pmesh << yComp << flush
                           << y_cmd.str() << endl << flush;

               MPI_Barrier(MPI_COMM_WORLD);

               ostringstream z_cmd;
               z_cmd << " window_title 'Eigenmode " << i+1 << '/' << nev
                     << " Z, Lambda = " << eigenvalues[i] - shift << "'"
                     << " valuerange -"<< max_r << ' ' << max_r;
               if (i == 0)
               {
                  z_cmd << " keys aa "
                        << " window_geometry 806 0 400 350";
               }

               mode_z_sock << "parallel " << num_procs << " " << myid << "\n"
                           << "solution\n" << pmesh << zComp << flush
                           << z_cmd.str() << endl << flush;

               MPI_Barrier(MPI_COMM_WORLD);

               VectorGridFunctionCoefficient dmodeCoef(&dx);
               InnerProductCoefficient dyCoef(yVecCoef, dmodeCoef);
               InnerProductCoefficient dzCoef(zVecCoef, dmodeCoef);

               dyComp.ProjectCoefficient(dyCoef);
               dzComp.ProjectCoefficient(dzCoef);

               real_t min_d = max_r / (bbMax[0] - bbMin[0]);

               max_y = GetScalarMax(dyComp);
               max_z = GetScalarMax(dzComp);
               max_r = std::max(std::max(max_y, max_z), min_d);

               ostringstream dy_cmd;
               dy_cmd << " window_title 'Curl Eigenmode "
                      << i+1 << '/' << nev
                      << " Y, Lambda = " << eigenvalues[i] - shift << "'"
                      << "valuerange -"<< max_r << ' ' << max_r;
               if (i == 0)
               {
                  dy_cmd << " keys aa"
                         << " window_geometry 403 375 400 350";
               }

               mode_dy_sock << "parallel " << num_procs << " " << myid << "\n"
                            << "solution\n" << pmesh << dyComp << flush
                            << dy_cmd.str() << endl << flush;

               MPI_Barrier(MPI_COMM_WORLD);

               ostringstream dz_cmd;
               dz_cmd << " window_title 'Curl Eigenmode "
                      << i+1 << '/' << nev
                      << " Z, Lambda = " << eigenvalues[i] - shift << "'"
                      << "valuerange -"<< max_r << ' ' << max_r;
               if (i == 0)
               {
                  dz_cmd << " keys aa"
                         << " window_geometry 806 375 400 350";
               }

               mode_dz_sock << "parallel " << num_procs << " " << myid << "\n"
                            << "solution\n" << pmesh << dzComp << flush
                            << dz_cmd.str() << endl << flush;

               MPI_Barrier(MPI_COMM_WORLD);
            }
            char c;
            if (Mpi::Root())
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
         ND_FECollection fec_nd_xy(order, dim);
         RT_FECollection fec_rt_xy(order-1, dim);
         L2_FECollection fec_l2(order-1, dim);

         ParFiniteElementSpace fes_h1(&pmesh, &fec_h1);
         ParFiniteElementSpace fes_nd(&pmesh, &fec_nd_xy);
         ParFiniteElementSpace fes_rt(&pmesh, &fec_rt_xy);
         ParFiniteElementSpace fes_l2(&pmesh, &fec_l2);

         ParGridFunction xyComp(&fes_nd);
         ParGridFunction zComp(&fes_h1);

         ParGridFunction dxyComp(&fes_rt);
         ParGridFunction dzComp(&fes_l2);

         for (int i=0; i<nev; i++)
         {
            if (Mpi::Root())
            {
               cout << "Eigenmode " << i+1 << '/' << nev
                    << ", Lambda = " << eigenvalues[i] - shift << endl;
            }

            // convert eigenvector from HypreParVector to ParGridFunction
            x = ame->GetEigenvector(i);
            curl.Mult(x, dx);

            {
               VectorGridFunctionCoefficient modeCoef(&x);
               MatrixVectorProductCoefficient xyCoef(xyMatCoef, modeCoef);
               InnerProductCoefficient zCoef(zVecCoef, modeCoef);

               xyComp.ProjectCoefficient(xyCoef);
               zComp.ProjectCoefficient(zCoef);

               real_t max_v = GetVectorMax(2, xyComp);
               real_t max_s = GetScalarMax(zComp);
               real_t max_r = std::max(max_v, max_s);

               ostringstream xy_cmd;
               xy_cmd << " window_title 'Eigenmode " << i+1 << '/' << nev
                      << " XY, Lambda = " << eigenvalues[i] - shift << "'"
                      << " valuerange 0.0 " << max_r;
               if (i == 0)
               {
                  xy_cmd << " keys aavvv"
                         << " window_geometry 0 0 400 350";
               }

               mode_xy_sock << "parallel " << num_procs << " " << myid << "\n"
                            << "solution\n" << pmesh << xyComp << flush
                            << xy_cmd.str() << endl << flush;

               MPI_Barrier(MPI_COMM_WORLD);

               ostringstream z_cmd;
               z_cmd << " window_title 'Eigenmode " << i+1 << '/' << nev
                     << " Z, Lambda = " << eigenvalues[i] - shift << "'"
                     << " valuerange -"<< max_r << ' ' << max_r;
               if (i == 0)
               {
                  z_cmd << " keys aa"
                        << " window_geometry 403 0 400 350";
               }

               mode_z_sock << "parallel " << num_procs << " " << myid << "\n"
                           << "solution\n" << pmesh << zComp << flush
                           << z_cmd.str() << endl << flush;

               MPI_Barrier(MPI_COMM_WORLD);

               VectorGridFunctionCoefficient dmodeCoef(&dx);
               MatrixVectorProductCoefficient dxyCoef(xyMatCoef, dmodeCoef);
               InnerProductCoefficient dzCoef(zVecCoef, dmodeCoef);

               dxyComp.ProjectCoefficient(dxyCoef);
               dzComp.ProjectCoefficient(dzCoef);

               real_t min_d = max_r / std::min(bbMax[0] - bbMin[0],
                                               bbMax[1] - bbMin[1]);

               max_v = GetVectorMax(2, dxyComp);
               max_s = GetScalarMax(dzComp);
               max_r = std::max(std::max(max_v, max_s), min_d);

               ostringstream dxy_cmd;
               dxy_cmd << " window_title 'Curl Eigenmode "
                       << i+1 << '/' << nev
                       << " XY, Lambda = " << eigenvalues[i] - shift << "'"
                       << " valuerange 0.0 " << max_r << '\n';
               if (i == 0)
               {
                  dxy_cmd << " keys aavvv "
                          << " window_geometry 0 375 400 350";

               }

               mode_dxy_sock << "parallel " << num_procs << " " << myid << "\n"
                             << "solution\n" << pmesh << dxyComp << flush
                             << dxy_cmd.str() << endl << flush;

               MPI_Barrier(MPI_COMM_WORLD);

               ostringstream dz_cmd;
               dz_cmd << " window_title 'Curl Eigenmode "
                      << i+1 << '/' << nev
                      << " Z, Lambda = " << eigenvalues[i] - shift << "'"
                      << " valuerange -" << max_r << ' ' << max_r;
               if (i == 0)
               {
                  dz_cmd << " keys aa"
                         << " window_geometry 403 375 400 350";
               }

               mode_dz_sock << "parallel " << num_procs << " " << myid << "\n"
                            << "solution\n" << pmesh << dzComp << flush
                            << dz_cmd.str() << endl << flush;

               MPI_Barrier(MPI_COMM_WORLD);
            }
            char c;
            if (Mpi::Root())
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
            if (Mpi::Root())
            {
               cout << "Eigenmode " << i+1 << '/' << nev
                    << ", Lambda = " << eigenvalues[i] - shift << endl;
            }

            // convert eigenvector from HypreParVector to ParGridFunction
            x = ame->GetEigenvector(i);
            curl.Mult(x, dx);

            mode_sock << "parallel " << num_procs << " " << myid << "\n"
                      << "solution\n" << pmesh << x << flush
                      << "window_title 'Eigenmode " << i+1 << '/' << nev
                      << ", Lambda = " << eigenvalues[i] - shift
                      << "'" << endl;

            MPI_Barrier(MPI_COMM_WORLD);

            mode_deriv_sock << "parallel " << num_procs << " " << myid << "\n"
                            << "solution\n" << pmesh << dx << flush
                            << "window_geometry 0 375 400 350 "
                            << "window_title 'Curl Eigenmode "
                            << i+1 << '/' << nev
                            << ", Lambda = " << eigenvalues[i] - shift
                            << "'" << endl;

            MPI_Barrier(MPI_COMM_WORLD);

            char c;
            if (Mpi::Root())
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

real_t GetVectorMax(int vdim, const ParGridFunction &x)
{
   Vector zeroVec(vdim); zeroVec = 0.0;
   VectorConstantCoefficient zero(zeroVec);
   real_t nrm = x.ComputeMaxError(zero);
   return nrm;
}

real_t GetScalarMax(const ParGridFunction &x)
{
   ConstantCoefficient zero(0.0);
   real_t nrm = x.ComputeMaxError(zero);
   return nrm;
}
