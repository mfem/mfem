//                       MFEM Example 13-cyl - Parallel Version
//
// Compile with: make ex13p-cyl
//
// Sample runs:  mpirun -np 4 ex13p-cyl
//               mpirun -np 4 ex13p-cyl -o 2
//               mpirun -np 4 ex13p-cyl -o 2 -e 0
//
// Description:  This example code demonstrates the use of MFEM to solve the
//               Maxwell (electromagnetic) eigenvalue problem on an
//               axisymmetric domain. The eigenvalue problem:
//                 curl curl E = lambda E
//               with homogeneous Dirichlet boundary conditions E x n = 0 is
//               solved on a cylindrical domain by meshing only a rectangle in
//               the rho, z plane.  In cylindrical coordinates the weak form of
//               the eigenvalue problem is given by:
//                  (rho Curl(u), Curl(v)) = lambda (rho u, v)
//
//               We compute the eight lowest nonzero eigenmodes by discretizing
//               the curl curl operator using a Nedelec FE space of the
//               specified order and compare to the known values.  Because the
//               eigenvalue spectrum of a domain is unique this provides a
//               reliable test that the axisymmetric domain is being faithfully
//               characterized.
//
//               In two dimensions the curl curl operator, with isotropic
//               material coefficients, splits into two separate PDEs. The rho
//               and z components form a 2D vector field in the rho-z plane
//               which is discretized with the 2D Nedelec vector basis
//               functions. The Maxwell eigenvalue problem for the rho-z field
//               can be written with cartesian operators as:
//                   curl (rho curl E_rz) = lambda rho E_rz
//
//               The angular component can be discretized with the 2D H1 scalar
//               basis functions. The angular portion of the eigenvalue problem
//               can be written with cartesian operators as:
//                   -div (rho grad E_phi) + (1/rho) E_phi = lambda rho E_phi
//
//               The example highlights the use of specialized coefficients
//               with existing operators to mimic axisymmetric domains.  The
//               curl of each eigenmode is also computed and displayed to
//               ilustrate that no special steps need to be taken to compute
//               the curl in this coordinate system.
//
//               We recommend viewing examples 13 and 11-cyl before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Zeros of Bessel function J_0
static double J0z[] = {2.404825557695773,
                       5.520078110286312,
                       8.653727912911013,
                       11.79153443901428
                      };

// Zeros of Bessel function J_1
static double J1z[] = {3.831705970207512,
                       7.015586669815622,
                       10.17346813506272,
                       13.32369193631422
                      };

// Modes numbers in the rho and z directions for the first five eigenmodes
static int mode_nums_rz[] = {0, 0,
                             0, 1,
                             1, 0,
                             1, 1,
                             0, 2,
			     1, 2,
			     2, 0,
			     2, 1
                            };

// Modes numbers in the phi direction for the first five eigenmodes
static int mode_nums_phi[] = {0, 1,
                              0, 2,
                              1, 1,
                              1, 2,
                              0, 3,
			      2, 1,
			      1, 3,
			      2, 2
                             };

// Mode polarizations for the first several modes; 0 - rz, 1 - phi
static int mode_type[] = {0,0,1,0,0,0,1,1,0,0,0,1,0,1,1,0,1,0,1};

double rhoFunc(const Vector &x)
{
   return x[0];
}

double rhoInvFunc(const Vector &x)
{
   return 1.0 / x[0];
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   int dim = 2;
   int nr = 1;
   int nz = 1;
   int el_type_flag = 1;
   Element::Type el_type;
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 1;
   int nev = 8;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&nz, "-nz", "--num-elements-z",
                  "Number of elements in z-direction.");
   args.AddOption(&nr, "-nr", "--num-elements-rho",
                  "Number of elements in radial direction.");
   args.AddOption(&el_type_flag, "-e", "--element-type",
                  "Element type: 0 - Triangle, 1 - Quadrilateral.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
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
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // The output mesh could be quadrilaterals or triangles
   el_type = (el_type_flag == 0) ? Element::TRIANGLE : Element::QUADRILATERAL;
   if (el_type != Element::TRIANGLE && el_type != Element::QUADRILATERAL)
   {
      cout << "Unsupported element type" << endl;
      exit(1);
   }

   // 3. Prepare a rectangular mesh with the desired dimensions and element
   //    type.  Other 2D meshes could be used but then we couldn't check the
   //    eigenvalues.
   ParMesh pmesh;
   {
      Mesh mesh = Mesh::MakeCartesian2D(nr, nz, el_type);

      // 4. Refine the serial mesh on all processors to increase the resolution. In
      //    this example we do 'ref_levels' of uniform refinement (2 by default, or
      //    specified on the command line with -rs).
      for (int lev = 0; lev < ser_ref_levels; lev++)
      {
         mesh.UniformRefinement();
      }

      // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
      //    this mesh further in parallel to increase the resolution (1 time by
      //    default, or specified on the command line with -rp). Once the parallel
      //    mesh is defined, the serial mesh can be deleted.
      pmesh = ParMesh(MPI_COMM_WORLD, mesh);
      for (int lev = 0; lev < par_ref_levels; lev++)
      {
         pmesh.UniformRefinement();
      }
      pmesh.ReorientTetMesh();
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements (ND) of the specified order. We also
   //    create an L2 space to represent the z-component of the curl of the
   //    modes.
   ND_FECollection fec_nd(order, dim);
   RT_FECollection fec_rt(order - 1, dim);
   H1_FECollection fec_ndp(order, dim);
   L2_FECollection fec_rtp(order - 1, dim,
                           BasisType::GaussLegendre, FiniteElement::INTEGRAL);
   L2_FECollection fec_l2(order - 1, dim);
   ParFiniteElementSpace fespace_nd(&pmesh, &fec_nd);
   ParFiniteElementSpace fespace_rt(&pmesh, &fec_rt);
   ParFiniteElementSpace fespace_ndp(&pmesh, &fec_ndp);
   ParFiniteElementSpace fespace_rtp(&pmesh, &fec_rtp);
   ParFiniteElementSpace fespace_l2(&pmesh, &fec_l2);
   HYPRE_Int size = fespace_nd.GlobalTrueVSize() +
                    fespace_ndp.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
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
   FunctionCoefficient rhoCoef(rhoFunc);
   FunctionCoefficient rhoInvCoef(rhoInvFunc);
   Array<int> ess_bdr_rz(pmesh.bdr_attributes.Size());
   Array<int> ess_bdr_phi(pmesh.bdr_attributes.Size());
   ess_bdr_rz = 1;  ess_bdr_rz[3] = 0;
   ess_bdr_phi = 1;

   ParBilinearForm *arz = new ParBilinearForm(&fespace_nd);
   arz->AddDomainIntegrator(new CurlCurlIntegrator(rhoCoef));
   arz->Assemble();
   arz->EliminateEssentialBCDiag(ess_bdr_rz, 1.0);
   arz->Finalize();

   ParBilinearForm *mrz = new ParBilinearForm(&fespace_nd);
   mrz->AddDomainIntegrator(new VectorFEMassIntegrator(rhoCoef));
   mrz->Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   mrz->EliminateEssentialBCDiag(ess_bdr_rz, numeric_limits<double>::min());
   mrz->Finalize();

   ParBilinearForm *aphi = new ParBilinearForm(&fespace_ndp);
   aphi->AddDomainIntegrator(new MassIntegrator(rhoInvCoef));
   aphi->AddDomainIntegrator(new DiffusionIntegrator(rhoCoef));
   aphi->Assemble();
   aphi->EliminateEssentialBCDiag(ess_bdr_phi, 1.0);
   aphi->Finalize();

   ParBilinearForm *mphi = new ParBilinearForm(&fespace_ndp);
   mphi->AddDomainIntegrator(new MassIntegrator(rhoCoef));
   mphi->Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   mphi->EliminateEssentialBCDiag(ess_bdr_phi, numeric_limits<double>::min());
   mphi->Finalize();

   HypreParMatrix *Arz = arz->ParallelAssemble();
   HypreParMatrix *Mrz = mrz->ParallelAssemble();

   HypreParMatrix *Aphi = aphi->ParallelAssemble();
   HypreParMatrix *Mphi = mphi->ParallelAssemble();

   delete arz;
   delete mrz;
   delete aphi;
   delete mphi;

   // 8. Define and configure the AME eigensolver and the AMS preconditioner for
   //    A to be used within the solver. Set the matrices which define the
   //    generalized eigenproblem A x = lambda M x.
   HypreAMS *ams = new HypreAMS(*Arz,&fespace_nd);
   ams->SetPrintLevel(0);
   ams->SetSingularProblem();

   HypreAME *ame = new HypreAME(MPI_COMM_WORLD);
   ame->SetNumModes(nev);
   ame->SetPreconditioner(*ams);
   ame->SetMaxIter(100);
   ame->SetTol(1e-8);
   ame->SetPrintLevel(1);
   ame->SetMassMatrix(*Mrz);
   ame->SetOperator(*Arz);

   HypreBoomerAMG *amg = new HypreBoomerAMG(*Aphi);
   amg->SetPrintLevel(0);

   HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);
   lobpcg->SetNumModes(nev);
   // lobpcg->SetRandomSeed(seed);
   lobpcg->SetPreconditioner(*amg);
   lobpcg->SetMaxIter(200);
   lobpcg->SetTol(1e-8);
   lobpcg->SetPrecondUsageMode(1);
   lobpcg->SetPrintLevel(1);
   lobpcg->SetMassMatrix(*Mphi);
   lobpcg->SetOperator(*Aphi);

   // 9. Compute the eigenmodes and extract the array of eigenvalues. Define a
   //    parallel grid function to represent each of the eigenmodes returned by
   //    the solver. Also, define a discrete curl operator.
   Array<double> eigenvalues_rz;
   Array<double> eigenvalues_phi;

   ame->Solve();
   ame->GetEigenvalues(eigenvalues_rz);

   lobpcg->Solve();
   lobpcg->GetEigenvalues(eigenvalues_phi);

   Array<double> eigenvalues;

   ParGridFunction x_rz(&fespace_nd);
   ParGridFunction x_phi(&fespace_ndp);
   ParGridFunction dx_rz(&fespace_rt);
   ParGridFunction dx_phi(&fespace_rtp);

   ParDiscreteLinearOperator curl(&fespace_nd, &fespace_rtp);
   curl.AddDomainInterpolator(new CurlInterpolator());
   curl.Assemble();
   curl.Finalize();

   ParDiscreteLinearOperator curl2(&fespace_ndp, &fespace_rt);
   curl2.AddDomainInterpolator(new CurlInterpolator());
   curl2.Assemble();
   curl2.Finalize();

   // This is one workaround for GLVis limitations
   ParGridFunction dx_l2(&fespace_l2);

   GridFunctionCoefficient dxCoef(&dx_phi);

   if ( myid == 0 )
   {
      cout << "\nRelative error in eigenvalues of RZ modes:\n";
      for (int i=0; i<nev; i++)
      {
         double lambda =
            pow(J0z[mode_nums_rz[2*i]], 2) +
            pow(M_PI * mode_nums_rz[2*i+1], 2);
         cout << "Lambda " << i+1 << '/' << nev << " = " << eigenvalues_rz[i]
              << ", rel err = " << fabs(eigenvalues_rz[i] - lambda) / lambda
              << endl;
      }
      cout << endl;
      cout << "\nRelative error in eigenvalues of Phi modes:\n";
      for (int i=0; i<nev; i++)
      {
         double lambda =
            pow(J1z[mode_nums_phi[2*i]], 2) +
            pow(M_PI * mode_nums_phi[2*i+1], 2);
         cout << "Lambda " << i+1 << '/' << nev << " = " << eigenvalues_phi[i]
              << ", rel err = " << fabs(eigenvalues_phi[i] - lambda) / lambda
              << endl;
      }
      cout << endl;
   }

   // 10. Save the refined mesh and the modes in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g mode".
   {
      ostringstream mesh_name, mode_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      for (int i=0; i<nev; i++)
      {
         // convert eigenvector from HypreParVector to ParGridFunction
         x_rz = ame->GetEigenvector(i);

         mode_name << "mode_rz_" << setfill('0') << setw(2) << i << "."
                   << setfill('0') << setw(6) << myid;

         ofstream mode_rz_ofs(mode_name.str().c_str());
         mode_rz_ofs.precision(8);
         x_rz.Save(mode_rz_ofs);
         mode_name.str("");

         // convert eigenvector from HypreParVector to ParGridFunction
         x_phi = lobpcg->GetEigenvector(i);

         mode_name << "mode_phi_" << setfill('0') << setw(2) << i << "."
                   << setfill('0') << setw(6) << myid;

         ofstream mode_phi_ofs(mode_name.str().c_str());
         mode_phi_ofs.precision(8);
         x_phi.Save(mode_phi_ofs);
         mode_name.str("");
      }
   }

   // 11. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_rz_sock(vishost, visport);
      socketstream curl_rz_sock(vishost, visport);
      socketstream mode_phi_sock(vishost, visport);
      socketstream curl_phi_sock(vishost, visport);
      mode_rz_sock.precision(8);
      curl_rz_sock.precision(8);
      mode_phi_sock.precision(8);
      curl_phi_sock.precision(8);

      int irz = 0;
      int iphi = 0;

      for (int i=0; i<nev; i++)
      {
         if (mode_type[i] == 0)
         {
            if ( myid == 0 )
            {
               cout << "Eigenmode " << i+1 << '/' << nev
                    << ", Lambda = " << eigenvalues_rz[irz] << endl;
            }

            // convert eigenvector from HypreParVector to ParGridFunction
            x_rz = ame->GetEigenvector(irz);

            curl.Mult(x_rz, dx_phi);
            dx_l2.ProjectCoefficient(dxCoef);

            mode_rz_sock << "parallel " << num_procs << " " << myid << "\n"
			 << "solution\n" << pmesh << x_rz << flush
			 << "window_title 'Eigenmode " << i+1 << '/' << nev
			 << ", Lambda = " << eigenvalues_rz[irz] << "'\n";
	    if (irz == 0)
	      {
		mode_rz_sock << "keys vvv\n";
	      }
	    mode_rz_sock << flush;
	    
            curl_rz_sock << "parallel " << num_procs << " " << myid << "\n"
			 << "solution\n" << pmesh << dx_l2 << flush
			 << "window_title 'Curl of Eigenmode " << i+1
			 << '/' << nev
			 << ", Lambda = " << eigenvalues_rz[irz] << "' "
			 << "window_geometry 400 0 400 350\n" << flush;

            irz++;
         }
         else
         {
            if ( myid == 0 )
            {
               cout << "Eigenmode " << i+1 << '/' << nev
                    << ", Lambda = " << eigenvalues_phi[iphi] << endl;
            }

            // convert eigenvector from HypreParVector to ParGridFunction
            x_phi = lobpcg->GetEigenvector(iphi);

            curl2.Mult(x_phi, dx_rz);

            mode_phi_sock << "parallel " << num_procs << " " << myid << "\n"
			  << "solution\n" << pmesh << x_phi << flush
			  << "window_title 'Eigenmode " << i+1 << '/' << nev
			  << ", Lambda = " << eigenvalues_phi[iphi] << "' "
			  << "window_geometry 0 375 400 350\n"
			  << flush;

            curl_phi_sock << "parallel " << num_procs << " " << myid << "\n"
			  << "solution\n" << pmesh << dx_rz << flush
			  << "window_title 'Curl of Eigenmode " << i+1
			  << '/' << nev
			  << ", Lambda = " << eigenvalues_phi[iphi] << "' "
			  << "window_geometry 400 375 400 350\n";
	    if (iphi == 0)
	    {
	      curl_phi_sock << "keys vvv\n";
	    }
	    curl_phi_sock << flush;
	    
            iphi++;
         }
         char c;
         if (myid == 0)
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
      mode_rz_sock.close();
      curl_rz_sock.close();
      mode_phi_sock.close();
      curl_phi_sock.close();
   }
   /*
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_sock(vishost, visport);
      mode_sock.precision(8);
      socketstream curl_sock(vishost, visport);
      curl_sock.precision(8);

      for (int i=0; i<nev; i++)
      {
         if ( myid == 0 )
         {
            cout << "Eigenmode " << i+1 << '/' << nev
                 << ", Lambda = " << eigenvalues_phi[i] << endl;
         }

         // convert eigenvector from HypreParVector to ParGridFunction
         x_phi = lobpcg->GetEigenvector(i);

         curl2.Mult(x_phi, dx_rz);
    // dx_l2.ProjectCoefficient(dxCoef);

         mode_sock << "parallel " << num_procs << " " << myid << "\n"
                   << "solution\n" << pmesh << x_phi << flush
                   << "window_title 'Eigenmode " << i+1 << '/' << nev
                   << ", Lambda = " << eigenvalues_phi[i] << "' "
                   << "keys vvv\n" << flush;

         // Limitations in the GridFunction and GLVis prevent this from working
         curl_sock << "parallel " << num_procs << " " << myid << "\n"
                   << "solution\n" << pmesh << dx_rz << flush
                   << "window_title 'Curl of Eigenmode " << i+1 << '/' << nev
                   << ", Lambda = " << eigenvalues_phi[i] << "' "
                   << "window_geometry 400 0 400 350\n" << flush;

         char c;
         if (myid == 0)
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
   */
   // 12. Free the used memory.
   delete ame;
   delete ams;
   delete lobpcg;
   delete amg;
   delete Mrz;
   delete Arz;
   delete Mphi;
   delete Aphi;

   MPI_Finalize();

   return 0;
}
