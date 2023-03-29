//                       MFEM Example 3 - Parallel Version
//
// Compile with: make ex3p
//
// Sample runs:  mpirun -np 4 ex3p -m ../data/star.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh -o 2 -pa
//               mpirun -np 4 ex3p -m ../data/escher.mesh
//               mpirun -np 4 ex3p -m ../data/escher.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/fichera.mesh
//               mpirun -np 4 ex3p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex3p -m ../data/fichera-q3.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc-nurbs.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex-nurbs.mesh
//               mpirun -np 4 ex3p -m ../data/amr-quad.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex3p -m ../data/ref-prism.mesh -o 1
//               mpirun -np 4 ex3p -m ../data/octahedron.mesh -o 1
//               mpirun -np 4 ex3p -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/mobius-strip.mesh -o 2 -f 0.1
//               mpirun -np 4 ex3p -m ../data/klein-bottle.mesh -o 2 -f 0.1
//
// Device sample runs:
//               mpirun -np 4 ex3p -m ../data/star.mesh -pa -d cuda
//               mpirun -np 4 ex3p -m ../data/star.mesh -no-pa -d cuda
//               mpirun -np 4 ex3p -m ../data/star.mesh -pa -d raja-cuda
//               mpirun -np 4 ex3p -m ../data/star.mesh -pa -d raja-omp
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Static condensation is
//               also illustrated.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/fichera-mixed.mesh";
   Array<int> cond_attr;
   Array<int> submesh_elems;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   int order = 1;
   double sigma_const = 1e-6;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
#ifdef MFEM_USE_AMGX
   bool useAmgX = false;
#endif

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&submesh_elems, "-sme", "--submesh-elems",
                  "Element indices of submesh.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&cond_attr, "-ca", "--cond-attr",
                  "Attributes of conductor");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&sigma_const, "-s", "--sigma", "Conductivity");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
#ifdef MFEM_USE_AMGX
   args.AddOption(&useAmgX, "-amgx", "--useAmgX", "-no-amgx",
                  "--no-useAmgX",
                  "Enable or disable AmgX in MatrixFreeAMS.");
#endif

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }
   kappa = freq * M_PI;

   if (submesh_elems.Size() == 0 &&
       strcmp(mesh_file, "../data/fichera-mixed.mesh") == 0)
   {
      submesh_elems.SetSize(5);
      submesh_elems[0] = 0;
      submesh_elems[1] = 2;
      submesh_elems[2] = 3;
      submesh_elems[3] = 4;
      submesh_elems[4] = 9;
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   // int sdim = mesh->SpaceDimension();

   int submesh_attr = -1;
   if (submesh_elems.Size() > 0)
   {
      int max_attr = mesh->attributes.Max();
      submesh_attr = max_attr + 1;

      for (int i=0; i<submesh_elems.Size(); i++)
      {
         mesh->SetAttribute(submesh_elems[i], submesh_attr);
      }
      mesh->SetAttributes();

      if (cond_attr.Size() == 0)
      {
         cond_attr.Append(submesh_attr);
      }
   }

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels = ser_ref_levels;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   cout << myid << ": Number of element in full domain:  " << pmesh->GetNE() << endl;
   cout << myid << ": Number of vertices in full domain: " << pmesh->GetNV() << endl;
   
   // 6b. Extract a submesh covering a portion of the domain
   ParSubMesh *pmesh_cond =
      new ParSubMesh(ParSubMesh::CreateFromDomain(*pmesh, cond_attr));

   cout << myid << ": Number of element in conductor:  " << pmesh_cond->GetNE() << endl;
   cout << myid << ": Number of vertices in conductor: " << pmesh_cond->GetNV() << endl;
   
   RT_FECollection fec_cond(order - 1, dim);
   ParFiniteElementSpace fes_cond(pmesh_cond, &fec_cond);
   ParGridFunction j_cond(&fes_cond);

   {
      /*
      Vector zVec(3); zVec = 0.0; zVec[2] = 1.0;
      VectorConstantCoefficient zVecCoef(zVec);
      j_cond.ProjectCoefficient(zVecCoef);
      */
      H1_FECollection fec_h1(order, dim);
      ParFiniteElementSpace fes_h1(pmesh_cond, &fec_h1);

      ConstantCoefficient sigmaCoef(1.0);
      Array<int> ess_bdr_phi(pmesh_cond->bdr_attributes.Max());
      Array<int> ess_bdr_tdof_phi;
      ess_bdr_phi = 0;
      ess_bdr_phi[1] = 1;
      ess_bdr_phi[22] = 1;
      fes_h1.GetEssentialTrueDofs(ess_bdr_phi, ess_bdr_tdof_phi);

      ParBilinearForm a_h1(&fes_h1);
      a_h1.AddDomainIntegrator(new DiffusionIntegrator(sigmaCoef));
      a_h1.Assemble();

      ParLinearForm b_h1(&fes_h1);
      b_h1 = 0.0;

      ConstantCoefficient one(1.0);
      ConstantCoefficient zero(0.0);
      ParGridFunction phi_h1(&fes_h1);
      phi_h1 = 0.0;

      Array<int> bdr0(pmesh_cond->bdr_attributes.Max()); bdr0 = 0; bdr0[1] = 1;
      phi_h1.ProjectBdrCoefficient(zero, bdr0);

      Array<int> bdr1(pmesh_cond->bdr_attributes.Max()); bdr1 = 0; bdr1[22] = 1;
      phi_h1.ProjectBdrCoefficient(one, bdr1);

      {
         OperatorPtr A;
         Vector B, X;
         a_h1.FormLinearSystem(ess_bdr_tdof_phi, phi_h1, b_h1, A, X, B);

         HypreBoomerAMG prec;
         CGSolver cg(MPI_COMM_WORLD);
         cg.SetRelTol(1e-12);
         cg.SetMaxIter(2000);
         cg.SetPrintLevel(1);
         cg.SetPreconditioner(prec);
         cg.SetOperator(*A);
         cg.Mult(B, X);
         a_h1.RecoverFEMSolution(X, b_h1, phi_h1);
      }

      ParBilinearForm m_rt(&fes_cond);
      m_rt.AddDomainIntegrator(new VectorFEMassIntegrator);
      m_rt.Assemble();

      ParMixedBilinearForm d_h1(&fes_h1, &fes_cond);
      d_h1.AddDomainIntegrator(new MixedVectorGradientIntegrator(sigmaCoef));
      d_h1.Assemble();

      ParLinearForm b_rt(&fes_cond);
      d_h1.Mult(phi_h1, b_rt);

      {
         Array<int> ess_bdr_tdof_rt;
         OperatorPtr M;
         Vector B, X;
         m_rt.FormLinearSystem(ess_bdr_tdof_rt, j_cond, b_rt, M, X, B);

         HypreDiagScale prec;

         CGSolver cg(MPI_COMM_WORLD);
         cg.SetRelTol(1e-12);
         cg.SetMaxIter(2000);
         cg.SetPrintLevel(1);
         cg.SetPreconditioner(prec);
         cg.SetOperator(*M);
         cg.Mult(B, X);
         m_rt.RecoverFEMSolution(X, b_rt, j_cond);
      }
   }

   {
      ostringstream mesh_name, cond_name;
      mesh_name << "cond_mesh." << setfill('0') << setw(6) << myid;
      cond_name << "cond_j." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh_cond->Print(mesh_ofs);

      ofstream cond_ofs(cond_name.str().c_str());
      cond_ofs.precision(8);
      j_cond.Save(cond_ofs);
   }
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream port_sock(vishost, visport);
      port_sock << "parallel " << num_procs << " " << myid << "\n";
      port_sock.precision(8);
      port_sock << "solution\n" << *pmesh_cond << j_cond
                << "window_title 'Conductor J'"
                << "window_geometry 0 0 400 350" << flush;
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   H1_FECollection fec_h1(order, dim);
   ND_FECollection fec_nd(order, dim);
   RT_FECollection fec_rt(order - 1, dim);
   ParFiniteElementSpace fespace_h1(pmesh, &fec_h1);
   ParFiniteElementSpace fespace_nd(pmesh, &fec_nd);
   ParFiniteElementSpace fespace_rt(pmesh, &fec_rt);
   HYPRE_BigInt size = fespace_nd.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   ParGridFunction j_full(&fespace_rt);
   j_full = 0.0;
   pmesh_cond->Transfer(j_cond, j_full);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << j_full
               << "window_title 'J Full'"
               << "window_geometry 200 0 400 350" << flush;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      {
         ess_bdr[ 8] = 0;
         ess_bdr[ 9] = 0;
         ess_bdr[10] = 0;
         ess_bdr[11] = 0;

         ess_bdr[12] = 0;
         ess_bdr[13] = 0;
         ess_bdr[14] = 0;
         ess_bdr[15] = 0;
      }
      /*
      ess_bdr = 0;
      {
      ess_bdr[ 8] = 1;
      ess_bdr[ 9] = 1;
      ess_bdr[10] = 1;
      ess_bdr[11] = 1;

      ess_bdr[12] = 1;
      ess_bdr[13] = 1;
      ess_bdr[14] = 1;
      ess_bdr[15] = 1;
           }
           */
      /*
      ess_bdr = 0;
      ess_bdr[0] = 1;
      ess_bdr[1] = 1;
      ess_bdr[2] = 1;
      // ess_bdr[3] = 1;
      ess_bdr[20] = 1;
      ess_bdr[21] = 1;
      ess_bdr[22] = 1;
      ess_bdr[23] = 1;
      */
      // ess_bdr[7] = 1;
      // ess_bdr[19] = 1;
      fespace_nd.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   if (false)
   {
      Array<int> ess_bdr8(pmesh->bdr_attributes.Max());
      ess_bdr8 = 0; ess_bdr8[7] = 1;
      ess_bdr8[9] = 1; ess_bdr8[10] = 1;

      Array<int> ess_bdr8_tdofs;
      fespace_nd.GetEssentialTrueDofs(ess_bdr8, ess_bdr8_tdofs);

      Array<int> ess_bdr19(pmesh->bdr_attributes.Max());
      ess_bdr19 = 0; // ess_bdr19[18] = 1;
      ess_bdr19[12] = 1; ess_bdr19[15] = 1;

      Array<int> ess_bdr19_tdofs;
      fespace_nd.GetEssentialTrueDofs(ess_bdr19, ess_bdr19_tdofs);

      std::set<int> tdofs8;
      std::set<int> tdofs19;

      for (int i=0; i<ess_bdr8_tdofs.Size(); i++)
      {
         tdofs8.insert(ess_bdr8_tdofs[i]);
      }
      for (int i=0; i<ess_bdr19_tdofs.Size(); i++)
      {
         tdofs19.insert(ess_bdr19_tdofs[i]);
      }

      std::set<int> edge_8_19;
      std::set_intersection(tdofs8.begin(), tdofs8.end(),
                            tdofs19.begin(), tdofs19.end(),
                            std::inserter(edge_8_19, edge_8_19.begin()));

      for (std::set<int>::iterator it=edge_8_19.begin();
           it!=edge_8_19.end(); it++)
      {
         ess_tdof_list.Append(*it);
      }
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   // VectorFunctionCoefficient f(sdim, f_exact);
   VectorGridFunctionCoefficient f(&j_full);
   ParLinearForm *b = new ParLinearForm(&fespace_nd);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();
   /*
   ParGridFunction dpsi_nd(&fespace_nd);

   {
     // Divergence cleaning
     ParDiscreteLinearOperator grad(&fespace_h1, &fespace_nd);
     grad.AddDomainInterpolator(new GradientInterpolator);
     grad.Assemble();
     grad.Finalize();

     ParLinearForm rho_h1(&fespace_h1);
     grad.MultTranspose(*b, rho_h1);

     ParBilinearForm a_h1(&fespace_h1);
     a_h1.AddDomainIntegrator(new DiffusionIntegrator);
     a_h1.Assemble();

     ParGridFunction psi_h1(&fespace_h1);
     psi_h1 = 0.0;

     {
     Array<int> ess_bdr_psi(pmesh->bdr_attributes.Max());
     Array<int> ess_bdr_tdof_psi;
     ess_bdr_psi = 0;
     ess_bdr_psi[1] = 1;
     ess_bdr_psi[22] = 1;
     fespace_h1.GetEssentialTrueDofs(ess_bdr_psi, ess_bdr_tdof_psi);

       OperatorPtr A;
       Vector B, X;
       a_h1.FormLinearSystem(ess_bdr_tdof_psi, psi_h1, rho_h1, A, X, B);

       HypreBoomerAMG prec;
       CGSolver cg(MPI_COMM_WORLD);
       cg.SetRelTol(1e-12);
       cg.SetMaxIter(2000);
       cg.SetPrintLevel(1);
       cg.SetPreconditioner(prec);
       cg.SetOperator(*A);
       cg.Mult(B, X);
       a_h1.RecoverFEMSolution(X, rho_h1, psi_h1);

       grad.Mult(psi_h1, dpsi_nd);
     }

     ParMixedBilinearForm d_h1(&fespace_h1, &fespace_nd);
     d_h1.AddDomainIntegrator(new MixedVectorGradientIntegrator);
     d_h1.Assemble();

     d_h1.AddMult(psi_h1, *b, -1.0);
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << dpsi_nd
          << "window_title 'Div Clean'"
          << "window_geometry 400 400 400 350" << flush;
   }
   */
   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x by projecting the exact
   //     solution. Note that only values from the boundary edges will be used
   //     when eliminating the non-homogeneous boundary condition to modify the
   //     r.h.s. vector b.
   ParGridFunction x(&fespace_nd);
   // VectorFunctionCoefficient E(sdim, E_exact);
   // x.ProjectCoefficient(E);
   x = 0.0;

   // 11. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(sigma_const);
   ParBilinearForm *a = new ParBilinearForm(&fespace_nd);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // 13. Solve the system AX=B using PCG with an AMS preconditioner.
   if (pa)
   {
#ifdef MFEM_USE_AMGX
      MatrixFreeAMS ams(*a, *A, fespace_nd, muinv, sigma, NULL, ess_bdr, useAmgX);
#else
      MatrixFreeAMS ams(*a, *A, fespace_nd, muinv, sigma, NULL, ess_bdr);
#endif
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(ams);
      cg.Mult(B, X);
   }
   else
   {
      if (myid == 0)
      {
         cout << "Size of linear system: "
              << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;
      }

      ParFiniteElementSpace *prec_fespace =
         (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : &fespace_nd);
      HypreAMS ams(*A.As<HypreParMatrix>(), prec_fespace);
      // ams.SetSingularProblem();
      HyprePCG pcg(*A.As<HypreParMatrix>());
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(ams);
      pcg.Mult(B, X);
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 15. Compute and print the L^2 norm of the error.
   /*
   {
      double error = x.ComputeL2Error(E);
      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^2} = " << error << '\n' << endl;
      }
   }
   */

   // 16. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
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

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x
               << "window_title 'Vector Potential'"
               << "window_geometry 400 0 400 350" << flush;
   }

   ParDiscreteLinearOperator curl(&fespace_nd, &fespace_rt);
   curl.AddDomainInterpolator(new CurlInterpolator);
   curl.Assemble();
   curl.Finalize();

   ParGridFunction dx(&fespace_rt);
   curl.Mult(x, dx);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << dx
               << "window_title 'Magnetic Flux'"
               << "window_geometry 800 0 400 350" << flush;
   }

   // 18. Free the used memory.
   delete a;
   delete sigma;
   delete muinv;
   delete b;
   // delete fespace;
   // delete fec;
   delete pmesh;

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
