//                       MFEM Example 34 - Parallel Version
//
// Compile with: make ex34p
//
// Sample runs:  mpirun -np 4 ex34p -o 2
//               mpirun -np 4 ex34p -o 2 -hex -pa
//
// Device sample runs:
//               mpirun -np 4 ex34p -o 2 -hex -pa -d cuda
//               mpirun -np 4 ex34p -o 2 -no-pa -d cuda
//
// Description:  This example code solves a simple magnetostatic problem
//               curl curl A = J where the current density J is computed on a
//               subset of the domain as J = -sigma grad phi. We discretize the
//               vector potential with Nedelec finite elements, the scalar
//               potential with Lagrange finite elements, and the current
//               density with Raviart-Thomas finite elements.
//
//               The example demonstrates the use of a SubMesh to compute the
//               scalar potential and its associated current density which is
//               then transferred to the original mesh and used as a source
//               function.
//
//               Note that this example takes certain liberties with the
//               current density which is not necessarily divergence free
//               as it should be. This was done to focus on the use of the
//               SubMesh to transfer information between a full mesh and a
//               sub-domain. A more rigorous implementation might employ an
//               H(div) saddle point solver to obtain a divergence free J on
//               the SubMesh. It would then also need to ensure that the r.h.s.
//               of curl curl A = J does in fact lie in the range of the weak
//               curl operator by performing a divergence cleaning procedure
//               before the solve. After divergence cleaning the delta
//               parameter would probably not be needed.
//
//               This example is designed to make use of a specific mesh which
//               has a known configuration of elements and boundary attributes.
//               Other meshes could be used but extra care would be required to
//               properly define the SubMesh and the necessary boundaries.
//
//               We recommend viewing examples 1 and 3 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void ComputeCurrentDensityOnSubMesh(int order,
                                    bool visualization,
                                    const Array<int> &phi0_attr,
                                    const Array<int> &phi1_attr,
                                    const Array<int> &jn_zero_attr,
                                    ParGridFunction &j_cond);

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
   Array<int> sym_plane_attr;
   Array<int> phi0_attr;
   Array<int> phi1_attr;
   Array<int> jn_zero_attr;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   int order = 1;
   real_t delta_const = 1e-6;
   bool mixed = true;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
#ifdef MFEM_USE_AMGX
   bool useAmgX = false;
#endif

   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&delta_const, "-mc", "--magnetic-cond",
                  "Magnetic Conductivity");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&mixed, "-mixed", "--mixed-mesh", "-hex",
                  "--hex-mesh", "Mixed mesh of hexahedral mesh.");
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

   if (!mixed || pa)
   {
      mesh_file = "../data/fichera.mesh";
   }

   if (submesh_elems.Size() == 0)
   {
      if (strcmp(mesh_file, "../data/fichera-mixed.mesh") == 0)
      {
         submesh_elems.SetSize(5);
         submesh_elems[0] = 0;
         submesh_elems[1] = 2;
         submesh_elems[2] = 3;
         submesh_elems[3] = 4;
         submesh_elems[4] = 9;
      }
      else if (strcmp(mesh_file, "../data/fichera.mesh") == 0)
      {
         submesh_elems.SetSize(7);
         submesh_elems[0] = 10;
         submesh_elems[1] = 14;
         submesh_elems[2] = 34;
         submesh_elems[3] = 36;
         submesh_elems[4] = 37;
         submesh_elems[5] = 38;
         submesh_elems[6] = 39;
      }
   }
   if (sym_plane_attr.Size() == 0)
   {
      if (strcmp(mesh_file, "../data/fichera-mixed.mesh") == 0 ||
          strcmp(mesh_file, "../data/fichera.mesh") == 0)
      {
         sym_plane_attr.SetSize(8);
         sym_plane_attr[0] =  9;
         sym_plane_attr[1] = 10;
         sym_plane_attr[2] = 11;
         sym_plane_attr[3] = 12;
         sym_plane_attr[4] = 13;
         sym_plane_attr[5] = 14;
         sym_plane_attr[6] = 15;
         sym_plane_attr[7] = 16;
      }
   }
   if (phi0_attr.Size() == 0)
   {
      if (strcmp(mesh_file, "../data/fichera-mixed.mesh") == 0 ||
          strcmp(mesh_file, "../data/fichera.mesh") == 0)
      {
         phi0_attr.Append(2);
      }
   }
   if (phi1_attr.Size() == 0)
   {
      if (strcmp(mesh_file, "../data/fichera-mixed.mesh") == 0 ||
          strcmp(mesh_file, "../data/fichera.mesh") == 0)
      {
         phi1_attr.Append(23);
      }
   }
   if (jn_zero_attr.Size() == 0)
   {
      if (strcmp(mesh_file, "../data/fichera-mixed.mesh") == 0 ||
          strcmp(mesh_file, "../data/fichera.mesh") == 0)
      {
         jn_zero_attr.Append(25);
      }
      for (int i=0; i<sym_plane_attr.Size(); i++)
      {
         jn_zero_attr.Append(sym_plane_attr[i]);
      }
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   if (!mixed || pa)
   {
      mesh->UniformRefinement();

      if (ser_ref_levels > 0)
      {
         ser_ref_levels--;
      }
      else
      {
         par_ref_levels--;
      }
   }

   int submesh_attr = -1;
   if (cond_attr.Size() == 0 && submesh_elems.Size() > 0)
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
   //    this example we do 'ref_levels' of uniform refinement.
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
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 6b. Extract a submesh covering a portion of the domain
   ParSubMesh pmesh_cond(ParSubMesh::CreateFromDomain(pmesh, cond_attr));

   // 7. Define a suitable finite element space on the SubMesh and compute
   //    the current density as an H(div) field.
   RT_FECollection fec_cond_rt(order - 1, dim);
   ParFiniteElementSpace fes_cond_rt(&pmesh_cond, &fec_cond_rt);
   ParGridFunction j_cond(&fes_cond_rt);

   ComputeCurrentDensityOnSubMesh(order, visualization,
                                  phi0_attr, phi1_attr, jn_zero_attr, j_cond);

   // 7a. Save the SubMesh and associated current density in parallel. This
   //     output can be viewed later using GLVis:
   //        "glvis -np <np> -m cond_mesh -g cond_j"
   {
      ostringstream mesh_name, cond_name;
      mesh_name << "cond_mesh." << setfill('0') << setw(6) << myid;
      cond_name << "cond_j." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh_cond.Print(mesh_ofs);

      ofstream cond_ofs(cond_name.str().c_str());
      cond_ofs.precision(8);
      j_cond.Save(cond_ofs);
   }

   // 7b. Send the current density, computed on the SubMesh, to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream port_sock(vishost, visport);
      port_sock << "parallel " << num_procs << " " << myid << "\n";
      port_sock.precision(8);
      port_sock << "solution\n" << pmesh_cond << j_cond
                << "window_title 'Conductor J'"
                << "window_geometry 400 0 400 350" << flush;
   }

   // 8. Define a parallel finite element space on the full mesh. Here we use
   //    the H(curl) finite elements for the vector potential and H(div) for the
   //    current density.
   ND_FECollection fec_nd(order, dim);
   RT_FECollection fec_rt(order - 1, dim);
   ParFiniteElementSpace fespace_nd(&pmesh, &fec_nd);
   ParFiniteElementSpace fespace_rt(&pmesh, &fec_rt);

   ParGridFunction j_full(&fespace_rt);
   j_full = 0.0;
   pmesh_cond.Transfer(j_cond, j_full);

   // 8a. Send the transferred current density to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << j_full
               << "window_title 'J Full'"
               << "window_geometry 400 430 400 350" << flush;
   }

   // 9. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes except for those on a symmetry
   //    plane as essential (Dirichlet) and converting them to a list of
   //    true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      for (int i=0; i<sym_plane_attr.Size(); i++)
      {
         ess_bdr[sym_plane_attr[i]-1] = 0;
      }
      fespace_nd.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 10. Set up the parallel linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system, which in this case is
   //     (J,W_i) where J is given by the function H(div) field transferred
   //     from the SubMesh and W_i are the basis functions in the finite
   //     element fespace.
   VectorGridFunctionCoefficient jCoef(&j_full);
   ParLinearForm b(&fespace_nd);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(jCoef));
   b.Assemble();

   // 11. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x to zero.
   ParGridFunction x(&fespace_nd);
   x = 0.0;

   // 12. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + delta I, by adding the curl-curl and the
   //     mass domain integrators. For standard magnetostatics equations choose
   //     delta << 1. Larger values of delta should make the linear system
   //     easier to solve at the expense of resembling a diffusive quasistatic
   //     magnetic field.  A reasonable balance must be found whenever the mesh
   //     or problem setup is altered.
   ConstantCoefficient muinv(1.0);
   ConstantCoefficient delta(delta_const);
   ParBilinearForm a(&fespace_nd);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(delta));

   // 13. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   if (myid == 0)
   {
      cout << "\nSolving for magnetic vector potential "
           << "using CG with AMS" << endl;
   }

   // 14. Solve the system AX=B using PCG with an AMS preconditioner.
   if (pa)
   {
#ifdef MFEM_USE_AMGX
      MatrixFreeAMS ams(a, *A, fespace_nd, &muinv, &delta, NULL, ess_bdr,
                        useAmgX);
#else
      MatrixFreeAMS ams(a, *A, fespace_nd, &muinv, &delta, NULL, ess_bdr);
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
         (a.StaticCondensationIsEnabled() ? a.SCParFESpace() : &fespace_nd);
      HypreAMS ams(*A.As<HypreParMatrix>(), prec_fespace);
      HyprePCG pcg(*A.As<HypreParMatrix>());
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(ams);
      pcg.Mult(B, X);
   }

   // 15. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   // 16. Save the refined mesh and the solution in parallel. This output can
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

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x
               << "window_title 'Vector Potential'"
               << "window_geometry 800 0 400 350" << flush;
   }

   // 18. Compute the magnetic flux as the curl of the solution
   ParDiscreteLinearOperator curl(&fespace_nd, &fespace_rt);
   curl.AddDomainInterpolator(new CurlInterpolator);
   curl.Assemble();
   curl.Finalize();

   ParGridFunction dx(&fespace_rt);
   curl.Mult(x, dx);

   // 19. Save the curl of the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g dsol".
   {
      ostringstream dsol_name;
      dsol_name << "dsol." << setfill('0') << setw(6) << myid;

      ofstream dsol_ofs(dsol_name.str().c_str());
      dsol_ofs.precision(8);
      dx.Save(dsol_ofs);
   }

   // 20. Send the curl of the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << dx
               << "window_title 'Magnetic Flux'"
               << "window_geometry 1200 0 400 350" << flush;
   }

   // 21. Clean exit
   return 0;
}

void ComputeCurrentDensityOnSubMesh(int order,
                                    bool visualization,
                                    const Array<int> &phi0_attr,
                                    const Array<int> &phi1_attr,
                                    const Array<int> &jn_zero_attr,
                                    ParGridFunction &j_cond)
{
   // Extract the finite element space and mesh on which j_cond is defined
   ParFiniteElementSpace &fes_cond_rt = *j_cond.ParFESpace();
   ParMesh &pmesh_cond = *fes_cond_rt.GetParMesh();
   int myid = fes_cond_rt.GetMyRank();
   int dim  = pmesh_cond.Dimension();

   // Define a parallel finite element space on the SubMesh. Here we use the
   // H1 finite elements for the electrostatic potential.
   H1_FECollection fec_h1(order, dim);
   ParFiniteElementSpace fes_cond_h1(&pmesh_cond, &fec_h1);

   // Define the conductivity coefficient and the boundaries associated with the
   // fixed potentials phi0 and phi1 which will drive the current.
   ConstantCoefficient sigmaCoef(1.0);
   Array<int> ess_bdr_phi(pmesh_cond.bdr_attributes.Max());
   Array<int> ess_bdr_j(pmesh_cond.bdr_attributes.Max());
   Array<int> ess_bdr_tdof_phi;
   ess_bdr_phi = 0;
   ess_bdr_j   = 0;
   for (int i=0; i<phi0_attr.Size(); i++)
   {
      ess_bdr_phi[phi0_attr[i]-1] = 1;
   }
   for (int i=0; i<phi1_attr.Size(); i++)
   {
      ess_bdr_phi[phi1_attr[i]-1] = 1;
   }
   for (int i=0; i<jn_zero_attr.Size(); i++)
   {
      ess_bdr_j[jn_zero_attr[i]-1] = 1;
   }
   fes_cond_h1.GetEssentialTrueDofs(ess_bdr_phi, ess_bdr_tdof_phi);

   // Setup the bilinear form corresponding to -Div(sigma Grad phi)
   ParBilinearForm a_h1(&fes_cond_h1);
   a_h1.AddDomainIntegrator(new DiffusionIntegrator(sigmaCoef));
   a_h1.Assemble();

   // Set the r.h.s. to zero
   ParLinearForm b_h1(&fes_cond_h1);
   b_h1 = 0.0;

   // Setup the boundary conditions on phi
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   ParGridFunction phi_h1(&fes_cond_h1);
   phi_h1 = 0.0;

   Array<int> bdr0(pmesh_cond.bdr_attributes.Max()); bdr0 = 0;
   for (int i=0; i<phi0_attr.Size(); i++)
   {
      bdr0[phi0_attr[i]-1] = 1;
   }
   phi_h1.ProjectBdrCoefficient(zero, bdr0);

   Array<int> bdr1(pmesh_cond.bdr_attributes.Max()); bdr1 = 0;
   for (int i=0; i<phi1_attr.Size(); i++)
   {
      bdr1[phi1_attr[i]-1] = 1;
   }
   phi_h1.ProjectBdrCoefficient(one, bdr1);

   // Solve the linear system using algebraic multigrid
   {
      if (myid == 0)
      {
         cout << "\nSolving for electric potential "
              << "using CG with AMG" << endl;
      }
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

   if (visualization)
   {
      int num_procs = fes_cond_h1.GetNRanks();
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream port_sock(vishost, visport);
      port_sock << "parallel " << num_procs << " " << myid << "\n";
      port_sock.precision(8);
      port_sock << "solution\n" << pmesh_cond << phi_h1
                << "window_title 'Conductor Potential'"
                << "window_geometry 0 0 400 350" << flush;
   }

   // Solve for the current density J = -sigma Grad phi with boundary conditions
   // J.n = 0 on the walls of the conductor but not on the ports where phi=0 and
   // phi=1.

   // J will be computed in H(div) so we need an RT mass matrix
   ParBilinearForm m_rt(&fes_cond_rt);
   m_rt.AddDomainIntegrator(new VectorFEMassIntegrator);
   m_rt.Assemble();

   // Assemble the (sigma Grad phi) operator
   ParMixedBilinearForm d_h1(&fes_cond_h1, &fes_cond_rt);
   d_h1.AddDomainIntegrator(new MixedVectorGradientIntegrator(sigmaCoef));
   d_h1.Assemble();

   // Compute the r.h.s, b_rt = sigma E = -sigma Grad phi
   ParLinearForm b_rt(&fes_cond_rt);
   d_h1.Mult(phi_h1, b_rt);
   b_rt *= -1.0;

   // Apply the necessary boundary conditions and solve for J in H(div)
   HYPRE_BigInt glb_size_rt = fes_cond_rt.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "\nSolving for current density in H(Div) "
           << "using diagonally scaled CG" << endl;
      cout << "Size of linear system: "
           << glb_size_rt << endl;
   }
   Array<int> ess_bdr_tdof_rt;
   OperatorPtr M;
   Vector B, X;

   fes_cond_rt.GetEssentialTrueDofs(ess_bdr_j, ess_bdr_tdof_rt);

   j_cond = 0.0;
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
