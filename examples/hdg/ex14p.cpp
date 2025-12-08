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
//               mpirun -np 4 ex14p -pa -rs 1 -rp 0 -o 3
//               mpirun -np 4 ex14p -pa -rs 1 -rp 0 -m ../data/fichera.mesh -o 3
//
// Device sample runs:
//               mpirun -np 4 ex14p -pa -rs 2 -rp 0 -d cuda -o 3
//               mpirun -np 4 ex14p -pa -rs 2 -rp 0 -d cuda -m ../data/fichera.mesh -o 3
//
// Description:  This example code demonstrates the use of MFEM to define a
//               discontinuous Galerkin (DG) finite element discretization of
//               the Poisson problem -Delta u = 1 with homogeneous Dirichlet
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

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ser_ref_levels = -1;
   int par_ref_levels = 2;
   int order = 1;
   bool dg = false;
   bool brt = false;
   real_t td = 0.5;
   bool reduction = false;
   bool hybridization = false;
   bool trace_h1 = false;
   bool pa = false;
   bool visualization = 1;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial,"
                  " -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction of DG flux.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default DG).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   // 3. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code. NURBS meshes are projected to second order meshes.
   Mesh mesh(mesh_file);
   int dim = mesh.Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ser_ref_levels' of uniform refinement. By default,
   //    or if ser_ref_levels < 0, we choose it to be the largest number that
   //    gives a final mesh with no more than 50,000 elements.
   {
      if (ser_ref_levels < 0)
      {
         ser_ref_levels = (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ser_ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use discontinuous finite elements of the specified order >= 0.
   FiniteElementCollection *R_coll;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      R_coll = new L2_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else if (brt)
   {
      R_coll = new BrokenRT_FECollection(order, dim);
   }
   else
   {
      R_coll = new RT_FECollection(order, dim);
   }
   FiniteElementCollection *W_coll = new L2_FECollection(order, dim);

   ParFiniteElementSpace *R_space = new ParFiniteElementSpace(&pmesh, R_coll,
                                                              (dg)?(dim):(1));
   ParFiniteElementSpace *W_space = new ParFiniteElementSpace(&pmesh, W_coll);


   HYPRE_BigInt q_size = R_space->GlobalTrueVSize();
   HYPRE_BigInt u_size = W_space->GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of flux unknowns: " << q_size << endl;
      cout << "Number of potential unknowns: " << u_size << endl;
   }

   // 7. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system.

   ParDarcyForm darcy(R_space, W_space);
   const Array<int> &block_offsets = darcy.GetOffsets();
   const Array<int> &block_trueOffsets = darcy.GetTrueOffsets();

   MemoryType mt = device.GetMemoryType();
   BlockVector b(block_offsets, mt);

   b.GetBlock(0) = 0.0;

   ConstantCoefficient one(1.0), negone(-1.0);

   ParLinearForm f;
   f.Update(W_space, b.GetBlock(1), 0);
   f.AddDomainIntegrator(new DomainLFIntegrator(negone));
   f.Assemble();
   f.SyncAliasMemory(b);

   // 8. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   BlockVector x(block_offsets, mt);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and the interior and boundary DG face integrators.
   //    Note that boundary conditions are imposed weakly in the form, so there
   //    is no need for dof elimination. After serial and parallel assembly we
   //    extract the corresponding parallel matrix A.
   ParBilinearForm *Mq = darcy.GetParFluxMassForm();
   ParMixedBilinearForm *Bq = darcy.GetParFluxDivForm();
   ParBilinearForm *Mu = (dg)?(darcy.GetParPotentialMassForm()):(NULL);

   if (dg)
   {
      Mq->AddDomainIntegrator(new VectorMassIntegrator());
      Bq->AddDomainIntegrator(new VectorDivergenceIntegrator());
      Bq->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                       new DGNormalTraceIntegrator(-1.)));
      Mu->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, td));
   }
   else
   {
      Mq->AddDomainIntegrator(new VectorFEMassIntegrator());
      Bq->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      if (brt)
      {
         Bq->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                          new DGNormalTraceIntegrator(-1.)));
      }
   }

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;

   FiniteElementCollection *trace_coll = NULL;
   ParFiniteElementSpace *trace_space = NULL;

   if (hybridization)
   {
      if (trace_h1)
      {
         trace_coll = new H1_Trace_FECollection(order+1, dim);
      }
      else
      {
         trace_coll = new DG_Interface_FECollection(order, dim);
      }
      trace_space = new ParFiniteElementSpace(&pmesh, trace_coll);
      darcy.EnableHybridization(trace_space,
                                new NormalTraceJumpIntegrator(),
                                ess_flux_tdofs_list);
   }
   else if (reduction && (dg || brt))
   {
      darcy.EnableFluxReduction();
   }

   if (pa) { darcy.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   darcy.Assemble();

   // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.

   OperatorHandle A;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux_tdofs_list, x, b, A, X, B);

   // 11. Depending on the symmetry of A, define and apply a parallel PCG or
   //     GMRES solver for AX=B using the BoomerAMG preconditioner from hypre.

   int maxIter(500);
   real_t rtol(1.0e-6);
   real_t atol(0.0);

   if (hybridization || (reduction && (dg || brt)))
   {
      // 12. Construct the preconditioner
      HypreBoomerAMG prec;

      // 13. Solve the linear system with GMRES.
      //     Check the norm of the unpreconditioned residual.
      GMRESSolver solver(MPI_COMM_WORLD);
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetPreconditioner(prec);
      solver.SetOperator(*A);
      solver.SetPrintLevel(1);

      solver.Mult(B, X);
   }
   else
   {
      // 12. Construct the operators for preconditioner
      //
      //                 P = [ diag(M)         0         ]
      //                     [  0       B diag(M)^-1 B^T ]
      //
      //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
      //     pressure Schur Complement
      HypreParMatrix *MinvBt = NULL;
      HypreParVector *Md = NULL;
      HypreParMatrix *S = NULL;
      Solver *invM, *invS;

      if (pa)
      {
         Vector Md(R_space->GetTrueVSize());
         Mq->AssembleDiagonal(Md);
         auto Md_host = Md.HostRead();
         Vector invMd(Md.Size());
         for (int i=0; i<Md.Size(); ++i)
         {
            invMd(i) = 1.0 / Md_host[i];
         }

         Vector BMBt_diag(W_space->GetTrueVSize());
         Bq->AssembleDiagonal_ADAt(invMd, BMBt_diag);

         Array<int> ess_tdof_list;  // empty

         invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
         invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
      }
      else
      {
         HypreParMatrix &M = *Mq->ParallelAssembleInternalMatrix();
         Md = new HypreParVector(MPI_COMM_WORLD, M.GetGlobalNumRows(),
                                 M.GetRowStarts());
         M.GetDiag(*Md);

         HypreParMatrix &B = *Bq->ParallelAssembleInternalMatrix();
         MinvBt = B.Transpose();
         MinvBt->InvScaleRows(*Md);
         S = ParMult(&B, MinvBt);

         if (Mu)
         {
            HypreParMatrix &Mt = *Mu->ParallelAssembleInternalMatrix();
            HypreParMatrix *Snew = ParAdd(&Mt, S);
            delete S;
            S = Snew;
         }

         invM = new HypreDiagScale(M);
         invS = new HypreBoomerAMG(*S);
      }

      invM->iterative_mode = false;
      invS->iterative_mode = false;

      BlockDiagonalPreconditioner darcyPrec(block_trueOffsets);
      darcyPrec.SetDiagonalBlock(0, invM);
      darcyPrec.SetDiagonalBlock(1, invS);

      // 13. Solve the linear system with MINRES.
      //     Check the norm of the unpreconditioned residual.

      MINRESSolver solver(MPI_COMM_WORLD);
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*A);
      solver.SetPreconditioner(darcyPrec);
      solver.SetPrintLevel(1);

      solver.Mult(B, X);

      delete invM;
      delete invS;
      delete S;
      delete Md;
      delete MinvBt;
   }

   darcy.RecoverFEMSolution(X, b, x);
   if (device.IsEnabled()) { x.HostRead(); }

   // 12. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".

   ParGridFunction u(W_space, x.GetBlock(1), 0);

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << Mpi::WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << Mpi::WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      u.Save(sol_ofs);
   }

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << u << flush;
   }

   delete W_space;
   delete R_space;
   delete trace_space;
   delete W_coll;
   delete R_coll;
   delete trace_coll;

   return 0;
}
