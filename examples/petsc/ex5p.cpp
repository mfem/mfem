//                       MFEM Example 5 - Parallel Version
//                              PETSc Modification
//
// Compile with: make ex5p
//
// Sample runs:
//    mpirun -np 4 ex5p -m ../../data/beam-tet.mesh --petscopts rc_ex5p_fieldsplit
//    mpirun -np 4 ex5p -m ../../data/star.mesh     --petscopts rc_ex5p_bddc --nonoverlapping
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//                                 k*u + grad p = f
//                                 - div u      = g
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockOperator class, as
//               well as the collective saving of several grid functions in a
//               VisIt (visit.llnl.gov) visualization format.
//
//               Two types of PETSc solvers can be used: BDDC or fieldsplit.
//               When using BDDC, the nonoverlapping assembly feature should be
//               used. This specific example needs PETSc compiled with support
//               for SuiteSparse and/or MUMPS for using BDDC.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
real_t pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
real_t gFun(const Vector & x);
real_t f_natural(const Vector & x);

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   bool verbose = (myid == 0);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ser_ref_levels = -1;
   int par_ref_levels = 2;
   int order = 1;
   bool par_format = false;
   bool visualization = 1;
   bool use_petsc = true;
   bool use_nonoverlapping = false;
   bool local_bdr_spec = false;
   const char *petscrc_file = "";
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&par_format, "-pf", "--parallel-format", "-sf",
                  "--serial-format",
                  "Format to use when saving the results for VisIt.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the linear system.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.AddOption(&use_nonoverlapping, "-nonoverlapping", "--nonoverlapping",
                  "-no-nonoverlapping", "--no-nonoverlapping",
                  "Use or not the block diagonal PETSc's matrix format "
                  "for non-overlapping domain decomposition.");
   args.AddOption(&local_bdr_spec, "-local-bdr", "--local-bdr", "-no-local-bdr",
                  "--no-local-bdr",
                  "Specify boundary dofs in local (Vdofs) ordering.");
   args.Parse();
   if (!args.Good())
   {
      if (verbose)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (verbose)
   {
      args.PrintOptions(cout);
   }

   // 2b. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 2c. We initialize PETSc
   if (use_petsc) { MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL); }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
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

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
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

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

   ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh, hdiv_coll);
   ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh, l2_coll);

   HYPRE_BigInt dimR = R_space->GlobalTrueVSize();
   HYPRE_BigInt dimW = W_space->GlobalTrueVSize();

   if (verbose)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(R) = " << dimR << "\n";
      std::cout << "dim(W) = " << dimW << "\n";
      std::cout << "dim(R+W) = " << dimR + dimW << "\n";
      std::cout << "***********************************************************\n";
   }

   // 7. Define the two BlockStructure of the problem.  block_offsets is used
   //    for Vector based on dof (like ParGridFunction or ParLinearForm),
   //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
   //    for the rhs and solution of the linear system).  The offsets computed
   //    here are local to the processor.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = R_space->GetVSize();
   block_offsets[2] = W_space->GetVSize();
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(3); // number of variables + 1
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = R_space->TrueVSize();
   block_trueOffsets[2] = W_space->TrueVSize();
   block_trueOffsets.PartialSum();

   // 8. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   FunctionCoefficient fnatcoeff(f_natural);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   // 9. Define the parallel grid function and parallel linear forms, solution
   //    vector and rhs.
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);
   BlockVector trueX(block_trueOffsets, mt), trueRhs(block_trueOffsets, mt);

   ParLinearForm *fform(new ParLinearForm);
   fform->Update(R_space, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
   fform->Assemble();
   fform->SyncAliasMemory(rhs);
   fform->ParallelAssemble(trueRhs.GetBlock(0));
   trueRhs.GetBlock(0).SyncAliasMemory(trueRhs);

   ParLinearForm *gform(new ParLinearForm);
   gform->Update(W_space, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform->Assemble();
   gform->SyncAliasMemory(rhs);
   gform->ParallelAssemble(trueRhs.GetBlock(1));
   trueRhs.GetBlock(1).SyncAliasMemory(trueRhs);

   // 10. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   ParBilinearForm *mVarf(new ParBilinearForm(R_space));
   ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(R_space, W_space));

   PetscParMatrix *pM = NULL, *pB = NULL, *pBT = NULL;
   HypreParMatrix *M = NULL, *B = NULL, *BT = NULL;
   Operator::Type tid =
      !use_petsc ? Operator::Hypre_ParCSR :
      (use_nonoverlapping ? Operator::PETSC_MATIS : Operator::PETSC_MATAIJ);
   OperatorHandle Mh(tid), Bh(tid);

   mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
   mVarf->Assemble();
   mVarf->Finalize();
   mVarf->ParallelAssemble(Mh);
   if (!use_petsc) { Mh.Get(M); }
   else { Mh.Get(pM); }
   Mh.SetOperatorOwner(false);

   bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf->Assemble();
   bVarf->Finalize();
   if (!use_petsc)
   {
      B = bVarf->ParallelAssemble();
      (*B) *= -1;
   }
   else
   {
      bVarf->ParallelAssemble(Bh);
      Bh.Get(pB);
      Bh.SetOperatorOwner(false);
      (*pB) *= -1;
   }

   if (!use_petsc) { BT = B->Transpose(); }
   else { pBT = pB->Transpose(); };

   Operator *darcyOp = NULL;
   if (!use_petsc)
   {
      BlockOperator *tdarcyOp = new BlockOperator(block_trueOffsets);
      tdarcyOp->SetBlock(0,0,M);
      tdarcyOp->SetBlock(0,1,BT);
      tdarcyOp->SetBlock(1,0,B);
      darcyOp = tdarcyOp;
   }
   else
   {
      // We construct the BlockOperator and we then convert it to a
      // PetscParMatrix to avoid any conversion in the construction of the
      // preconditioners.
      BlockOperator *tdarcyOp = new BlockOperator(block_trueOffsets);
      tdarcyOp->SetBlock(0,0,pM);
      tdarcyOp->SetBlock(0,1,pBT);
      tdarcyOp->SetBlock(1,0,pB);
      darcyOp = new PetscParMatrix(pM->GetComm(),tdarcyOp,
                                   use_nonoverlapping ? Operator::PETSC_MATIS :
                                   Operator::PETSC_MATAIJ);
      delete tdarcyOp;
   }

   // 11. Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     pressure Schur Complement.
   PetscPreconditioner *pdarcyPr = NULL;
   BlockDiagonalPreconditioner *darcyPr = NULL;
   HypreSolver *invM = NULL, *invS = NULL;
   HypreParMatrix *S = NULL;
   HypreParMatrix *MinvBt = NULL;
   HypreParVector *Md = NULL;
   if (!use_petsc)
   {
      MinvBt = B->Transpose();
      Md = new HypreParVector(MPI_COMM_WORLD, M->GetGlobalNumRows(),
                              M->GetRowStarts());
      M->GetDiag(*Md);

      MinvBt->InvScaleRows(*Md);
      S = ParMult(B, MinvBt);

      invM = new HypreDiagScale(*M);
      invS = new HypreBoomerAMG(*S);

      invM->iterative_mode = false;
      invS->iterative_mode = false;

      darcyPr = new BlockDiagonalPreconditioner(block_trueOffsets);
      darcyPr->SetDiagonalBlock(0, invM);
      darcyPr->SetDiagonalBlock(1, invS);
   }
   else
   {
      if (use_nonoverlapping)
      {
         PetscBDDCSolverParams opts;

         // For saddle point problems, we need to provide BDDC the list of
         // boundary dofs either essential or natural.
         // Since R_space is the only space that may have boundary dofs and it
         // is ordered first then W_space, we don't need any local offset when
         // specifying the dofs.
         Array<int> bdr_tdof_list;
         if (pmesh->bdr_attributes.Size())
         {
            Array<int> bdr(pmesh->bdr_attributes.Max());
            bdr = 1;

            if (!local_bdr_spec)
            {
               // Essential dofs in global ordering
               R_space->GetEssentialTrueDofs(bdr, bdr_tdof_list);
            }
            else
            {
               // Alternatively, you can also provide the list of dofs in local
               // ordering
               R_space->GetEssentialVDofs(bdr, bdr_tdof_list);
               bdr_tdof_list.SetSize(R_space->GetVSize()+W_space->GetVSize(),0);
            }
            opts.SetNatBdrDofs(&bdr_tdof_list,local_bdr_spec);
         }
         else
         {
            MFEM_WARNING("Missing boundary dofs. This may cause solver failures.");
         }

         // See also command line options rc_ex5p_bddc
         pdarcyPr = new PetscBDDCSolver(MPI_COMM_WORLD,*darcyOp,opts,"prec_");
      }
      else
      {
         // With PETSc, we can construct the (same) block-diagonal solver with
         // command line options (see rc_ex5p_fieldsplit)
         pdarcyPr = new PetscFieldSplitSolver(MPI_COMM_WORLD,*darcyOp,"prec_");
      }
   }

   // 12. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.

   int maxIter(500);
   real_t rtol(1.e-6);
   real_t atol(1.e-10);

   chrono.Clear();
   chrono.Start();

   trueX = 0.0;
   if (!use_petsc)
   {
      MINRESSolver solver(MPI_COMM_WORLD);
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*darcyOp);
      solver.SetPreconditioner(*darcyPr);
      solver.SetPrintLevel(1);
      solver.Mult(trueRhs, trueX);
      if (verbose)
      {
         if (solver.GetConverged())
         {
            std::cout << "MINRES converged in " << solver.GetNumIterations()
                      << " iterations with a residual norm of "
                      << solver.GetFinalNorm() << ".\n";
         }
         else
         {
            std::cout << "MINRES did not converge in "
                      << solver.GetNumIterations()
                      << " iterations. Residual norm is "
                      << solver.GetFinalNorm() << ".\n";
         }
         std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
      }

   }
   else
   {
      std::string solvertype;
      PetscLinearSolver *solver;
      if (use_nonoverlapping)
      {
         // We can use conjugate gradients to solve the problem
         solver = new PetscPCGSolver(MPI_COMM_WORLD);
         solvertype = "PCG";
      }
      else
      {
         solver = new PetscLinearSolver(MPI_COMM_WORLD);
         solvertype = "MINRES";
      }
      solver->SetOperator(*darcyOp);
      solver->SetPreconditioner(*pdarcyPr);
      solver->SetAbsTol(atol);
      solver->SetRelTol(rtol);
      solver->SetMaxIter(maxIter);
      solver->SetPrintLevel(2);
      solver->Mult(trueRhs, trueX);
      if (verbose)
      {
         if (solver->GetConverged())
         {
            std::cout << solvertype << " converged in "
                      << solver->GetNumIterations()
                      << " iterations with a residual norm of "
                      << solver->GetFinalNorm() << ".\n";
         }
         else
         {
            std::cout << solvertype << " did not converge in "
                      << solver->GetNumIterations()
                      << " iterations. Residual norm is "
                      << solver->GetFinalNorm() << ".\n";
         }
         std::cout << solvertype << " solver took "
                   << chrono.RealTime() << "s. \n";
      }
      delete solver;
   }
   chrono.Stop();

   // 13. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor. Compute
   //     L2 error norms.
   ParGridFunction *u(new ParGridFunction);
   ParGridFunction *p(new ParGridFunction);
   u->MakeRef(R_space, x.GetBlock(0), 0);
   p->MakeRef(W_space, x.GetBlock(1), 0);
   u->Distribute(&(trueX.GetBlock(0)));
   p->Distribute(&(trueX.GetBlock(1)));

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   real_t err_u  = u->ComputeL2Error(ucoeff, irs);
   real_t norm_u = ComputeGlobalLpNorm(2, ucoeff, *pmesh, irs);
   real_t err_p  = p->ComputeL2Error(pcoeff, irs);
   real_t norm_p = ComputeGlobalLpNorm(2, pcoeff, *pmesh, irs);

   if (verbose)
   {
      std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
      std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
   }

   // 14. Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol_*".
   {
      ostringstream mesh_name, u_name, p_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      u_name << "sol_u." << setfill('0') << setw(6) << myid;
      p_name << "sol_p." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream u_ofs(u_name.str().c_str());
      u_ofs.precision(8);
      u->Save(u_ofs);

      ofstream p_ofs(p_name.str().c_str());
      p_ofs.precision(8);
      p->Save(p_ofs);
   }

   // 15. Save data in the VisIt format
   VisItDataCollection visit_dc("Example5-Parallel", pmesh);
   visit_dc.RegisterField("velocity", u);
   visit_dc.RegisterField("pressure", p);
   visit_dc.SetFormat(!par_format ?
                      DataCollection::SERIAL_FORMAT :
                      DataCollection::PARALLEL_FORMAT);
   visit_dc.Save();

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *pmesh << *u << "window_title 'Velocity'"
             << endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << num_procs << " " << myid << "\n";
      p_sock.precision(8);
      p_sock << "solution\n" << *pmesh << *p << "window_title 'Pressure'"
             << endl;
   }

   // 17. Free the used memory.
   delete fform;
   delete gform;
   delete u;
   delete p;
   delete darcyOp;
   delete darcyPr;
   delete pdarcyPr;
   delete invM;
   delete invS;
   delete S;
   delete BT;
   delete B;
   delete M;
   delete pBT;
   delete pB;
   delete pM;
   delete MinvBt;
   delete Md;
   delete mVarf;
   delete bVarf;
   delete W_space;
   delete R_space;
   delete l2_coll;
   delete hdiv_coll;
   delete pmesh;

   // We finalize PETSc
   if (use_petsc) { MFEMFinalizePetsc(); }

   return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(0.0);
   if (x.Size() == 3)
   {
      zi = x(2);
   }

   u(0) = - exp(xi)*sin(yi)*cos(zi);
   u(1) = - exp(xi)*cos(yi)*cos(zi);

   if (x.Size() == 3)
   {
      u(2) = exp(xi)*sin(yi)*sin(zi);
   }
}

// Change if needed
real_t pFun_ex(const Vector & x)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(0.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
   f = 0.0;
}

real_t gFun(const Vector & x)
{
   if (x.Size() == 3)
   {
      return -pFun_ex(x);
   }
   else
   {
      return 0;
   }
}

real_t f_natural(const Vector & x)
{
   return (-pFun_ex(x));
}
