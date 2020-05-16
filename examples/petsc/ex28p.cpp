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
//               The example demonstrates the use of the BlockMatrix class, as
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
//#include <slepceps.h>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   bool verbose = (myid == 0);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 1;
   int nev = 5;
   bool par_format = false;
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
   args.AddOption(&par_format, "-pf", "--parallel-format", "-sf",
                  "--serial-format",
                  "Format to use when saving the results for VisIt.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (verbose)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (verbose)
   {
      args.PrintOptions(cout);
   }
   // 2b. We initialize SLEPc
   MFEMInitializeSlepc(NULL,NULL,(char*)0,NULL);

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }
   pmesh->ReorientTetMesh();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Raviart-Thomas finite elements of the specified order.
   std::cout << "dim: " << dim << "\n";
   FiniteElementCollection *hcurl_coll = new ND_FECollection(order, dim);
   FiniteElementCollection *h1_coll = new H1_FECollection(order, dim);

   ParFiniteElementSpace *N_space = new ParFiniteElementSpace(pmesh, hcurl_coll);
   ParFiniteElementSpace *L_space = new ParFiniteElementSpace(pmesh, h1_coll);

   HYPRE_Int dimN = N_space->GlobalTrueVSize();
   HYPRE_Int dimL = L_space->GlobalTrueVSize();

   if (verbose)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(N) = " << dimN << "\n";
      std::cout << "dim(L) = " << dimL << "\n";
      std::cout << "dim(N+L) = " << dimN + dimL << "\n";
      std::cout << "***********************************************************\n";
   }

   // 7. Define the two BlockStructure of the problem.  block_offsets is used
   //    for Vector based on dof (like ParGridFunction or ParLinearForm),
   //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
   //    for the rhs and solution of the linear system).  The offsets computed
   //    here are local to the processor.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = N_space->GetVSize();
   block_offsets[2] = L_space->GetVSize();
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(3); // number of variables + 1
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = N_space->TrueVSize();
   block_trueOffsets[2] = L_space->TrueVSize();
   block_trueOffsets.PartialSum();

   // 8. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient u_r_func(1.0);
   Vector e_r(2);
   double k0 = M_PI*2/1.55;
   e_r(0) = -pow(k0*1.45,2);//-k0^2*e_r
   e_r(1) = -pow(k0*3.45,2);
   PWConstCoefficient e_r_func(e_r);

   // 9. Define the parallel grid function and parallel linear forms, solution
   //    vector and rhs.
   BlockVector x(block_offsets);
   BlockVector trueX(block_trueOffsets);

   //boundary attributes
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      std::cout << "mesh bdr " << pmesh->bdr_attributes.Size() << " " <<
                pmesh->bdr_attributes.Max() << "\n";
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
   }



   // 10. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   ParBilinearForm *att = new ParBilinearForm(N_space);
   ParBilinearForm *btt = new ParBilinearForm(N_space);
   ParBilinearForm *azz = new ParBilinearForm(L_space);
   ParBilinearForm *bzz = new ParBilinearForm(L_space);
   ParMixedBilinearForm *btz = new ParMixedBilinearForm(N_space, L_space);

   PetscParMatrix *pAtt = NULL, *pBtt = NULL, *pBzz = NULL;
   PetscParMatrix *pAzz = NULL, *pBtz = NULL, *pBzt = NULL;
   Operator::Type tid = Operator::PETSC_MATAIJ;
   OperatorHandle Atth(tid), Btth(tid), Bzzh(tid), Btzh(tid), Azzh(tid);

   att->AddDomainIntegrator(new CurlCurlIntegrator(u_r_func));
   att->AddDomainIntegrator(new VectorFEMassIntegrator(e_r_func));
   att->Assemble();
   att->EliminateEssentialBCDiag(ess_bdr, 1.0);
   att->Finalize();
   att->ParallelAssemble(Atth);
   Atth.Get(pAtt);
   Atth.SetOperatorOwner(false);

   azz->Assemble();
   azz->EliminateEssentialBCDiag(ess_bdr, 1.0);
   azz->Finalize();
   azz->ParallelAssemble(Azzh);
   Azzh.Get(pAzz);
   Azzh.SetOperatorOwner(false);

   btt->AddDomainIntegrator(new VectorFEMassIntegrator(u_r_func));
   btt->Assemble();
   btt->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
   btt->Finalize();
   btt->ParallelAssemble(Btth);
   Btth.Get(pBtt);
   Btth.SetOperatorOwner(false);
   (*pBtt) *= -1;

   bzz->AddDomainIntegrator(new DiffusionIntegrator(u_r_func));
   bzz->AddDomainIntegrator(new MassIntegrator(e_r_func));
   bzz->Assemble();
   btt->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
   bzz->Finalize();
   bzz->ParallelAssemble(Bzzh);
   Bzzh.Get(pBzz);
   Bzzh.SetOperatorOwner(false);
   (*pBzz) *= -1;

   ParLinearForm dummy(N_space);
   btz->AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(u_r_func));
   btz->Assemble();
   btz->EliminateTestDofs(ess_bdr);
   btz->EliminateTrialDofs(ess_bdr,x.GetBlock(0),dummy);
   btz->Finalize();
   btz->ParallelAssemble(Btzh);
   Btzh.Get(pBtz);
   Btzh.SetOperatorOwner(false);
   //(*pBtz) *= -1;

   pBzt = pBtz->Transpose();

   PetscParMatrix *LHSOp = NULL, *RHSOp = NULL;
   // We construct the BlockOperator and we then convert it to a
   // PetscParMatrix to avoid any conversion in the construction of the
   // preconditioners.
   BlockOperator *tLHSOp = new BlockOperator(block_trueOffsets);
   tLHSOp->SetBlock(0,0,pAtt);
   tLHSOp->SetBlock(1,1,pAzz);
   LHSOp = new PetscParMatrix(MPI_COMM_WORLD,tLHSOp,Operator::PETSC_MATAIJ);
   delete tLHSOp;

   BlockOperator *tRHSOp = new BlockOperator(block_trueOffsets);
   tRHSOp->SetBlock(0,0,pBtt);
   tRHSOp->SetBlock(1,1,pBzz);
   tRHSOp->SetBlock(1,0,pBtz);
   tRHSOp->SetBlock(0,1,pBzt);
   RHSOp = new PetscParMatrix(MPI_COMM_WORLD,tRHSOp,Operator::PETSC_MATAIJ);
   delete tRHSOp;

   // 12. Solve the linear system with slepc.
   std::cout << "Solving...\n";
   int maxIter(500);
   double rtol(1.e-6);
   double atol(1.e-10);

   trueX = 0.0;

   /*PetscParVector *X = new PetscParVector(*LHSOp, true, false);
   X->PlaceArray(trueX.GetData());
   EPS eps;
   EPSCreate(PETSC_COMM_WORLD,&eps);
   EPSSetDimensions(eps,1,PETSC_DECIDE,PETSC_DECIDE);
   EPSSetTolerances(eps,1e-12,500);
   EPSSetOperators(eps,*LHSOp,*RHSOp);
   EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE);
   EPSSetTarget(eps,pow(k0*3.44,2));
   ST st;
   EPSGetST(eps,&st);
   STSetType(st,STSINVERT);
   EPSSetFromOptions(eps);
   EPSSolve(eps);*/
   SlepcEigenSolver *solver = new SlepcEigenSolver(MPI_COMM_WORLD);
   solver->SetOperators(*LHSOp,*RHSOp);
   solver->SetTol(1e-12);
   solver->SetMaxIter(500);
   solver->SetNumModes(1);
   solver->SetWhichEigenpairs(SlepcEigenSolver::TARGET_MAGNITUDE);
   solver->SetTarget(pow(k0*3.44,2));
   solver->SetSpectralTransformation(SlepcEigenSolver::SHIFT_INVERT);
   solver->Solve();
   /*PetscInt num_converged;
   PetscInt it_num;
   PetscScalar lr, lc;
   EPSGetConverged(eps,&num_converged);
   std::cout << "num_converged = " << num_converged << "\n";
   EPSGetIterationNumber(eps,&it_num);
   std::cout <<"it_num = " << it_num << "\n";
   EPSReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);
   Vec xi;
   MatCreateVecs(*LHSOp,NULL,&xi);
   EPSGetEigenpair(eps,0,&lr,&lc,*X,xi);
   X->ResetArray();
   PetscReal re,im;
   re = lr;
   im = lc;*/
   double re;
   solver->GetEigenvalue(0,re);
   Vector dummy2(block_trueOffsets[2]);
   solver->GetEigenvector(0,trueX);
   std::cout << sqrt(re)/k0 << "\n";

   // 13. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor. Compute
   //     L2 error norms.
   ParGridFunction *exy(new ParGridFunction);
   ParGridFunction *ez(new ParGridFunction);
   exy->MakeRef(N_space, x.GetBlock(0), 0);
   ez->MakeRef(L_space, x.GetBlock(1), 0);
   exy->Distribute(&(trueX.GetBlock(0)));
   ez->Distribute(&(trueX.GetBlock(1)));

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

      ofstream exy_ofs(u_name.str().c_str());
      exy_ofs.precision(8);
      exy->Save(exy_ofs);

      ofstream ez_ofs(p_name.str().c_str());
      ez_ofs.precision(8);
      ez->Save(ez_ofs);
   }

   // 15. Save data in the VisIt format
   VisItDataCollection visit_dc("Example5-Parallel", pmesh);
   visit_dc.RegisterField("Exy", exy);
   visit_dc.RegisterField("Ez", ez);
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
      u_sock << "solution\n" << *pmesh << *exy << "window_title 'Velocity'"
             << endl;
      u_sock << "keys Rjl!\n";
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << num_procs << " " << myid << "\n";
      p_sock.precision(8);
      p_sock << "solution\n" << *pmesh << *ez << "window_title 'Pressure'"
             << endl;
      p_sock << "keys Rjl!\n";
   }

   // 17. Free the used memory.
   delete exy;
   delete ez;
   delete N_space;
   delete L_space;
   delete h1_coll;
   delete hcurl_coll;
   delete pmesh;

   // We finalize SLEPc
   MFEMFinalizeSlepc();
   MPI_Finalize();

   return 0;
}

