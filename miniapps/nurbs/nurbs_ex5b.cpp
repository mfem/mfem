//                                MFEM Example 5b -- modified for NURBS FE Darcy converted to stokes
//
// Compile with: make nurbs_ex5b
//
// Sample runs:  nurbs_ex5b -m ../../data/square-nurbs.mesh -o 3
//               nurbs_ex5b -m ../../data/cube-nurbs.mesh -r 3
//               nurbs_ex5b -m ../../data/pipe-nurbs-2d.mesh
//               nurbs_ex5b -m ../../data/beam-tet.mesh
//               nurbs_ex5b -m ../../data/beam-hex.mesh
//               nurbs_ex5b -m ../../data/escher.mesh
//               nurbs_ex5b -m ../../data/fichera.mesh
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//
//                          -k*Delta*u + grad p = f
//                                 - div u      = g
//
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               NURBS-based H(div) spaces only implemented for meshes
//               consisting of a single patch.
//
//               The example demonstrates the use of the BlockOperator class, as
//               well as the collective saving of several grid functions in
//               VisIt (visit.llnl.gov) and ParaView (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.




#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

void sol_fun(const Vector & x, Vector &sol)
{
   sol = 0.0;
   if ((x[1] - 0.99 > 0.0) &&
       (fabs(x[0] - 0.5) < 0.49) )
   {
      sol[0] = 1.0;
   }
}

void force_fun(const Vector & x, Vector &f)
{
   f = 0.0;
}

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // Parse command-line options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   int ref_levels = -1;
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // Refine the mesh to increase the resolution
   {
      if (ref_levels < 0)
      {
         ref_levels =
            (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   //  Define the mixed finite element space on the mesh.
   FiniteElementCollection *hdiv_coll = nullptr;
   FiniteElementCollection *l2_coll = nullptr;
   NURBSExtension *NURBSext = nullptr;

   if (mesh->NURBSext)
   {
      hdiv_coll = new NURBS_HDivFECollection(order,dim);
      l2_coll   = new NURBSFECollection(order);
      NURBSext  = new NURBSExtension(mesh->NURBSext, order);
      mfem::out<<"Create NURBS fec and ext"<<std::endl;
   }
   else
   {
      hdiv_coll = new RT_FECollection(order, dim);
      l2_coll   = new L2_FECollection(order, dim);
      mfem::out<<"Create Normal fec"<<std::endl;
   }

   FiniteElementSpace *space_p = new FiniteElementSpace(mesh, NURBSext,
                                                        l2_coll); //W
   FiniteElementSpace *space_u = new FiniteElementSpace(mesh,  //R
                                                        space_p->StealNURBSext(),
                                                        hdiv_coll);

   // Define the BlockStructure of the problem
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = space_u->GetVSize();
   block_offsets[2] = space_p->GetVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << " # Velocity dofs = " << block_offsets[1] - block_offsets[0] <<
             "\n";
   std::cout << " # Pressure dofs = " << block_offsets[2] - block_offsets[1] <<
             "\n";
   std::cout << " # Total    dofs = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // Define the gridfunctions
   BlockVector xp(block_offsets);

   GridFunction x_u(space_u);
   GridFunction x_p(space_p);

   x_u.MakeTRef(space_u, xp.GetBlock(0), 0);
   x_p.MakeTRef(space_p, xp.GetBlock(1), 0);

   VectorFunctionCoefficient sol(dim, sol_fun);

   x_u.ProjectCoefficient(sol);
   x_p = 0.0;

   x_u.SetTrueVector();
   x_p.SetTrueVector();

   // Define the output
   VisItDataCollection visit_dc("stokes", mesh);
   visit_dc.RegisterField("u", &x_u);
   visit_dc.RegisterField("p", &x_p);
   visit_dc.SetCycle(0);
   visit_dc.Save();

   // Set boundary conditions
   {
      Array<int> ess_tdof_list;
      if (mesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(mesh->bdr_attributes.Max());
         ess_bdr = 1;
         space_u->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      cout << "Number boundary dofs in H(div): "
           << ess_tdof_list.Size() << endl;
   }

   {
      Array<int> ess_tdof_list;
      if (mesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(mesh->bdr_attributes.Max());
         ess_bdr = 0;
         space_p->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      cout << "Number boundary dofs in H1: "
           << ess_tdof_list.Size() << endl;
   }

   // Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient kappa(1.0);
   VectorFunctionCoefficient force_cf(dim, force_fun);


   // Compute the rhs
   BlockVector rhs(block_offsets);

   LinearForm *fform(new LinearForm);
   fform->Update(space_u, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(force_cf));
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   LinearForm *gform(new LinearForm);
   gform->Update(space_p, rhs.GetBlock(1), 0);
   //gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   // Assemble the finite element matrices for the Stokes operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k \nabla u_h \cdot \nabla v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   BilinearForm *mVarf(new BilinearForm(space_u));
   MixedBilinearForm *bVarf(new MixedBilinearForm(space_u, space_p));


   mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(
                                 kappa)); // NEEDS SOMETHING ELSE
   mVarf->Assemble();
   mVarf->Finalize();

   bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf->Assemble();
   bVarf->Finalize();

   BlockOperator stokesOp(block_offsets);


   SparseMatrix &K(mVarf->SpMat());
   SparseMatrix &B(bVarf->SpMat());
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1; 
    Array<int> ess_tdof_list;
   space_u->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      int ii = ess_tdof_list[i];
      K.EliminateRowCol(ii, xp[ii], rhs.GetBlock(0));
    //  B.EliminateCol   (ii, x_p[ii], rhs.GetBlock(1));
   }
   ess_tdof_list.Print();
   
   B.EliminateCols   (ess_tdof_list, &xp.GetBlock(0), &rhs.GetBlock(1));
 //  B *= -1.;
   TransposeOperator *Bt = new TransposeOperator(&B);

   stokesOp.SetBlock(0,0, &K);
   stokesOp.SetBlock(0,1, Bt);
   stokesOp.SetBlock(1,0, &B);


  // Array<Array<int> *> ess_bdr(2);
 //  Array<Array<int> *> ess_bdr(2);

//   Array<int> ess_bc_dofs;


   //Array<int> ess_bdr_u(space_u->GetMesh()->bdr_attributes.Max());
 //  Array<int> ess_bdr_p(spaces[1]->GetMesh()->bdr_attributes.Max());

  // ess_bdr_p = 0;
 //  ess_bdr_u = 1;

  // ess_bdr[0] = &ess_bdr_u;
  // ess_bdr[1] = &ess_bdr_p;

//space_u->GetEssentialTrueDofs(ess_bdr_u, ess_bc_dofs);
//space_p->GetEssentialTrueDofs(ess_bdr_p, ess_tdof_list_p);


  // stokesOp.EliminateRowCol(ess_bc_dofs, xp, rhs)

   // 10. Construct the operators for preconditioner
   //
   //                 P = [ diag(K)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     pressure Schur Complement
   Vector Kd(mVarf->Height());

   BlockDiagonalPreconditioner stokesPrec(block_offsets);
   K.GetDiag(Kd);

   SparseMatrix *KinvBt = Transpose(B);

   for (int i = 0; i < Kd.Size(); i++)
   {
      KinvBt->ScaleRow(i, 1./Kd(i));
   }
   SparseMatrix *S = Mult(B, *KinvBt);

   Solver *invK = new DSmoother(K);
   Solver *invS =
#ifndef MFEM_USE_SUITESPARSE
      new GSSmoother(*S);
#else
      new UMFPackSolver(*S);
#endif
   invK->iterative_mode = false;
   invS->iterative_mode = false;

   stokesPrec.SetDiagonalBlock(0, invK);
   stokesPrec.SetDiagonalBlock(1, invS);

   // 11. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.
   int maxIter(10000);
   real_t rtol(1.e-10);
   real_t atol(1.e-10);

   chrono.Clear();
   chrono.Start();
   MINRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(stokesOp);
   solver.SetPreconditioner(stokesPrec);
   solver.SetPrintLevel(1);
   xp = 0.0;
   solver.Mult(rhs, xp);
   chrono.Stop();

   if (solver.GetConverged())
   {
      std::cout << "MINRES converged in " << solver.GetNumIterations()
                << " iterations with a residual norm of "
                << solver.GetFinalNorm() << ".\n";
   }
   else
   {
      std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                << " iterations. Residual norm is " << solver.GetFinalNorm()
                << ".\n";
   }
   std::cout << "MINRES solver took " << chrono.RealTime() << "s.\n";


   // 13. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
   //     sol_p.gf".
   {
      ofstream mesh_ofs("ex5.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream u_ofs("sol_u.gf");
      u_ofs.precision(8);
      x_u.Save(u_ofs);

      ofstream p_ofs("sol_p.gf");
      p_ofs.precision(8);
      x_p.Save(p_ofs);
   }

   // 14. Save data in the VisIt format
   visit_dc.RegisterField("u", &x_u);
   visit_dc.RegisterField("p", &x_p);
   visit_dc.SetCycle(1);
   visit_dc.Save();

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n" << *mesh << x_u << "window_title 'Velocity'" << endl;
      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "solution\n" << *mesh << x_p << "window_title 'Pressure'" << endl;
   }

   // 16. Free the used memory.
   delete fform;
   delete gform;
   delete invK;
   delete invS;
   delete S;
   delete Bt;
   delete KinvBt;
   delete mVarf;
   delete bVarf;
   delete space_u;
   delete space_p;
   delete l2_coll;
   delete hdiv_coll;
   delete mesh;

   return 0;
}




