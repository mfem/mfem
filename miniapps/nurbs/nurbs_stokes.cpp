//                                MFEM Example 5 -- modified for NURBS FE
//
// Compile with: make nurbs_ex5
//
// Sample runs:  nurbs_ex5 -m ../../data/square-nurbs.mesh -o 3
//               nurbs_ex5 -m ../../data/cube-nurbs.mesh -r 3
//               nurbs_ex5 -m ../../data/pipe-nurbs-2d.mesh
//               nurbs_ex5 -m ../../data/beam-tet.mesh
//               nurbs_ex5 -m ../../data/beam-hex.mesh
//               nurbs_ex5 -m ../../data/escher.mesh
//               nurbs_ex5 -m ../../data/fichera.mesh
//
// Device sample runs -- do not work for NURBS:
//               nurbs_ex5 -m ../../data/escher.mesh -pa -d cuda
//               nurbs_ex5 -m ../../data/escher.mesh -pa -d raja-cuda
//               nurbs_ex5 -m ../../data/escher.mesh -pa -d raja-omp
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//
//                                 k*u + grad p = f
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

// Sample runs:  nurbs_ex3 -m ../../data/square-nurbs.mesh
//               nurbs_ex3 -m ../../data/square-nurbs.mesh -o 2
//               nurbs_ex3 -m ../../data/cube-nurbs.mesh


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void f_fun(const Vector & x, Vector & f)
{
   f = 0.0;
}

real_t g_fun(const Vector & x)
{
   return 0;
}

void u_fun(const Vector & x, Vector & u)
{
   u = 0.0;

   if ((x[1] - 0.99 > 0.0) &&
       (fabs(x[0] - 0.5) < 0.49) )
   {
      u[0] = 1.0;
   }
}

real_t p_fun(const Vector & x)
{
   return 0.0;
}




//! @class SadlePointLUPreconditioner
/**
 * \brief A class to handle Block lower triangular preconditioners in a
 * matrix-free implementation.
 *
 * Usage:
 * - Use the constructors to define the block structure
 * - Use SetBlock() to fill the BlockLowerTriangularOperator
 * - Diagonal blocks of the preconditioner should approximate the inverses of
 *   the diagonal block of the matrix
 * - Off-diagonal blocks of the preconditioner should match/approximate those of
 *   the original matrix
 * - Use the method Mult() and MultTranspose() to apply the operator to a vector.
 *
 * If a diagonal block is not set, it is assumed to be an identity block, if an
 * off-diagonal block is not set, it is assumed to be a zero block.
 *
 */
class SadlePointLUPreconditioner : public Solver
{
public:
   //! Constructor for BlockLUPreconditioner%s with the same
   //! block-structure for rows and columns.
   /**
    *  @param offsets  Offsets that mark the start of each row/column block
    *                  (size nBlocks+1).
    *
    *  @note BlockLUPreconditioner will not own/copy the data
    *  contained in @a offsets.
    */
   SadlePointLUPreconditioner(const Array<int> & offsets_)
      : Solver(offsets_.Last()),
        nBlocks(offsets_.Size() - 1),
        offsets(0),
        ops(nBlocks, nBlocks)
   {
      MFEM_VERIFY(offsets_.Size() == 3, "Wrong number of offsets");

      ops = static_cast<Operator *>(NULL);
      offsets.MakeRef(offsets_);

      tmp0.SetSize(offsets[1] - offsets[0]);
      tmp1.SetSize(offsets[2] - offsets[1]);
   }

   //! Add a block opt in the block-entry (iblock, jblock).
   /**
    * @param iRow, iCol  The block will be inserted in location (iRow, iCol).
    * @param op          The Operator to be inserted.
    */
   void SetBlock(int iRow, int iCol, Operator *op)
   {
      MFEM_VERIFY(offsets[iRow+1] - offsets[iRow] == op->NumRows() &&
                  offsets[iCol+1] - offsets[iCol] == op->NumCols(),
                  "incompatible Operator dimensions");

      ops(iRow, iCol) = op;
   }

   //! This method is present since required by the abstract base class Solver
   virtual void SetOperator(const Operator &op) { }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const
   {
      MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
      MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

      sol.Update(y.GetData(),offsets);
      rhs.Update(x.GetData(),offsets);

      y = 0.0;
      ops(0,0)->Mult(rhs.GetBlock(0), sol.GetBlock(0)); // u = K^-1 f

      tmp1 = rhs.GetBlock(1);
      tmp1.Neg();
      ops(1,0)->AddMult(sol.GetBlock(0), tmp1);         // tmp = -g + Du
      ops(1,1)->Mult(tmp1, sol.GetBlock(1));            // p = S^{-1} (-g + Du)

      // tmp1 = rhs.GetBlock(1);
      // ops(1,0)->AddMult(sol.GetBlock(0), tmp1, -1.0);   // tmp = g - Du
      // ops(1,1)->Mult(tmp1, sol.GetBlock(1));            // p = S^{-1} (g - Du)



      ops(0,1)->Mult(sol.GetBlock(1), tmp0);
      ops(0,0)->AddMult(tmp0, sol.GetBlock(0), -1.0);    // u = u - K^-1 G p
   }

private:
   //! Number of block rows/columns
   int nBlocks;
   //! Offsets for the starting position of each block
   Array<int> offsets;
   //! 2D array that stores each block of the operator.
   Array2D<Operator *> ops;

   //! Temporary Vectors used to efficiently apply the Mult and MultTranspose
   //! methods.
   mutable BlockVector sol;
   mutable BlockVector rhs;
   mutable Vector tmp0;
   mutable Vector tmp1;
};


int main(int argc, char *argv[])
{
   StopWatch chrono;

   // Parse command-line options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   int ref_levels = -1;
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;
   real_t kappa = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&kappa, "-k", "--kappa",
                  "Diffusion parameter.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // Refine the mesh to increase the resolution.
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

   // Define a finite element space on the mesh.
   FiniteElementCollection *hdiv_coll = nullptr;
   FiniteElementCollection *l2_coll = nullptr;
   NURBSExtension *NURBSext = nullptr;

   if (mesh->NURBSext && !pa)
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

   FiniteElementSpace p_space(mesh, NURBSext, l2_coll);
   FiniteElementSpace u_space(mesh,
                              p_space.StealNURBSext(),
                              hdiv_coll);

   // Define the BlockStructure of the problem
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = u_space.GetVSize();
   block_offsets[2] = p_space.GetVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << "dim(R)   = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W)   = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // 7. Define the coefficients, analytical solution, gridfunctions.
   ConstantCoefficient k_c(kappa);

   VectorFunctionCoefficient f_cf(dim, f_fun);
   FunctionCoefficient g_cf(g_fun);

   VectorFunctionCoefficient u_cf(dim, u_fun);
   FunctionCoefficient p_cf(p_fun);

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt);

   GridFunction u, p;
   u.MakeRef(&u_space, x.GetBlock(0), 0);
   p.MakeRef(&p_space, x.GetBlock(1), 0);

   u.ProjectCoefficient(u_cf);
   p = 0.0;

   VisItDataCollection visit_dc0("Stokes_ic", mesh);
   visit_dc0.RegisterField("velocity", &u);
   visit_dc0.RegisterField("pressure", &p);
   visit_dc0.Save();

   // Assemble the right hand side via the linear forms (fform, gform).
   BlockVector rhs(block_offsets, mt);
   LinearForm *fform(new LinearForm);
   fform->Update(&u_space, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_cf));

   bool weakBC  = false;
   real_t penalty = 8.0;

   //   if (weakBC) fform->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(u_cf, -1.0, penalty)); Vector extension....
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   LinearForm *gform(new LinearForm);
   gform->Update(&p_space, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(g_cf));
   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   // Assemble the finite element matrices for the Stokes operator
   //
   //                            A = [ K   G ]
   //                                [ D   0 ]
   //     where:
   //
   //     K = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     D = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   //     G = D^t
   BilinearForm kVarf(&u_space);
   kVarf.AddDomainIntegrator(new VectorFEDiffusionIntegrator(k_c));
   // if (weakBC) kVarf.AddBdrFaceIntegrator(new DGDiffusionIntegrator(k_c, -1.0, penalty)); Vector extension....
   kVarf.Assemble();
   if (!weakBC) { kVarf.EliminateEssentialBC(ess_bdr, u, *fform); }
   kVarf.Finalize();
   SparseMatrix &K(kVarf.SpMat());

   MixedBilinearForm bVarf(&u_space, &p_space);
   bVarf.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   //  if (weakBC) bVarf.AddBoundaryIntegrator(new NormalTraceIntegrator(-1.0)); Not for mixed???

   bVarf.Assemble();
   if (!weakBC) { bVarf.EliminateTrialDofs(ess_bdr, u, *gform); }
   bVarf.Finalize();
   SparseMatrix &D(bVarf.SpMat());
   TransposeOperator G(&D);

   BlockOperator stokesOp(block_offsets);
   stokesOp.SetBlock(0,0, &K);
   stokesOp.SetBlock(0,1, &G);
   stokesOp.SetBlock(1,0, &D);

   // Construct the Schur Complement
#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver invK(K);
#else
   DSmoother invK(K);
#endif
   invK.iterative_mode = false;
   ProductOperator KiG(&invK, &G, false, false);
   ProductOperator S(&D, &KiG, false, false);

   // Construct an approximate Schur Complement
   real_t h = 1.0/sqrt(real_t(mesh->GetNE()));
   cout<<"h = "<<h<<endl;
   ConstantCoefficient tau_c(h*h/kappa);
   BilinearForm pVarf(&p_space);

   pVarf.AddDomainIntegrator(new DiffusionIntegrator(tau_c));
   pVarf.Assemble();
   pVarf.Finalize();

   SparseMatrix &Sp(pVarf.SpMat());

   // Construct an approximate Schur Complement inverse
#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver invSp(Sp);
#else
   DSmoother invSp(Sp);
#endif
   invSp.iterative_mode = false;
   int smaxIter(1000);
   real_t srtol(1.e-3);
   real_t satol(1.e-3);

   FGMRESSolver invS;
   invS.SetAbsTol(satol);
   invS.SetRelTol(srtol);
   invS.SetMaxIter(smaxIter);
   invS.SetKDim(smaxIter); // restart!!!
   invS.SetOperator(S);
   invS.SetPreconditioner(invSp);
   invS.SetPrintLevel(0);
   invS.iterative_mode = false;

   // Construct the operators for preconditioner
   //
   //       P = [ K   0        ] [ I   K^-1 G ]
   //           [ D  -D K^-1 G ] [ 0   I      ]
   //
   SadlePointLUPreconditioner stokesPrec(block_offsets);
   //BlockLowerTriangularPreconditioner stokesPrec(block_offsets);
   // BlockLUPreconditioner stokesPrec(block_offsets);

   stokesPrec.SetBlock(0,0, &invK);
   stokesPrec.SetBlock(0,1, &G);

   stokesPrec.SetBlock(1,1, &invS);
   stokesPrec.SetBlock(1,0, &D);

   // 11. Solve the linear system with a Krylov.
   //     Check the norm of the unpreconditioned residual.
   int maxIter(10000);
   real_t rtol(1.e-10);
   real_t atol(1.e-10);

   chrono.Clear();
   chrono.Start();
   //   MINRESSolver solver;
   FGMRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetKDim(maxIter); // restart!!!
   solver.SetOperator(stokesOp);
   solver.SetPreconditioner(stokesPrec);
   solver.SetPrintLevel(1);
   x = 0.0;
   solver.Mult(rhs, x);
   if (device.IsEnabled()) { x.HostRead(); }
   chrono.Stop();
#ifdef MFEM_USE_SUITESPARSE
   cout<<"MFEM_USE_SUITESPARSE\n";
#endif
   if (solver.GetConverged())
   {
      std::cout << "Solver converged in " << solver.GetNumIterations()
                << " iterations with a residual norm of "
                << solver.GetFinalNorm() << ".\n";
   }
   else
   {
      std::cout << "Solver did not converge in " << solver.GetNumIterations()
                << " iterations. Residual norm is " << solver.GetFinalNorm()
                << ".\n";
   }
   std::cout << " Solver took " << chrono.RealTime() << "s.\n";

   // 13. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
   //     sol_p.gf".
   {
      ofstream mesh_ofs("ex5.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream u_ofs("sol_u.gf");
      u_ofs.precision(8);
      u.Save(u_ofs);

      ofstream p_ofs("sol_p.gf");
      p_ofs.precision(8);
      p.Save(p_ofs);
   }

   // 14. Save data in the VisIt format
   VisItDataCollection visit_dc("Stokes", mesh);
   visit_dc.RegisterField("velocity", &u);
   visit_dc.RegisterField("pressure", &p);
   visit_dc.Save();

   // 15. Save data in the ParaView format
   if (false)
   {
      ParaViewDataCollection paraview_dc("Stokes", mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("velocity",&u);
      paraview_dc.RegisterField("pressure",&p);
      paraview_dc.Save();
   }
   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n" << *mesh << u << "window_title 'Velocity'" << endl;
      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "solution\n" << *mesh << p << "window_title 'Pressure'" << endl;
   }

   // 17. Free the used memory.
   // delete fform;
   // delete gform;
   //  delete invSp;
   // delete kVarf;
   //delete bVarf;
   delete l2_coll;
   delete hdiv_coll;
   delete mesh;

   return 0;
}



