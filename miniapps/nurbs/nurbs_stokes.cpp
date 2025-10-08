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

// Placeholder till can be removed it PR #4326 is accepted
void ProjectCoefficientGlobalL2(VectorCoefficient &vcoeff,
                                real_t rtol, int iter, GridFunction &gf)
{
   // Define and assemble linear form
   LinearForm b(gf.FESpace());
   BilinearForm a(gf.FESpace());

   if (gf.FESpace()->GetTypicalFE()->GetRangeType() == mfem::FiniteElement::VECTOR)
   {
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vcoeff));
      a.AddDomainIntegrator(new VectorFEMassIntegrator());
   }
   else
   {
      b.AddDomainIntegrator(new VectorDomainLFIntegrator(vcoeff));
      a.AddDomainIntegrator(new VectorMassIntegrator());
   }
   a.Assemble();
   b.Assemble();

   // Set solver and preconditioner
   SparseMatrix A(a.SpMat());
   GSSmoother  prec(A);
   CGSolver cg;
   cg.SetOperator(A);
   cg.SetPreconditioner(prec);
   cg.SetRelTol(rtol);
   cg.SetMaxIter(iter);
   cg.SetPrintLevel(0);

   // Solve and get solution
   gf = 0.0;
   cg.Mult(b,gf);
}


// Define the analytical solution and forcing terms / boundary conditions
// Functions that define the problem for a lid driven cavity
// No forcing, homogenous BC troughout, except the top.
namespace lidDrivenCavity
{

void u_fun(const Vector & x, Vector & u)
{
   // Placeholder smooth function, sharp solution can be used when PR #4326 is accepted
   u[0] = sin(M_PI*x[0])*x[1];
   u[1] = 0;

   /* u = 0.0;

    if (fabs(x[0] - 0.5) < 0.4999999)
    {
       u[0] = x[1];
    }*/
}

void f_fun(const Vector & x, Vector & f)
{
   f = 0.0;
}

real_t p_fun(const Vector & x)
{
   return 0.0;
}

real_t g_fun(const Vector & x)
{
   return 0;
}

}

// Functions that define the problem for a rotating domain
// The domain rotates around x = [0.5, -0.1]
namespace rotatingDomain
{
real_t omega = 1.0;
real_t x0 = 0.5;
real_t y0 = -0.1;

//
void u_fun(const Vector & x, Vector & u)
{
   u = 0.0;
   u[0] =  omega*(x[1] - y0);
   u[1] = -omega*(x[0] - x0);
}

// f = -omega^2*r
void f_fun(const Vector & x, Vector & f)
{
   f = 0.0;
   f[0] = omega*omega*(x[0] - x0);
   f[1] = omega*omega*(x[1] - y0);
}

real_t p_fun(const Vector & x)
{
   return 0.0;
}

real_t g_fun(const Vector & x)
{
   return 0.0;
}

}


// Functions that define the problem for a closed rotating box
// The box rotates around x = [0.5, -0.1]
namespace rotatingBox
{
real_t omega = 1.0;
real_t x0 = 0.5;
real_t y0 = -0.1;

// f = -omega^2*r

void u_fun(const Vector & x, Vector & u)
{
   u = 0.0;
}

void f_fun(const Vector & x, Vector & f)
{
   f[0] = omega*omega*(x[0] - x0);
   f[1] = omega*omega*(x[1] - y0);
}

real_t p_fun(const Vector & x)
{
   return 0.0;
}

real_t g_fun(const Vector & x)
{
   return 0.0;
}

}

/// Function to remove the undetermined pressure mode
void MeanZero(GridFunction &p_gf)
{
   ConstantCoefficient one_cf(1.0);

   LinearForm mass_lf(p_gf.FESpace());
   mass_lf.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
   mass_lf.Assemble();

   GridFunction one_gf(p_gf.FESpace());
   one_gf.ProjectCoefficient(one_cf);

   real_t volume = mass_lf.operator()(one_gf);
   real_t integ = mass_lf.operator()(p_gf);

   p_gf -= integ / volume;
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

/** Construct an exact LU decomposition
*
*       P = [ K   0        ] [ I   K^-1 G ]
*           [ D  -D K^-1 G ] [ 0   I      ]
*
* of the ubiquitous sadle point problem
*
*       A = [ K   G ]
*           [ D   0 ]
*
*
* Usage:
* - Use the constructors to define the block structure
* - Use SetBlock() to fill the LU Decomposition
*       A = [ K^-1   G ]
*           [ D     -S^-1 ]
*   where S is the schur complement S = D K^-1 G
* - Use the method Mult()
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
   bool weakBC = true;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;
   real_t mu = 1.0;
   real_t penalty = -1.0;
   int problem = 0;
   int maxiter = 200;
   real_t rtol = 1.e-6;
   bool schur_complement = true;
   int schur_maxiter = 10000;
   real_t schur_rtol = 1.e-8;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&weakBC, "-wbc", "--weak-bc", "-sbc", "--strong-bc",
                  "Weak boundary conditions.");
   args.AddOption(&penalty, "-p", "--lambda",
                  "Penalty parameter for enforcing weak Dirichlet boundary conditions.");
   args.AddOption(&mu, "-mu", "--diffusion",
                  "Diffusion parameter.");
   args.AddOption(&problem, "-c", "--problem",
                  "Problem to solve:\n."
                  "\t 0 = Lid Driven Cavity (detault)"
                  "\t 1 = Rotating domain"
                  "\t 2 = Rotating box");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   // Solver parameters
   args.AddOption(&rtol, "-lt", "--linear-tolerance",
                  "Relative tolerance for the GMRES solver.");
   args.AddOption(&maxiter, "-li", "--linear-itermax",
                  "Maximum iteration count for the GMRES solver.");

   // Schur_complement  parameters
   args.AddOption(&schur_complement, "-sc", "--schur", "-nsc", "--no-schur",
                  "Use a schur-complement or not.");
   args.AddOption(&schur_rtol, "-slt", "--schur_linear-tolerance",
                  "Relative tolerance for the Schur complement GMRES solver.");
   args.AddOption(&schur_maxiter, "-sli", "--schur_linear-itermax",
                  "Maximum iteration count for the Schur complement GMRES solver.");

#ifdef MFEM_USE_SUITESPARSE
   bool UMFPack = true;
   args.AddOption(&UMFPack, "-umf", "--direct", "-gs", "--gssmoother",
                  "Type of preconditioner to use.");
#endif

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (schur_complement)
   {
#ifdef MFEM_USE_SUITESPARSE
      if (UMFPack)
      {
         std::cout<<"Schur complement is exact"<<std::endl;
      }
      else
      {
         std::cout<<"Schur complement is approximate!!"<<std::endl;
      }
#else
      std::cout<<"Schur complement is approximate"<<std::endl;
      std::cout<<" - Compile with SuiteSparse to have an exact Schur complement"<<std::endl;
#endif
   }

   if (penalty < 0)
   {
      penalty = 10*(order+1)*(order+1);
   }

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
      hdiv_coll = new NURBS_HDivH1FECollection(order,dim);
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

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   u_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Define the BlockStructure of the problem
   Array<int> bOffsets(3); // number of variables + 1
   bOffsets[0] = 0;
   bOffsets[1] = u_space.GetVSize();
   bOffsets[2] = p_space.GetVSize();
   bOffsets.PartialSum();

   std::cout<<"===========================================================\n";
   std::cout<<"Velocity dofs     = "<<bOffsets[1] - bOffsets[0]<<endl;
   std::cout<<"Pressure dofs     = "<<bOffsets[2] - bOffsets[1]<<endl;
   std::cout<<"Total # of dofs   = "<<bOffsets.Last()<<endl;
   if (!weakBC)
   {
      std::cout << "-----------------------------------------------------------\n";
      Array<int> ess_tdof_list;
      u_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      int nbc_dofs = ess_tdof_list.Size();
      std::cout<<"Velocity BC dofs  = "<<ess_tdof_list.Size()<<endl;
      std::cout<<"Net Velocity dofs = "<<bOffsets[1]- bOffsets[0]- nbc_dofs<<endl;
      std::cout<<"Net # of dofs     = "<<bOffsets.Last() - nbc_dofs<<endl;
   }
   std::cout<<"===========================================================\n";

   // Define the viscosity, solution and forcing coefficients
   ConstantCoefficient mu_cf(mu);

   VectorFunctionCoefficient *u_cf, *f_cf;
   FunctionCoefficient *p_cf, *g_cf;

   if (problem == 0)
   {
      cout<<"Lid driven cavity\n";
      f_cf = new VectorFunctionCoefficient (dim, lidDrivenCavity::f_fun);
      u_cf = new VectorFunctionCoefficient (dim, lidDrivenCavity::u_fun);

      g_cf = new FunctionCoefficient (lidDrivenCavity::g_fun);
      p_cf = new FunctionCoefficient (lidDrivenCavity::p_fun);
   }
   else if (problem == 1)
   {
      cout<<"Rotating domain\n";
      f_cf = new VectorFunctionCoefficient (dim, rotatingDomain::f_fun);
      u_cf = new VectorFunctionCoefficient (dim, rotatingDomain::u_fun);

      g_cf = new FunctionCoefficient (rotatingDomain::g_fun);
      p_cf = new FunctionCoefficient (rotatingDomain::p_fun);
   }
   else if (problem == 2)
   {
      cout<<"Rotating box\n";
      f_cf = new VectorFunctionCoefficient (dim, rotatingBox::f_fun);
      u_cf = new VectorFunctionCoefficient (dim, rotatingBox::u_fun);

      g_cf = new FunctionCoefficient (rotatingBox::g_fun);
      p_cf = new FunctionCoefficient (rotatingBox::p_fun);
   }
   else
   {
      mfem_error("Incorrect problem selected");
   }

   // Define the gridfunctions and set the initial/boundary conditions
   MemoryType mt = device.GetMemoryType();
   BlockVector x(bOffsets, mt);

   GridFunction u_gf, p_gf;
   u_gf.MakeRef(&u_space, x.GetBlock(0), 0);
   p_gf.MakeRef(&p_space, x.GetBlock(1), 0);

   u_gf.ProjectCoefficient(
      *u_cf);//,GridFunction::ProjType::ELEMENT); // use when PR #4326 is accepted
   //ProjectCoefficientGlobalL2(*u_cf, 1e-10, 250,
   //                          u_gf); //remove when PR #4326 is accepted
   p_gf = 0.0;

   VectorGridFunctionCoefficient uh_cf(&u_gf);

   VisItDataCollection visit_dc0("Stokes_ic", mesh);
   visit_dc0.RegisterField("velocity", &u_gf);
   visit_dc0.RegisterField("pressure", &p_gf);
   visit_dc0.Save();

   // Assemble the right hand side via the linear forms (fform, gform).
   BlockVector rhs(bOffsets, mt);
   LinearForm *fform(new LinearForm);
   fform->Update(&u_space, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*f_cf));
   if (weakBC) { fform->AddBdrFaceIntegrator(new VectorFEDGDirichletLFIntegrator(uh_cf, mu_cf, -1.0, penalty)); }
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   LinearForm *gform(new LinearForm);
   gform->Update(&p_space, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(*g_cf));
   if (weakBC) { gform->AddBdrFaceIntegrator(new BoundaryNormalLFIntegrator(uh_cf)); }
   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   // Assemble the finite element matrices for the Stokes operator
   //
   //                            A = [ K   G ]
   //                                [ D   0 ]
   //     where:
   //
   //     K = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     D = \int_\Omega u_h grad q_h d\Omega   u_h \in R_h, q_h \in W_h
   //     G = D^t
   if (weakBC)
   {
      cout<<"Weak Dirichlet BCs("<<penalty<<")\n";
   }
   else
   {
      cout<<"Strong Dirichlet BC\n";
   }

   chrono.Clear();
   chrono.Start();
   BilinearForm kVarf(&u_space);
   kVarf.AddDomainIntegrator(new VectorFEDiffusionIntegrator(mu_cf));
   if (weakBC) { kVarf.AddBdrFaceIntegrator(new DGDiffusionIntegrator(mu_cf, -1.0, penalty)); }
   kVarf.Assemble();
   if (!weakBC) { kVarf.EliminateEssentialBC(ess_bdr, u_gf, *fform); }
   kVarf.Finalize();
   SparseMatrix &K(kVarf.SpMat());
   chrono.Stop();
   std::cout<<" Assembly of diffusion matrix took "<<chrono.RealTime()<<"s\n";

   chrono.Clear();
   chrono.Start();
   Operator *D, *G;
   ConstantCoefficient minus(-1.0);
   if (weakBC)
   {
      MixedBilinearForm gVarf(&p_space, &u_space);
      gVarf.AddDomainIntegrator(new TransposeIntegrator(new
                                                        VectorFEDivergenceIntegrator(-1.0)));
      gVarf.AddBdrTraceFaceIntegrator(new NormalTraceIntegrator(1.0));
      gVarf.Assemble();
      gVarf.Finalize();
      G = new SparseMatrix(gVarf.SpMat());
      D = new TransposeOperator(G);
   }
   else
   {
      MixedBilinearForm dVarf(&u_space, &p_space);
      dVarf.AddDomainIntegrator(new VectorFEDivergenceIntegrator(-1.0));
      dVarf.Assemble();
      dVarf.EliminateTrialEssentialBC(ess_bdr, u_gf, *gform);
      dVarf.Finalize();
      D = new SparseMatrix(dVarf.SpMat());
      G = new TransposeOperator(D);
   }
   chrono.Stop();
   std::cout<<" Assembly of grad & div matrices took "<<chrono.RealTime()<<"s\n";

   BlockOperator stokesOp(bOffsets);
   stokesOp.SetBlock(0,0, &K);
   stokesOp.SetBlock(0,1, G);
   stokesOp.SetBlock(1,0, D);

   // Construct the momentum-velocity block preconditioner
   Solver *invK;
#ifdef MFEM_USE_SUITESPARSE
   if (UMFPack)
   {
      invK = new UMFPackSolver(K);
   }
   else
   {
      invK = new GSSmoother(K);
   }
#else
   invK = new GSSmoother(K);
#endif
   invK->iterative_mode = false;

   // Construct an approximate Schur complement operator
   // This is the continuity-pressure PSPG matrix
   real_t h = std::pow(real_t(mesh->GetNE()),-1.0/real_t(mesh->Dimension()));
   cout<<"h = "<<h<<endl;
   double Cinv = 1.0;
   ConstantCoefficient tau_c(Cinv*h*h/mu);
   BilinearForm pVarf(&p_space);

   pVarf.AddDomainIntegrator(new DiffusionIntegrator(tau_c));
   pVarf.Assemble();
   pVarf.Finalize();

   SparseMatrix &Sp(pVarf.SpMat());

   // Construct an Schur complement preconditioner
   // This uses the approximate Schur complement operator
   Solver *invSp;
#ifdef MFEM_USE_SUITESPARSE
   if (UMFPack)
   {
      invSp = new UMFPackSolver(Sp);
   }
   else
   {
      invSp = new GSSmoother(Sp);
   }
#else
   invSp = new GSSmoother(Sp);
#endif

   invSp->iterative_mode = false;

   // Construct the Schur complement operator
   ProductOperator KiG(invK, G, false, false);
   ProductOperator S(D, &KiG, false, false);

   // Construct the exact Schur complement inverse operator
   GMRESSolver invS;
   invS.SetRelTol(schur_rtol);
   invS.SetMaxIter(schur_maxiter);
   invS.SetKDim(schur_maxiter+1); // restart!!!
   // if (weakBC)
   // {
   invS.SetOperator(S);
   // }
   // else
   // {
   //    invS.SetOperator(Sp);
   // }
   invS.SetPreconditioner(*invSp);
   //invS.SetPrintLevel(IterativeSolver::PrintLevel().FirstAndLast());
   invS.SetPrintLevel(0);
   invS.iterative_mode = false;

   // Construct the operators for preconditioner
   //
   //       P = [ K   0        ] [ I   K^-1 G ]
   //           [ D  -D K^-1 G ] [ 0   I      ]
   //
   SadlePointLUPreconditioner stokesPrec(bOffsets);

   stokesPrec.SetBlock(0,0, invK);
   stokesPrec.SetBlock(0,1, G);

   if (schur_complement)
   {
      stokesPrec.SetBlock(1,1, &invS);
   }
   else
   {
      stokesPrec.SetBlock(1,1, invSp);
   }
   stokesPrec.SetBlock(1,0, D);

   // Solve the linear system with a Krylov solver.
   // Check the norm of the unpreconditioned residual.
   chrono.Clear();
   chrono.Start();
   FGMRESSolver solver;
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxiter);
   solver.SetKDim(maxiter+1); // restart!!!
   solver.SetOperator(stokesOp);
   solver.SetPreconditioner(stokesPrec);
   solver.SetPrintLevel(1);
   x = 0.0;
   solver.Mult(rhs, x);
   if (device.IsEnabled()) { x.HostRead(); }
   chrono.Stop();

   MeanZero(p_gf);

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

   // Save the mesh and the solution. This output can be viewed later using
   // GLVis: "glvis -m stokes.mesh -g sol_u.gf" or "glvis -m stokes.mesh -g
   // sol_p.gf".
   {
      ofstream mesh_ofs("stokes.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream u_ofs("sol_u.gf");
      u_ofs.precision(8);
      u_gf.Save(u_ofs);

      ofstream p_ofs("sol_p.gf");
      p_ofs.precision(8);
      p_gf.Save(p_ofs);
   }

   // Save data in the VisIt format
   if (false)
   {
      // Define a traditional NURBS space as long as ViSit does not read to new Vector FEs
      FiniteElementSpace ui_space(mesh, p_space.StealNURBSext(), l2_coll, dim);
      GridFunction ui_gf(&ui_space);
      ui_gf.ProjectCoefficient(uh_cf);

      VisItDataCollection visit_dc("Stokes", mesh);
      visit_dc.RegisterField("velocity", &ui_gf);
      visit_dc.RegisterField("pressure", &p_gf);
      visit_dc.Save();
   }

   // Save data in the ParaView format
   if (false)
   {
      // Define a traditional NURBS space as long as Paraview does not read to new Vector FEs
      FiniteElementSpace ui_space(mesh, p_space.StealNURBSext(), l2_coll, dim);
      GridFunction ui_gf(&ui_space);
      ui_gf.ProjectCoefficient(uh_cf);

      ParaViewDataCollection paraview_dc("Stokes", mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("velocity",&u_gf);
      paraview_dc.RegisterField("pressure",&p_gf);
      paraview_dc.Save();
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n" << *mesh << u_gf << "window_title 'Velocity'" << endl;
      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "solution\n" << *mesh << p_gf << "window_title 'Pressure'" << endl;
   }

   // 17. Free the used memory.
   delete l2_coll;
   delete hdiv_coll;
   delete mesh;
   delete u_cf;
   delete f_cf;
   delete p_cf;
   delete g_cf;
   delete invK;
   delete invSp;
   delete G;
   delete D;

   return 0;
}

