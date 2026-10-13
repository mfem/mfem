//                                MFEM Example 5
//
// Compile with: make ex5
//
// Sample runs:  ex5 -m ../data/square-disc.mesh
//               ex5 -m ../data/square-disc.mesh -hb
//               ex5 -m ../data/star.mesh
//               ex5 -m ../data/star.mesh -pa
//               ex5 -m ../data/star.mesh -hb
//               ex5 -m ../data/beam-tet.mesh
//               ex5 -m ../data/beam-hex.mesh
//               ex5 -m ../data/beam-hex.mesh -pa
//               ex5 -m ../data/beam-hex.mesh -hb
//               ex5 -m ../data/escher.mesh
//               ex5 -m ../data/fichera.mesh
//               ex5 -m ../data/fichera.mesh -hb
//
// Device sample runs:
//               ex5 -m ../data/star.mesh -pa -d cuda
//               ex5 -m ../data/star.mesh -pa -d raja-cuda
//               ex5 -m ../data/star.mesh -pa -d raja-omp
//               ex5 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//
//                                 sigma*E - curl B = f
//                                 curl E  + mu*B   = 0
//
//               with natural boundary condition -B = <given pressure>.
//               Here, we use a given exact solution (E,B) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity E) and piecewise discontinuous
//               polynomials (pressure B).
//
//               The example demonstrates the use of the DarcyForm class, as
//               well as hybridization of mixed systems and the collective
//               saving of several grid functions in VisIt (visit.llnl.gov) and
//               ParaView (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void EFun_ex(const Vector & x, Vector & E);
void BFun_ex(const Vector & x, Vector & B);
void fFun(const Vector & x, Vector & f);
real_t freq = 1.0, kappa;

/** Integrator for the DG form:
    $$
      \alpha \langle \rho_u \{v\},[w \times n] \rangle
      + \beta \langle \rho_u (u \cdot n) / |u \cdot n| [v],[w \times n] \rangle,
    $$
    where $v$ is the scalar trial variable and $w$ is the vector test variable.
    $\rho$/$u$ are given scalar/vector coefficients. $\{v\}$ represents the
    average value of $v$ on the face and $[v]$ is the jump such that
    $\{v\}=(v_1+v_2)/2$ and $[v]=(v_1-v_2)$ for the face between elements $1$
    and $2$. For boundary elements, $v_2=0$. The vector coefficient, $u$, is
    assumed to be continuous across the faces and when given the scalar
    coefficient, $\rho$, is assumed to be discontinuous. The integrator uses
    the upwind value of $\rho$, denoted by $\rho_u$, which is value from the
    side into which the vector coefficient, $u$, points.
    */
class DGTangentialTraceIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *rho = nullptr;
   VectorCoefficient *u = nullptr;
   real_t alpha, beta;

private:
   DenseMatrix te_vshape1, te_vshape2;
   Vector tr_shape1, te_shape1, tr_shape2, te_shape2;

public:
   /// Construct integrator with $\rho = 1$, $\beta = 0$.
   DGTangentialTraceIntegrator(real_t a) : alpha(a), beta(0.) { }

   /// Construct integrator with $\rho = 1$, $\beta = \alpha/2$.
   DGTangentialTraceIntegrator(VectorCoefficient &u_, real_t a)
      : u(&u_), alpha(a), beta(0.5*a) { }

   /// Construct integrator with $\rho = 1$.
   DGTangentialTraceIntegrator(VectorCoefficient &u_, real_t a, real_t b)
      : u(&u_), alpha(a), beta(b) { }

   DGTangentialTraceIntegrator(Coefficient &rho_, VectorCoefficient &u_,
                               real_t a, real_t b)
      : rho(&rho_), u(&u_), alpha(a), beta(b) { }

   void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                           const FiniteElement &test_fe1,
                           const FiniteElement &trial_fe2,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

void DGTangentialTraceIntegrator::AssembleFaceMatrix(const FiniteElement
                                                     &trial_fe1,
                                                     const FiniteElement &test_fe1,
                                                     const FiniteElement &trial_fe2,
                                                     const FiniteElement &test_fe2,
                                                     FaceElementTransformations &Trans,
                                                     DenseMatrix &elmat)
{
   int tr_ndof1, te_ndof1, tr_ndof2, te_ndof2;

   real_t un, a, b, w;

   const int dim = test_fe1.GetDim();
   const int dim_curl = dim == 3 ? 3 : 1;
   tr_ndof1 = trial_fe1.GetDof();
   te_ndof1 = test_fe1.GetDof();
   bool l2 = test_fe1.GetRangeType() == FiniteElement::SCALAR;
   Vector vu(dim), nor(dim);

   if (Trans.Elem2No >= 0)
   {
      tr_ndof2 = trial_fe2.GetDof();
      te_ndof2 = test_fe2.GetDof();
   }
   else
   {
      tr_ndof2 = 0;
      te_ndof2 = 0;
   }

   tr_shape1.SetSize(tr_ndof1);
   te_shape1.SetSize(te_ndof1);
   tr_shape2.SetSize(tr_ndof2);
   te_shape2.SetSize(te_ndof2);
   if (l2)
   {
      elmat.SetSize((te_ndof1 + te_ndof2) * dim, (tr_ndof1 + tr_ndof2) * dim_curl);
   }
   else
   {
      te_vshape1.SetSize(te_ndof1, dim);
      te_vshape2.SetSize(te_ndof2, dim);
      elmat.SetSize(te_ndof1 + te_ndof2, (tr_ndof1 + tr_ndof2) * dim_curl);
   }
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  max(trial_fe1.GetOrder(), trial_fe2.GetOrder()) +
                  max(test_fe1.GetOrder(), test_fe2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + trial_fe1.GetOrder() + test_fe1.GetOrder();
      }
      if (trial_fe1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);

      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      trial_fe1.CalcPhysShape(*Trans.Elem1, tr_shape1);

      Trans.Face->SetIntPoint(&ip);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      if (l2)
      {
         test_fe1.CalcPhysShape(*Trans.Elem1, te_shape1);
      }
      else
      {
         test_fe1.CalcPhysVShape(*Trans.Elem1, te_vshape1);
         te_vshape1.Mult(nor, te_shape1);
      }

      a = 0.5 * alpha;
      if (beta != 0.)
      {
         u->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
         b = (un != 0.)?(beta * un / fabs(un)):(0.);
      }
      else
      {
         un = 0.;
         b = 0.;
      }

      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      if (rho)
      {
         real_t rho_p;
         // upwinding
         if (un > 0.0 && tr_ndof2 && te_ndof2)
         {
            Trans.Elem2->SetIntPoint(&eip2);
            rho_p = rho->Eval(*Trans.Elem2, eip2);
         }
         else
         {
            rho_p = rho->Eval(*Trans.Elem1, eip1);
         }
         // averaging
         if (beta == 0. && tr_ndof2 && te_ndof2)
         {
            Trans.Elem2->SetIntPoint(&eip2);
            rho_p += rho->Eval(*Trans.Elem2, eip2);
            rho_p *= 0.5;
         }
         a *= rho_p;
         b *= rho_p;
      }

      auto kernel = [&elmat, &dim, &nor, &l2](const real_t w, const Vector &te_shape,
                                              const Vector &tr_shape, int ioff, int joff)
      {
         const int tr_ndof = tr_shape.Size();
         const int te_ndof = te_shape.Size();

         MFEM_VERIFY(l2, "Not implemtented");

         if (dim < 3)
         {
            for (int i = 0; i < te_ndof; i++)
               for (int j = 0; j < tr_ndof; j++)
               {
                  elmat(ioff + i + 0*te_ndof, joff + j) += w * te_shape(i) * tr_shape(j) * nor(1);
                  elmat(ioff + i + 1*te_ndof, joff + j) -= w * te_shape(i) * tr_shape(j) * nor(0);
               }
         }
         else
         {
            for (int i = 0; i < te_ndof; i++)
               for (int j = 0; j < tr_ndof; j++)
               {
                  elmat(ioff + i + 1*te_ndof,
                        joff + j + 0*tr_ndof) += w * te_shape(i) * tr_shape(j) * nor(2);
                  elmat(ioff + i + 2*te_ndof,
                        joff + j + 0*tr_ndof) -= w * te_shape(i) * tr_shape(j) * nor(1);
                  elmat(ioff + i + 2*te_ndof,
                        joff + j + 1*tr_ndof) += w * te_shape(i) * tr_shape(j) * nor(0);
                  elmat(ioff + i + 0*te_ndof,
                        joff + j + 1*tr_ndof) -= w * te_shape(i) * tr_shape(j) * nor(2);
                  elmat(ioff + i + 0*te_ndof,
                        joff + j + 2*tr_ndof) += w * te_shape(i) * tr_shape(j) * nor(1);
                  elmat(ioff + i + 1*te_ndof,
                        joff + j + 2*tr_ndof) -= w * te_shape(i) * tr_shape(j) * nor(0);
               }
         }
      };

      w = ip.weight * (a+b);
      if (w != 0.0)
      {
         kernel(w, te_shape1, tr_shape1, 0, 0);
      }

      if (tr_ndof2 && te_ndof2)
      {
         trial_fe2.CalcPhysShape(*Trans.Elem2, tr_shape2);
         if (l2)
         {
            test_fe2.CalcPhysShape(*Trans.Elem2, te_shape2);
         }
         else
         {
            test_fe2.CalcPhysVShape(*Trans.Elem2, te_vshape2);
            te_vshape2.Mult(nor, te_shape2);
         }

         if (w != 0.0)
         {
            kernel(-w, te_shape2, tr_shape1, te_ndof1*dim, 0);
         }

         w = ip.weight * (b-a);
         if (w != 0.0)
         {
            kernel(+w, te_shape2, tr_shape2, te_ndof1*dim, tr_ndof1*dim_curl);
            kernel(-w, te_shape1, tr_shape2, 0, tr_ndof1*dim_curl);
         }
      }
   }
}

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-tet.mesh";
   int order = 1;
   bool hybridization = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
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
   kappa = freq * M_PI;


   if (pa && hybridization)
   {
      cerr << "Partial assembly is not supported for hybridization."
           << endl;
      return 1;
   }

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   const int dim = mesh->Dimension();
   const int dim_curl = dim == 3 ? 3 : 1;

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *E_coll(new L2_FECollection(order, dim,
                                                       BasisType::GaussLobatto));
   FiniteElementCollection *B_coll(new L2_FECollection(order, dim));

   FiniteElementSpace *E_space = new FiniteElementSpace(mesh, E_coll, dim);
   FiniteElementSpace *B_space = new FiniteElementSpace(mesh, B_coll, dim_curl);

   // 6. Define the block structure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   DarcyForm *darcy = new DarcyForm(E_space, B_space);
   const Array<int> &block_offsets = darcy->GetOffsets();
   const Array<int> &block_trueOffsets = darcy->GetTrueOffsets();

   std::cout << "***********************************************************\n";
   std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient sigmacoeff(1.0); // Diffusivity
   ConstantCoefficient mucoeff(1.0); // Permeability

   VectorFunctionCoefficient fcoeff(dim, fFun); // Electric rhs

   VectorFunctionCoefficient Ecoeff(dim, EFun_ex); // Electric field
   VectorFunctionCoefficient Bcoeff(dim_curl, BFun_ex); // Magnetic field
   ScalarVectorProductCoefficient fnatcoeff(-1., Ecoeff);

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction E,B for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (E,B) and the linear forms (fform, gform).
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt);

   LinearForm *fform = darcy->GetFluxRHS();
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
   if (dim_curl > 1)
   {
      MFEM_ABORT("Not implemented");
   }
   else
   {
      fform->AddBdrFaceIntegrator(new VectorBoundaryTangentialLFIntegrator(
                                     fnatcoeff));
   }

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   BilinearForm *mVarf = darcy->GetFluxMassForm();
   MixedBilinearForm *cVarf = darcy->GetFluxDivForm();
   BilinearForm *mbVarf = darcy->GetPotentialMassForm();

   mVarf->AddDomainIntegrator(new VectorMassIntegrator(sigmacoeff));

   cVarf->AddDomainIntegrator(new MixedCurlIntegrator(
                                 MixedCurlIntegrator::Mode::OutOfPlane));
   cVarf->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                       new DGTangentialTraceIntegrator(-1.)));
   //mbVarf->AddInteriorFaceIntegrator()

   if (dim_curl > 1)
   {
      mbVarf->AddDomainIntegrator(new VectorMassIntegrator(mucoeff));
   }
   else
   {
      mbVarf->AddDomainIntegrator(new MassIntegrator(mucoeff));
   }

   // Set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;

   chrono.Clear();
   chrono.Start();

   if (hybridization)
   {
      trace_coll = new DG_Interface_FECollection(order, dim);
      trace_space = new FiniteElementSpace(mesh, trace_coll);
      darcy->EnableHybridization(trace_space,
                                 new TangentTraceJumpIntegrator(),
                                 ess_flux_tdofs_list);
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   darcy->Assemble();

   OperatorPtr A;
   Vector X, RHS;
   x = 0.;
   darcy->FormLinearSystem(ess_flux_tdofs_list, x, A, X, RHS);

   chrono.Stop();
   std::cout << "Assembly took " << chrono.RealTime() << "s.\n";


   constexpr int maxIter(1000);
   constexpr real_t rtol(1.e-6);
   constexpr real_t atol(1.e-10);

   if (hybridization)
   {
      // 10. Construct the preconditioner
      GSSmoother prec;

      // 11. Solve the linear system with GMRES.
      //     Check the norm of the unpreconditioned residual.
      chrono.Clear();
      chrono.Start();
      GMRESSolver solver;
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetPreconditioner(prec);
      solver.SetOperator(*A);
      solver.SetPrintLevel(1);

      solver.Mult(RHS, X);
      darcy->RecoverFEMSolution(X, x);

      chrono.Stop();

      if (solver.GetConverged())
      {
         std::cout << "GMRES converged in " << solver.GetNumIterations()
                   << " iterations with a residual norm of "
                   << solver.GetFinalNorm() << ".\n";
      }
      else
      {
         std::cout << "GMRES did not converge in " << solver.GetNumIterations()
                   << " iterations. Residual norm is " << solver.GetFinalNorm()
                   << ".\n";
      }
      std::cout << "GMRES solver took " << chrono.RealTime() << "s.\n";
   }
   else
   {
      // 10. Construct the operators for preconditioner
      //
      //                 P = [ diag(M)         0         ]
      //                     [  0       B diag(M)^-1 B^T ]
      //
      //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
      //     pressure Schur Complement
      SparseMatrix *MinvBt = NULL;
      Vector Md(mVarf->Height());

      Solver *invM, *invS;
      SparseMatrix *S = NULL;

      if (pa)
      {
         mVarf->AssembleDiagonal(Md);
         auto Md_host = Md.HostRead();
         Vector invMd(mVarf->Height());
         for (int i=0; i<mVarf->Height(); ++i)
         {
            invMd(i) = 1.0 / Md_host[i];
         }

         Vector BMBt_diag(cVarf->Height());
         cVarf->AssembleDiagonal_ADAt(invMd, BMBt_diag);

         Array<int> ess_tdof_list;  // empty

         invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
         invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
      }
      else
      {
         SparseMatrix &M(mVarf->SpMat());
         M.GetDiag(Md);
         Md.HostReadWrite();

         SparseMatrix &Bm(cVarf->SpMat());
         MinvBt = Transpose(Bm);

         for (int i = 0; i < Md.Size(); i++)
         {
            MinvBt->ScaleRow(i, 1./Md(i));
         }

         S = Mult(Bm, *MinvBt);

         invM = new DSmoother(M);

#ifndef MFEM_USE_SUITESPARSE
         invS = new GSSmoother(*S);
#else
         invS = new UMFPackSolver(*S);
#endif
      }

      invM->iterative_mode = false;
      invS->iterative_mode = false;

      BlockDiagonalPreconditioner darcyPrec(block_trueOffsets);
      darcyPrec.SetDiagonalBlock(0, invM);
      darcyPrec.SetDiagonalBlock(1, invS);

      // 11. Solve the linear system with MINRES.
      //     Check the norm of the unpreconditioned residual.

      chrono.Clear();
      chrono.Start();
      GMRESSolver solver;
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*A);
      solver.SetPreconditioner(darcyPrec);
      solver.SetPrintLevel(1);

      solver.Mult(RHS, X);
      darcy->RecoverFEMSolution(X, x);

      if (device.IsEnabled()) { x.HostRead(); }
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

      delete invM;
      delete invS;
      delete S;
      delete MinvBt;
   }

   // 12. Create the grid functions E and B. Compute the L2 error norms.
   GridFunction E, B;
   E.MakeRef(E_space, x.GetBlock(0), 0);
   B.MakeRef(B_space, x.GetBlock(1), 0);

   GridFunction E_anal(E_space);
   GridFunction B_anal(B_space);
   E_anal.ProjectCoefficient(Ecoeff);
   B_anal.ProjectCoefficient(Bcoeff);

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   real_t err_u  = E.ComputeL2Error(Ecoeff, irs);
   real_t norm_u = ComputeLpNorm(2., Ecoeff, *mesh, irs);
   real_t err_p  = B.ComputeL2Error(Bcoeff, irs);
   real_t norm_p = ComputeLpNorm(2., Bcoeff, *mesh, irs);

   std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
   std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";

   // 13. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_E.gf" or "glvis -m ex5.mesh -g
   //     sol_B.gf".
   {
      ofstream mesh_ofs("ex5.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream E_ofs("sol_E.gf");
      E_ofs.precision(8);
      E.Save(E_ofs);

      ofstream B_ofs("sol_B.gf");
      B_ofs.precision(8);
      B.Save(B_ofs);
   }

   // 14. Save data in the VisIt format
   VisItDataCollection visit_dc("Example5", mesh);
   visit_dc.RegisterField("electric field", &E);
   visit_dc.RegisterField("magnetic field", &B);
   visit_dc.Save();

   // 15. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("Example5", mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("electric field",&E);
   paraview_dc.RegisterField("magnetic field",&B);
   paraview_dc.Save();

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream E_sock(vishost, visport);
      E_sock.precision(8);
      E_sock << "solution\n" << *mesh << E << "window_title 'Electric field'" << endl;
      socketstream B_sock(vishost, visport);
      B_sock.precision(8);
      B_sock << "solution\n" << *mesh << B << "window_title 'Magnetic field'" << endl;
      socketstream Ea_sock(vishost, visport);
      Ea_sock.precision(8);
      Ea_sock << "solution\n" << *mesh << E_anal <<
              "window_title 'Electric field anal'" << endl;
      socketstream Ba_sock(vishost, visport);
      Ba_sock.precision(8);
      Ba_sock << "solution\n" << *mesh << B_anal <<
              "window_title 'Magnetic field anal'" << endl;
   }

   // 17. Free the used memory.
   delete darcy;
   delete B_space;
   delete E_space;
   delete trace_space;
   delete B_coll;
   delete E_coll;
   delete trace_coll;
   delete mesh;

   return 0;
}


void EFun_ex(const Vector &x, Vector &E)
{
   const int dim = x.Size();
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
      //if (dim == 3) { E(2) = 0.0; }
   }
}

// Change if needed
void BFun_ex(const Vector & x, Vector &B)
{
   const int dim = x.Size();
   if (dim < 3)
   {
      B(0) = kappa * (-cos(kappa * x(0)) + cos(kappa * x(1)));
   }
   else
   {
      B(0) = kappa * (-cos(kappa * x(0)) + cos(kappa * x(2)));
      B(1) = kappa * (cos(kappa * x(0)));
      B(2) = kappa * (cos(kappa * x(1)));
   }
}

void fFun(const Vector & x, Vector & f)
{
   const int dim = x.Size();
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
