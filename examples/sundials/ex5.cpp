//                                MFEM Example 5
//                             SUNDIALS Modification
//
// Compile with:
//    make ex5             (GNU make)
//    make sundials_ex5    (CMake)
//
// Sample runs:  ex5
//               ex5 -o 2
//               ex5 -o 2 -tf 2.0 -dt 0.05
//               ex5 -m ../../data/star.mesh -o 3 -r 0
//               ex5 -m ../../data/beam-tet.mesh -r 0
//               ex5 -m ../../data/fichera.mesh -r 0
//               ex5 -rtol 1e-10 -atol 1e-12
//               ex5 -no-sa       (expected to abort; see below)
//               ex5 -calcic      (expected to abort; see below)
//
// Description:  This example solves the TRANSIENT mixed Darcy problem
//
//                                 du/dt + k u + grad p = f(x,t)
//                                 - div u              = g(x,t)
//
//               with natural boundary condition -p = <given pressure>,
//               discretized with Raviart-Thomas finite elements (velocity u)
//               and piecewise discontinuous polynomials (pressure p) exactly
//               as in the steady Example 5.  Writing M for the velocity mass
//               matrix (u,v), Mk for the Darcy matrix (k u, v) and B for the
//               divergence matrix (div u, w), the semi-discrete system is
//
//                                 M u' + Mk u - B^T p = F(t)
//                                       - B u         = G(t).
//
//               This is a differential-algebraic equation and not an ODE,
//               and that is the entire point of the example.  IDA integrates
//               the fully implicit residual form R(t, y, y') = 0 and, unlike
//               CVODE and ARKODE, does not need the mass matrix to be
//               invertible.  Here
//
//                                 dR/dy' = diag(M, 0)
//
//               is structurally singular -- the pressure carries no time
//               derivative -- so no rearrangement turns this system into
//               y' = f(y,t) and there is nothing for CVODE or ARKODE to
//               integrate.  Moreover the pressure does not appear in the
//               constraint row at all: the (1,1) block is exactly zero.  The
//               constraint must therefore be differentiated twice before an
//               equation for p' appears, which makes the system Hessenberg
//               index 2.
//
//               Contrast examples/sundials/ex16.cpp, where IDA is pointed at
//               the heat equation.  There the mass matrix is non-singular,
//               the problem is a genuine ODE, and IDA integrates it as a
//               trivial DAE -- every component is differential, so nothing
//               there exercises SetDifferentialComponents(),
//               SetSuppressAlgebraic() or ComputeConsistentIC().  This
//               example is the case where those three earn their place.
//
//               The exact solution is chosen to lie in the discrete spaces,
//
//                    u(x,t) = a(t) u0(x),   u0(x) = x
//                    p(x,t) = a(t) p0(x),   p0(x) = x_0 - x_1
//                    a(t)   = 1 + sin(t)/2,
//
//               so u0 is in RT_k and p0 is in L2_k for every k >= 1 and the
//               semi-discrete solution is a(t) times the coefficient vector
//               of (u0, p0) exactly.  There is therefore NO spatial
//               discretization error at all, and the L2 errors reported at
//               the end measure the time integration alone: they track the
//               requested tolerances rather than sitting at O(h^{k+1}), and
//               tightening -rtol / -atol drives them towards round-off.
//               That is this example's own check on itself, and it is much
//               sharper than a convergence rate.
//
//               The example demonstrates the use of the BlockOperator class
//               to state a DAE Jacobian, of MINRES with a block-diagonal
//               preconditioner to solve the resulting symmetric indefinite
//               system, and of the collective saving of several grid
//               functions in VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) formats.
//
//               We recommend viewing examples 5 and 16 before this one.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// The exact solution, its time derivative, and the data that follow from it.
//
// Derivation, which is short enough to check by eye:
//
//    u = a(t) u0,  u0(x) = x       =>  div u0 = dim,  du/dt = a'(t) u0
//    p = a(t) p0,  p0(x) = x_0 - x_1  =>  grad p0 = (1, -1, 0, ...)
//
//    row 2:  - div u = g           =>  g(x,t) = - a(t) * dim
//    row 1:  du/dt + k u + grad p = f
//                                  =>  f(x,t) = a'(t) u0(x)
//                                             + a(t) (k u0(x) + grad p0(x))
//
// and the natural boundary condition -p = <given pressure> supplies the
// datum -p_ex = -a(t) p0(x) to VectorFEBoundaryFluxLFIntegrator, which
// assembles <datum, v.n> -- the boundary term left over from integrating
// (grad p, v) by parts as -(p, div v) + <p, v.n>.
real_t aFun(real_t t);
real_t daFun(real_t t);
void u0Fun(const Vector &x, Vector &u);
real_t p0Fun(const Vector &x);
void uFun_ex(const Vector &x, real_t t, Vector &u);
real_t pFun_ex(const Vector &x, real_t t);
void dudtFun_ex(const Vector &x, real_t t, Vector &u);
real_t dpdtFun_ex(const Vector &x, real_t t);
void fFun(const Vector &x, real_t t, Vector &f);
real_t gFun(const Vector &x, real_t t);
real_t f_natural(const Vector &x, real_t t);

/** The transient mixed Darcy system as a DAE residual.

    The operator is declared HOMOGENEOUS, so G(u,t) = 0 and the whole of

        R(t, y, y') = | M y_0' + Mk y_0 - B^T y_1 - F(t) |
                      |          - B y_0          - G(t) |

    is assembled by ImplicitMult().  IDA drives that to zero.

    The blocks are the ones the steady Example 5 assembles.  B here is the
    divergence matrix (div u, w) exactly as VectorFEDivergenceIntegrator
    produces it; Example 5 negates it on assembly and writes its steady
    operator as [Mk, B^T; B, 0] with that negated B, which is the same
    operator as [Mk, -B^T; -B, 0] is here.  Either way it is symmetric, and
    that is what admits MINRES on the Jacobian below. */
class TransientDarcyOperator : public TimeDependentOperator
{
private:
   Array<int> block_offsets;

   FiniteElementSpace &R_space, &W_space;

   /** The Darcy coefficient k.  It is 1.0 here, exactly as in the steady
       Example 5, so Mk and M below are numerically the same matrix.  They
       are kept as separate names anyway because they play different roles
       and the reader needs to see which is which: M multiplies y' and is
       therefore the whole of dR/dy', while Mk multiplies y and belongs to
       dR/dy.  Give k any other value and they part company. */
   ConstantCoefficient kcoeff;

   // F(t) and G(t) are time dependent, so their coefficients are re-timed
   // and the linear forms re-assembled whenever the residual is asked for at
   // a new time.  Mutable because ImplicitMult() is const.
   mutable VectorFunctionCoefficient fcoeff;
   mutable FunctionCoefficient fnatcoeff;
   mutable FunctionCoefficient gcoeff;

   BilinearForm mVarf;        // M  = (u, v)
   BilinearForm mkVarf;       // Mk = (k u, v)
   MixedBilinearForm bVarf;   // B  = (div u, w)

   SparseMatrix *M, *Mk, *B;  // owned by the three forms above

   LinearForm *fform, *gform;
   mutable BlockVector rhs;   // [F(t); G(t)], aliased by fform and gform
   mutable real_t rhs_time;   // the time rhs currently holds

   // The Jacobian J = [ cj*M + Mk, -B^T ; -B, 0 ] and everything that solves
   // with it.  All of it depends on cj and is rebuilt when cj moves.
   real_t cj_cached;
   SparseMatrix *A;                     // cj*M + Mk
   TransposeOperator *Bt;
   BlockOperator *jacOp;
   SparseMatrix *MinvBt, *S;            // S = B diag(A)^-1 B^T
   Solver *invA, *invS;
   BlockDiagonalPreconditioner *jacPrec;
   MINRESSolver *jacSolver;

   /// Re-assemble [F(t); G(t)] if it is not already held at time @a t.
   void AssembleRHS(real_t t) const;

   /// Release the Jacobian, its preconditioner and its solver.
   void FreeJacobian();

public:
   TransientDarcyOperator(FiniteElementSpace &R, FiniteElementSpace &W,
                          const Array<int> &offsets, real_t k_value);

   /** There is no explicit form.  dR/dy' is singular, so this system cannot
       be written as y' = f(y,t) and Mult() has nothing to return. */
   void Mult(const Vector &y, Vector &k) const override;

   /// Compute the DAE residual R(t, @a y, @a yp) at the currently set time.
   void ImplicitMult(const Vector &y, const Vector &yp,
                     Vector &r) const override;

   /// Assemble and factor J = dR/dy + @a cj dR/dy'.
   int SUNImplicitSetupDAE(const Vector &y, const Vector &yp,
                           const Vector &res, real_t cj) override;

   /// Solve J @a x = @a b with the solver SUNImplicitSetupDAE() built.
   int SUNImplicitSolveDAE(const Vector &b, Vector &x, real_t tol) override;

   virtual ~TransientDarcyOperator();
};


int main(int argc, char *argv[])
{
   // 0. Initialize SUNDIALS.
   Sundials::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   int ref_levels = 1;
   real_t t_final = 1.0;
   real_t dt = 0.1;
   real_t reltol = 1e-8;
   real_t abstol = 1e-10;
   bool suppress_alg = true;
   bool calcic = false;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree). Must be at "
                  "least 1: the exact pressure x_0 - x_1 is linear and has "
                  "to lie in the L2 space for the example to be exact.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Output time step. IDA takes as many internal steps as it "
                  "needs between two output times.");
   args.AddOption(&reltol, "-rtol", "--relative-tolerance",
                  "Relative tolerance for IDA.");
   args.AddOption(&abstol, "-atol", "--absolute-tolerance",
                  "Absolute tolerance for IDA.");
   args.AddOption(&suppress_alg, "-sa", "--suppress-algebraic",
                  "-no-sa", "--no-suppress-algebraic",
                  "Exclude the pressure from IDA's local error test. On by "
                  "default; -no-sa is EXPECTED TO ABORT, and that is the "
                  "demonstration -- see the comment where it is set.");
   args.AddOption(&calcic, "-calcic", "--compute-consistent-ic",
                  "-no-calcic", "--no-compute-consistent-ic",
                  "Spoil the initial pressure and ask ComputeConsistentIC() "
                  "to recover it. EXPECTED TO ABORT: IDACalcIC is for "
                  "index-one systems and this one is index two.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles",
                  "-no-visit", "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles",
                  "-no-paraview", "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) "
                  "visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   // 2. Read the mesh from the given mesh file and refine it. We can handle
   //    triangular, quadrilateral, tetrahedral and hexahedral meshes with the
   //    same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // The manufactured solution is only in the discrete spaces if the spaces
   // are rich enough to hold it, and it is only defined if there are two
   // coordinates to subtract.
   MFEM_VERIFY(order >= 1, "order must be at least 1: the exact pressure "
               "p0 = x_0 - x_1 is a degree-one polynomial and L2_0 cannot "
               "represent it, so the example would lose its exactness.");
   MFEM_VERIFY(dim >= 2, "the exact pressure p0 = x_0 - x_1 needs at least "
               "two space dimensions.");

   // 3. Define the Raviart-Thomas velocity space and the discontinuous L2
   //    pressure space, and the block structure of the problem.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

   FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, l2_coll);

   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = R_space->GetVSize();
   block_offsets[2] = W_space->GetVSize();
   block_offsets.PartialSum();

   mfem::out << "***********************************************************\n";
   mfem::out << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
   mfem::out << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   mfem::out << "dim(R+W) = " << block_offsets.Last() << "\n";
   mfem::out << "***********************************************************\n";

   // 4. Define the DAE residual operator.
   TransientDarcyOperator *oper =
      new TransientDarcyOperator(*R_space, *W_space, block_offsets, 1.0);

   // 5. Build the initial condition, which is the part of this example that
   //    most needs explaining.
   //
   //    IDA needs R(t0, y0, y'0) = 0 to start. For an index-two DAE that is
   //    TWO conditions, not one:
   //
   //      (a) the constraint itself,        - B u_0  = G(0);
   //      (b) the HIDDEN constraint got by differentiating it in time,
   //                                        - B u'_0 = G'(0).
   //
   //    (b) is what determines p_0. The pressure appears nowhere in the
   //    constraint row, and the momentum row -- with u'_0 unknown too --
   //    only pins the COMBINATION M u'_0 - B^T p_0. So for any p_0
   //    whatsoever there is a u'_0 making R(t0, y0, y'0) vanish exactly: a
   //    state can satisfy the residual identically and still carry a wrong
   //    pressure. The hidden constraint is what closes it, (a) and (b)
   //    together being the saddle-point problem
   //
   //        [  M   -B^T ] [ u'_0 ]   [ F(0) - Mk u_0 ]
   //        [ -B     0  ] [  p_0 ] = [     G'(0)     ]
   //
   //    whose solution is the unique consistent pair. That is what "index
   //    two" costs, and it is why a consistent initialisation here is not a
   //    solve of the algebraic rows.
   //
   //    Both conditions hold in closed form here, because the exact solution
   //    lies in the discrete spaces:
   //
   //        u_0 = a(0) u0,   p_0 = a(0) p0,
   //        u'_0 = a'(0) u0, p'_0 = a'(0) p0.
   //
   //    So the pair is projected rather than computed.
   real_t t = 0.0;

   BlockVector x(block_offsets), xp(block_offsets);

   GridFunction u, p;
   u.MakeRef(R_space, x.GetBlock(0), 0);
   p.MakeRef(W_space, x.GetBlock(1), 0);

   GridFunction du, dp;
   du.MakeRef(R_space, xp.GetBlock(0), 0);
   dp.MakeRef(W_space, xp.GetBlock(1), 0);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);
   VectorFunctionCoefficient ducoeff(dim, dudtFun_ex);
   FunctionCoefficient dpcoeff(dpdtFun_ex);

   ucoeff.SetTime(t);
   pcoeff.SetTime(t);
   ducoeff.SetTime(t);
   dpcoeff.SetTime(t);

   u.ProjectCoefficient(ucoeff);
   p.ProjectCoefficient(pcoeff);
   du.ProjectCoefficient(ducoeff);
   dp.ProjectCoefficient(dpcoeff);

   u.SyncAliasMemory(x);
   p.SyncAliasMemory(x);
   du.SyncAliasMemory(xp);
   dp.SyncAliasMemory(xp);

   // 6. Mark which unknowns carry a time derivative: 1 on every velocity dof
   //    and 0 on every pressure dof, read straight off the block offsets.
   Array<int> is_differential(block_offsets.Last());
   is_differential = 0;
   for (int i = block_offsets[0]; i < block_offsets[1]; i++)
   {
      is_differential[i] = 1;
   }

   // 7. Set up IDA.
   IDASolver ida;
   ida.Init(*oper);
   ida.SetSStolerances(reltol, abstol);
   ida.SetDifferentialComponents(is_differential);

   // For an index-two system this is not a tuning knob. IDA's BDF local
   // error estimate for an algebraic variable is not an estimate of any
   // error -- the pressure has no dynamics for a truncation error to be
   // taken of -- so leaving it in the test controls the step size by a
   // quantity that means nothing. Measured on a 2x2 Hessenberg index-two
   // DAE through IDA 7.9.0: with the algebraic variable suppressed the
   // integration reaches t = 1 in 20 steps with errors at 1e-16; with it
   // left in, IDASolve FAILS at the very first output time with
   // IDA_ERR_FAIL, "the error test failed repeatedly or with |h| = hmin".
   // So the step size is not merely mis-controlled, it collapses.
   //
   // Running with -no-sa is therefore expected to ABORT, in IDASolver::Step's
   // MFEM_VERIFY on the IDASolve() flag. That is the demonstration and not a
   // bug in this example.
   //
   // Measured here: -no-sa aborts at t = 0 with h driven to 3.9e-15, and
   // this problem reports the CORRECTOR failing rather than the error test
   // -- "the corrector convergence failed repeatedly or with |h| = hmin".
   // Either way the step collapses before the first output time; which of
   // the two tests gives out first is not the point and is not worth
   // relying on.
   ida.SetSuppressAlgebraic(suppress_alg);

   ida.SetInitialDerivative(xp);

   // ComputeConsistentIC() is not used here and cannot be. IDACalcIC is
   // documented for index-one systems: given the differential components it
   // solves the algebraic rows for the algebraic components and for y'. On
   // this problem the algebraic unknown -- the pressure -- does not appear in
   // the algebraic row at all, so there is nothing there to solve for it and
   // the Newton/linesearch it runs has a singular Jacobian.
   //
   // Measured on a 2x2 Hessenberg index-two DAE through IDA 7.9.0: from an
   // inconsistent start IDACalcIC returns IDA_LINESEARCH_FAIL (-4),
   // "Newton/Linesearch algorithm failed to converge". From an ALREADY
   // consistent start it returns 0 -- trivially, having had nothing to do.
   // That second row is the subtle part: a zero return proves nothing unless
   // the state it was given was actually inconsistent, so a demonstration
   // has to hand it real work. -calcic does that by discarding the initial
   // pressure first; the initial velocity, which is the differential
   // component, is left alone, which is precisely the input IDA_YA_YDP_INIT
   // documents.
   //
   // Measured here: it aborts inside ComputeConsistentIC()'s own check on
   // the IDACalcIC() flag, with "Newton/Linesearch algorithm failed to
   // converge" -- IDA_LINESEARCH_FAIL, the same outcome as on a 2x2
   // Hessenberg index-2 system. The failure is the demonstration.
   if (calcic)
   {
      mfem::out << "Discarding the initial pressure and calling "
                << "ComputeConsistentIC()...\n";
      x.GetBlock(1) = 0.0;
      ida.ComputeConsistentIC(x, t + dt);
   }

   // IDA's default step mode is IDA_NORMAL, which interpolates back to the
   // requested output time. Keep it: the error report below compares against
   // the exact solution at t_final, and a one-step mode would step past
   // t_final and compare two different times. That is a real trap -- it
   // shows up as an error that does not shrink with the tolerances.

   // 8. Prepare the visualization and data collections.
   VisItDataCollection visit_dc("Example5-Transient", mesh);
   visit_dc.RegisterField("velocity", &u);
   visit_dc.RegisterField("pressure", &p);
   if (visit)
   {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(t);
      visit_dc.Save();
   }

   ParaViewDataCollection paraview_dc("Example5-Transient", mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.RegisterField("velocity", &u);
   paraview_dc.RegisterField("pressure", &p);
   if (paraview)
   {
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(t);
      paraview_dc.Save();
   }

   socketstream u_sock, p_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      u_sock.open(vishost, visport);
      p_sock.open(vishost, visport);
      if (!u_sock || !p_sock)
      {
         mfem::out << "Unable to connect to GLVis server at "
                   << vishost << ':' << visport << endl;
         visualization = false;
         mfem::out << "GLVis visualization disabled.\n";
      }
      else
      {
         u_sock.precision(8);
         p_sock.precision(8);
         u_sock << "solution\n" << *mesh << u
                << "window_title 'Velocity'" << endl;
         p_sock << "solution\n" << *mesh << p
                << "window_title 'Pressure'" << endl;
      }
   }

   // 9. Integrate the DAE.
   mfem::out << "Integrating the DAE ..." << endl;
   tic_toc.Clear();
   tic_toc.Start();

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      real_t dt_real = min(dt, t_final - t);

      ida.Step(x, t, dt_real);

      last_step = (t >= t_final - 1e-8*dt);

      mfem::out << "step " << ti << ", t = " << t << endl;

      if (visualization)
      {
         u_sock << "solution\n" << *mesh << u << flush;
         p_sock << "solution\n" << *mesh << p << flush;
      }
      if (visit)
      {
         visit_dc.SetCycle(ti);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }
      if (paraview)
      {
         paraview_dc.SetCycle(ti);
         paraview_dc.SetTime(t);
         paraview_dc.Save();
      }
   }
   tic_toc.Stop();
   mfem::out << "Done, " << tic_toc.RealTime() << "s.\n";

   ida.PrintInfo();

   // 10. Compute the L2 errors against the exact solution at the final time.
   //     The exact solution lies in the discrete spaces, so there is no
   //     spatial discretization error to hide behind: what is reported here
   //     is the time integration error alone.
   //
   //     Measured, -r 1 -o 1 -tf 1.0 -dt 0.1, relative L2 errors:
   //
   //       -rtol   -atol    u            p            steps
   //       1e-4    1e-6     7.86e-05     1.39e-04     16
   //       1e-6    1e-8     1.09e-06     1.29e-07     46
   //       1e-8    1e-10    1.32e-08     2.39e-09     66
   //       1e-10   1e-12    corrector convergence fails at t = 0
   //
   //     So the error tracks the tolerance over three decades and then the
   //     method stops, at around 1e-9. THE LINEAR SOLVER IS NOT WHY, and
   //     that was established by eliminating it rather than by argument:
   //     dropping the MINRES relative-tolerance floor from 1e-12 to zero
   //     changes nothing, raising its iteration cap from 1e3 to 5e4 changes
   //     nothing, and a run that does succeed reports "LS fails = 0"
   //     alongside "NLS fails = 6" -- MINRES never fails at all.
   //
   //     What is left is intrinsic to index two. SetSuppressAlgebraic()
   //     removes the algebraic block from the ERROR test, but IDA's
   //     CORRECTOR test is a weighted norm over the whole correction, and
   //     the algebraic component of a Newton correction for an index-2 DAE
   //     scales like 1/h. Shrink the step and that test stops being
   //     satisfiable, whatever the linear solve does. This is the standard
   //     reason index-2 systems are integrated at moderate tolerances, and
   //     it is worth a reader's attention precisely because it looks like a
   //     linear solver problem and is not.
   ucoeff.SetTime(t);
   pcoeff.SetTime(t);

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   real_t err_u  = u.ComputeL2Error(ucoeff, irs);
   real_t norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs);
   real_t err_p  = p.ComputeL2Error(pcoeff, irs);
   real_t norm_p = ComputeLpNorm(2., pcoeff, *mesh, irs);

   mfem::out << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
   mfem::out << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";

   // 11. Save the mesh and the final solution. This output can be viewed
   //     later using GLVis: "glvis -m ex5.mesh -g sol_u.gf" or
   //     "glvis -m ex5.mesh -g sol_p.gf".
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

   // 12. Free the used memory. The operator owns forms built on the two
   //     spaces, so it goes first.
   delete oper;
   delete W_space;
   delete R_space;
   delete l2_coll;
   delete hdiv_coll;
   delete mesh;

   return 0;
}


TransientDarcyOperator::TransientDarcyOperator(FiniteElementSpace &R,
                                               FiniteElementSpace &W,
                                               const Array<int> &offsets,
                                               real_t k_value)
   : TimeDependentOperator(offsets.Last(), 0.0,
                           TimeDependentOperator::HOMOGENEOUS),
     block_offsets(offsets),
     R_space(R), W_space(W),
     kcoeff(k_value),
     fcoeff(R.GetMesh()->Dimension(), fFun),
     fnatcoeff(f_natural),
     gcoeff(gFun),
     mVarf(&R), mkVarf(&R), bVarf(&R, &W),
     M(NULL), Mk(NULL), B(NULL),
     fform(NULL), gform(NULL),
     rhs(block_offsets), rhs_time(0.0),
     cj_cached(-1.0),
     A(NULL), Bt(NULL), jacOp(NULL), MinvBt(NULL), S(NULL),
     invA(NULL), invS(NULL), jacPrec(NULL), jacSolver(NULL)
{
   // M carries the time derivative: it is the whole of dR/dy'. Mk is the
   // Darcy term k u. With k = 1 the two matrices are equal entry for entry;
   // they are assembled separately so that the residual below reads as the
   // equation does.
   mVarf.AddDomainIntegrator(new VectorFEMassIntegrator());
   mVarf.Assemble();
   mVarf.Finalize();
   M = &mVarf.SpMat();

   mkVarf.AddDomainIntegrator(new VectorFEMassIntegrator(kcoeff));
   mkVarf.Assemble();
   mkVarf.Finalize();
   Mk = &mkVarf.SpMat();

   // B = (div u, w). The constraint row - div u = g is - B u = G below, and
   // the momentum row carries - B^T p, which is the -(p, div v) left over
   // from integrating (grad p, v) by parts.
   bVarf.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf.Assemble();
   bVarf.Finalize();
   B = &bVarf.SpMat();

   // The two linear forms alias the blocks of rhs, exactly as in the steady
   // Example 5, so assembling them fills [F(t); G(t)] in place.
   fform = new LinearForm;
   fform->Update(&R_space, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform->AddBoundaryIntegrator(
      new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));

   gform = new LinearForm;
   gform->Update(&W_space, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));

   // Force the first AssembleRHS() to do the work: rhs_time is 0.0 and the
   // integration starts there, so a "same time" test would otherwise skip it.
   rhs = 0.0;
   rhs_time = -1.0;
}

void TransientDarcyOperator::AssembleRHS(real_t t) const
{
   // IDA evaluates the residual several times per step -- once per Newton
   // iteration, plus the error-test and initial-step machinery -- and almost
   // all of those calls are at a time that was already assembled for. So the
   // assembly is cached on the time it was last done at. Note that this is a
   // cache on the TIME and not on the state: F and G depend on t only.
   if (t == rhs_time) { return; }

   fcoeff.SetTime(t);
   fnatcoeff.SetTime(t);
   gcoeff.SetTime(t);

   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   rhs_time = t;
}

void TransientDarcyOperator::Mult(const Vector &, Vector &) const
{
   MFEM_ABORT("This system has no explicit form y' = f(y,t): dR/dy' is "
              "diag(M, 0), which is singular, and the pressure has no time "
              "derivative to return. Only the residual, ImplicitMult(), is "
              "defined. Use IDA rather than CVODE or ARKODE.");
}

void TransientDarcyOperator::ImplicitMult(const Vector &y, const Vector &yp,
                                          Vector &r) const
{
   // IDA sets the operator's time before every residual evaluation, so
   // GetTime() is the time this residual is wanted at.
   AssembleRHS(GetTime());

   // Views onto the caller's storage; writing through rb writes into r.
   const BlockVector yb(y.GetData(), block_offsets);
   const BlockVector ypb(yp.GetData(), block_offsets);
   BlockVector rb(r.GetData(), block_offsets);

   // r_0 = M yp_0 + Mk y_0 - B^T y_1 - F(t)
   M->Mult(ypb.GetBlock(0), rb.GetBlock(0));
   Mk->AddMult(yb.GetBlock(0), rb.GetBlock(0));
   B->AddMultTranspose(yb.GetBlock(1), rb.GetBlock(0), -1.0);
   rb.GetBlock(0) -= rhs.GetBlock(0);

   // r_1 = - B y_0 - G(t)
   B->Mult(yb.GetBlock(0), rb.GetBlock(1));
   rb.GetBlock(1).Neg();
   rb.GetBlock(1) -= rhs.GetBlock(1);
}

int TransientDarcyOperator::SUNImplicitSetupDAE(const Vector &, const Vector &,
                                                const Vector &, real_t cj)
{
   // J = dR/dy + cj dR/dy' = [ cj*M + Mk   -B^T ]
   //                         [    -B         0  ]
   //
   // Nothing in it depends on the state -- the system is linear in (y, y') --
   // so cj is the only thing that can invalidate it. IDA lags its Jacobian
   // setup and only calls this when cj has moved far enough to matter, but it
   // does call it again at the same cj after a convergence failure, and the
   // (0,0) block and hence the whole preconditioner would be rebuilt and
   // refactored for nothing.
   if (jacSolver && cj == cj_cached) { return SUN_SUCCESS; }

   FreeJacobian();
   cj_cached = cj;

   // (0,0) block: cj*M + Mk. This is where cj enters, and it is the only
   // place: the constraint row has no time derivative in it.
   A = mfem::Add(cj, *M, 1.0, *Mk);

   Bt = new TransposeOperator(B);

   jacOp = new BlockOperator(block_offsets);
   jacOp->SetBlock(0, 0, A);
   jacOp->SetBlock(0, 1, Bt, -1.0);
   jacOp->SetBlock(1, 0, B, -1.0);
   // The (1,1) block is left unset, i.e. exactly zero. That zero is what
   // makes this index two rather than index one.

   // Preconditioner, following the steady Example 5:
   //
   //      P = [ diag(A)          0          ]
   //          [    0     B diag(A)^-1 B^T   ]
   //
   // with the (0,0) block cj*M + Mk rather than Example 5's Mk, so it has to
   // be rebuilt whenever cj moves. Symmetric Gauss-Seidel (or UMFPACK where
   // SuiteSparse is available) approximates the inverse of the pressure Schur
   // complement. The sign of B does not reach S, since it appears twice.
   Vector Ad(A->Height());
   A->GetDiag(Ad);
   Ad.HostReadWrite();

   MinvBt = mfem::Transpose(*B);
   for (int i = 0; i < Ad.Size(); i++)
   {
      MinvBt->ScaleRow(i, 1./Ad(i));
   }
   S = mfem::Mult(*B, *MinvBt);

   invA = new DSmoother(*A);
#ifndef MFEM_USE_SUITESPARSE
   invS = new GSSmoother(*S);
#else
   invS = new UMFPackSolver(*S);
#endif
   invA->iterative_mode = false;
   invS->iterative_mode = false;

   jacPrec = new BlockDiagonalPreconditioner(block_offsets);
   jacPrec->SetDiagonalBlock(0, invA);
   jacPrec->SetDiagonalBlock(1, invS);

   // J is symmetric -- [A, -B^T; -B, 0] with A symmetric -- and indefinite,
   // the zero (1,1) block guaranteeing negative eigenvalues. So MINRES, not
   // CG: CG would be applied outside its hypotheses and would not converge.
   jacSolver = new MINRESSolver;
   jacSolver->iterative_mode = false;
   jacSolver->SetAbsTol(0.0);
   jacSolver->SetMaxIter(1000);
   jacSolver->SetPrintLevel(-1);
   jacSolver->SetOperator(*jacOp);
   jacSolver->SetPreconditioner(*jacPrec);

   return SUN_SUCCESS;
}

int TransientDarcyOperator::SUNImplicitSolveDAE(const Vector &b, Vector &x,
                                                real_t tol)
{
   MFEM_VERIFY(jacSolver, "SUNImplicitSetupDAE() has not been called");

   // IDA asks for whatever residual reduction its Newton convergence test
   // wants, and on a tight integration tolerance that can be a number MINRES
   // on this preconditioner will not deliver -- it would simply run to
   // max_iter every solve. Floor it. The floor is well below the tolerances
   // this example is run at, so it does not limit the answer; it limits the
   // work spent chasing a target the preconditioner cannot reach.
   const real_t rtol_floor = 1e-12;
   jacSolver->SetRelTol(std::max(tol, rtol_floor));

   jacSolver->Mult(b, x);

   if (jacSolver->GetConverged())
   {
      return SUN_SUCCESS;
   }
   else
   {
      // Returning a recoverable failure lets IDA cut the step and try again,
      // which is the right response to a linear solve that did not converge.
      return SUNLS_CONV_FAIL;
   }
}

void TransientDarcyOperator::FreeJacobian()
{
   delete jacSolver; jacSolver = NULL;
   delete jacPrec;   jacPrec = NULL;
   delete invS;      invS = NULL;
   delete invA;      invA = NULL;
   delete S;         S = NULL;
   delete MinvBt;    MinvBt = NULL;
   delete jacOp;     jacOp = NULL;
   delete Bt;        Bt = NULL;
   delete A;         A = NULL;
}

TransientDarcyOperator::~TransientDarcyOperator()
{
   FreeJacobian();
   delete gform;
   delete fform;
}


// The time modulation and its derivative.
real_t aFun(real_t t)
{
   return 1.0 + 0.5*sin(t);
}

real_t daFun(real_t t)
{
   return 0.5*cos(t);
}

// u0(x) = x. Its divergence is the space dimension, and it is in RT_k for
// every k >= 0 -- RT_0 already contains the radial field x.
void u0Fun(const Vector & x, Vector & u)
{
   for (int i = 0; i < u.Size(); i++)
   {
      u(i) = x(i);
   }
}

// p0(x) = x_0 - x_1, in L2_k for every k >= 1. Its gradient is (1,-1,0,...).
real_t p0Fun(const Vector & x)
{
   return x(0) - x(1);
}

void uFun_ex(const Vector & x, real_t t, Vector & u)
{
   u0Fun(x, u);
   u *= aFun(t);
}

real_t pFun_ex(const Vector & x, real_t t)
{
   return aFun(t)*p0Fun(x);
}

void dudtFun_ex(const Vector & x, real_t t, Vector & u)
{
   u0Fun(x, u);
   u *= daFun(t);
}

real_t dpdtFun_ex(const Vector & x, real_t t)
{
   return daFun(t)*p0Fun(x);
}

// f = a'(t) u0 + a(t) (k u0 + grad p0).
//
// k is 1.0 here, as in the steady Example 5, and it MUST match the value
// main() hands to TransientDarcyOperator: the whole verification rests on the
// data being the data of the equation actually assembled.
void fFun(const Vector & x, real_t t, Vector & f)
{
   const real_t k = 1.0;

   Vector u0(f.Size());
   u0Fun(x, u0);

   Vector gradp0(f.Size());
   gradp0 = 0.0;
   gradp0(0) =  1.0;
   gradp0(1) = -1.0;

   for (int i = 0; i < f.Size(); i++)
   {
      f(i) = daFun(t)*u0(i) + aFun(t)*(k*u0(i) + gradp0(i));
   }
}

// g = - div u = - a(t) div u0 = - a(t) * dim.
real_t gFun(const Vector & x, real_t t)
{
   return -aFun(t)*real_t(x.Size());
}

// The natural boundary condition is -p = <given pressure>, and
// VectorFEBoundaryFluxLFIntegrator assembles <datum, v.n>, which is the
// boundary term of -(p, div v) + <p, v.n>. So the datum is -p_ex.
real_t f_natural(const Vector & x, real_t t)
{
   return -pFun_ex(x, t);
}
