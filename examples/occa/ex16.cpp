//                                MFEM Example 16
//
// Compile with: make ex16
//
// Sample runs:  ex16
//               ex16 -m ../data/inline-tri.mesh
//               ex16 -m ../data/disc-nurbs.mesh -tf 2
//               ex16 -s 1 -a 0.0 -k 1.0
//               ex16 -s 2 -a 1.0 -k 0.0
//               ex16 -s 3 -a 0.5 -k 0.5 -o 4
//               ex16 -s 14 -dt 1.0e-4 -tf 4.0e-2 -vs 40
//               ex16 -m ../data/fichera-q2.mesh
//               ex16 -m ../data/escher.mesh
//               ex16 -m ../data/beam-tet.mesh -tf 10 -dt 0.1
//               ex16 -m ../data/amr-quad.mesh -o 4 -r 0
//               ex16 -m ../data/amr-hex.mesh -o 2 -r 0
//
// Description:  This example solves a time dependent nonlinear heat equation
//               problem of the form du/dt = C(u), with a non-linear diffusion
//               operator C(u) = \nabla \cdot (\kappa + \alpha u) \nabla u.
//
//               The example demonstrates the use of nonlinear operators (the
//               class ConductionOperator defining C(u)), as well as their
//               implicit time integration. Note that implementing the method
//               ConductionOperator::ImplicitSolve is the only requirement for
//               high-order implicit (SDIRK) time integration.
//
//               We recommend viewing examples 2, 9 and 10 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#ifndef MFEM_USE_ACROTENSOR
typedef OccaMassIntegrator      AcroMassIntegrator;
typedef OccaDiffusionIntegrator AcroDiffusionIntegrator;
#endif

class GradientUpdate : public Operator {
  Operator *mOper, *kOper;
  const double dt;
  mutable OccaVector update;

public:
  GradientUpdate(Operator *mOper_,
                 Operator *kOper_,
                 const double dt_) :
    Operator(mOper_->Width(), mOper_->Height()),
    mOper(mOper_),
    kOper(kOper_),
    dt(dt_) {}

  virtual void Mult(const OccaVector &x, OccaVector &y) const {
    static occa::kernelBuilder builder =
      makeCustomBuilder("gradient_update",
                        "v0[i] += c0*v1[i];");

    occa::device dev = x.GetDevice();
    occa::kernel kernel = builder.build(dev);

    if (update.Size() == 0) {
      update.SetSize(x.GetDevice(), x.Size());
    }
    mOper->Mult(x, y);
    kOper->Mult(x, update);

    kernel((int) y.Size(), dt, y, update);
  }
};

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator {
protected:
  OccaFiniteElementSpace &ofespace;
  Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

  OccaBilinearForm *M;
  OccaBilinearForm *K;

  Operator *Moper, *Koper;
  GradientUpdate *T;        // T = M + dt K
  double current_dt;

  OccaCGSolver M_solver;    // Krylov solver for inverting the mass matrix M
  DSmoother M_prec;         // Preconditioner for the mass matrix M

  OccaCGSolver T_solver;    // Implicit solver for T = M + dt K
  DSmoother T_prec;         // Preconditioner for the implicit solver

  double alpha, kappa;

  mutable OccaVector z;    // auxiliary vector

  bool use_acrotensor;    // Use Acrotensor integrators

public:
  ConductionOperator(OccaFiniteElementSpace &fespace_,
                     double alpha_,
                     double kappa_,
                     const OccaVector &u,
                     const bool use_acrotensor_);

  virtual void Mult(const OccaVector &u, OccaVector &du_dt) const;
  /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
      This is the only requirement for high-order SDIRK implicit integration.*/
  virtual void ImplicitSolve(const double dt, const OccaVector &u, OccaVector &k);

  /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
  void SetParameters(const OccaVector &u);

  virtual ~ConductionOperator();
};

double InitialTemperature(const Vector &x);

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file = "../../data/star.mesh";
  int ref_levels = 2;
  int order = 2;
  const char *basis_type = "G"; // Gauss-Lobatto
  int ode_solver_type = 3;
  double t_final = 0.5;
  double dt = 1.0e-2;
  double alpha = 1.0e-2;
  double kappa = 0.5;
  const char *device_info = "mode: 'Serial'";
  bool occa_verbose = false;
  bool use_acrotensor = false;
  bool visualization = true;
  int vis_steps = 5;

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use.");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order",
                 "Order (degree) of the finite elements.");
  args.AddOption(&basis_type, "-b", "--basis-type",
                 "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                 "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4.");
  args.AddOption(&t_final, "-tf", "--t-final",
                 "Final time; start time is 0.");
  args.AddOption(&dt, "-dt", "--time-step",
                 "Time step.");
  args.AddOption(&alpha, "-a", "--alpha",
                 "Alpha coefficient.");
  args.AddOption(&kappa, "-k", "--kappa",
                 "Kappa coefficient offset.");
  args.AddOption(&device_info, "-d", "--device-info",
                 "Device information to run example on (default: \"mode: 'Serial'\").");
  args.AddOption(&occa_verbose,
                 "-ov", "--occa-verbose",
                 "--no-ov", "--no-occa-verbose",
                 "Print verbose information about OCCA kernel compilation.");
  args.AddOption(&use_acrotensor,
                 "-ac", "--use-acro",
                 "--no-ac", "--no-acro",
                 "Use Acrotensor.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                 "Visualize every n-th timestep.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

#ifndef MFEM_USE_ACROTENSOR
  if (use_acrotensor) {
    cout << "MFEM not compiled with Acrotensor, reverting to OCCA\n";
    use_acrotensor = false;
  }
#endif

  // Set the OCCA device to run example in
  occa::setDevice(device_info);

  // Load cached kernels
  occa::loadKernels();
  occa::loadKernels("mfem");

  // Set as the background device
  occa::settings()["verboseCompilation"] = occa_verbose;

  // 2. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  // See class BasisType in fem/fe_coll.hpp for available basis types
  int basis = BasisType::GetType(basis_type[0]);
  cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;

  // 3. Define the ODE solver used for time integration. Several implicit
  //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
  //    explicit Runge-Kutta methods are available.
  ODESolver *ode_solver;
  switch (ode_solver_type) {
    // Implicit L-stable methods
  case 1:  ode_solver = new OccaBackwardEulerSolver; break;
  case 2:  ode_solver = new OccaSDIRK23Solver(2); break;
  case 3:  ode_solver = new OccaSDIRK33Solver; break;
    // Explicit methods
  case 11: ode_solver = new OccaForwardEulerSolver; break;
  case 12: ode_solver = new OccaRK2Solver(0.5); break; // midpoint method
  case 13: ode_solver = new OccaRK3SSPSolver; break;
  case 14: ode_solver = new OccaRK4Solver; break;
    // Implicit A-stable methods (not L-stable)
  case 22: ode_solver = new OccaImplicitMidpointSolver; break;
  case 23: ode_solver = new OccaSDIRK23Solver; break;
  case 24: ode_solver = new OccaSDIRK34Solver; break;
  default:
    cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
    return 3;
  }

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement, where 'ref_levels' is a
  //    command-line parameter.
  for (int lev = 0; lev < ref_levels; lev++) {
    mesh->UniformRefinement();
  }

  // 5. Define the vector finite element space representing the current and the
  //    initial temperature, u_ref.
  H1_FECollection fe_coll(order, dim, basis);
  OccaFiniteElementSpace ofespace(mesh, &fe_coll);
  FiniteElementSpace &fespace = *(ofespace.GetFESpace());

  int fe_size = fespace.GetTrueVSize();
  cout << "Number of temperature unknowns: " << fe_size << endl;

  // 6. Set the initial conditions for u. All boundaries are considered
  //    natural.
  FunctionCoefficient u_0(InitialTemperature);
  GridFunction u_f(&fespace);
  u_f.ProjectCoefficient(u_0);

  OccaGridFunction u_gf(&ofespace);
  u_gf = u_f;

  OccaVector u;
  u_gf.GetTrueDofs(u);

  // 7. Initialize the conduction operator and the visualization.
  ConductionOperator oper(ofespace, alpha, kappa, u, use_acrotensor);

  u_gf.SetFromTrueDofs(u);

  {
    ofstream omesh("ex16.mesh");
    omesh.precision(precision);
    mesh->Print(omesh);
    ofstream osol("ex16-init.gf");
    osol.precision(precision);
    u_f = u_gf;
    u_f.Save(osol);
  }

  socketstream sout;
  if (visualization) {
    char vishost[] = "localhost";
    int  visport   = 19916;
    sout.open(vishost, visport);
    if (!sout) {
      cout << "Unable to connect to GLVis server at "
           << vishost << ':' << visport << endl;
      visualization = false;
      cout << "GLVis visualization disabled.\n";
    } else {
      sout.precision(precision);
      sout << "solution\n" << *mesh << u_f;
      sout << "pause\n";
      sout << flush;
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
    }
  }

  // 8. Perform time-integration (looping over the time iterations, ti, with a
  //    time-step dt).
  ode_solver->Init(oper);
  double t = 0.0;

  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    if (t + dt >= t_final - dt/2) {
      last_step = true;
    }

    ode_solver->Step(u, t, dt);

    if (last_step || (ti % vis_steps) == 0) {
      cout << "step " << ti << ", t = " << t << endl;

      u_gf.SetFromTrueDofs(u);
      if (visualization) {
        // [MISSING] u_gf
        sout << "solution\n" << *mesh << u_f << flush;
      }
    }
    // Set the
    oper.SetParameters(u);
  }

  // 9. Save the final solution. This output can be viewed later using GLVis:
  //    "glvis -m ex16.mesh -g ex16-final.gf".
  {
    ofstream osol("ex16-final.gf");
    osol.precision(precision);
    u_f = u_gf;
    u_f.Save(osol);
  }

  // 10. Free the used memory.
  delete ode_solver;
  delete mesh;

  return 0;
}

ConductionOperator::ConductionOperator(OccaFiniteElementSpace &ofespace_,
                                       double alpha_,
                                       double kappa_,
                                       const OccaVector &u,
                                       const bool use_acrotensor_) :
  TimeDependentOperator(ofespace_.GetFESpace()->GetTrueVSize(), 0.0),
  ofespace(ofespace_),
  M(NULL),
  K(NULL),
  T(NULL),
  current_dt(0.0),
  alpha(alpha_),
  kappa(kappa_),
  z(height),
  use_acrotensor(use_acrotensor_) {

  const double rel_tol = 1e-8;

  M = new OccaBilinearForm(&ofespace);
  if (use_acrotensor) {
    M->AddDomainIntegrator(new AcroMassIntegrator(1.0));
  } else {
    M->AddDomainIntegrator(new OccaMassIntegrator(1.0));
  }
  M->Assemble();
  M->FormOperator(ess_tdof_list, Moper);

  M_solver.iterative_mode = false;
  M_solver.SetRelTol(rel_tol);
  M_solver.SetAbsTol(0.0);
  M_solver.SetMaxIter(4000);
  M_solver.SetPrintLevel(0);
  // M_solver.SetPreconditioner(M_prec);
  M_solver.SetOperator(*Moper);

  T_solver.iterative_mode = false;
  T_solver.SetRelTol(rel_tol);
  T_solver.SetAbsTol(0.0);
  T_solver.SetMaxIter(4000);
  T_solver.SetPrintLevel(0);
  // T_solver.SetPreconditioner(T_prec);

  SetParameters(u);
}

void ConductionOperator::Mult(const OccaVector &u, OccaVector &du_dt) const {
  // Compute:
  //    du_dt = M^{-1}*-K(u)
  // for du_dt
  Koper->Mult(u, z);
  z.Neg();
  M_solver.Mult(z, du_dt);
}

void ConductionOperator::ImplicitSolve(const double dt,
                                       const OccaVector &u, OccaVector &du_dt) {
  // Solve the equation:
  //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
  // for du_dt
  if (!T) {
    T = new GradientUpdate(Moper, Koper, dt);
    current_dt = dt;
    T_solver.SetOperator(*T);
  }
  MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt
  Koper->Mult(u, z);
  z.Neg();
  T_solver.Mult(z, du_dt);
}

void ConductionOperator::SetParameters(const OccaVector &u) {
  delete K;

  OccaGridFunction u_alpha_gf(&ofespace);
  u_alpha_gf.SetFromTrueDofs(u);

  OccaCoefficient u_coeff("(kappa + alpha*u(q, e))");
  u_coeff
    .AddDefine("kappa", kappa)
    .AddDefine("alpha", alpha)
    .AddGridFunction("u", u_alpha_gf, true);

  K = new OccaBilinearForm(&ofespace);

  if (use_acrotensor) {
    K->AddDomainIntegrator(new AcroDiffusionIntegrator(u_coeff));
  } else {
    K->AddDomainIntegrator(new OccaDiffusionIntegrator(u_coeff));
  }
  K->Assemble();
  K->FormOperator(ess_tdof_list, Koper);

  delete T;
  T = NULL; // re-compute T on the next ImplicitSolve
}

ConductionOperator::~ConductionOperator() {
  delete T;
  delete M;
  delete K;
}

double InitialTemperature(const Vector &x)
{
   if (x.Norml2() < 0.5)
   {
      return 2.0;
   }
   else
   {
      return 1.0;
   }
}
