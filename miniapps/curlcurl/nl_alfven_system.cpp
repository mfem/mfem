// 
// Put it in mfem/example and compile  
//
// we solve system: start with this: (M + nu dt K2) B^{n+1} = M * B0 + K1 * V0
//
// Sample run: mpirun -np 4 alfven -dt 0.01 -o 2
//

#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

#include "mfem.hpp"

using namespace mfem;

double c0 = 10.0;
void computeB0(const Vector& x, Vector& B0);
void computeV0(const Vector& x, Vector& V0);

// next three functions are used in earlier versions of the code
void computeCrossB0(const Vector& x, DenseMatrix& CrossB0);
void computeCrossB0transpose(const Vector& x, DenseMatrix& CrossB0);
void computeOmega(const Vector& x, DenseMatrix& omega);


/*******************************************************************
* Alfven time-dependent non-linear operator. Explicit interface 
* solves the following system of ODEs:
*
*   dV/dt = inv(Mv) * [-B x (\nabla x B)]
*   dB/dt = inv(Mb) * [\nabla x (V x B) - 1/S \nabla x \nabla x B]
*
* where V in H_0(div) and B in H(curl). 
****************************************************************** */
class AlfvenOperator : public TimeDependentOperator {
public:
  AlfvenOperator(std::shared_ptr<ParFiniteElementSpace>& vspace,
                 std::shared_ptr<ParFiniteElementSpace>& bspace,
                 Array<int>& ess_tdof_list,
                 double gmres_tol,
                 int myid);
  ~AlfvenOperator() {
    // if (A11_) delete A11_;
  }

  // required interface
  virtual void Mult(const Vector& u, Vector& du_dt) const;
  virtual void ImplicitSolve(const double dt, const Vector& u, Vector& du_dt);

  // other member functions
  void Init(double eta);
  void UpdateMatrices(const Vector& u) const;
  void UpdateLinearSolver(double dt, const Vector& u);
  void FormBlockSystem(int mode, double dt=0) const;
  void MassMult(const Vector& u, Vector& v);

  // getters / setters
  std::shared_ptr<BlockOperator> get_system() { return system_; }
  Array<int> get_block_offsets() { return block_offsets_; }
  Array<int> get_block_true_offsets() { return block_true_offsets_; }
  HypreParMatrix* get_system_block(int i, int j) { return static_cast<HypreParMatrix*>(&system_->GetBlock(i, j)); }

  std::pair<int, int> get_explicit_iterations() { return std::make_pair(itrs0_, itrs1_); }
  int get_implicit_iterations() { return itrs2_; }

private:
  std::shared_ptr<ParFiniteElementSpace> vspace_, bspace_;
  int ntv_, ntb_, myid_;

  mutable std::shared_ptr<BlockOperator> system_;
  Array<int> block_offsets_, block_true_offsets_; // number of variables + 1
  Array<int>& ess_tdof_list_;

  double eta_;

  std::shared_ptr<HypreParMatrix> M00_, M11_, S11_, P11add_;
  mutable std::shared_ptr<HypreParMatrix> A11_, B01_;
  mutable TransposeOperator *B10_;

  std::shared_ptr<GMRESSolver> gmres_; // solver for M u + dt K u, where u = (V, B)
  double gmres_tol_;

  std::shared_ptr<BlockDiagonalPreconditioner> pc_;
  std::shared_ptr<HypreSolver> inv00_, inv11_;

  // statistics
  mutable int itrs0_, itrs1_, itrs2_;
};


/*******************************************************************
* Alfven operator constructor
****************************************************************** */
AlfvenOperator::AlfvenOperator(std::shared_ptr<ParFiniteElementSpace>& vspace,
                               std::shared_ptr<ParFiniteElementSpace>& bspace,
                               Array<int>& ess_tdof_list,
                               double gmres_tol,
                               int myid)
  : TimeDependentOperator(vspace->TrueVSize() + bspace->TrueVSize(), 0.0), 
    vspace_(vspace),
    bspace_(bspace),
    ess_tdof_list_(ess_tdof_list),
    gmres_tol_(gmres_tol),
    myid_(myid)
{
  // A11_ = nullptr; // FIXME (shares ptr?)
}


/*******************************************************************
* Alfven operator API for explicit time integration
****************************************************************** */
void AlfvenOperator::Mult(const Vector& u, Vector& du_dt) const
{
  // Update nonlinear operators
  UpdateMatrices(u);
  FormBlockSystem(1);

  Vector f(u);
  system_->Mult(u, f); 

  CGSolver M00_solver(MPI_COMM_WORLD), M11_solver(MPI_COMM_WORLD);
  HypreSmoother M00_prec, M11_prec;

  M00_solver.iterative_mode = false;
  M00_solver.SetRelTol(1e-6);
  M00_solver.SetAbsTol(0.0);
  M00_solver.SetMaxIter(30);
  M00_solver.SetPrintLevel(0);

  M00_prec.SetType(HypreSmoother::Jacobi);
  M00_solver.SetPreconditioner(M00_prec);
  M00_solver.SetOperator(*M00_);

  M11_solver.iterative_mode = false;
  M11_solver.SetRelTol(1e-6);
  M11_solver.SetAbsTol(0.0);
  M11_solver.SetMaxIter(30);
  M11_solver.SetPrintLevel(0);

  M11_prec.SetType(HypreSmoother::Jacobi);
  M11_solver.SetPreconditioner(M11_prec);
  M11_solver.SetOperator(*M11_);

  Vector v(f.GetData(), ntv_);
  Vector dv_dt(du_dt.GetData(), ntv_);
  M00_solver.Mult(v, dv_dt);
  itrs0_ = M00_solver.GetNumIterations();

  Vector b(f.GetData() + ntv_, ntb_);
  Vector db_dt(du_dt.GetData() + ntv_, ntb_);
  M11_solver.Mult(b, db_dt);
  itrs1_ = M11_solver.GetNumIterations();

  // negate the time rate, see the definition of the block system for 
  // explicit time integrator
  du_dt.Neg();

  if (myid_ == 0) {
    std::cout << "  mass matrix solvers: " << itrs0_ << " " << itrs1_ << std::endl;
  }
}

/*******************************************************************
* Alfven operator API for implicit time integration
****************************************************************** */
void AlfvenOperator::ImplicitSolve(const double dt, const Vector& u, Vector& du_dt)
{
  // Here we use Picard iterations to solve a nonlinear equation
  // for the Runge-Kutta stage vector k,
  //
  //    M*k = N(u+dt*k)                     (1)
  //
  // for nonlinear operator N. We assume N can be written as
  //
  //    N(u+dt*k) := L[u+dt*k](u+dt*k) + f(t)
  //
  // where L is a matrix-valued operator evaluated at u+dt*k and f(t)
  // a (potentially zero) time-dependent forcing vector. (1) can be
  // rewritten as a fixed-point equation
  //
  //    x = (M - dt*L[x])^{-1} (Mu + f)     (2)
  //
  // where x := u + dt*k, which can be solved using a Picard iteration,
  // where a function G(x) = x is solved via iterations x_{k+1} = G(x_k).
  //
  //    Note, system_ = M + dtK

  double tol = 1e-6;
  int maxiter = 100;

  // Right-hand side for nonlinear iteration
  Vector z(u);   // Vector for right-hand side
  Vector temp(u);   // Vector to measure error
  MassMult(u, z);   // NOTE : Add forcing function here if one exists
  du_dt = u;        // Set u as initial guess for x (2)
  temp = u;
  double error = 1;
  int iter = 0;
  while (error > tol) {
    iter ++;
    UpdateMatrices(du_dt);         // Update linearized nonlinear operator L[x]
    FormBlockSystem(2, dt);        // Form matrix (M - dt*L[x])
    UpdateLinearSolver(dt, du_dt); // Construct preconditioner
    gmres_->Mult(z, du_dt);        // Solve linearized system
    temp -= du_dt;                 // Measure error
    error = std::sqrt(InnerProduct(MPI_COMM_WORLD, temp, temp));
    temp = du_dt;
    itrs2_ = gmres_->GetNumIterations();
    if (myid_ == 0) {
      std::cout << "  Picard iteration " << iter << ", error = " <<
        std::setprecision(5) << error << ", gmres iterations: " << itrs2_ << std::endl;
    }

    if (iter >= maxiter) {
      mfem_warning("Nonlinear iteration did not converge!");
      break;
    }
  }

  // Above we solved for x = u + dt*k, where k is the desired update
  // Map du_dt -> k.
  du_dt -= u;
  du_dt /= dt;
}


/*******************************************************************
* Define 2x2 block system
****************************************************************** */
void AlfvenOperator::Init(double eta)
{
  block_offsets_.Append(0);
  block_offsets_.Append(vspace_->GetVSize());
  block_offsets_.Append(bspace_->GetVSize());
  block_offsets_.PartialSum();

  ntv_ = vspace_->TrueVSize();
  ntb_ = bspace_->TrueVSize();

  block_true_offsets_.Append(0);
  block_true_offsets_.Append(ntv_);
  block_true_offsets_.Append(ntb_);
  block_true_offsets_.PartialSum();

  system_ = std::make_shared<BlockOperator>(block_true_offsets_);

  eta_ = eta;

  // ---------- Precompute operators fixed for all time ---------- //
  // block 00 is mass matrix for velocity uses the unit coefficient
  ConstantCoefficient one(1.0);
  ParBilinearForm *form00 = new ParBilinearForm(vspace_.get());
  form00->AddDomainIntegrator(new VectorFEMassIntegrator(one));
  form00->Assemble();
  form00->Finalize();

  // -- block 11 is mass matrix plus (1/S) curl-curl matrix for B
  ParBilinearForm *form11m = new ParBilinearForm(bspace_.get());
  form11m->AddDomainIntegrator(new VectorFEMassIntegrator(one));
  form11m->Assemble();
  form11m->Finalize();

  ParBilinearForm *form11s = new ParBilinearForm(bspace_.get());
  form11s->AddDomainIntegrator(new CurlCurlIntegrator(one));
  form11s->Assemble();
  form11s->Finalize();

  // TODO : why eliminate only for form00, not form11m/form11s?
  form00->EliminateEssentialBC(ess_tdof_list_);

  M00_ = std::shared_ptr<HypreParMatrix>(form00->ParallelAssemble());
  M11_ = std::shared_ptr<HypreParMatrix>(form11m->ParallelAssemble());
  S11_ = std::shared_ptr<HypreParMatrix>(form11s->ParallelAssemble());

  delete form00;
  delete form11m;
  delete form11s;
}

/*******************************************************************
* Apply mass matrices on both spaces, v = Mu, where u,v are block vectors
****************************************************************** */
void AlfvenOperator::MassMult(const Vector& u, Vector& v)
{
  Vector u0(u.GetData(), ntv_);
  Vector u1(u.GetData() + ntv_, ntb_);
  Vector v0(v.GetData(), ntv_);
  Vector v1(v.GetData() + ntv_, ntb_);

  M00_->Mult(u0, v0);  
  M11_->Mult(u1, v1);  
}

/*******************************************************************
* Update matrices using the previous time-step solution
****************************************************************** */
void AlfvenOperator::UpdateMatrices(const Vector& u) const
{
  // block 10 is <T0^t curl B, V'> = <B x curl B, V'>
  // MatrixFunctionCoefficient tensor01(3, computeCrossB0transpose);
  // auto form01 = std::make_shared<ParMixedBilinearForm>(bspace_.get(), vspace_.get());
  // form01->AddDomainIntegrator(new MixedVectorCurlIntegrator(tensor01));

  ParGridFunction bfun(bspace_.get());
  Vector u_b(u.GetData() + ntv_, ntb_); // shallow subvector
  bfun.SetFromTrueDofs(u_b);
  VectorGridFunctionCoefficient vector10(&bfun);
  auto form01 = std::make_shared<ParMixedBilinearForm>(bspace_.get(), vspace_.get());
  form01->AddDomainIntegrator(new MixedCrossCurlIntegrator(vector10));
  form01->Assemble();
  form01->Finalize();

  form01->EliminateTestDofs(ess_tdof_list_);
  B01_ = std::shared_ptr<HypreParMatrix>(form01->ParallelAssemble());

  // -- block 10 is negative transport of block 01
  //    we use transpose, but should revisit due to different boundary conditions
  B10_ = new TransposeOperator(B01_.get());
}


/*******************************************************************
* Update solver and preconditioner using the previous solution
****************************************************************** */
void AlfvenOperator::UpdateLinearSolver(double dt, const Vector& u)
{
  // update matrices
  ParGridFunction bfun(bspace_.get());
  Vector u_b(u.GetData() + ntv_, ntb_);
  bfun.SetFromTrueDofs(u_b);
  VectorGridFunctionCoefficient vector11(&bfun);
  CrossCrossCoefficient tensorOmega(1.0, vector11);

  // NOTE: integrator takes ownership of the input pointer
  ParBilinearForm *form11p = new ParBilinearForm(bspace_.get());
  form11p->AddDomainIntegrator(new CurlCurlIntegrator(tensorOmega));
  form11p->Assemble();
  form11p->Finalize();

  double alpha = dt * dt;
  HypreParMatrix *P11;
  P11add_ = std::shared_ptr<HypreParMatrix>(form11p->ParallelAssemble());
  P11 = Add(1.0, *A11_, alpha, *P11add_);

  // form block-diagonal preconditioner
  inv00_ = std::make_shared<HypreDiagScale>(*M00_);
  inv11_ = std::make_shared<HypreAMS>(*P11, bspace_.get()); // no Schur complement: use A11
   
  inv00_->iterative_mode = false;
  inv11_->iterative_mode = false;
 
  pc_ = std::make_shared<BlockDiagonalPreconditioner>(block_true_offsets_);
  pc_->SetDiagonalBlock(0, inv00_.get());
  pc_->SetDiagonalBlock(1, inv11_.get());

  // linearized solver for implicit time integrator
  gmres_ = std::make_shared<GMRESSolver>(MPI_COMM_WORLD);
  gmres_->SetKDim(100);
  gmres_->SetOperator(*system_);
  gmres_->SetPreconditioner(*pc_);
  gmres_->SetRelTol(gmres_tol_);
  gmres_->SetMaxIter(5000);
  gmres_->SetPrintLevel(0);

  // cleaning
  delete form11p;
}


/*******************************************************************
* Form block system from elemental matrices
* mode:
*   0 : Qi's hardcoded implicit solver, requires dt
*   1 : explicit mult, no dt required
*   2 : implicit solve, dt required
* Only provide dt for type 0 or type 2.
****************************************************************** */
void AlfvenOperator::FormBlockSystem(int mode, double dt) const
{
  // a stand-alone one-time step solver, line in Qi's solver
  if (mode == 0) {
    A11_ = std::shared_ptr<HypreParMatrix>(Add(1.0, *M11_, eta_ * dt, *S11_));

    system_->SetBlock(0, 0, M00_.get());
    system_->SetBlock(0, 1, B01_.get(), dt);
    system_->SetBlock(1, 0, B10_, -dt);
    system_->SetBlock(1, 1, A11_.get());

  // an explicit time integrator du/dt + A(u) = 0
  } else if (mode == 1) {
    system_->SetBlock(0, 0, M00_.get(), 0.0); // FIXME
    system_->SetBlock(0, 1, B01_.get());
    system_->SetBlock(1, 0, B10_, -1.0);
    system_->SetBlock(1, 1, S11_.get(), eta_);

  // an implicit time integrator uses linearized operator T = M + dt K
  } else if (mode == 2) {
    A11_ = std::shared_ptr<HypreParMatrix>(Add(1.0, *M11_, eta_ * dt, *S11_));

    system_->SetBlock(0, 0, M00_.get()); // FIXME
    system_->SetBlock(0, 1, B01_.get(), dt);
    system_->SetBlock(1, 0, B10_, -dt);
    system_->SetBlock(1, 1, A11_.get());

  } else {
    exit(0);
  }
}


/*******************************************************************
* Alfven linearized system
****************************************************************** */
int main(int argc, char *argv[])
{
  Mpi::Init(argc, argv);
  // int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  // default values for skipped input
  int nz = 3; // 3 is the minimal value for a periodic mesh
  int dim(3), order(1), refinement(1), ode_solver_type(11);
  double dt(1.0/64), tfin(1.0), io_freq(0.002), eta(1e-5), gmres_tol(1e-6);
  bool visualization(false);

  // Parse command line
  OptionsParser args(argc, argv);
  args.AddOption(&nz, "-nz", "--num-elem", "number of elements.");
  args.AddOption(&order, "-o", "--order", "Finite element order RT(o-1) + ND(o).");
  args.AddOption(&refinement, "-r", "--refinement", "Refinement leveles for parallel mesh.");

  args.AddOption(&c0, "-c0", "--c0", "set c0 in the background B.");
  args.AddOption(&dt, "-dt", "--dt", "set dt.");
  args.AddOption(&tfin, "-tfin", "--tfin", "set final time.");
  args.AddOption(&io_freq, "-io", "--io-frequency", "time frequency of i/o snapshots.");
  args.AddOption(&eta, "-eta", "--eta", "set eta (i.e., 1/S).");

  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                 "            11 - Forward Euler, 12 - RK2,\n\t"
                 "            13 - RK3 SSP, 14 - RK4."
                 "            22 - Implicit Midpoint Method,\n\t"
                 "            23 - SDIRK23 (A-stable), 24 - SDIRK34");

  args.AddOption(&gmres_tol, "-gmres_tol", "--gmres_tol", "set tolerance for linear solver (GMRES).");

  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization", "GLVis visualization.");
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) args.PrintUsage(std::cout);
    return 1;
  }
  if (myid == 0) args.PrintOptions(std::cout);

  // Generate a base mesh
  Mesh orig_mesh = Mesh::MakeCartesian3D(nz, nz, nz, Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);

  // Make z direction periodic
  std::vector<Vector> translations = { Vector({0.0, 0.0, 1.0}) };

  Mesh mesh = Mesh::MakePeriodic(orig_mesh,
                                 orig_mesh.CreatePeriodicVertexMapping(translations));
  mesh.RemoveInternalBoundaries();

  // Refine serial mesh once
  mesh.UniformRefinement();

  // Save the final serial mesh
  std::ofstream mesh_ofs("periodic-cube-z.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);
 
  // Define a parallel mesh by a partitioning of the serial mesh. Refine
  // this mesh further in parallel to increase the resolution. Once the
  // parallel mesh is defined, the serial mesh can be deleted.
  auto pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, mesh);
  auto ne0 = pmesh->GetGlobalNE();

  for (int l = 0; l < refinement; l++) {
    pmesh->UniformRefinement();
  }
  auto ne1 = pmesh->GetGlobalNE();
  if (myid == 0) {
    std::cout << "Number of cells in initial and final meshes: " << ne0 << " " << ne1 << std::endl;
  }

  FiniteElementCollection *rt_coll = new RT_FECollection(order - 1, dim);
  FiniteElementCollection *nd_coll = new ND_FECollection(order, dim);

  auto vspace = std::make_shared<ParFiniteElementSpace>(pmesh.get(), rt_coll);
  auto bspace = std::make_shared<ParFiniteElementSpace>(pmesh.get(), nd_coll);

  HYPRE_BigInt sizeb = bspace->GlobalTrueVSize();
  HYPRE_BigInt sizev = vspace->GlobalTrueVSize();
  if (myid == 0) {
    std::cout << "Number of finite element unknowns (B&V): " << sizeb << " " << sizev << std::endl;
  }

  // set up zero velocity on the whole boundary
  Array<int> ess_tdof_list;
  if (pmesh->bdr_attributes.Size() > 0) {
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 1;
    vspace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    if (myid == 0) {
      std::cout << "Number of velocity dofs slated for essential BCs: " << ess_tdof_list.Size() << std::endl;
    }
  }

  // parse initial conditions
  // -- velocity field
  ParGridFunction V0(vspace.get());
  VectorFunctionCoefficient Vcoeff(dim, computeV0);
  V0.ProjectCoefficient(Vcoeff);

  // -- magnetic field
  ParGridFunction B0(bspace.get());
  VectorFunctionCoefficient Bcoeff(dim, computeB0);
  B0.ProjectCoefficient(Bcoeff);

  // explicit time-stepping
  ODESolver *ode_solver;
  switch (ode_solver_type) {
    // Implicit L-stable methods
    case 1:  ode_solver = new BackwardEulerSolver; break;
    case 2:  ode_solver = new SDIRK23Solver(2); break;
    case 3:  ode_solver = new SDIRK33Solver; break;

    // Explicit methods
    case 11: ode_solver = new ForwardEulerSolver; break;
    case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
    case 13: ode_solver = new RK3SSPSolver; break;
    case 14: ode_solver = new RK4Solver; break;
    case 15: ode_solver = new GeneralizedAlphaSolver(0.5); break;

    // Implicit A-stable methods (not L-stable)
    case 22: ode_solver = new ImplicitMidpointSolver; break;
    case 23: ode_solver = new SDIRK23Solver; break;
    case 24: ode_solver = new SDIRK34Solver; break;

    default:
      if (myid == 0) std::cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      exit(1);
  }
  bool implicit = (ode_solver_type < 4 || ode_solver_type > 21) ? true : false;

  // initialized block system
  AlfvenOperator op(vspace, bspace, ess_tdof_list, gmres_tol, myid);
  op.Init(eta);
  op.SetTime(0.0);
  ode_solver->Init(op);

  // initialzie solution vectors
  auto block_offsets = op.get_block_offsets();
  auto block_true_offsets = op.get_block_true_offsets();

  BlockVector sol(block_offsets), rhs(block_offsets);
  BlockVector trueSol(block_true_offsets), trueRhs(block_true_offsets);
   
  V0.GetTrueDofs(trueSol.GetBlock(0));
  B0.GetTrueDofs(trueSol.GetBlock(1));

  double vnorm = GlobalLpNorm(2.0, trueSol.GetBlock(0).Norml2(), MPI_COMM_WORLD);
  double bnorm = GlobalLpNorm(2.0, trueSol.GetBlock(1).Norml2(), MPI_COMM_WORLD);
  if (myid == 0) {
    std::cout << "  ||B0|| = " << bnorm << ", ||V0|| = " << vnorm << std::endl;
  }

  // initialize io
  int nloop(0);
  double t(0.0), dt_limited, t_io(dt);

  ParaViewDataCollection io("alfven_system", pmesh.get());
  io.SetPrefixPath("ParaView");
  io.SetLevelsOfDetail(order);
  io.SetDataFormat(VTKFormat::BINARY);
  io.SetHighOrderOutput(true);

  while (t < tfin) {
    dt_limited = std::min(dt, tfin - t);
    if (myid == 0) {
      std::cout << "\nCycle " << nloop << ":  t=" << t << " dt=" << dt_limited << std::endl;
    }

    ode_solver->Step(trueSol, t, dt_limited);

    double vnorm = GlobalLpNorm(2.0, trueSol.GetBlock(0).Norml2(), MPI_COMM_WORLD);
    double bnorm = GlobalLpNorm(2.0, trueSol.GetBlock(1).Norml2(), MPI_COMM_WORLD);
    if (myid == 0) {
      std::cout << "  ||B|| = " << bnorm << ", ||V|| = " << vnorm << std::endl;
    }

    if (visualization && std::fabs(t - t_io) < dt) {
      if (myid == 0) std::cout << "  paraview io..." << std::endl;
      ParGridFunction Vfinal(vspace.get());
      ParGridFunction Bfinal(bspace.get());

      Vfinal.SetFromTrueDofs(trueSol.GetBlock(0));
      Bfinal.SetFromTrueDofs(trueSol.GetBlock(1));

      io.SetCycle(nloop);
      io.SetTime(t);
      io.RegisterField("B", &Bfinal);
      io.RegisterField("V", &Vfinal);
      io.Save();

      t_io += io_freq;
    }

    nloop++;
  }
  if (myid == 0) std::cout << "SIMULATION SUCCESSFUL\n";

  delete rt_coll;
  delete nd_coll;
  delete ode_solver;

  return 0;
}


/*******************************************************************
* Compute B0
****************************************************************** */
void computeB0(const Vector& x, Vector& B0)
{
  double x0(0.5), y0(0.5);
  double Az = 0.5 * std::pow(x(0) - x0, 2.0) + 1.0/32.0 * pow(sin(2*M_PI * (x(1) - y0)), 2.0);
  B0(0) = M_PI/8.0 * std::sin(2 * M_PI * (x(1) - y0)) * cos(2*M_PI * (x(1) - y0));
  B0(1) = -x(0) + x0;
  B0(2) = c0 * fabs(Az);
}


/*******************************************************************
* Compute V0
****************************************************************** */
void computeV0(const Vector& x, Vector& V0)
{
  V0(0) = 0.0;
  V0(1) = 0.0;
  V0(2) = 0.0;
}


/*******************************************************************
* OBSOLETE. Compute tensor T in T w = (w x B0) or its transpose
****************************************************************** */
void computeCrossB0(const Vector& x, DenseMatrix& crossB0)
{
   crossB0.SetSize(3);
   Vector B0(3);

   computeB0(x, B0);
   
   crossB0(0, 1) = B0(2);
   crossB0(0, 2) =-B0(1);
   crossB0(1, 0) =-B0(2);
   crossB0(1, 2) = B0(0);
   crossB0(2, 0) = B0(1);
   crossB0(2, 1) =-B0(0);
}

void computeCrossB0transpose(const Vector& x, DenseMatrix& crossB0)
{
   crossB0.SetSize(3);
   Vector B0(3);

   computeB0(x, B0);
   
   crossB0(0, 1) =-B0(2);
   crossB0(0, 2) = B0(1);
   crossB0(1, 0) = B0(2);
   crossB0(1, 2) =-B0(0);
   crossB0(2, 0) =-B0(1);
   crossB0(2, 1) = B0(0);
}


/*******************************************************************
* OBSOLETE. Compute Omega = |B0|^2 I - B0 B0^T
****************************************************************** */
void computeOmega(const Vector& x, DenseMatrix& omega)
{
  omega.SetSize(3);
  Vector B0(3);

  computeB0(x, B0);

  double norm2 = B0.Norml2();
  norm2 *= norm2;
  
  double tol = norm2 * 0.00; // % of B0-norml
   
  omega(0, 0) = norm2 - B0(0) * B0(0) + tol; 
  omega(1, 1) = norm2 - B0(1) * B0(1) + tol;
  omega(2, 2) = norm2 - B0(2) * B0(2) + tol;
  omega(0, 1) =-B0(0) * B0(1);
  omega(0, 2) =-B0(0) * B0(2);
  omega(1, 2) =-B0(1) * B0(2);

  // symmetrize
  omega(1, 0) = omega(0, 1);
  omega(2, 0) = omega(0, 2);
  omega(2, 1) = omega(1, 2);
}
