//                      Example 41 - parallel version
//
// clang-format off
// Sample runs:  mpirun -np 4 ex41p -dim 2 -p 1 -res 160 -o 1 -dt 0.025 --glvis
//               mpirun -np 4 ex41p -dim 2 -p 1 -res 40 -o 3 -dt 0.025 --glvis
//               mpirun -np 4 ex41p -dim 3 -p 1 -res 40 -o 1 -dt 0.025 --glvis
// clang-format on
//
// Description:
//
// This example implements a poroelasticity problem in 2D. The problem is
// described by a system of PDEs that involve the displacement field u, the
// Darcy velocity field z, and the pressure field p. Additionally, the problem
// is formulated with the total stress tensor σ and the total strain
// displacement ε. These two are given by
// (1) σ = λ tr(ε) I + 2με - αpI   (I is the identity)
// (2) ε = (∇u + ∇u^T)/2
// where λ and μ are the Lame constants, α is the Biot coefficient, and p is the
// pressure field. The problem is defined on the domain D = (0,1)^2 and the
// equations read
// (3) - ∇·σ = f in D x (0,T]
// (4) z = -K ∇p in D x (0,T]
// (5) ∂/∂t( c0 p + α ∇·u ) + ∇·z = 0 in D x (0,T]
// The boundary conditions are
// (6) u = 0 on ∂D x (0,T]
// (7) p = 0 on ∂D x (0,T]
//
// The problem is discretized using the mixed finite element method. The
// displacement field u is in the (H1)^d space, the Darcy velocity field z is in
// the H(div) space, and the pressure field p is in the L2 space. The mixed
// method is implemented using the BlockVector and BlockOperator classes. The
// time discretization is done using the backward Euler method. The
// discretization leads to a symmetric system equation that is solved using
// a preconditioned MINRES solver. It reads
// clang-format off
// ⌈ A00    0     A20^T ⌉ ⌈ u(n+1) ⌉     ⌈  0  0   0  ⌉ ⌈ u(n) ⌉   ⌈ b0(n+1) ⌉
// |  0    A11    A21^T | | z(n+1) |  =  |  0  0   0  | | z(n) | + |   0     |
// ⌊ A20   A21   -A22^T ⌋ ⌊ p(n+1) ⌋     ⌊ A20 0 -A22 ⌋ ⌊ p(n) ⌋   ⌊ b2(n+1) ⌋
// clang-format on
// where with a slight abuse of notation we use z = dt/α z and p = - p/α
// (rescaling). Let us define the basis fuctions φ_i, ψ_i, and χ_i for (H1)^2,
// H(div), and L2, respectively. Then the matrices are defined as follows:
// A00_ij = (σ(φ_j):ε(φ_i))            where φ_i and φ_j are in (H1)^2
// A11_ij = (α^2 /dt K^(-1)ψ_j, ψ_i )  where ψ_i and ψ_j are in H(div)
// A22_ij = (c0/α^2 χ_j, χ_i)          where χ_i and χ_j are in L2
// A20_ij = (∇·φ_j, χ_i)               where φ_j is in (H1)^2 and χ_i is in L2
// A21_ij = (∇·ψ_j, χ_i)               where ψ_j is in H(div) and χ_i is in L2
// b0_i = (f, φ_i)                     where φ_i is in (H1)^2
// b2_i = (dt * s / α, χ_i)            where χ_i is in L2
// (·,·) denotes the L2 inner product and subject to modifications due to
// boundaries.
//
// This implements a standard three-field poroelasticity problem with the
// example taken from
// "Wheeler, M., Xue, G. & Yotov, I. Coupling multipoint flux mixed finite
// element methods with continuous Galerkin methods for poroelasticity.
// Comput Geosci 18, 57–75 (2014). https://doi.org/10.1007/s10596-013-9382-y "
//

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include "ex41.hpp"
#include "mfem.hpp"

using namespace std;
using namespace mfem;

// ----------------------------------------------------------------------
// Helper functions
// ----------------------------------------------------------------------

/// @brief Find the maximum absolute value in a vector.
real_t FindMax(std::vector<real_t> &vec);

/// @brief Compute the L2 inner product of a vector with itself.
real_t Computel2InnerProduct(std::vector<real_t> &vec);

/// @brief Visualize the deformed mesh and the field in GLVis.
void visualize(ostream &os, ParMesh *mesh, ParGridFunction *field,
               const char *field_name = NULL, bool init_vis = false,
               int offset = 0);

int main(int argc, char *argv[]) {
  /// ----------------------------------------------------------------------
  // 1. Initialize MPI and HYPRE.
  /// ----------------------------------------------------------------------

  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();
  const bool verbose =
      (myid == 0);  // Short variable for which process is printing

  /// ----------------------------------------------------------------------
  // 2. Parse command-line options.
  /// ----------------------------------------------------------------------

  // Define simulation parameters
  int order = 1;        // Polynomial order of the finite element space
  int resolution = 20;  // Resolution of the mesh (1/h)
  const char *device_config = "cpu";  // Device configuration string
  bool visualization = true;          // Enable or disable visualization
  double dt = 1e-2;    // Time step size for the backward Euler method
  double T = 0.5;      // Final time for the backward Euler method
  int problem = 1;     // Integer to define problem 1 or 2 (different RHS)
  int dim = 2;         // Dimension of the problem (2D or 3D)
  double alpha = 0.1;  // Biot coefficient
  double nu = 0.2;     // Poisson's ratio
  double E = 1.0;      // Young's modulus
  double c0 = 0.1;     // Storage coefficient
  bool compute_error = true;  // Flag to compute error
  bool paraview = false;      // Flag to enable Paraview output
  bool glvis = false;         // Flag to enable GLVis output

  // Parse command-line options
  OptionsParser args(argc, argv);
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&resolution, "-res", "--resolution",
                 "Resolution of the mesh. (1/h)");
  args.AddOption(&device_config, "-d", "--device",
                 "Device configuration string, see Device::Configure().");
  args.AddOption(&dt, "-dt", "--time-step",
                 "Time step size for the backward Euler method.");
  args.AddOption(&T, "-T", "--final-time",
                 "Final time for the backward Euler method.");
  args.AddOption(&problem, "-p", "--problem",
                 "Problem to solve: 0 = P1, 1 = P2.");
  args.AddOption(&dim, "-dim", "--dimension",
                 "Dimension of the problem: 2 or 3.");
  args.AddOption(&alpha, "-alpha", "--biot-coefficient", "Biot coefficient.");
  args.AddOption(&nu, "-nu", "--poisson-ratio", "Poisson's ratio.");
  args.AddOption(&E, "-E", "--young-modulus", "Young's modulus.");
  args.AddOption(&c0, "-c0", "--storage-coefficient", "Storage coefficient.");
  args.AddOption(&compute_error, "-error", "--compute-error", "-no-error",
                 "--no-compute-error", "Enable or disable error computation.");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Enable Paraview output for visualization.");
  args.AddOption(&glvis, "-gl", "--glvis", "-no-gl", "--no-glvis",
                 "Enable GLVis output for visualization.");
  args.Parse();

  // Check if the command-line options are valid
  if (!args.Good()) {
    if (verbose) {
      args.PrintUsage(cout);
    }
    return 1;
  }
  if (verbose) {
    args.PrintOptions(cout);
  }
  if (problem != 1 && problem != 2) {
    std::cerr << "Error: Problem must be 1 or 2." << std::endl;
    return 1;
  }

  if (paraview || glvis) {
    visualization = true;  // Enable visualization if Paraview or GLVis is
                           // requested
    if (verbose) {
      std::cout << "Visualization enabled." << std::endl;
    }
  }

  /// ----------------------------------------------------------------------
  // 3. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  /// ----------------------------------------------------------------------
  Device device(device_config);
  if (myid == 0) {
    device.Print();
  }

  /// ----------------------------------------------------------------------
  /// 4. Define Domain (0,1)^d with triangular / tetrahedron mesh
  /// ----------------------------------------------------------------------
  ParMesh *pmesh{nullptr};
  if (dim == 2) {
    Mesh mesh = Mesh::MakeCartesian2D(resolution, resolution, Element::TRIANGLE,
                                      true, 1.0, 1.0);
    pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
  } else if (dim == 3) {
    Mesh mesh =
        Mesh::MakeCartesian3D(resolution, resolution, resolution,
                              Element::TETRAHEDRON, true, 1.0, 1.0, 1.0);
    pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
  } else {
    std::cerr << "Error: Dimension not supported." << std::endl;
    return 1;
  }

  /// ----------------------------------------------------------------------
  /// 5. Define Finite Element Spaces
  /// ----------------------------------------------------------------------

  // Appropriate finite elements for the displacement field u, the Darcy
  // velocity field z, and the pressure field p (see Wheeler et al. 2014)
  H1_FECollection h1_coll(order, dim);
  RT_FECollection hdiv_coll(order - 1, dim);
  L2_FECollection l2_coll(order - 1, dim);

  // Create the corresponding FEM spaces
  ParFiniteElementSpace V_space(pmesh, &h1_coll, dim);
  ParFiniteElementSpace Z_space(pmesh, &hdiv_coll);
  ParFiniteElementSpace W_space(pmesh, &l2_coll);

  // Print the number of degrees of freedom
  auto dimV = V_space.GlobalTrueVSize();
  auto dimZ = Z_space.GlobalTrueVSize();
  auto dimW = W_space.GlobalTrueVSize();
  if (verbose) {
    std::cout
        << "***********************************************************\n";
    std::cout << "dim(V) = " << dimV << "\n";
    std::cout << "dim(Z) = " << dimZ << "\n";
    std::cout << "dim(W) = " << dimW << "\n";
    std::cout << "dim(V+Z+W) = " << dimV + dimZ + dimW << "\n";
    std::cout
        << "***********************************************************\n";
  }

  // Define essential boundary conditions (needed for homogeneous dirichlet
  // boundary conditions on the displacement field u and pressure field p)
  Array<int> empty;  // empty array for the essential boundary conditions
  Array<int> ess_tdof_list_h1;
  Array<int> ess_tdof_list_l2;
  if (pmesh->bdr_attributes.Size()) {
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 1;
    V_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_h1);
    W_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_l2);
  }

  /// ----------------------------------------------------------------------
  /// 6. Define Block structure for FEM & true dofs
  /// ----------------------------------------------------------------------

  Array<int> offsets(4);  // number of variables + 1
  offsets[0] = 0;
  offsets[1] = V_space.GetVSize();
  offsets[2] = Z_space.GetVSize();
  offsets[3] = W_space.GetVSize();
  offsets.PartialSum();

  Array<int> toffsets(4);  // number of variables + 1
  toffsets[0] = 0;
  toffsets[1] = V_space.TrueVSize();
  toffsets[2] = Z_space.TrueVSize();
  toffsets[3] = W_space.TrueVSize();
  toffsets.PartialSum();

  // Define BlockVectors for the solution and rhs (x, tx, rhs, trhs)
  auto mem_type = device.GetMemoryType();
  BlockVector x(offsets, mem_type);
  BlockVector tx(toffsets, mem_type);
  BlockVector rhs(offsets, mem_type);
  BlockVector trhs(toffsets, mem_type);
  x = 0.0;
  tx = 0.0;
  rhs = 0.0;
  trhs = 0.0;

  // Define the BlockOperator for the Biot operator
  BlockOperator biot_operator(toffsets);

  /// ----------------------------------------------------------------------
  /// 7. Define the Coefficients of the problem
  /// ----------------------------------------------------------------------

  PoroelasticityReferenceSolution ref_sol(dim, alpha, nu, E, c0, problem);

  // Coefficient for the volume force f(x,t) in the displacement equation
  auto VolumeForce = [&ref_sol](const Vector &x, real_t t, Vector &u) {
    ref_sol.VolumeForce(x, t, u);
  };
  VectorFunctionCoefficient force_coeff(dim, VolumeForce);

  // Coefficient for the source term s(x,t) in the pressure equation
  auto SourcesAndSinks = [&ref_sol](const Vector &x, real_t t) {
    return ref_sol.SourceTerm(x, t);
  };
  FunctionCoefficient source_coeff(SourcesAndSinks);

  // Solutions
  auto AnalyticDisplacementSolution = [&ref_sol](const Vector &x, real_t t,
                                                 Vector &u) {
    ref_sol.AnalyticDisplacementSolution(x, t, u);
  };
  VectorFunctionCoefficient c_u_solution(dim, AnalyticDisplacementSolution);

  auto AnalyticDarcySolution = [&ref_sol](const Vector &x, real_t t,
                                          Vector &u) {
    ref_sol.AnalyticDarcySolution(x, t, u);
  };
  VectorFunctionCoefficient c_z_solution(dim, AnalyticDarcySolution);

  auto AnalyticPressureSolution = [&ref_sol](const Vector &x, real_t t) {
    return ref_sol.AnalyticPressureSolution(x, t);
  };
  FunctionCoefficient c_p_solution(AnalyticPressureSolution);

  if (verbose) {
    std::cout << "Problem " << problem << " selected.\n";
    if (problem == 1 && alpha != 0.1 && c0 != 0.1) {
      std::cerr
          << "Warning: alpha is not set to 0.1 or c0 is not set to 0.1 for "
             "problem 1."
          << std::endl;
    } else if (problem == 2 && alpha != 10.0 && c0 != 100.0) {
      std::cerr << "Warning: alpha is not set to 10.0 for problem 2.\n"
                << std::endl;
    }
  }

  // Set time to dt for all coefficients
  force_coeff.SetTime(dt);
  source_coeff.SetTime(dt);

  /// ----------------------------------------------------------------------
  /// 8. Define & assemble (bi)linear forms and the corresponding true dofs
  /// ----------------------------------------------------------------------

  ParBilinearForm a00(&V_space);
  ParBilinearForm a11(&Z_space);
  ParBilinearForm a22(&W_space);
  ParMixedBilinearForm a20(&V_space, &W_space);
  ParMixedBilinearForm a21(&Z_space, &W_space);

  // H1 space -- linear elasticity
  ConstantCoefficient c_lambda(E * nu / ((1 + nu) * (1 - 2 * nu)));
  ConstantCoefficient c_mu(E / (2 * (1 + nu)));
  a00.AddDomainIntegrator(new ElasticityIntegrator(c_lambda, c_mu));
  a00.Assemble();
  HypreParMatrix A00;
  a00.FormLinearSystem(ess_tdof_list_h1, x.GetBlock(0), rhs.GetBlock(0), A00,
                       tx.GetBlock(0), trhs.GetBlock(0));

  // H(div) space -- Darcy velocity mass matrix
  // Note: you may adjust the matrix K but then the reference solution will
  // no longer be valid and the computed errors are wrong.
  DenseMatrix K(dim);
  K(0, 0) = 3;
  K(0, 1) = 1;
  K(1, 0) = 1;
  K(1, 1) = 2;
  if (dim == 3) {
    K(2, 2) = 1.0;
  }
  K.Invert();
  K *= (alpha * alpha) / dt;
  MatrixConstantCoefficient K_coeff(K);
  a11.AddDomainIntegrator(new VectorFEMassIntegrator(K_coeff));
  a11.Assemble();
  a11.Finalize();
  HypreParMatrix A11;
  a11.FormSystemMatrix(empty, A11);

  // L2 space -- pressure mass matrix
  ConstantCoefficient c_c0(c0 / (alpha * alpha));
  a22.AddDomainIntegrator(new MassIntegrator(c_c0));
  a22.Assemble();
  HypreParMatrix A22;
  a22.FormLinearSystem(ess_tdof_list_l2, x.GetBlock(2), rhs.GetBlock(2), A22,
                       tx.GetBlock(2), trhs.GetBlock(2));

  // ((H1)^2, L2) space -- divergence of displacement
  a20.AddDomainIntegrator(new VectorDivergenceIntegrator);
  a20.Assemble();
  HypreParMatrix A20;
  a20.FormRectangularLinearSystem(ess_tdof_list_h1, ess_tdof_list_l2,
                                  x.GetBlock(0), rhs.GetBlock(2), A20,
                                  tx.GetBlock(0), trhs.GetBlock(2));
  auto *A02 = A20.Transpose();  // A02 = A20^T (second block)

  // ((H(div)), L2) space -- divergence of Darcy velocity
  a21.AddDomainIntegrator(new MixedScalarDivergenceIntegrator);
  a21.Assemble();
  HypreParMatrix A21;
  a21.FormRectangularLinearSystem(empty, ess_tdof_list_l2, x.GetBlock(1),
                                  rhs.GetBlock(2), A21, tx.GetBlock(1),
                                  trhs.GetBlock(2));
  auto *A12 = A21.Transpose();  // A12 = A21^T (scend block)

  // Fill the block operator
  biot_operator.SetBlock(0, 0, &A00);
  biot_operator.SetBlock(1, 1, &A11);
  biot_operator.SetBlock(2, 2, &A22, -1.0);
  biot_operator.SetBlock(2, 0, &A20);
  biot_operator.SetBlock(2, 1, &A21);
  biot_operator.SetBlock(0, 2, A02);
  biot_operator.SetBlock(1, 2, A12);

  /// ----------------------------------------------------------------------
  /// 9. Construct the operators for preconditioner
  /// TODO: improve preconditioner
  /// ----------------------------------------------------------------------

  // HypreBoomerAMG for the displacement field (elasticity)
  HypreBoomerAMG P00(A00);
  P00.SetPrintLevel(0);
  P00.iterative_mode = false;

  // Symmetric Gauss-Seidel to approximate the inverse of the
  // pressure Schur Complement for Darcy velocity & pressure
  HypreParVector A11_d(MPI_COMM_WORLD, A11.GetGlobalNumRows(),
                       A11.GetRowStarts());
  A11.GetDiag(A11_d);
  auto *MinvBt = A21.Transpose();
  MinvBt->InvScaleRows(A11_d);
  auto *S = ParMult(&A21, MinvBt);

  HypreDiagScale P11(A11);
  HypreBoomerAMG P22(*S);
  P11.iterative_mode = false;
  P22.iterative_mode = false;
  P22.SetPrintLevel(0);

  // Block diagonal preconditioner
  BlockDiagonalPreconditioner prec(toffsets);
  prec.SetDiagonalBlock(0, &P00);
  prec.SetDiagonalBlock(1, &P11);
  prec.SetDiagonalBlock(2, &P22);

  /// ----------------------------------------------------------------------
  /// 10. Setup the visualization
  /// ----------------------------------------------------------------------

  // Define fields for visualization
  ParGridFunction u;
  ParGridFunction z;
  ParGridFunction p;
  u.MakeRef(&V_space, x.GetBlock(0), 0);
  z.MakeRef(&Z_space, x.GetBlock(1), 0);
  p.MakeRef(&W_space, x.GetBlock(2), 0);
  u = 0.0;
  z = 0.0;
  p = 0.0;
  ParGridFunction f_applied(&V_space);
  ParGridFunction s_applied(&W_space);

  // Prepare the data collection
  ParaViewDataCollection paraview_dc("PoroElasticity", pmesh);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(order);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  // Register the fields
  paraview_dc.RegisterField("displacement", &u);
  paraview_dc.RegisterField("darcy", &z);
  paraview_dc.RegisterField("pressure", &p);
  paraview_dc.RegisterField("force", &f_applied);
  paraview_dc.RegisterField("source", &s_applied);

  // TODO: GLVis export is still somewhat buggy and for unknown reason
  // sometimes fails here, particulary in the 2D case. For whatever reason,
  // first running the 3D example and then the 2D example seems to bypass this.
  socketstream vis_v;
  socketstream vis_z;
  socketstream vis_w;
  if (glvis) {
    // GLVis standard visualization setup
    char vishost[] = "localhost";
    int visport = 19916;

    // Displacement
    vis_v.open(vishost, visport);
    vis_v.precision(8);
    visualize(vis_v, pmesh, &u, "Displacement", true, 0);
    MPI_Barrier(pmesh->GetComm());

    // Darcy velocity
    vis_z.open(vishost, visport);
    vis_z.precision(8);
    visualize(vis_z, pmesh, &z, "Darcy Velocity", true, 1);
    MPI_Barrier(pmesh->GetComm());

    // Pressure
    vis_w.open(vishost, visport);
    vis_w.precision(8);
    visualize(vis_w, pmesh, &p, "Pressure", true, 2);
    MPI_Barrier(pmesh->GetComm());

    if (verbose) {
      std::cout << "GLVis visualization enabled." << std::endl;
      // Wait for 2 seconds to allow GLVis to connect
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  }

  // ----------------------------------------------------------------------
  // 11. Integrate the equations in time using the backward Euler method
  // ----------------------------------------------------------------------

  // 11.1 Objects for (manual) time integration
  real_t t{0.0};

  // Get stopwatch for timing
  StopWatch timeintegration_measurement;
  timeintegration_measurement.Clear();
  timeintegration_measurement.Start();

  // Linear solver parameter
  int maxIter = 50000;
  real_t rtol(1.e-9);
  real_t atol(1.e-12);

  // 11.2 Setup MINRES to solve the symmetric linear system
  MINRESSolver solver(MPI_COMM_WORLD);
  solver.SetAbsTol(atol);
  solver.SetRelTol(rtol);
  solver.SetMaxIter(maxIter);
  solver.SetOperator(biot_operator);
  solver.SetPreconditioner(prec);
  solver.SetPrintLevel(3);

  // Time keeping
  tx = 0.0;
  int cntr = 0;

  // Error tracking
  std::vector<real_t> err_u;
  std::vector<real_t> err_z;
  std::vector<real_t> err_p;

  // Vector for loop updates capturing the old solution on the right-hand side
  // of the pressure equation
  Vector p_rhs_history;
  p_rhs_history.SetSize(W_space.TrueVSize());

  // 11.3 Time integration with backward Euler method
  while (t < T) {
    // Update the time
    t += dt;
    if (verbose) {
      std::cout << "\nTime: " << t << std::endl;
    }

    // Start timer
    StopWatch chrono_step;
    chrono_step.Clear();
    chrono_step.Start();

    // Construct the right-hand side for the elasticity equation
    ParLinearForm b0;
    force_coeff.SetTime(t);
    b0.Update(&V_space, rhs.GetBlock(0), 0);
    b0.AddDomainIntegrator(new VectorDomainLFIntegrator(force_coeff));
    b0.Assemble();

    // Construct the right-hand side for the pressure equation
    ParLinearForm b2;
    source_coeff.SetTime(t);
    b2.Update(&W_space, rhs.GetBlock(2), 0);
    b2.AddDomainIntegrator(new DomainLFIntegrator(source_coeff));
    b2.Assemble();

    // Capture the influence of the previous time step on the pressure equation
    // Pt. 1/2
    A20.Mult(tx.GetBlock(0), p_rhs_history);
    A22.AddMult(tx.GetBlock(2), p_rhs_history, -1.0);

    // Enforce boundary conditions (TODO: improve this)
    a00.FormLinearSystem(ess_tdof_list_h1, x.GetBlock(0), rhs.GetBlock(0), A00,
                         tx.GetBlock(0), trhs.GetBlock(0), true);
    a22.FormLinearSystem(ess_tdof_list_l2, x.GetBlock(2), rhs.GetBlock(2), A22,
                         tx.GetBlock(2), trhs.GetBlock(2), true);

    // Capture the influence of the previous time step on the pressure equation
    // Pt. 2/2
    trhs.GetBlock(2) *= (dt / alpha);
    trhs.GetBlock(2) += p_rhs_history;
    p_rhs_history = 0.0;

    // Print the time for the linear forms & rhs; reset the timer
    chrono_step.Stop();
    if (verbose) {
      std::cout << "Time for right-hand side: " << chrono_step.RealTime()
                << "s. \n";
    }
    chrono_step.Clear();

    // Solve
    chrono_step.Start();
    solver.Mult(trhs, tx);
    chrono_step.Stop();
    trhs = 0.0;  // Reset trhs
    rhs = 0.0;   // Reset rhs

    if (verbose) {
      if (!solver.GetConverged()) {
        std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                  << " iterations. Residual norm is " << solver.GetFinalNorm()
                  << ".\n";
        return 1;
      } else {
        std::cout << "MINRES converged in " << solver.GetNumIterations()
                  << " iterations. Residual norm is " << solver.GetFinalNorm()
                  << ".\n";
        std::cout << "Time for linear solver: " << chrono_step.RealTime()
                  << "s. \n";
      }
    }

    // Update the solution
    u.Distribute(&(tx.GetBlock(0)));
    z.Distribute(&(tx.GetBlock(1)));
    p.Distribute(&(tx.GetBlock(2)));

    // Transform solution to the original space -- the poroelasticity
    // formulation is rescaled to achieve a symmetric system
    // (see the description above)
    z *= (alpha / dt);
    p *= (-1.0 / alpha);

    // Update the coefficients
    c_u_solution.SetTime(t);
    c_z_solution.SetTime(t);
    c_p_solution.SetTime(t);

    // Save paraview data
    if (paraview) {
      // Project the coefficients to the grid functions
      s_applied.ProjectCoefficient(source_coeff);
      f_applied.ProjectCoefficient(force_coeff);
      paraview_dc.SetCycle(cntr);
      paraview_dc.SetTime(t);
      paraview_dc.Save();
    }

    // Save GLVis data
    if (glvis) {
      visualize(vis_v, pmesh, &u);
      visualize(vis_z, pmesh, &z);
      visualize(vis_w, pmesh, &p);
    }

    if (compute_error) {
      // Compute the error (note: the error is computed in the original space)
      err_u.push_back(u.ComputeL2Error(c_u_solution));
      err_z.push_back(z.ComputeL2Error(c_z_solution));
      err_p.push_back(p.ComputeL2Error(c_p_solution));

      // Print the error
      if (verbose) {
        std::cout << "errors (t = " << t
                  << ") : || u_h - u_ex ||_(max) = " << err_u.back();
        std::cout << "; || z_h - z_ex ||_(l2) = " << err_z.back();
        std::cout << "; || p_h - p_ex ||_(max) = " << err_p.back() << "\n";
      }
    }

    // Increment the counter
    cntr++;
  }

  if (device.IsEnabled()) {
    tx.HostRead();
  }

  // Stop the timer
  timeintegration_measurement.Stop();

  // Print the results
  if (verbose) {
    std::cout << "Time integration took "
              << timeintegration_measurement.RealTime() << "s. \n";
  }

  /// ----------------------------------------------------------------------
  /// 12. Compute final error
  /// ----------------------------------------------------------------------
  if (compute_error) {
    real_t u_error = FindMax(err_u);
    real_t z_error = Computel2InnerProduct(err_z);
    real_t p_error = FindMax(err_p);

    if (verbose) {
      std::cout << "Final errors: || u_h - u_ex ||_(max) = " << u_error;
      std::cout << "; || z_h - z_ex ||_(l2) = " << z_error;
      std::cout << "; || p_h - p_ex ||_(max) = " << p_error << "\n";
    }
  }

  // Free the used memory.
  delete pmesh;
  delete A02;  // Transpose operator allocates with new
  delete A12;  // Transpose operator allocates with new
  delete MinvBt;
  delete S;

  return 0;
}

// ----------------------------------------------------------------------
// Helper functions
// ----------------------------------------------------------------------
real_t FindMax(std::vector<real_t> &vec) {
  if (vec.empty()) {
    return std::numeric_limits<real_t>::quiet_NaN();
  } else {
    auto it = std::max_element(vec.begin(), vec.end(), [](double a, double b) {
      return std::abs(a) < std::abs(b);
    });
    return std::abs(*it);
  }
};

real_t Computel2InnerProduct(std::vector<real_t> &vec) {
  if (vec.empty()) {
    return std::numeric_limits<real_t>::quiet_NaN();
  } else {
    auto it = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    return std::sqrt(it / vec.size());
  }
};

void visualize(ostream &os, ParMesh *mesh, ParGridFunction *field,
               const char *field_name, bool init_vis, int offset) {
  if (!os) {
    return;
  }

  os << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
  os << "solution\n" << *mesh << *field;

  if (init_vis) {
    offset *= 403;
    offset += 50;
    os << "window_geometry " << offset << " 20 400 350\n";
    os << "window_title '" << field_name << "'\n";
    if (mesh->SpaceDimension() == 2) {
      os << "view 0 0\n";  // view from top
      os << "keys jl\n";   // turn off perspective and light
      os << "keys c\n";    // show colorbar and mesh
    } else {
      os << "keys cm\n";  // show colorbar and mesh
    }
    // os << "keys c\n";           // show colorbar
    os << "autoscale value\n";  // update value-range; keep mesh-extents
  }
  os << flush;
  // MPI barrier to ensure all processes have sent their data
  MPI_Barrier(mesh->GetComm());
}