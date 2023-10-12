//                                MFEM Example 98 seismic
//
// Compile with: make ex98_test_seismic (modified from Example 23)
//
// Sample runs:  ex98_test_seismic
//               ex98_test_seismic -o 4 -tf 5
//               ex98_test_seismic -m ../data/square-disc.mesh -o 2 -tf 2 --neumann
//               ex98_test_seismic -m ../data/disc-nurbs.mesh -r 3 -o 4 -tf 2
//               ex98_test_seismic -m ../data/inline-hex.mesh -o 1 -tf 2 --neumann
//               ex98_test_seismic -m ../data/inline-tet.mesh -o 1 -tf 2 --neumann
//
// Device sample runs:
//               ex98_test_seismic -m ../data/inline-quad.mesh -o 3 -p 1 -pa -d cuda
//               ex98_test_seismic -m ../data/inline-hex.mesh -o 2 -p 2 -pa -d cuda
//               ex98_test_seismic -m ../data/star.mesh -r 1 -o 2 -sigma 10.0 -pa -d cuda
//
// Description:  This example solves the wave equation problem of the form:
//
//                               d^2u/dt^2 = c^2 \Delta u.
//
//               The example demonstrates the use of time dependent operators,
//               implicit solvers and second order time integration.
//
//               We recommend viewing examples 9 and 10 before viewing this
//               example.

// add MKL Pardiso solver from ex3
// added the anisotropic material from ex31
// added the point source
// added mesh from Palace (meshio.hpp)
// add Laplace B.C. from ex27

#include "mfem.hpp"
#include "mkl.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <array>
#include <limits>
#include <map>
#include <filesystem>
#include <thread>
#include <chrono>
#include <cstdlib> // for env

using namespace std;
using namespace mfem;

const auto MSH_FLT_PRECISION = std::numeric_limits<double>::max_digits10;

class WaveOperator : public SecondOrderTimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   BilinearForm *M;
   BilinearForm *K;

   SparseMatrix Mmat, Kmat, Kmat0;
   SparseMatrix *T; // T = M + dt K
   double current_dt;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver T_solver; // Implicit solver for T = M + fac0*K
   DSmoother T_prec;  // Preconditioner for the implicit solver

   Coefficient *c2;
   mutable Vector z; // auxiliary vector

public:
   WaveOperator(FiniteElementSpace &f, Array<int> &ess_bdr,double speed);

   using SecondOrderTimeDependentOperator::Mult;
   virtual void Mult(const Vector &u, const Vector &du_dt,
                     Vector &d2udt2) const;

   /** Solve the Backward-Euler equation:
       d2udt2 = f(u + fac0*d2udt2,dudt + fac1*d2udt2, t),
       for the unknown d2udt2. */
   using SecondOrderTimeDependentOperator::ImplicitSolve;
   virtual void ImplicitSolve(const double fac0, const double fac1,
                              const Vector &u, const Vector &dudt, Vector &d2udt2);

   ///
   void SetParameters(const Vector &u);

   virtual ~WaveOperator();
};


WaveOperator::WaveOperator(FiniteElementSpace &f,
                           Array<int> &ess_bdr, double speed)
   : SecondOrderTimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL),
     K(NULL),
     T(NULL), current_dt(0.0), z(height)
{
   const double rel_tol = 1e-8;

   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   c2 = new ConstantCoefficient(speed*speed);

   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(*c2));
   K->Assemble();

   Array<int> dummy;
   K->FormSystemMatrix(dummy, Kmat0);
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(100);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);

   T = NULL;
}

void WaveOperator::Mult(const Vector &u, const Vector &du_dt,
                        Vector &d2udt2)  const
{
   // Compute:
   //    d2udt2 = M^{-1}*-K(u)
   // for d2udt2
   Kmat.Mult(u, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, d2udt2);
}

void WaveOperator::ImplicitSolve(const double fac0, const double fac1,
                                 const Vector &u, const Vector &dudt, Vector &d2udt2)
{
   // Solve the equation:
   //    d2udt2 = M^{-1}*[-K(u + fac0*d2udt2)]
   // for d2udt2
   if (!T)
   {
      T = Add(1.0, Mmat, fac0, Kmat);
      T_solver.SetOperator(*T);
   }
   Kmat0.Mult(u, z);
   z.Neg();

   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      z[ess_tdof_list[i]] = 0.0;
   }
   T_solver.Mult(z, d2udt2);
}

void WaveOperator::SetParameters(const Vector &u)
{
   delete T;
   T = NULL; // re-compute T on the next ImplicitSolve
}

WaveOperator::~WaveOperator()
{
   delete T;
   delete M;
   delete K;
   delete c2;
}

Mesh* LoadMeshNew(const std::string& path);

void source(const Vector& x, Vector& f);

void PrintMatrixConstantCoefficient(mfem::MatrixConstantCoefficient& coeff);

/// Convert a set of attribute numbers to a marker array
/** The marker array will be of size max_attr and it will contain only zeroes
    and ones. Ones indicate which attribute numbers are present in the attrs
    array. In the special case when attrs has a single entry equal to -1 the
    marker array will contain all ones. */
void AttrToMarker(int max_attr, const Array<int>& attrs, Array<int>& marker);

void PrintArray2D(const Array2D<double>& arr);

double InitialSolution(const Vector &x)
{
   return exp(-x.Norml2()*x.Norml2()*30);
}

double InitialRate(const Vector &x)
{
   return 0.0;
}

int dim;

int main(int argc, char* argv[])
{
    std::cout << mkl_get_max_threads() << std::endl; // in VS2022, check properties->intel library for OneAPI->Use oneMKL (Parallel)
    mkl_set_dynamic(0);
    mkl_set_num_threads(30);
    std::cout << mkl_get_max_threads() << std::endl; // in VS2022, check properties->intel library for OneAPI->Use oneMKL (Parallel)

    int result = 0;
    #if defined(_WIN32) || defined(_WIN64)
        result = _putenv("MKL_PARDISO_OOC_MAX_CORE_SIZE=60000");
    #elif defined(__unix__) || defined(__APPLE__)
        result = setenv("MKL_PARDISO_OOC_MAX_CORE_SIZE", "60000", 1);
    #else
        #error "Unknown compiler";
    #endif

    if (result != 0) {
        std::cerr << "Failed to set environment variable." << std::endl;
        return 2;
    }
    else {
        std::cout << "MKL_PARDISO_OOC_MAX_CORE_SIZE=" << getenv("MKL_PARDISO_OOC_MAX_CORE_SIZE") << std::endl;
    }

    // 1. Parse command-line options.
    //const char* mesh_file = "../data/em_sphere_mfem_ex0_coarse.mphtxt";
    //const char* mesh_file = "../data/em_sphere_mfem_ex0.mphtxt";
    //const char* mesh_file = "../data/simple_cube.mphtxt";
    //const char* mesh_file = "../data/cube_comsol_pml.mphtxt";
     //const char* mesh_file = "../data/cube_comsol_coarse.mphtxt";
    //const char* mesh_file = "../data/cube_comsol_ex_coarse.mphtxt";
    const char* mesh_file = "../data/comsol_submarine_acoustic.mphtxt";
    //const char* mesh_file = "../data/inline-tet.mesh";
    int ref_levels = 2;
    int order = 2;
    int ode_solver_type = 10;
    double t_final = 0.5;
    double dt = 1.0e-2;
    double speed = 1.0;
    bool visualization = true;
    bool visit = true;
    bool dirichlet = true;
    int vis_steps = 5;

    int logging_ = 1;
    bool comp_solver = true;
    int bprint = 1;

    std::vector<int> values = { 1, 2, 4, 5, 6, 9, 11, 13, 16, 21, 22, 28, 33, 59, 60, 66, 71, 73, 74, 75, 76, 77 };
    Array<int> abcs(values.data(), values.size());
    Array<int> dbcs;
    std::vector<int> values_pml = { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15 };
    Array<int> pmls(values_pml.data(), values_pml.size());

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "Mesh file to use.");
    args.AddOption(&ref_levels, "-r", "--refine",
                    "Number of times to refine the mesh uniformly.");
    args.AddOption(&order, "-o", "--order",
                    "Order (degree) of the finite elements.");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                    "ODE solver: [0--10] - GeneralizedAlpha(0.1 * s),\n\t"
                    "\t   11 - Average Acceleration, 12 - Linear Acceleration\n"
                    "\t   13 - CentralDifference, 14 - FoxGoodwin");
    args.AddOption(&t_final, "-tf", "--t-final",
                    "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
                    "Time step.");
    args.AddOption(&speed, "-c", "--speed",
                    "Wave speed.");
    args.AddOption(&dirichlet, "-dir", "--dirichlet", "-neu",
                    "--neumann",
                    "BC switch.");
    args.AddOption(&ref_dir, "-r", "--ref",
                    "Reference directory for checking final solution.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                    "--no-visualization",
                    "Enable or disable GLVis visualization.");
    args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                    "--no-visit-datafiles",
                    "Save data files for VisIt (visit.llnl.gov) visualization.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                    "Visualize every n-th timestep.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    ComplexOperator::Convention conv =
        herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

    // 2. Enable hardware devices such as GPUs, and programming models such as
    //    CUDA, OCCA, RAJA and OpenMP based on command line options.
    Device device(device_config);
    device.Print();

    // 3. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes
    //    with the same code.
    //Mesh* mesh = new Mesh(mesh_file, 1, 1);
    Mesh* mesh = LoadMeshNew(mesh_file);
    dim = mesh->Dimension();

    // 3a. Define the ODE solver used for time integration. Several second order
    //    time integrators are available.
    SecondOrderODESolver *ode_solver;
    switch (ode_solver_type)
    {
        // Implicit methods
        case 0: ode_solver = new GeneralizedAlpha2Solver(0.0); break;
        case 1: ode_solver = new GeneralizedAlpha2Solver(0.1); break;
        case 2: ode_solver = new GeneralizedAlpha2Solver(0.2); break;
        case 3: ode_solver = new GeneralizedAlpha2Solver(0.3); break;
        case 4: ode_solver = new GeneralizedAlpha2Solver(0.4); break;
        case 5: ode_solver = new GeneralizedAlpha2Solver(0.5); break;
        case 6: ode_solver = new GeneralizedAlpha2Solver(0.6); break;
        case 7: ode_solver = new GeneralizedAlpha2Solver(0.7); break;
        case 8: ode_solver = new GeneralizedAlpha2Solver(0.8); break;
        case 9: ode_solver = new GeneralizedAlpha2Solver(0.9); break;
        case 10: ode_solver = new GeneralizedAlpha2Solver(1.0); break;

        case 11: ode_solver = new AverageAccelerationSolver(); break;
        case 12: ode_solver = new LinearAccelerationSolver(); break;
        case 13: ode_solver = new CentralDifferenceSolver(); break;
        case 14: ode_solver = new FoxGoodwinSolver(); break;

        default:
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
            delete mesh;
            return 3;
    }

    // 4. Refine the mesh to increase resolution. In this example we do
    //    'ref_levels' of uniform refinement where the user specifies
    //    the number of levels with the '-r' option.
    for (int l = 0; l < ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

    // 4a. Set element attributes in order to distinguish elements in the
    //    PML region
    // Setup PML length
    Array2D<double> length(dim, 2); 
    length = 1.5;
    PML * pml = new PML(mesh,length);
    comp_domain_bdr = pml->GetCompDomainBdr();
    domain_bdr = pml->GetDomainBdr();
    pml->SetAttributes(mesh,pmls);

    // 5. Define a finite element space on the mesh. Here we use continuous
    //    Lagrange, Nedelec, or Raviart-Thomas finite elements of the specified
    //    order.
    if (dim == 1 && prob != 0)
    {
        cout << "Switching to problem type 0, H1 basis functions, "
            << "for 1 dimensional mesh." << endl;
        prob = 0;
    }

    FiniteElementCollection* fec = NULL;
    switch (prob)
    {
    case 0:  fec = new H1_FECollection(order, dim);      break;
    case 1:  fec = new ND_FECollection(order, dim);      break;
    case 2:  fec = new RT_FECollection(order - 1, dim);  break;
    default: break; // This should be unreachable
    }
    FiniteElementSpace* fespace = new FiniteElementSpace(mesh, fec);
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl;

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined based on the type
    //    of mesh and the problem type.
    Array<int> ess_tdof_list;
    Array<int> ess_bdr;
    if (mesh->bdr_attributes.Size())
    {
        ess_bdr.SetSize(mesh->bdr_attributes.Max());
        AttrToMarker(mesh->bdr_attributes.Max(), abcs, ess_bdr);
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Array of 0's and 1's marking the location of absorbing surfabc_marker_ces
    Array<int> abc_marker_;
    Coefficient* etaInvCoef_; // Admittance Coefficient
    AttrToMarker(mesh->bdr_attributes.Max(), abcs, abc_marker_);
    etaInvCoef_ = new ConstantCoefficient(omega_*sqrt(epsilon0_ / mu0_));


    // 7. Set up the linear form b(.) which corresponds to the right-hand side of
    //    the FEM linear system.
    VectorDeltaCoefficient* delta_one; // add point source
    double src_scalar = omega_ * 1.0;
    double position = 0.00;
    if (dim == 1)
    {
        Vector dir(1);
        dir[0] = 1.0;
        delta_one = new VectorDeltaCoefficient(dir, position, src_scalar);
    }
    else if (dim == 2)
    {
        Vector dir(2);
        dir[0] = 0.0; dir[1] = 1.0;
        delta_one = new VectorDeltaCoefficient(dir, position, position, src_scalar);
    }
    else if (dim == 3)
    {
        Vector dir(3);
        dir[0] = 0;
        dir[1] = 0;
        dir[2] = 1;
        delta_one = new VectorDeltaCoefficient(dir, position, position, position, src_scalar);
    }
    ComplexLinearForm b(fespace, conv);
    VectorFunctionCoefficient f(dim, source);
    b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(*delta_one)); // add delta point source
    //b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f)); // add Gaussian point source
    //b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(m_rbcBCoef), NULL, rbc_bdr);
    b.Vector::operator=(0.0);
    // Access and print the imaginary part
    
    //imagPart.Print();

    b.Assemble();

    // Access and print the real part
    //const Vector& imagPart = b.imag();
    //imagPart.Print();

    // 8. Define the solution vector u as a complex finite element grid function
    //    corresponding to fespace. Initialize u with initial guess of 1+0i or
    //    the exact solution if it is known.
    ComplexGridFunction u(fespace);
    ComplexGridFunction* u_exact = NULL;
    if (exact_sol) { u_exact = new ComplexGridFunction(fespace); }

    FunctionCoefficient u0_r(u0_real_exact);
    FunctionCoefficient u0_i(u0_imag_exact);
    VectorFunctionCoefficient u1_r(dim, u1_real_exact);
    VectorFunctionCoefficient u1_i(dim, u1_imag_exact);
    VectorFunctionCoefficient u2_r(dim, u2_real_exact);
    VectorFunctionCoefficient u2_i(dim, u2_imag_exact);

    ConstantCoefficient zeroCoef(0.0);
    ConstantCoefficient oneCoef(1.0);

    Vector zeroVec(dim); zeroVec = 0.0;
    Vector oneVec(dim);  oneVec = 0.0; oneVec[(prob == 2) ? (dim - 1) : 0] = 1.0;
    VectorConstantCoefficient zeroVecCoef(zeroVec);
    VectorConstantCoefficient oneVecCoef(oneVec);

    // 9. Set up the sesquilinear form a(.,.) on the finite element space
    //    corresponding to the damped harmonic oscillator operator of the
    //    appropriate type:
    //
    //    0) A scalar H1 field
    //       -Div(a Grad) - omega^2 b + i omega c
    //
    //    1) A vector H(Curl) field
    //       Curl(a Curl) - omega^2 b + i omega c
    //
    //    2) A vector H(Div) field
    //       -Grad(a Div) - omega^2 b + i omega c
    //
    ConstantCoefficient stiffnessCoef(1.0 / mu_ / mu0_);
    ConstantCoefficient massCoef(-omega_ * omega_ * epsilon_ * epsilon0_); std::cout << std::setw(20) << ( - omega_ * omega_ * epsilon_ * epsilon0_) << std::endl;
    ConstantCoefficient lossCoef(omega_ * sigma_); std::cout << std::setw(20) << (omega_ * sigma_) << std::endl;
    ConstantCoefficient negMassCoef(omega_ * omega_ * epsilon_ * epsilon0_);

    DenseMatrix sigmaMat(3);
    sigmaMat(0, 0) = 3.0; sigmaMat(1, 1) = 2.0; sigmaMat(2, 2) = 4.0;
    sigmaMat(0, 2) = 0.0; sigmaMat(2, 0) = 0.0;
    sigmaMat(0, 1) = M_SQRT1_2; sigmaMat(1, 0) = M_SQRT1_2; // 1/sqrt(2) in cmath
    sigmaMat(1, 2) = M_SQRT1_2; sigmaMat(2, 1) = M_SQRT1_2;
    Vector omega(dim); omega = omega_;
    sigmaMat.LeftScaling(omega);
    MatrixConstantCoefficient aniLossCoef(sigmaMat);
    //PrintMatrixConstantCoefficient(aniLossCoef);

    DenseMatrix epsilonMat(3);
    epsilonMat(0, 0) = 2.0; epsilonMat(1, 1) = 3.0; epsilonMat(2, 2) = 4.0;
    epsilonMat(0, 2) = 0.0; epsilonMat(2, 0) = 0.0;
    epsilonMat(0, 1) = 0.0; epsilonMat(1, 0) = 0.0; // 1/sqrt(2) in cmath
    epsilonMat(1, 2) = 0.0; epsilonMat(2, 1) = 0.0;
    omega = -omega_ * omega_ * epsilon0_;
    epsilonMat.LeftScaling(omega);
    MatrixConstantCoefficient aniMassCoef(epsilonMat);
    // PrintMatrixConstantCoefficient(aniMassCoef);

    Array<int> attr; //active attr for domain
    Array<int> attrPML; // active attr for PML
    if (mesh->attributes.Size())
    {
        attr.SetSize(mesh->attributes.Max());
        attrPML.SetSize(mesh->attributes.Max());
        attr = 1;
        attrPML = 0;
        if (mesh->attributes.Max() > 1)
            for(int i = 0; i<mesh->attributes.Max(); ++i)
            {
                bool found = false;
                for (int j = 0; j < pmls.Size(); j++) {
                    if (i == pmls[j]-1) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    attr[i] = 0;
                    attrPML[i] = 1;
                }
            }
    }

    for (int i = 0; i < attr.Size(); i++)
    {
        std::cout << i << "th domain: " << attr[i] << " " << std::endl;
        std::cout << i << "th PML: " << attrPML[i] << " " << std::endl;
    }
    std::cout << std::endl;

    // Print comp_domain_bdr
    std::cout << "comp_domain_bdr:" << std::endl;
    PrintArray2D(comp_domain_bdr);

    // Print domain_bdr
    std::cout << "domain_bdr:" << std::endl;
    PrintArray2D(domain_bdr);

    ConstantCoefficient muinv(1.0 / mu_ / mu0_);
    ConstantCoefficient omeg(-pow(omega_, 2) * epsilon_ * epsilon0_);
    ConstantCoefficient loss(omega_ * sigma_);
    RestrictedCoefficient restr_muinv(muinv,attr);
    RestrictedCoefficient restr_omeg(omeg,attr);
    RestrictedCoefficient restr_loss(loss,attr);

    int cdim = (dim == 2) ? 1 : dim;
    PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pml);
    PMLDiagMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pml);
    ScalarVectorProductCoefficient c1_Re(muinv,pml_c1_Re);
    ScalarVectorProductCoefficient c1_Im(muinv,pml_c1_Im);
    VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
    VectorRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

    PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,pml);
    PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,pml);
    ScalarVectorProductCoefficient c2_Re(omeg,pml_c2_Re);
    ScalarVectorProductCoefficient c2_Im(omeg,pml_c2_Im);
    VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
    VectorRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

    SesquilinearForm* a = new SesquilinearForm(fespace, conv);
    if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    switch (prob)
    {
    case 0:
        a->AddDomainIntegrator(new DiffusionIntegrator(stiffnessCoef),
            NULL);
        a->AddDomainIntegrator(new MassIntegrator(massCoef),
            new MassIntegrator(lossCoef));
        break;
    case 1:
        a->AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv),
            NULL);
        a->AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg),
            new VectorFEMassIntegrator(restr_loss));
        // Integrators inside the PML region
        a->AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                                new CurlCurlIntegrator(restr_c1_Im));
        a->AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                                new VectorFEMassIntegrator(restr_c2_Im));
        // if (etaInvCoef_)
        // {
        //     if (logging_ > 0)
        //     {
        //         cout << "Adding boundary integrator for absorbing boundary" << endl;
        //     }
        //     a->AddBoundaryIntegrator(
        //         NULL, new VectorFEMassIntegrator(*etaInvCoef_), abc_marker_);
        // }
        break;
    case 2:
        a->AddDomainIntegrator(new DivDivIntegrator(stiffnessCoef),
            NULL);
        a->AddDomainIntegrator(new VectorFEMassIntegrator(massCoef),
            new VectorFEMassIntegrator(lossCoef));
        break;
    default: break; // This should be unreachable
    }

    // 9a. Set up the bilinear form for the preconditioner corresponding to the
    //     appropriate operator
    //
    //      0) A scalar H1 field
    //         -Div(a Grad) - omega^2 b + omega c
    //
    //      1) A vector H(Curl) field
    //         Curl(a Curl) + omega^2 b + omega c
    //
    //      2) A vector H(Div) field
    //         -Grad(a Div) - omega^2 b + omega c
    //
#ifndef MFEM_USE_SUITESPARSE
    BilinearForm* pcOp = new BilinearForm(fespace);
    if (pa) { pcOp->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

    switch (prob)
    {
    case 0:
        pcOp->AddDomainIntegrator(new DiffusionIntegrator(stiffnessCoef));
        pcOp->AddDomainIntegrator(new MassIntegrator(massCoef));
        pcOp->AddDomainIntegrator(new MassIntegrator(lossCoef));
        break;
    case 1:
        pcOp->AddDomainIntegrator(new CurlCurlIntegrator(stiffnessCoef));
        pcOp->AddDomainIntegrator(new VectorFEMassIntegrator(negMassCoef));
        pcOp->AddDomainIntegrator(new VectorFEMassIntegrator(lossCoef));
        break;
    case 2:
        pcOp->AddDomainIntegrator(new DivDivIntegrator(stiffnessCoef));
        pcOp->AddDomainIntegrator(new VectorFEMassIntegrator(massCoef));
        pcOp->AddDomainIntegrator(new VectorFEMassIntegrator(lossCoef));
        break;
    default: break; // This should be unreachable
    }
#endif

    // 10. Assemble the form and the corresponding linear system, applying any
    //     necessary transformations such as: assembly, eliminating boundary
    //     conditions, conforming constraints for non-conforming AMR, etc.
    //a->SetDiagonalPolicy(mfem::Operator::DiagonalPolicy::DIAG_KEEP);
    a->Assemble(0);
#ifndef MFEM_USE_SUITESPARSE
    pcOp->Assemble();
#endif

    OperatorHandle A;
    Vector B, U;

    a->FormLinearSystem(ess_tdof_list, u, b, A, U, B);
    std::cout << "Size of linear system: " << A->Width() << endl << endl;
    //std::cout << "Printing Matrix A..." << std::endl;
    //std::ofstream A_file("Asp_matrix.txt");
    //A->PrintMatlab(A_file);

    ComplexSparseMatrix* Asp_blk = a->AssembleComplexSparseMatrix();
    SparseMatrix* Asp = Asp_blk->GetSystemMatrix();
     //std::cout << "Printing Matrix Asp..." << std::endl;
     //std::ofstream Asp_file("Asp_matrix.txt");
     //Asp->PrintMatlab(Asp_file);

     //std::cout << "Printing Matrix B..." << std::endl;
     //std::ofstream B_file("B_matrix.txt");
     //B.Print(B_file);
     //exit(0);

    // std::cout << "Printing Matrix b..." << std::endl;
    // std::ofstream bb_file("bb_matrix.txt");
    // b.Print(bb_file);


    // 11. Define and apply a GMRES solver for AU=B with a block diagonal
    //     preconditioner based on the appropriate sparse smoother.
    mfem::StopWatch timer;
#ifndef MFEM_USE_SUITESPARSE
    {
        Array<int> blockOffsets;
        blockOffsets.SetSize(3);
        blockOffsets[0] = 0;
        blockOffsets[1] = A->Height() / 2;
        blockOffsets[2] = A->Height() / 2;
        blockOffsets.PartialSum();

        BlockDiagonalPreconditioner BDP(blockOffsets);

        Operator* pc_r = NULL;
        Operator* pc_i = NULL;

        if (pa)
        {
            pc_r = new OperatorJacobiSmoother(*pcOp, ess_tdof_list);
        }
        else
        {
            OperatorHandle PCOp;
            pcOp->SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
            pcOp->FormSystemMatrix(ess_tdof_list, PCOp);
            switch (prob)
            {
            case 0:
                pc_r = new DSmoother(*PCOp.As<SparseMatrix>());
                break;
            case 1:
                pc_r = new GSSmoother(*PCOp.As<SparseMatrix>());
                break;
            case 2:
                pc_r = new DSmoother(*PCOp.As<SparseMatrix>());
                break;
            default:
                break; // This should be unreachable
            }
        }
        double s = (prob != 1) ? 1.0 : -1.0;
        pc_i = new ScaledOperator(pc_r,
            (conv == ComplexOperator::HERMITIAN) ?
            s : -s);

        BDP.SetDiagonalBlock(0, pc_r);
        BDP.SetDiagonalBlock(1, pc_i);
        BDP.owns_blocks = 1;

        if (use_gmres)
        {
            GMRESSolver gmres;
            gmres.SetPreconditioner(BDP);
            gmres.SetOperator(*A.Ptr());
            gmres.SetRelTol(1e-9);
            gmres.SetMaxIter(pa ? 5000 * order * order : 6000 * order * order);
            gmres.SetPrintLevel(1);
            gmres.SetAbsTol(0.0);
            gmres.SetKDim(200);
            gmres.Mult(B, U);
        }
        else
        {
            FGMRESSolver fgmres;
            fgmres.SetPreconditioner(BDP);
            fgmres.SetOperator(*A.Ptr());
            fgmres.SetRelTol(1e-12);
            fgmres.SetMaxIter(1000);
            fgmres.SetPrintLevel(1);
            fgmres.Mult(B, U);
        }
    }
#elif !defined(MFEM_USE_MKL_PARDISO)
    {
        /*ComplexUMFPackSolver csolver(*A.As<ComplexSparseMatrix>());
        csolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
        csolver.SetPrintLevel(3);
        csolver.Mult(B, U);*/

         timer.Start();
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*Asp);
         umf_solver.SetPrintLevel(1);
         umf_solver.Mult(B, U);
         timer.Stop();
         double elapsed_time = timer.RealTime();
         mfem::out << "UMFPACK solver took " << elapsed_time << " seconds." << std::endl;
    }
#else //defined MKL Pardiso
    {
        if (!comp_solver) {
            timer.Start();
            PardisoSolver pardiso_solver;
            pardiso_solver.SetPrintLevel(bprint); // set to 1 if want to see details
            pardiso_solver.SetOperator(*Asp);
            pardiso_solver.Mult(B, U);
            timer.Stop();
            double elapsed_time = timer.RealTime();
            mfem::out << "Pardiso real solver took " << elapsed_time << " seconds." << std::endl;
        }
        else {
            timer.Start();
            PardisoCompSolver pardiso_comp_solver;
            //pardiso_comp_solver.SetMatrixType(PardisoCompSolver::MatType::COMPLEX_SYMMETRIC); // MKL needs only the upper triangular part of the system
            pardiso_comp_solver.SetPrintLevel(bprint); // set to 1 if want to see details
            pardiso_comp_solver.SetOperator(*Asp_blk);
            pardiso_comp_solver.Mult(B, U);
            timer.Stop();
            double elapsed_time = timer.RealTime();
            mfem::out << "Pardiso complex solver took " << elapsed_time << " seconds." << std::endl;
        }
    }
#endif

    // 12. Recover the solution as a finite element grid function and compute the
    //     errors if the exact solution is known.
    a->RecoverFEMSolution(U, b, u);

    //std::cout << "Printing solution u..." << std::endl;
    //std::ofstream u_file("u_field.txt");
    //u.Print(u_file);

    if (exact_sol)
    {
        double err_r = -1.0;
        double err_i = -1.0;

        switch (prob)
        {
        case 0:
            err_r = u.real().ComputeL2Error(u0_r);
            err_i = u.imag().ComputeL2Error(u0_i);
            break;
        case 1:
            err_r = u.real().ComputeL2Error(u1_r);
            err_i = u.imag().ComputeL2Error(u1_i);
            break;
        case 2:
            err_r = u.real().ComputeL2Error(u2_r);
            err_i = u.imag().ComputeL2Error(u2_i);
            break;
        default: break; // This should be unreachable
        }

        cout << endl;
        cout << "|| Re (u_h - u) ||_{L^2} = " << err_r << endl;
        cout << "|| Im (u_h - u) ||_{L^2} = " << err_i << endl;
        cout << endl;
    }

    // 13. Save the refined mesh and the solution. This output can be viewed
    //     later using GLVis: "glvis -m mesh -g sol".
    {
        ofstream mesh_ofs("refined_11.mesh");
        mesh_ofs.precision(8);
        mesh->Print(mesh_ofs);

        ofstream sol_r_ofs("sol_r.gf");
        ofstream sol_i_ofs("sol_i.gf");
        sol_r_ofs.precision(8);
        sol_i_ofs.precision(8);
        u.real().Save(sol_r_ofs);
        u.imag().Save(sol_i_ofs);
    }

    // 14. Send the solution by socket to a GLVis server.
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport = 19916;
        socketstream sol_sock_r(vishost, visport);
        socketstream sol_sock_i(vishost, visport);
        sol_sock_r.precision(8);
        sol_sock_i.precision(8);
        sol_sock_r << "solution\n" << *mesh << u.real()
            << "window_title 'Solution: Real Part'" << flush;
        sol_sock_i << "solution\n" << *mesh << u.imag()
            << "window_title 'Solution: Imaginary Part'" << flush;
    }
    if (visualization && exact_sol)
    {
        *u_exact -= u;

        char vishost[] = "localhost";
        int  visport = 19916;
        socketstream sol_sock_r(vishost, visport);
        socketstream sol_sock_i(vishost, visport);
        sol_sock_r.precision(8);
        sol_sock_i.precision(8);
        sol_sock_r << "solution\n" << *mesh << u_exact->real()
            << "window_title 'Error: Real Part'" << flush;
        sol_sock_i << "solution\n" << *mesh << u_exact->imag()
            << "window_title 'Error: Imaginary Part'" << flush;
    }
    if (visualization)
    {
        GridFunction u_t(fespace);
        u_t = u.real();
        char vishost[] = "localhost";
        int  visport = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n" << *mesh << u_t
            << "window_title 'Harmonic Solution (t = 0.0 T)'"
            << "pause\n" << flush;

        cout << "GLVis visualization paused."
            << " Press space (in the GLVis window) to resume it.\n";
        int num_frames = 32;
        int i = 0;
        while (sol_sock)
        {
            double t = (double)(i % num_frames) / num_frames;
            ostringstream oss;
            oss << "Harmonic Solution (t = " << t << " T)";

            add(cos(2.0 * M_PI * t), u.real(),
                sin(-2.0 * M_PI * t), u.imag(), u_t);
            sol_sock << "solution\n" << *mesh << u_t
                << "window_title '" << oss.str() << "'" << flush;
            i++;

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // 14b. Save data in the ParaView format
    ParaViewDataCollection paraview_dc("ex98_test_seismic", mesh);
    paraview_dc.SetDataFormat(VTKFormat::ASCII);
    paraview_dc.SetPrefixPath("ParaView");
    paraview_dc.SetLevelsOfDetail(order>1 ? order-1 : order);
    //paraview_dc.SetCycle(0);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    //paraview_dc.SetHighOrderOutput(true);
    paraview_dc.SetTime(0.0); // set the time
    paraview_dc.RegisterField("real", &u.real());
    paraview_dc.RegisterField("imag", &u.imag());
    paraview_dc.Save();

    ////14c. Save data in the Paraview format with tet only mesh
    //// Create a new mesh with only the tetrahedra
    //std::vector<mfem::Element*> tets;
    //std::vector<int> tetIndices;
    //// Iterate through the elements and collect the tetrahedra
    //for (int i = 0; i < mesh->GetNE(); i++)
    //{
    //    if (mesh->GetElementType(i) == mfem::Element::TETRAHEDRON)
    //    {
    //        tets.push_back(mesh->GetElement(i));
    //        tetIndices.push_back(i);
    //    }
    //}
    //Mesh tetMesh(3, tets.size(), 0, 0, 3);
    //for (size_t i = 0; i < tets.size(); i++)
    //{
    //    tetMesh.AddElement(tets[i]);
    //}
    //tetMesh.FinalizeTopology();
    //FiniteElementSpace tetFespace(&tetMesh, &fec);
    //GridFunction* tetSol(&tetFespace);
    //// Map the solution values
    //for (size_t i = 0; i < tetIndices.size(); i++)
    //{
    //    tetSol->GetBlock(i) = u.imag()->GetBlock(tetIndices[i]);
    //}

    // 15. Free the used memory.
    delete a;
    delete u_exact;
#ifndef MFEM_USE_MKL_PARDISO
    delete pcOp;
#endif
    delete fespace;
    delete fec;
    // delete mesh;

    return 0;
}

bool check_for_inline_mesh(const char* mesh_file)
{
    string file(mesh_file);
    size_t p0 = file.find_last_of("/");
    string s0 = file.substr((p0 == string::npos) ? 0 : (p0 + 1), 7);
    return s0 == "inline-";
}

complex<double> u0_exact(const Vector& x)
{
    int dim = x.Size();
    complex<double> i(0.0, 1.0);
    complex<double> alpha = (epsilon_ * omega_ - i * sigma_);
    complex<double> kappa = std::sqrt(mu_ * omega_ * alpha);
    return std::exp(-i * kappa * x[dim - 1]);
}

double u0_real_exact(const Vector& x)
{
    return u0_exact(x).real();
}

double u0_imag_exact(const Vector& x)
{
    return u0_exact(x).imag();
}

void u1_real_exact(const Vector& x, Vector& v)
{
    int dim = x.Size();
    v.SetSize(dim); v = 0.0; v[0] = u0_real_exact(x);
}

void u1_imag_exact(const Vector& x, Vector& v)
{
    int dim = x.Size();
    v.SetSize(dim); v = 0.0; v[0] = u0_imag_exact(x);
}

void u2_real_exact(const Vector& x, Vector& v)
{
    int dim = x.Size();
    v.SetSize(dim); v = 0.0; v[dim - 1] = u0_real_exact(x);
}

void u2_imag_exact(const Vector& x, Vector& v)
{
    int dim = x.Size();
    v.SetSize(dim); v = 0.0; v[dim - 1] = u0_imag_exact(x);
}

Mesh* LoadMeshNew(const std::string& path)
{
    // Read the (serial) mesh from the given mesh file. Handle preparation for refinement and
    // orientations here to avoid possible reorientations and reordering later on. MFEM
    // supports a native mesh format (.mesh), VTK/VTU, Gmsh, as well as some others. We use
    // built-in converters for the types we know, otherwise rely on MFEM to do the conversion
    // or error out if not supported.
    std::filesystem::path mfile(path);
    if (mfile.extension() == ".mphtxt" || mfile.extension() == ".mphbin" ||
        mfile.extension() == ".nas" || mfile.extension() == ".bdf")
    {
        // Put translated mesh in temporary string buffer.
        std::stringstream fi(std::stringstream::in | std::stringstream::out);
        // fi << std::fixed;
        fi << std::scientific;
        fi.precision(MSH_FLT_PRECISION);

        Mesh* tempmesh = new Mesh();

        if (mfile.extension() == ".mphtxt" || mfile.extension() == ".mphbin")
        {
            tempmesh->ConvertMeshComsol(path, fi);
        }
        else
        {
            tempmesh->ConvertMeshNastran(path, fi);
        }

        return new Mesh(fi, 1, 1, true);
    }
    // Otherwise, just rely on MFEM load the mesh.
    named_ifgzstream fi(path);
    if (!fi.good())
    {
        MFEM_ABORT("Unable to open mesh file \"" << path << "\"!");
    }
    Mesh* mesh = new Mesh(fi, 1, 1, true);
    mesh->EnsureNodes();
    return mesh;
}

void PrintMatrixConstantCoefficient(mfem::MatrixConstantCoefficient& coeff)
{
    const mfem::DenseMatrix& mat = coeff.GetMatrix();
    int height = mat.Height();
    int width = mat.Width();

    // Set the output format
    std::cout << std::scientific << std::setprecision(4) << std::right;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            std::cout << std::setw(16) << mat(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Function to print Array2D<double>
void PrintArray2D(const mfem::Array2D<double>& arr)
{
    for (int i = 0; i < arr.NumRows(); i++)
    {
        for (int j = 0; j < arr.NumCols(); j++)
        {
            std::cout << arr(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void AttrToMarker(int max_attr, const Array<int>& attrs, Array<int>& marker)
{
    MFEM_ASSERT(attrs.Max() <= max_attr, "Invalid attribute number present.");

    marker.SetSize(max_attr);
    if (attrs.Size() == 1 && attrs[0] == -1)
    {
        marker = 1;
    }
    else
    {
        marker = 0;
        for (int j = 0; j < attrs.Size(); j++)
        {
            int attr = attrs[j];
            MFEM_VERIFY(attr > 0, "Attribute number less than one!");
            marker[attr - 1] = 1;
        }
    }
}


void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(dxs[i], 2)).real();
   }
}

void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(dxs[i], 2)).imag();
   }
}

void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = abs(det / pow(dxs[i], 2));
   }
}

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   // in the 2D case the coefficient is scalar 1/det(J)
   if (dim == 2)
   {
      D = (1.0 / det).real();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow(dxs[i], 2) / det).real();
      }
   }
}

void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = (1.0 / det).imag();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow(dxs[i], 2) / det).imag();
      }
   }
}

void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = abs(1.0 / det);
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = abs(pow(dxs[i], 2) / det);
      }
   }
}

PML::PML(Mesh *mesh_, Array2D<double> length_)
   : mesh(mesh_), length(length_)
{
   dim = mesh->Dimension();
   SetBoundaries();
}

void PML::SetBoundaries()
{
   comp_dom_bdr.SetSize(dim, 2);
   dom_bdr.SetSize(dim, 2);
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   for (int i = 0; i < dim; i++)
   {
      dom_bdr(i, 0) = pmin(i);
      dom_bdr(i, 1) = pmax(i);
      comp_dom_bdr(i, 0) = dom_bdr(i, 0) + length(i, 0);
      comp_dom_bdr(i, 1) = dom_bdr(i, 1) - length(i, 1);
   }
}

void PML::SetAttributes(Mesh *mesh_)
{
   // Initialize bdr attributes
   for (int i = 0; i < mesh_->GetNBE(); ++i)
   {
      mesh_->GetBdrElement(i)->SetAttribute(i+1);
   }

   int nrelem = mesh_->GetNE();

   elems.SetSize(nrelem);

   // Loop through the elements and identify which of them are in the PML
   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = mesh_->GetElement(i);
      Array<int> vertices;

      // Initialize attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      // Check if any vertex is in the PML
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         double *coords = mesh_->GetVertex(vert_idx);
         for (int comp = 0; comp < dim; ++comp)
         {
            if (coords[comp] > comp_dom_bdr(comp, 1) ||
                coords[comp] < comp_dom_bdr(comp, 0))
            {
               in_pml = true;
               break;
            }
         }
      }
      if (in_pml)
      {
         elems[i] = 0;
         el->SetAttribute(2);
      }
   }
   mesh_->SetAttributes();
}

void PML::SetAttributes(Mesh *mesh_, Array<int> pmlmaker_)
{
   int nrelem = mesh_->GetNE();

   elems.SetSize(nrelem);

   // Loop through the elements and identify which of them are in the PML
   for (int i = 0; i < nrelem; ++i)
   {
        elems[i] = 1;
        bool in_pml = false;
        Element *el = mesh_->GetElement(i);
        Array<int> vertices;

        // Initialize attribute
        el->GetVertices(vertices);
        int nrvert = vertices.Size();

        // Check if element attribute is in the PML
        int j = el->GetAttribute();
        //std::cout << "element: " << i << ", attribute : " << j << std::endl;
        for (int k = 0; k < pmlmaker_.Size(); k++) {
            if (j == pmlmaker_[k]) {
                in_pml = true;
                //std::cout << "pml is true." << std::endl;
                break;
            }
        }
        if (in_pml)
        {
            elems[i] = 0;
        }
   }
   mesh_->SetAttributes();
}

void PML::StretchFunction(const Vector &x,
                          vector<complex<double>> &dxs)
{
   complex<double> zi = complex<double>(0., 1.);

   double n = 2.0;
   double c = 20.0;
   double coeff;
   double k = omega_ * sqrt(epsilon0_ * epsilon_ * mu_ * mu0_);

   // Stretch in each direction independently
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_domain_bdr(i, 1))
      {
         coeff = n * c / k / pow(length(i, 1), n);
         dxs[i] = 1.0 + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 1), n - 1.0));
      }
      if (x(i) <= comp_domain_bdr(i, 0))
      {
         coeff = n * c / k / pow(length(i, 0), n);
         dxs[i] = 1.0 + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 0), n - 1.0));
      }
   }
}

void source(const Vector& x, Vector& f)
{
    Vector center(dim);
    double r = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        center(i) = 0.5 * (comp_domain_bdr(i, 0) + comp_domain_bdr(i, 1));
        r += pow(x[i] - center[i], 2.);
    }
    double n = 500.0 * omega_ * sqrt(epsilon0_ * epsilon_ * mu0_ * mu_) / M_PI;
    double coeff = pow(n, 2) / M_PI;
    double alpha = -pow(n, 2) * r;
    f = 0.0;
    f[1] = coeff * exp(alpha);
}
