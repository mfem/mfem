//                                MFEM Example 22
//
// Compile with: make ex22
//
// Sample runs:  ex22 -m ../data/inline-segment.mesh -o 3
//               ex22 -m ../data/inline-tri.mesh -o 3
//               ex22 -m ../data/inline-quad.mesh -o 3
//               ex22 -m ../data/inline-quad.mesh -o 3 -p 1
//               ex22 -m ../data/inline-quad.mesh -o 3 -p 1 -pa
//               ex22 -m ../data/inline-quad.mesh -o 3 -p 2
//               ex22 -m ../data/inline-tet.mesh -o 2
//               ex22 -m ../data/inline-hex.mesh -o 2
//               ex22 -m ../data/inline-hex.mesh -o 2 -p 1
//               ex22 -m ../data/inline-hex.mesh -o 2 -p 2
//               ex22 -m ../data/inline-hex.mesh -o 2 -p 2 -pa
//               ex22 -m ../data/inline-wedge.mesh -o 1
//               ex22 -m ../data/inline-pyramid.mesh -o 1
//               ex22 -m ../data/star.mesh -r 1 -o 2 -sigma 10.0
//
// Device sample runs:
//               ex22 -m ../data/inline-quad.mesh -o 3 -p 1 -pa -d cuda
//               ex22 -m ../data/inline-hex.mesh -o 2 -p 2 -pa -d cuda
//               ex22 -m ../data/star.mesh -r 1 -o 2 -sigma 10.0 -pa -d cuda
//
// Description:  This example code demonstrates the use of MFEM to define and
//               solve simple complex-valued linear systems. It implements three
//               variants of a damped harmonic oscillator:
//
//               1) A scalar H1 field
//                  -Div(a Grad u) - omega^2 b u + i omega c u = 0
//
//               2) A vector H(Curl) field
//                  Curl(a Curl u) - omega^2 b u + i omega c u = 0
//
//               3) A vector H(Div) field
//                  -Grad(a Div u) - omega^2 b u + i omega c u = 0
//
//               In each case the field is driven by a forced oscillation, with
//               angular frequency omega, imposed at the boundary or a portion
//               of the boundary.
//
//               In electromagnetics, the coefficients are typically named the
//               permeability, mu = 1/a, permittivity, epsilon = b, and
//               conductivity, sigma = c. The user can specify these constants
//               using either set of names.
//
//               The example also demonstrates how to display a time-varying
//               solution as a sequence of fields sent to a single GLVis socket.
//
//               We recommend viewing examples 1, 3 and 4 before viewing this
//               example.

// added the anisotropic material
// added the point source
// added mesh

#include "mfem.hpp"
#include "meshio.hpp"
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

using namespace std;
using namespace mfem;

const auto MSH_FLT_PRECISION = std::numeric_limits<double>::max_digits10;

// Permittivity of free space [F/m].
static constexpr double epsilon0_ = 8.8541878176e-12;

// Permeability of free space [H/m].
static constexpr double mu0_ = 4.0e-7 * M_PI;

static double mu_ = 1.0;
static double epsilon_ = 1.0;
static double sigma_ = 0.0;
static double omega_ = 10.0;

double u0_real_exact(const Vector&);
double u0_imag_exact(const Vector&);

void u1_real_exact(const Vector&, Vector&);
void u1_imag_exact(const Vector&, Vector&);

void u2_real_exact(const Vector&, Vector&);
void u2_imag_exact(const Vector&, Vector&);

bool check_for_inline_mesh(const char* mesh_file);

Mesh* LoadMeshNew(const std::string& path);

void PrintMatrixConstantCoefficient(mfem::MatrixConstantCoefficient& coeff);

int main(int argc, char* argv[])
{
    std::cout << mkl_get_max_threads() << std::endl; // in VS2022, check properties->intel library for OneAPI->Use oneMKL (Parallel)

    // 1. Parse command-line options.
    //const char* mesh_file = "../data/em_sphere_mfem_ex0_coarse.mphtxt";
    //const char* mesh_file = "../data/em_sphere_mfem_ex0.mphtxt";
    //const char* mesh_file = "../data/simple_cube.mphtxt";
     const char* mesh_file = "../data/cube_comsol.mphtxt";
    //const char* mesh_file = "../data/inline-tet.mesh";
    int ref_levels = 1;
    int order = 1;
    int prob = 1;
    double freq = 100.0e6;
    double a_coef = 0.0;
    bool visualization = 1;
    bool herm_conv = true;
    bool exact_sol = true;
    bool pa = false;
    const char* device_config = "cpu";
    bool use_gmres = true;
    bool save_pv = false;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
        "Mesh file to use.");
    args.AddOption(&ref_levels, "-r", "--refine",
        "Number of times to refine the mesh uniformly.");
    args.AddOption(&order, "-o", "--order",
        "Finite element order (polynomial degree).");
    args.AddOption(&prob, "-p", "--problem-type",
        "Choose between 0: H_1, 1: H(Curl), or 2: H(Div) "
        "damped harmonic oscillator.");
    args.AddOption(&a_coef, "-a", "--stiffness-coef",
        "Stiffness coefficient (spring constant or 1/mu).");
    args.AddOption(&epsilon_, "-b", "--mass-coef",
        "Mass coefficient (or epsilon).");
    args.AddOption(&sigma_, "-c", "--damping-coef",
        "Damping coefficient (or sigma).");
    args.AddOption(&mu_, "-mu", "--permeability",
        "Permeability of free space (or 1/(spring constant)).");
    args.AddOption(&epsilon_, "-eps", "--permittivity",
        "Permittivity of free space (or mass constant).");
    args.AddOption(&sigma_, "-sigma", "--conductivity",
        "Conductivity (or damping constant).");
    args.AddOption(&freq, "-f", "--frequency",
        "Frequency (in Hz).");
    args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
        "--no-hermitian", "Use convention for Hermitian operators.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
        "--no-visualization",
        "Enable or disable GLVis visualization.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
        "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&device_config, "-d", "--device",
        "Device configuration string, see Device::Configure().");
    args.AddOption(&use_gmres, "-g", "--gmres", "-no-g",
        "--no-grems", "Use GMRES solver or FGRMES solver.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    MFEM_VERIFY(prob >= 0 && prob <= 2,
        "Unrecognized problem type: " << prob);

    if (a_coef != 0.0)
    {
        mu_ = 1.0 / a_coef;
    }
    if (freq > 0.0)
    {
        omega_ = 2.0 * M_PI * freq;
    }

    exact_sol = check_for_inline_mesh(mesh_file);
    if (exact_sol)
    {
        cout << "Identified a mesh with known exact solution" << endl;
    }

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
    int dim = mesh->Dimension();

    // 4. Refine the mesh to increase resolution. In this example we do
    //    'ref_levels' of uniform refinement where the user specifies
    //    the number of levels with the '-r' option.
    for (int l = 0; l < ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

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
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // 7. Set up the linear form b(.) which corresponds to the right-hand side of
    //    the FEM linear system.
    VectorDeltaCoefficient* delta_one; // add point source
    double src_scalar = omega_ * 1.0;
    double position = 0.50;
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
        dir[0] = 0.0; dir[1] = 0.0; dir[2] = 1.0;
        delta_one = new VectorDeltaCoefficient(dir, position, position, position, src_scalar);
    }
    ComplexLinearForm b(fespace, conv);
    b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(*delta_one));
    b.Vector::operator=(0.0);
    b.Assemble();

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
    Vector  oneVec(dim);  oneVec = 0.0; oneVec[(prob == 2) ? (dim - 1) : 0] = 1.0;
    VectorConstantCoefficient zeroVecCoef(zeroVec);
    VectorConstantCoefficient oneVecCoef(oneVec);

    switch (prob)
    {
    case 0:
        if (exact_sol)
        {
            u.ProjectBdrCoefficient(u0_r, u0_i, ess_bdr);
            u_exact->ProjectCoefficient(u0_r, u0_i);
        }
        else
        {
            u.ProjectBdrCoefficient(oneCoef, zeroCoef, ess_bdr);
        }
        break;
    case 1:
        if (exact_sol)
        {
            u.ProjectBdrCoefficientTangent(u1_r, u1_i, ess_bdr);
            u_exact->ProjectCoefficient(u1_r, u1_i);
        }
        else
        {
            u.ProjectBdrCoefficientTangent(oneVecCoef, zeroVecCoef, ess_bdr);
        }
        break;
    case 2:
        if (exact_sol)
        {
            u.ProjectBdrCoefficientNormal(u2_r, u2_i, ess_bdr);
            u_exact->ProjectCoefficient(u2_r, u2_i);
        }
        else
        {
            u.ProjectBdrCoefficientNormal(oneVecCoef, zeroVecCoef, ess_bdr);
        }
        break;
    default: break; // This should be unreachable
    }

    if (visualization && exact_sol)
    {
        char vishost[] = "localhost";
        int  visport = 19916;
        socketstream sol_sock_r(vishost, visport);
        socketstream sol_sock_i(vishost, visport);
        sol_sock_r.precision(8);
        sol_sock_i.precision(8);
        sol_sock_r << "solution\n" << *mesh << u_exact->real()
            << "window_title 'Exact: Real Part'" << flush;
        sol_sock_i << "solution\n" << *mesh << u_exact->imag()
            << "window_title 'Exact: Imaginary Part'" << flush;
    }

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
    sigmaMat(0, 0) = 2.0; sigmaMat(1, 1) = 3.0; sigmaMat(2, 2) = 4.0;
    sigmaMat(0, 2) = 0.0; sigmaMat(2, 0) = 0.0;
    sigmaMat(0, 1) = 0.0; sigmaMat(1, 0) = 0.0; // 1/sqrt(2) in cmath
    sigmaMat(1, 2) = 0.0; sigmaMat(2, 1) = 0.0;
    Vector omega(dim); omega = omega_;
    sigmaMat.LeftScaling(omega);
    MatrixConstantCoefficient aniLossCoef(sigmaMat);
    PrintMatrixConstantCoefficient(aniLossCoef);

    DenseMatrix epsilonMat(3);
    epsilonMat(0, 0) = 2.0; epsilonMat(1, 1) = 3.0; epsilonMat(2, 2) = 4.0;
    epsilonMat(0, 2) = 0.0; epsilonMat(2, 0) = 0.0;
    epsilonMat(0, 1) = 0.0; epsilonMat(1, 0) = 0.0; // 1/sqrt(2) in cmath
    epsilonMat(1, 2) = 0.0; epsilonMat(2, 1) = 0.0;
    omega = -omega_ * omega_ * epsilon0_;
    epsilonMat.LeftScaling(omega);
    MatrixConstantCoefficient aniMassCoef(epsilonMat);
    PrintMatrixConstantCoefficient(aniMassCoef);

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
        a->AddDomainIntegrator(new CurlCurlIntegrator(stiffnessCoef),
            NULL);
        a->AddDomainIntegrator(new VectorFEMassIntegrator(aniMassCoef),
            new VectorFEMassIntegrator(lossCoef));
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

    // 10. Assemble the form and the corresponding linear system, applying any
    //     necessary transformations such as: assembly, eliminating boundary
    //     conditions, conforming constraints for non-conforming AMR, etc.
    a->SetDiagonalPolicy(mfem::Operator::DiagonalPolicy::DIAG_KEEP);
    a->Assemble();
    pcOp->Assemble();

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
#else
    {
        timer.Start();
        PardisoSolver pardiso_solver;
        pardiso_solver.SetPrintLevel(0); // set to 1 if want to see details
        pardiso_solver.SetOperator(*Asp);
        pardiso_solver.Mult(B, U);
        timer.Stop();
        double elapsed_time = timer.RealTime();
        mfem::out << "Pardiso solver took " << elapsed_time << " seconds." << std::endl;
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
        ofstream mesh_ofs("refined.mesh");
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
        GridFunction abs_u(fespace);  // Assuming fespace is your finite element space
        GridFunction u_imag = u.imag();
        for (int i = 0; i < u_imag.Size(); i++) {
            double value = u_imag[i];
            abs_u[i] = std::abs(value);
            if (abs_u[i] < 1e-16) abs_u[i] = 1e-16;
        }

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

            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    // 14b. Save data in the ParaView format
    if (save_pv) {
        ParaViewDataCollection paraview_dc("ex0_point_source_em_test1_o2", mesh);
        paraview_dc.SetDataFormat(VTKFormat::ASCII);
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        //paraview_dc.SetCycle(0);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.SetTime(0.0); // set the time
        paraview_dc.RegisterField("real", &u.real());
        paraview_dc.RegisterField("imag", &u.imag());
        paraview_dc.Save();
    }

    // 15. Free the used memory.
    delete a;
    delete u_exact;
    delete pcOp;
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

        if (mfile.extension() == ".mphtxt" || mfile.extension() == ".mphbin")
        {
            palace::mesh::ConvertMeshComsol(path, fi);
            // mesh::ConvertMeshComsol(path, fo);
        }
        else
        {
            palace::mesh::ConvertMeshNastran(path, fi);
            // mesh::ConvertMeshNastran(path, fo);
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

