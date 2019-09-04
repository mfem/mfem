//                       MFEM Example 5 - Parallel Version
//
// Compile with: make ex5p
//
// Sample runs:  mpirun -np 4 ex5p -m ../data/square-disc.mesh
//               mpirun -np 4 ex5p -m ../data/star.mesh
//               mpirun -np 4 ex5p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex5p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex5p -m ../data/escher.mesh
//               mpirun -np 4 ex5p -m ../data/fichera.mesh
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//                                 k*u + grad p = f
//                                 - div u      = g
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockMatrix class, as
//               well as the collective saving of several grid functions in a
//               VisIt (visit.llnl.gov) visualization format.
//
//               We recommend viewing example 5 before viewing this miniapp.

#include "mfem.hpp"
#include "mixed_fe_solvers.hpp"
#include <fstream>
#include <iostream>
#include <assert.h>
#include <memory>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);

//     Assemble the finite element matrices for the Darcy problem
//
//                            D = [ M  B^T ]
//                                [ B   0  ]
//     where:
//
//     M = \int_\Omega u_h \cdot v_h d\Omega   u_h, v_h \in R_h
//     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
class DarcyProblem
{
    OperatorPtr M_;
    OperatorPtr B_;
    Vector rhs_;
    Vector ess_data_;
    ParGridFunction u_;
    ParGridFunction p_;
    DFSDataCollector collector_;
    VectorFunctionCoefficient ucoeff_;
    FunctionCoefficient pcoeff_;
    const IntegrationRule *irs_[Geometry::NumGeom];
    bool verbose_;
public:
    DarcyProblem(ParMesh* mesh, int num_refines, int order, bool verbose,
                 Array<int>& ess_bdr, DFSParameters param);

    HypreParMatrix& GetM() { return *M_.As<HypreParMatrix>(); }
    HypreParMatrix& GetB() { return *B_.As<HypreParMatrix>(); }
    const Vector& GetRHS() { return rhs_; }
    const Vector& GetBC() { return ess_data_; }
    const DFSDataCollector& GetDFSDataCollector() const { return collector_; }

    void ShowError(const Vector& sol, ParMesh* mesh);
};

DarcyProblem::DarcyProblem(ParMesh* mesh, int num_refines, int order, bool verbose,
                           Array<int>& ess_bdr, DFSParameters dfs_param)
    : collector_(order, num_refines, mesh, ess_bdr, dfs_param),
      ucoeff_(mesh->Dimension(), uFun_ex), pcoeff_(pFun_ex), verbose_(verbose)
{
    for (int l = 0; l < num_refines; l++)
    {
        mesh->UniformRefinement();
        collector_.CollectData(mesh);
    }

    // 8. Define the coefficients, analytical solution, and rhs of the PDE.
    VectorFunctionCoefficient fcoeff(mesh->Dimension(), fFun);
    FunctionCoefficient fnatcoeff(f_natural);
    FunctionCoefficient gcoeff(gFun);

    u_.SetSpace(collector_.hdiv_fes_.get());
    p_.SetSpace(collector_.l2_fes_.get());
    u_ = 0.0;
    u_.ProjectBdrCoefficientNormal(ucoeff_, ess_bdr);

    ParLinearForm fform(collector_.hdiv_fes_.get());
    fform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
    fform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
    fform.Assemble();

    ParLinearForm gform(collector_.l2_fes_.get());
    gform.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
    gform.Assemble();

    ParBilinearForm mVarf(collector_.hdiv_fes_.get());
    ParMixedBilinearForm bVarf(&(*collector_.hdiv_fes_), &(*collector_.l2_fes_));

    mVarf.AddDomainIntegrator(new VectorFEMassIntegrator);
    mVarf.Assemble();
    mVarf.EliminateEssentialBC(ess_bdr, u_, fform);
    mVarf.Finalize();
    M_.Reset(mVarf.ParallelAssemble());

    bVarf.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    bVarf.Assemble();
    bVarf.SpMat() *= -1.0;
    bVarf.EliminateTrialDofs(ess_bdr, u_, gform);
    bVarf.Finalize();
    B_.Reset(bVarf.ParallelAssemble());

    rhs_.SetSize(M_->NumRows() + B_->NumRows());
    Vector rhs_block0(rhs_.GetData(), M_->NumRows());
    Vector rhs_block1(rhs_.GetData()+M_->NumRows(), B_->NumRows());
    fform.ParallelAssemble(rhs_block0);
    gform.ParallelAssemble(rhs_block1);

    ess_data_.SetSize(M_->NumRows() + B_->NumRows());
    Vector ess_data_block0(ess_data_.GetData(), M_->NumRows());
    u_.ParallelProject(ess_data_block0);

    if (verbose)
    {
        cout << "***********************************************************\n";
        cout << "dim(R) = " << M_.As<HypreParMatrix>()->M() << ", ";
        cout << "dim(W) = " << B_.As<HypreParMatrix>()->M() << ", ";
        cout << "dim(N) = " << collector_.hcurl_fes_->GlobalTrueVSize() << "\n";
        cout << "***********************************************************\n";
    }

    int order_quad = max(2, 2*order+1);
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
        irs_[i] = &(IntRules.Get(i, order_quad));
    }
}

void DarcyProblem::ShowError(const Vector &sol, ParMesh* mesh)
{
    u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
    p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

    double err_u  = u_.ComputeL2Error(ucoeff_, irs_);
    double norm_u = ComputeGlobalLpNorm(2, ucoeff_, *mesh, irs_);
    double err_p  = p_.ComputeL2Error(pcoeff_, irs_);
    double norm_p = ComputeGlobalLpNorm(2, pcoeff_, *mesh, irs_);

    if (!verbose_) return;
    cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
    cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
}

int main(int argc, char *argv[])
{
    StopWatch chrono;
    auto ResetTimer = [&chrono]() { chrono.Clear(); chrono.Start(); };

    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    bool verbose = (myid == 0);

    // 2. Parse command-line options.
    int order = 0;
    int num_refines = 2;
    bool show_error = false;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&num_refines, "-r", "--ref",
                   "Number of parallel refinement steps.");
    args.AddOption(&show_error, "-se", "--show-error", "-no-se", "--no-show-error",
                   "Show approximation error with respect to exact solution.");
    args.Parse();
    if (!args.Good())
    {
        if (verbose)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (verbose)
    {
        args.PrintOptions(cout);
    }

    Mesh *mesh = new Mesh(2, 2, 2, mfem::Element::HEXAHEDRON, true);
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[1] = 1;

    bool is_tet = pmesh->GetElementType(0) == Element::TETRAHEDRON;
    IterSolveParameters param;
    DFSParameters dfs_param;
    dfs_param.MG_type = order > 0 && is_tet ? AlgebraicMG : GeometricMG;
    dfs_param.B_has_nullity_one = (ess_bdr.Sum() == ess_bdr.Size());

    ResetTimer();
    DarcyProblem* darcy_problem = new DarcyProblem(pmesh, num_refines, order,
                                                   verbose, ess_bdr, dfs_param);
    if (verbose) cout << "Algebraic system assembled in " << chrono.RealTime() << "s.\n";

    HypreParMatrix& M = darcy_problem->GetM();
    HypreParMatrix& B = darcy_problem->GetB();
    const DFSDataCollector& collector = darcy_problem->GetDFSDataCollector();

    std::map<const DarcySolver*, double> solver_setup_time;

    ResetTimer();
    DarcySolver* dfs = new DivFreeSolver(M, B, collector.hcurl_fes_.get(),
                                         collector.GetData());
    solver_setup_time[dfs] = chrono.RealTime();

    ResetTimer();
    DarcySolver* bdp = new BDPMinresSolver(M, B, param);
    solver_setup_time[dfs] = chrono.RealTime();

    std::map<const DarcySolver*, std::string> solver_to_name;
    solver_to_name[dfs] = "Divergence free";
    solver_to_name[bdp] = "Block-diagonal-preconditioned MINRES";

    for (const auto& solver_pair : solver_to_name)
    {
        auto& solver = solver_pair.first;
        auto& name = solver_pair.second;

        if (verbose) cout << name << " solver:\n";

        const Vector& rhs = darcy_problem->GetRHS();
        Vector sol = darcy_problem->GetBC();
        ResetTimer();
        solver->Mult(rhs, sol);

        if (verbose) cout << "  setup time: " << solver_setup_time[solver] << "s.\n";
        if (verbose) cout << "  solve time: " << chrono.RealTime() << "s.\n";
        if (show_error) darcy_problem->ShowError(sol, pmesh);
    }

    delete dfs;
    delete bdp;
    delete darcy_problem;
    delete pmesh;
    MPI_Finalize();
    return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    if (x.Size() == 3)
    {
        zi = x(2);
    }

    u(0) = - exp(xi)*sin(yi)*cos(zi);
    u(1) = - exp(xi)*cos(yi)*cos(zi);

    if (x.Size() == 3)
    {
        u(2) = exp(xi)*sin(yi)*sin(zi);
    }
}

// Change if needed
double pFun_ex(const Vector & x)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);

    if (x.Size() == 3)
    {
        zi = x(2);
    }

    return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
    f = 0.0;
}

double gFun(const Vector & x)
{
    if (x.Size() == 3)
    {
        return -pFun_ex(x);
    }
    else
    {
        return 0;
    }
}

double f_natural(const Vector & x)
{
    return (-pFun_ex(x));
}
