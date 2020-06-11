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
#include "div_free_solver.hpp"
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
    ParMesh mesh_;
    VectorFunctionCoefficient ucoeff_;
    FunctionCoefficient pcoeff_;
    DFSDataCollector collector_;
    const IntegrationRule *irs_[Geometry::NumGeom];
public:
    DarcyProblem(Mesh& mesh, int num_refines, int order,
                 Array<int>& ess_bdr, DFSParameters param);

    HypreParMatrix& GetM() { return *M_.As<HypreParMatrix>(); }
    HypreParMatrix& GetB() { return *B_.As<HypreParMatrix>(); }
    const Vector& GetRHS() { return rhs_; }
    const Vector& GetBC() { return ess_data_; }
    const DFSDataCollector& GetDFSDataCollector() const { return collector_; }

    void ShowError(const Vector& sol, bool verbose);
};

DarcyProblem::DarcyProblem(Mesh& mesh, int num_refines, int order,
                           Array<int>& ess_bdr, DFSParameters dfs_param)
    : mesh_(MPI_COMM_WORLD, mesh), ucoeff_(mesh.Dimension(), uFun_ex),
      pcoeff_(pFun_ex), collector_(order, num_refines, &mesh_, ess_bdr, dfs_param)
{
    for (int l = 0; l < num_refines; l++)
    {
        mesh_.UniformRefinement();
        collector_.CollectData();
    }

    // 8. Define the coefficients, analytical solution, and rhs of the PDE.
    VectorFunctionCoefficient fcoeff(mesh_.Dimension(), fFun);
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
    ess_data_ = 0.0;
    Vector ess_data_block0(ess_data_.GetData(), M_->NumRows());
    u_.ParallelProject(ess_data_block0);

    int order_quad = max(2, 2*order+1);
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
        irs_[i] = &(IntRules.Get(i, order_quad));
    }
}

void DarcyProblem::ShowError(const Vector &sol, bool verbose)
{
    u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
    p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

    double err_u  = u_.ComputeL2Error(ucoeff_, irs_);
    double norm_u = ComputeGlobalLpNorm(2, ucoeff_, mesh_, irs_);
    double err_p  = p_.ComputeL2Error(pcoeff_, irs_);
    double norm_p = ComputeGlobalLpNorm(2, pcoeff_, mesh_, irs_);

    if (!verbose) return;
    cout << "\n|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
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
    bool use_tet_mesh = false;
    bool coupled_solve = true;
    bool show_error = false;
    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&num_refines, "-r", "--ref",
                   "Number of parallel refinement steps.");
    args.AddOption(&use_tet_mesh, "-tet", "--tet-mesh", "-hex", "--hex-mesh",
                   "Use a tetrahedral or hexahedral mesh (on unit cube).");
    args.AddOption(&coupled_solve, "-cs", "--coupled-solve", "-ss", "--separate-solve",
                   "Whether to solve all unknowns together in divergence free solver.");
    args.AddOption(&show_error, "-se", "--show-error", "-no-se", "--no-show-error",
                   "Show or not show approximation error.");
    args.Parse();
    if (!args.Good())
    {
        if (verbose) args.PrintUsage(cout);
        MPI_Finalize();
        return 1;
    }
    if (verbose) args.PrintOptions(cout);

    auto elem_type = use_tet_mesh ? Element::TETRAHEDRON : Element::HEXAHEDRON;
    Mesh mesh(2, 2, 2, elem_type, true);
    for (int i = 0; i < (int)(log(num_procs)/log(8)); ++i)
         mesh.UniformRefinement();

    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[1] = 1;

    DFSParameters param;
    param.B_has_nullity_one = (ess_bdr.Sum() == ess_bdr.Size());
    param.coupled_solve = coupled_solve;

    string line = "\n*******************************************************\n";
    {
        ResetTimer();
        DarcyProblem darcy(mesh, num_refines, order, ess_bdr, param);
        HypreParMatrix& M = darcy.GetM();
        HypreParMatrix& B = darcy.GetB();
        const DFSDataCollector& collector = darcy.GetDFSDataCollector();

        if (verbose)
        {
            cout << line << "dim(R) = " << M.M() << ", dim(W) = " << B.M() << ", ";
            cout << "dim(N) = " << collector.hcurl_fes_->GlobalTrueVSize() << "\n";
            cout << "System assembled in " << chrono.RealTime() << "s.\n";
        }

        std::map<const DarcySolver*, double> setup_time;
        ResetTimer();
        DivFreeSolver dfs(M, B, collector.hcurl_fes_.get(), collector.GetData());
        setup_time[&dfs] = chrono.RealTime();

        ResetTimer();
        BDPMinresSolver bdp(M, B, false, param);
        setup_time[&bdp] = chrono.RealTime();

        std::map<const DarcySolver*, std::string> solver_to_name;
        solver_to_name[&dfs] = "Divergence free";
        solver_to_name[&bdp] = "Block-diagonal-preconditioned MINRES";

        for (const auto& solver_pair : solver_to_name)
        {
            auto& solver = solver_pair.first;
            auto& name = solver_pair.second;

            if (verbose) cout << line << name << " solver:\n";
            if (verbose) cout << "  Setup time: " << setup_time[solver] << "s.\n";

            const Vector& rhs = darcy.GetRHS();
            Vector sol = darcy.GetBC();
            ResetTimer();
            solver->Mult(rhs, sol);
            chrono.Stop();

            if (verbose) cout << "  Solve time: " << chrono.RealTime() << "s.\n";
            if (verbose) cout << "  Total time: " <<
                                 setup_time[solver] + chrono.RealTime() << "s.\n";
            if (verbose) cout << "  Iteration count: "
                              << solver->GetNumIterations() <<"\n";
            if (show_error) darcy.ShowError(sol, verbose);
        }
    }

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
