/// @file pde_filter_taylor_test.cpp
/// @brief Taylor remainder and random-perturbation directional derivative tests.
///
/// All objectives and sensitivities operate on true-dof Vectors via the
/// mfem::Operator interface (PDEFilter::Mult / PDEFilter::MultTranspose).
///
/// Two objectives:
///
///   J1(rho) = 0.5 * (rho_tilde, M * rho_tilde)    [L2 norm squared]
///             dJ1/d(rho_tilde) = M * rho_tilde
///
///   J2(rho) = <ones, M * g>  where g_i = a(x_i)/(c + rho_tilde_i^p)
///             dJ2/d(rho_tilde) = M * dg  where dg_i = -p*a*rho^{p-1}/(c+rho^p)^2
///
/// Both gradient vectors live in the filter true-dof space; the full
/// sensitivity in control true-dof space is:
///
///   sens = F^T(dJ/d(rho_tilde))  =  MultTranspose(lambda_tdof)
///
/// Taylor remainder test (h = sens/||sens||):
///   R0(eps) ~ O(eps),   R1(eps) ~ O(eps^2)
///
/// Random-h directional derivative test:
///   FD(eps) = (J(rho+eps*h) - J(rho)) / eps  ->  <sens, h>   rate O(eps)

#include "pde_filter.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;
using namespace toopt;

// ===========================================================================
//  Diffusion coefficients
// ===========================================================================

class ScalarDiffCoeff : public Coefficient
{
public:
    double Eval(ElementTransformation& T, const IntegrationPoint& ip) override
    {
        double x[3]; Vector xv(x, T.GetSpaceDim()); T.Transform(ip, xv);
        return 1.0 + 0.5 * std::sin(M_PI*x[0]) * std::sin(M_PI*x[1]);
    }
};

class VectorDiffCoeff : public VectorCoefficient
{
public:
    VectorDiffCoeff(int dim) : VectorCoefficient(dim) {}
    void Eval(Vector& V, ElementTransformation&, const IntegrationPoint&) override
    { V.SetSize(vdim); V[0]=2.0; V[1]=0.5; if(vdim>2) V[2]=1.0; }
};

class MatrixDiffCoeff : public MatrixCoefficient
{
public:
    MatrixDiffCoeff(int dim) : MatrixCoefficient(dim) {}
    void Eval(DenseMatrix& M, ElementTransformation&, const IntegrationPoint&) override
    {
        M.SetSize(height,width); M=0.0;
        M(0,0)=2.0; M(0,1)=0.5; M(1,0)=0.5; M(1,1)=1.0;
        if(height>2) M(2,2)=1.0;
    }
};

class BaseCoeff : public Coefficient
{
public:
    double Eval(ElementTransformation& T, const IntegrationPoint& ip) override
    {
        double x[3]; Vector xv(x, T.GetSpaceDim()); T.Transform(ip, xv);
        return 0.5 + 0.4 * std::sin(M_PI*x[0]) * std::cos(M_PI*x[1]);
    }
};

class WeightCoeff : public Coefficient
{
public:
    double Eval(ElementTransformation& T, const IntegrationPoint& ip) override
    {
        double x[3]; Vector xv(x, T.GetSpaceDim()); T.Transform(ip, xv);
        return 1.0 + 0.3 * std::cos(M_PI*x[0]) * std::cos(M_PI*x[1]);
    }
};

// ===========================================================================
//  Global inner product on true-dof vectors
// ===========================================================================
double GlobalDot(MPI_Comm comm, const Vector& a, const Vector& b)
{
    double loc = a * b, glb = 0.0;
    MPI_Allreduce(&loc, &glb, 1, MPI_DOUBLE, MPI_SUM, comm);
    return glb;
}

double GlobalNorm(MPI_Comm comm, const Vector& v)
{
    return std::sqrt(GlobalDot(comm, v, v));
}

// ===========================================================================
//  Mass matrix helper — assembled once per filter FE space
// ===========================================================================
struct MassOp
{
    std::unique_ptr<ParBilinearForm> bf;
    std::unique_ptr<HypreParMatrix>  mat;

    void Setup(ParFiniteElementSpace& fes)
    {
        bf = std::make_unique<ParBilinearForm>(&fes);
        bf->AddDomainIntegrator(new MassIntegrator());
        bf->Assemble(); bf->Finalize();
        mat.reset(bf->ParallelAssemble());
    }

    /// y = M * x  (true-dof)
    void Mult(const Vector& x, Vector& y) const
    { y.SetSize(x.Size()); mat->Mult(x, y); }
};

// ===========================================================================
//  Objective 1: J1 = 0.5 * ||rho_tilde||^2_{L2}
//
//  All inputs/outputs are true-dof Vectors on the filter space.
//
//  Eval(rho_tilde_tdof)      -> scalar J1
//  AdjLoad(rho_tilde_tdof)   -> lambda_tdof = M * rho_tilde   (filter space)
// ===========================================================================
struct Objective1
{
    const MassOp& mass;
    MPI_Comm      comm;

    Objective1(const MassOp& m, MPI_Comm c) : mass(m), comm(c) {}

    double Eval(const Vector& rho_tilde) const
    {
        Vector Mrho;
        mass.Mult(rho_tilde, Mrho);
        return 0.5 * GlobalDot(comm, rho_tilde, Mrho);
    }

    void AdjLoad(const Vector& rho_tilde, Vector& lambda) const
    {
        mass.Mult(rho_tilde, lambda);   // lambda = M * rho_tilde
    }
};

// ===========================================================================
//  Objective 2: J2 = integral_Omega a(x)/(c + rho_tilde^p) dx
//
//  Pointwise coefficients are applied via ProjectCoefficient onto a
//  ParGridFunction, then multiplied by the mass matrix.
//
//  The ParGridFunction is only used for projection; arithmetic is in true-dofs.
// ===========================================================================

/// g(x) = a(x) / (c + f^p)  where f is a ParGridFunction
class IntegrandJ2Coeff : public Coefficient
{
public:
    IntegrandJ2Coeff(const ParGridFunction& f, Coefficient& a, double c, double p)
        : f_(f), a_(a), c_(c), p_(p) {}
    double Eval(ElementTransformation& T, const IntegrationPoint& ip) override
    {
        return a_.Eval(T,ip) / (c_ + std::pow(f_.GetValue(T,ip), p_));
    }
private:
    const ParGridFunction& f_; Coefficient& a_; double c_, p_;
};

/// dg/df = -p*a*f^{p-1} / (c + f^p)^2
class AdjLoadJ2Coeff : public Coefficient
{
public:
    AdjLoadJ2Coeff(const ParGridFunction& f, Coefficient& a, double c, double p)
        : f_(f), a_(a), c_(c), p_(p) {}
    double Eval(ElementTransformation& T, const IntegrationPoint& ip) override
    {
        double fv = f_.GetValue(T,ip), av = a_.Eval(T,ip);
        double d = c_ + std::pow(fv, p_);
        return -p_ * av * std::pow(fv, p_-1.0) / (d*d);
    }
private:
    const ParGridFunction& f_; Coefficient& a_; double c_, p_;
};

struct Objective2
{
    const MassOp&     mass;
    MPI_Comm          comm;
    ParFiniteElementSpace& fes_filter;   // for GridFunction projection
    double c, p;
    mutable WeightCoeff weight;          // mutable: Coefficient::Eval not const

    Objective2(const MassOp& m, MPI_Comm cm,
               ParFiniteElementSpace& fes,
               double c_, double p_)
        : mass(m), comm(cm), fes_filter(fes), c(c_), p(p_) {}

    double Eval(const Vector& rho_tilde_tdof) const
    {
        // Lift true-dofs to GridFunction for pointwise evaluation
        ParGridFunction gf(&fes_filter);
        gf.SetFromTrueDofs(rho_tilde_tdof);

        // Project integrand onto FE space
        ParGridFunction g(&fes_filter);
        IntegrandJ2Coeff coeff(gf, weight, c, p);
        g.ProjectCoefficient(coeff);

        // J2 = <M*g, ones> = GlobalDot(M*g_tdof, ones_tdof)
        Vector g_tdof, Mg_tdof;
        g.GetTrueDofs(g_tdof);
        mass.Mult(g_tdof, Mg_tdof);

        Vector ones(g_tdof.Size()); ones = 1.0;
        return GlobalDot(comm, Mg_tdof, ones);
    }

    void AdjLoad(const Vector& rho_tilde_tdof, Vector& lambda) const
    {
        ParGridFunction gf(&fes_filter);
        gf.SetFromTrueDofs(rho_tilde_tdof);

        ParGridFunction dg(&fes_filter);
        AdjLoadJ2Coeff coeff(gf, weight, c, p);
        dg.ProjectCoefficient(coeff);

        // lambda = M * dg_tdof
        Vector dg_tdof;
        dg.GetTrueDofs(dg_tdof);
        mass.Mult(dg_tdof, lambda);
    }
};

// ===========================================================================
//  Formatting helper
// ===========================================================================
static std::string fmtRate(double v)
{
    if (std::isnan(v)) return "  ---";
    std::ostringstream s; s << std::fixed << std::setprecision(3) << v;
    return s.str();
}

// ===========================================================================
//  Core Taylor remainder test — true-dof Vectors throughout
// ===========================================================================
template<typename ObjType>
bool RunTaylorTest(const std::string&      label,
                   PDEFilter&              filter,
                   const Vector&           rho0_tdof,
                   ObjType&                obj,
                   int                     n_steps)
{
    const bool root = (Mpi::WorldRank() == 0);
    MPI_Comm   comm = filter.GetComm();
    if (root) std::cout << "\n--- " << label << " ---\n";

    // J0 and adjoint sensitivity at rho0
    Vector rho_tilde0(filter.Height());
    filter.Mult(rho0_tdof, rho_tilde0);
    const double J0 = obj.Eval(rho_tilde0);

    Vector lambda(filter.Height());
    obj.AdjLoad(rho_tilde0, lambda);

    Vector sens(filter.Width());
    filter.MultTranspose(lambda, sens);

    // Perturbation h = sens / ||sens||  (guarantees <sens,h> = ||sens|| > 0)
    const double sens_norm = GlobalNorm(comm, sens);
    MFEM_VERIFY(sens_norm > 0.0, "sens is zero for: " + label);

    Vector h(sens); h /= sens_norm;

    const double dJ_dh = GlobalDot(comm, sens, h);   // = ||sens||
    const double rel   = std::abs(dJ_dh) / (std::abs(J0) + 1.0);

    if (root)
    {
        std::cout << std::scientific << std::setprecision(4)
                  << "  J0=" << J0 << "  ||sens||=" << sens_norm
                  << "  <sens,h>=" << dJ_dh << "  rel=" << rel << "\n";
        if (rel < 1e-6) std::cout << "  WARNING: near-zero directional derivative\n";
        std::cout << std::setw(10) << "eps"
                  << std::setw(14) << "R0"
                  << std::setw(14) << "R1"
                  << std::setw(9)  << "rateR0"
                  << std::setw(9)  << "rateR1" << "\n";
    }

    std::vector<double> R0v, R1v;
    double eps = 0.1;
    Vector rho_pert(filter.Width()), rho_tilde_pert(filter.Height());

    for (int i = 0; i < n_steps; ++i, eps *= 0.5)
    {
        rho_pert = rho0_tdof; rho_pert.Add(eps, h);
        filter.Mult(rho_pert, rho_tilde_pert);
        const double Jp = obj.Eval(rho_tilde_pert);

        const double R0 = std::abs(Jp - J0);
        const double R1 = std::abs(Jp - J0 - eps * dJ_dh);
        R0v.push_back(R0); R1v.push_back(R1);

        double r0 = std::numeric_limits<double>::quiet_NaN();
        double r1 = std::numeric_limits<double>::quiet_NaN();
        if (i>0 && R0v[i-1]>1e-15 && R0v[i]>1e-15)
            r0 = std::log(R0v[i-1]/R0v[i])/std::log(2.0);
        if (i>0 && R1v[i-1]>1e-15 && R1v[i]>1e-15)
            r1 = std::log(R1v[i-1]/R1v[i])/std::log(2.0);

        if (root)
            std::cout << std::scientific << std::setprecision(3)
                      << std::setw(10) << eps
                      << std::setw(14) << R0
                      << std::setw(14) << R1
                      << std::setw(9)  << fmtRate(r0)
                      << std::setw(9)  << fmtRate(r1) << "\n";
    }

    double avg_r1=0.0; int cnt=0;
    for (int i=std::max(1,n_steps-3); i<n_steps; ++i)
        if (R1v[i-1]>1e-15 && R1v[i]>1e-15)
        { avg_r1 += std::log(R1v[i-1]/R1v[i])/std::log(2.0); ++cnt; }
    if (cnt) avg_r1 /= cnt;

    const bool ok = (rel >= 1e-6) && (avg_r1 >= 1.95);
    if (root)
        std::cout << std::fixed << std::setprecision(3)
                  << "  avg R1 rate=" << avg_r1
                  << "  non-trivial=" << (rel>=1e-6?"yes":"NO")
                  << "  => " << (ok ? "PASSED ✓" : "FAILED ✗") << "\n";
    return ok;
}

// ===========================================================================
//  Random perturbation directional derivative test — true-dof Vectors
//
//  FD(eps) = (J(rho+eps*h) - J(rho)) / eps  ->  <sens,h>  at rate O(eps)
// ===========================================================================
template<typename ObjType>
bool RunRandomPerturbTest(const std::string& label,
                          PDEFilter&         filter,
                          const Vector&      rho0_tdof,
                          ObjType&           obj,
                          int                n_steps,
                          int                seed = 42)
{
    const bool root = (Mpi::WorldRank() == 0);
    MPI_Comm   comm = filter.GetComm();
    if (root) std::cout << "\n--- " << label << " (random h) ---\n";

    // Gradient at rho0
    Vector rho_tilde0(filter.Height());
    filter.Mult(rho0_tdof, rho_tilde0);
    const double J0 = obj.Eval(rho_tilde0);

    Vector lambda(filter.Height());
    obj.AdjLoad(rho_tilde0, lambda);

    Vector sens(filter.Width());
    filter.MultTranspose(lambda, sens);

    // Random unit perturbation, identical across ranks via deterministic seed
    // Each rank generates its slice by advancing the RNG to its global offset.
    Vector h(filter.Width());
    {
        ParFiniteElementSpace* fes = filter.GetControlFESpace();
        std::mt19937 rng(static_cast<unsigned>(seed));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        HYPRE_BigInt offset = fes->GetMyTDofOffset();
        rng.discard(static_cast<unsigned long long>(offset));
        for (int i = 0; i < h.Size(); i++) h[i] = dist(rng);
        h /= GlobalNorm(comm, h);
    }

    const double dJ_adj = GlobalDot(comm, sens, h);
    const double rel    = std::abs(dJ_adj) / (std::abs(J0) + 1.0);

    if (root)
    {
        std::cout << std::scientific << std::setprecision(4)
                  << "  J0=" << J0 << "  <sens,h>_adj=" << dJ_adj
                  << "  rel=" << rel << "\n";
        if (rel < 1e-8) std::cout << "  WARNING: near-zero adjoint projection on h\n";
        std::cout << std::setw(10) << "eps"
                  << std::setw(18) << "FD=(J(rho+eh)-J)/e"
                  << std::setw(16) << "err=|FD-adj|"
                  << std::setw(10) << "rate" << "\n";
    }

    std::vector<double> errs;
    double eps = 0.1;
    Vector rho_pert(filter.Width()), rho_tilde_pert(filter.Height());

    for (int i = 0; i < n_steps; ++i, eps *= 0.5)
    {
        rho_pert = rho0_tdof; rho_pert.Add(eps, h);
        filter.Mult(rho_pert, rho_tilde_pert);
        const double Jp  = obj.Eval(rho_tilde_pert);
        const double fd  = (Jp - J0) / eps;
        const double err = std::abs(fd - dJ_adj);
        errs.push_back(err);

        double rate = std::numeric_limits<double>::quiet_NaN();
        if (i>0 && errs[i-1]>1e-15 && errs[i]>1e-15)
            rate = std::log(errs[i-1]/errs[i])/std::log(2.0);

        if (root)
            std::cout << std::scientific << std::setprecision(3)
                      << std::setw(10) << eps
                      << std::setw(18) << fd
                      << std::setw(16) << err
                      << std::setw(10) << fmtRate(rate) << "\n";
    }

    double avg=0.0; int cnt=0;
    for (int i=std::max(1,n_steps-3); i<n_steps; ++i)
        if (errs[i-1]>1e-15 && errs[i]>1e-15)
        { avg += std::log(errs[i-1]/errs[i])/std::log(2.0); ++cnt; }
    if (cnt) avg /= cnt;

    const bool ok = (rel >= 1e-8) && (avg >= 0.90);
    if (root)
        std::cout << std::fixed << std::setprecision(3)
                  << "  avg FD rate=" << avg
                  << "  non-trivial=" << (rel>=1e-8?"yes":"NO")
                  << "  => " << (ok ? "PASSED ✓" : "FAILED ✗") << "\n";
    return ok;
}

// ===========================================================================
//  main
// ===========================================================================
int main(int argc, char* argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();

    int    ref_levels = 3;
    int    order      = 1;
    int    n_steps    = 6;
    double j2_c       = 0.1;
    double j2_p       = 3.0;

    OptionsParser args(argc, argv);
    args.AddOption(&ref_levels, "-r", "--refine",   "Uniform refinement levels");
    args.AddOption(&order,      "-o", "--order",    "FE order (>= 1)");
    args.AddOption(&n_steps,    "-n", "--n-steps",  "Number of epsilon halvings");
    args.AddOption(&j2_c,       "-c", "--j2-const", "Constant c in J2");
    args.AddOption(&j2_p,       "-p", "--j2-exp",   "Exponent p in J2");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(std::cout); return 1; }
    if (Mpi::Root())
    {
        args.PrintOptions(std::cout);
        std::cout << "  J2 parameters: c=" << j2_c << "  p=" << j2_p << "\n";
    }

    // Mesh
    Mesh smesh = Mesh::MakeCartesian2D(8,8,Element::QUADRILATERAL,true,1.0,1.0);
    for (int l=0; l<ref_levels; l++) smesh.UniformRefinement();
    ParMesh pmesh(MPI_COMM_WORLD, smesh); smesh.Clear();
    const int dim = pmesh.Dimension();

    // FE spaces
    H1_FECollection fec_h1(order,     dim);
    L2_FECollection fec_l2(order - 1, dim);

    ParFiniteElementSpace fes_h1_filt(&pmesh, &fec_h1);
    ParFiniteElementSpace fes_h1_ctrl(&pmesh, &fec_h1);
    ParFiniteElementSpace fes_l2_ctrl(&pmesh, &fec_l2);

    MPI_Comm comm = MPI_COMM_WORLD;

    // Mass matrix on filter space (shared by both objectives)
    MassOp mass; mass.Setup(fes_h1_filt);

    // Objectives
    Objective1 obj1(mass, comm);
    Objective2 obj2(mass, comm, fes_h1_filt, j2_c, j2_p);

    // Diffusion coefficients
    ScalarDiffCoeff sc_diff;
    VectorDiffCoeff vc_diff(dim);
    MatrixDiffCoeff mc_diff(dim);

    PDEFilterOptions opts;
    opts.filter_radius = 0.05;
    opts.print_level   = 0;

    // Base density as true-dof Vector — projected via GridFunction
    auto makeRho0 = [&](ParFiniteElementSpace& fes_ctrl) -> Vector
    {
        ParGridFunction gf(&fes_ctrl);
        BaseCoeff bc; gf.ProjectCoefficient(bc);
        Vector v; gf.GetTrueDofs(v);
        return v;
    };

    // Helper: build filter and run both test types for one configuration
    int n_pass=0, n_total=0;
    auto run = [&](const std::string& label,
                   ParFiniteElementSpace& fes_ctrl,
                   Coefficient* sc, VectorCoefficient* vc, MatrixCoefficient* mc,
                   auto& obj)
    {
        PDEFilter filter(fes_h1_filt, fes_ctrl, opts);
        if      (sc) filter.SetDiffusionCoeff(*sc);
        else if (vc) filter.SetDiffusionCoeff(*vc);
        else if (mc) filter.SetDiffusionCoeff(*mc);
        filter.Assemble();

        Vector rho0 = makeRho0(fes_ctrl);

        ++n_total; if (RunTaylorTest       (label, filter, rho0, obj, n_steps)) ++n_pass;
        ++n_total; if (RunRandomPerturbTest(label, filter, rho0, obj, n_steps)) ++n_pass;
    };

    if (Mpi::Root())
        std::cout << "\n========================================\n"
                  << " PDE Filter Taylor remainder tests\n"
                  << "========================================\n";

    // ---- J1, L2 control ----
    if (Mpi::Root()) std::cout << "\n==== J1 = 0.5*||rho_tilde||^2_L2 | L2 control ====\n";
    run("J1|L2|default", fes_l2_ctrl, nullptr,  nullptr,  nullptr,  obj1);
    run("J1|L2|scalar",  fes_l2_ctrl, &sc_diff, nullptr,  nullptr,  obj1);
    run("J1|L2|vector",  fes_l2_ctrl, nullptr,  &vc_diff, nullptr,  obj1);
    run("J1|L2|matrix",  fes_l2_ctrl, nullptr,  nullptr,  &mc_diff, obj1);

    // ---- J1, H1 control ----
    if (Mpi::Root()) std::cout << "\n==== J1 = 0.5*||rho_tilde||^2_L2 | H1 control ====\n";
    run("J1|H1|default", fes_h1_ctrl, nullptr,  nullptr,  nullptr,  obj1);
    run("J1|H1|scalar",  fes_h1_ctrl, &sc_diff, nullptr,  nullptr,  obj1);
    run("J1|H1|vector",  fes_h1_ctrl, nullptr,  &vc_diff, nullptr,  obj1);
    run("J1|H1|matrix",  fes_h1_ctrl, nullptr,  nullptr,  &mc_diff, obj1);

    // ---- J2, L2 control ----
    if (Mpi::Root())
    {
        std::ostringstream oss; oss << std::fixed << std::setprecision(2);
        oss << "\n==== J2 = int a/(c+rho^p), c=" << j2_c << " p=" << j2_p
            << " | L2 control ====\n";
        std::cout << oss.str();
    }
    run("J2|L2|default", fes_l2_ctrl, nullptr,  nullptr,  nullptr,  obj2);
    run("J2|L2|scalar",  fes_l2_ctrl, &sc_diff, nullptr,  nullptr,  obj2);
    run("J2|L2|vector",  fes_l2_ctrl, nullptr,  &vc_diff, nullptr,  obj2);
    run("J2|L2|matrix",  fes_l2_ctrl, nullptr,  nullptr,  &mc_diff, obj2);

    // ---- J2, H1 control ----
    if (Mpi::Root())
    {
        std::ostringstream oss; oss << std::fixed << std::setprecision(2);
        oss << "\n==== J2 = int a/(c+rho^p), c=" << j2_c << " p=" << j2_p
            << " | H1 control ====\n";
        std::cout << oss.str();
    }
    run("J2|H1|default", fes_h1_ctrl, nullptr,  nullptr,  nullptr,  obj2);
    run("J2|H1|scalar",  fes_h1_ctrl, &sc_diff, nullptr,  nullptr,  obj2);
    run("J2|H1|vector",  fes_h1_ctrl, nullptr,  &vc_diff, nullptr,  obj2);
    run("J2|H1|matrix",  fes_h1_ctrl, nullptr,  nullptr,  &mc_diff, obj2);

    if (Mpi::Root())
        std::cout << "\n========================================\n"
                  << " Summary: " << n_pass << " / " << n_total
                  << " tests passed\n"
                  << "========================================\n";

    return (n_pass == n_total) ? 0 : 1;
}
