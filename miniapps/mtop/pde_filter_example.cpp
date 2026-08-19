/// @file pde_filter_example.cpp
/// @brief Demonstrates PDEFilter with externally constructed FE spaces.
///
/// Build (adapt to your MFEM install):
///   mpicxx -O2 -std=c++17                        \
///     -I${MFEM_DIR}/include                       \
///     -L${MFEM_DIR}/lib -lmfem -lHYPRE -lmetis   \
///     pde_filter.cpp pde_filter_example.cpp -o pde_filter_ex
///
/// Run:
///   mpirun -np 4 ./pde_filter_ex -r 3 -o 1

#include "pde_filter.hpp"
#include <iostream>

using namespace mfem;
using namespace toopt;

// ---------------------------------------------------------------------------
// Coefficients
// ---------------------------------------------------------------------------

/// a(x) = 1 + 0.5 sin(πx) sin(πy)
class SinScalarCoeff : public Coefficient
{
public:
    double Eval(ElementTransformation& T, const IntegrationPoint& ip) override
    {
        double x[3]; Vector xv(x, T.GetSpaceDim());
        T.Transform(ip, xv);
        return 1.0 + 0.5 * std::sin(M_PI*x[0]) * std::sin(M_PI*x[1]);
    }
};

/// v(x) = (2, 0.5)  — axis-aligned anisotropic
class DiagVecCoeff : public VectorCoefficient
{
public:
    DiagVecCoeff(int dim) : VectorCoefficient(dim) {}
    void Eval(Vector& V, ElementTransformation&,
              const IntegrationPoint&) override
    {
        V.SetSize(vdim);
        V[0] = 2.0; V[1] = 0.5;
        if (vdim > 2) V[2] = 1.0;
    }
};

/// M = [[2, 0.5],[0.5, 1]]  — full tensor
class FullMatCoeff : public MatrixCoefficient
{
public:
    FullMatCoeff(int dim) : MatrixCoefficient(dim) {}
    void Eval(DenseMatrix& M, ElementTransformation&,
              const IntegrationPoint&) override
    {
        M.SetSize(height, width); M = 0.0;
        M(0,0)=2.0; M(0,1)=0.5; M(1,0)=0.5; M(1,1)=1.0;
        if (height > 2) M(2,2)=1.0;
    }
};

/// Checkerboard-like input density
class CheckerCoeff : public Coefficient
{
public:
    double Eval(ElementTransformation& T, const IntegrationPoint& ip) override
    {
        double x[3]; Vector xv(x, T.GetSpaceDim());
        T.Transform(ip, xv);
        return 0.5 + 0.5*std::sin(20.0*M_PI*x[0])*std::sin(20.0*M_PI*x[1]);
    }
};

// ---------------------------------------------------------------------------
// Helper: run one demo, print stats, save ParaView output
// ---------------------------------------------------------------------------
void RunDemo(const std::string& label,
             ParFiniteElementSpace& fes_filter,
             ParFiniteElementSpace& fes_control,
             const PDEFilterOptions& opts,
             std::function<void(PDEFilter&)> configure)
{
    PDEFilter filter(fes_filter, fes_control, opts);
    configure(filter);
    filter.Assemble();

    ParGridFunction rho(&fes_control);
    CheckerCoeff input;
    rho.ProjectCoefficient(input);

    ParGridFunction rho_tilde(&fes_filter);
    filter.Mult(rho, rho_tilde);

    ParGridFunction sens(&fes_control);
    filter.MultTranspose(rho_tilde, sens);

    if (Mpi::Root())
    {
        std::cout << "\n=== " << label << " ===\n"
                  << "  rho       : [" << rho.Min()       << ", " << rho.Max()       << "]\n"
                  << "  rho_tilde : [" << rho_tilde.Min() << ", " << rho_tilde.Max() << "]\n"
                  << "  sens      : [" << sens.Min()       << ", " << sens.Max()       << "]\n";
    }

    ParaViewDataCollection pv(label, fes_filter.GetParMesh());
    pv.SetPrefixPath("output");
    pv.RegisterField("rho",       &rho);
    pv.RegisterField("rho_tilde", &rho_tilde);
    pv.RegisterField("sens",      &sens);
    pv.Save();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();

    int ref_levels = 3;
    int order      = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&ref_levels, "-r", "--refine", "Uniform refinement levels");
    args.AddOption(&order,      "-o", "--order",  "FE polynomial order (>= 1)");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(std::cout); return 1; }
    if (Mpi::Root())  { args.PrintOptions(std::cout); }

    // ------------------------------------------------------------------
    // Mesh
    // ------------------------------------------------------------------
    Mesh smesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL,
                                        true, 1.0, 1.0);
    for (int l = 0; l < ref_levels; l++) smesh.UniformRefinement();
    ParMesh pmesh(MPI_COMM_WORLD, smesh);
    smesh.Clear();
    const int dim = pmesh.Dimension();

    // ------------------------------------------------------------------
    // FE collections — built once, shared across all demos
    // ------------------------------------------------------------------
    H1_FECollection fec_h1(order,   dim);
    L2_FECollection fec_l2(order-1, dim);

    ParFiniteElementSpace fes_h1_filter (&pmesh, &fec_h1);  // solution
    ParFiniteElementSpace fes_l2_control(&pmesh, &fec_l2);  // L2 control
    ParFiniteElementSpace fes_h1_control(&pmesh, &fec_h1);  // H1 control

    PDEFilterOptions opts;
    opts.filter_radius = 0.05;
    opts.print_level   = 0;

    // ------------------------------------------------------------------
    // Demo 1: default (identity) diffusion, L2 control
    // ------------------------------------------------------------------
    RunDemo("Default_L2control", fes_h1_filter, fes_l2_control, opts,
            [](PDEFilter&) {});   // no diffusion coeff → r^2 I

    // ------------------------------------------------------------------
    // Demo 2: scalar diffusion, L2 control
    // ------------------------------------------------------------------
    {
        SinScalarCoeff sc;
        RunDemo("ScalarDiff_L2control", fes_h1_filter, fes_l2_control, opts,
                [&](PDEFilter& f) { f.SetDiffusionCoeff(sc); });
    }

    // ------------------------------------------------------------------
    // Demo 3: diagonal (vector) diffusion, L2 control
    // ------------------------------------------------------------------
    {
        DiagVecCoeff vc(dim);
        RunDemo("VectorDiff_L2control", fes_h1_filter, fes_l2_control, opts,
                [&](PDEFilter& f) { f.SetDiffusionCoeff(vc); });
    }

    // ------------------------------------------------------------------
    // Demo 4: full-tensor (matrix) diffusion, L2 control
    // ------------------------------------------------------------------
    {
        FullMatCoeff mc(dim);
        RunDemo("MatrixDiff_L2control", fes_h1_filter, fes_l2_control, opts,
                [&](PDEFilter& f) { f.SetDiffusionCoeff(mc); });
    }


    // ------------------------------------------------------------------
    // Demo 5: H1 control space (same filter space, different control)
    // ------------------------------------------------------------------
    {
        SinScalarCoeff sc;
        RunDemo("ScalarDiff_H1control", fes_h1_filter, fes_h1_control, opts,
                [&](PDEFilter& f) { f.SetDiffusionCoeff(sc); });
    }

    if (Mpi::Root())
        std::cout << "\nAll demos complete. ParaView output in ./output/\n";

    return 0;
}
