#include <cassert>
#include <functional>
#include <iostream>
#include <variant>
#include <vector>

#include "dfem.hpp"

#include "eigen-3.4.0/Eigen/Eigen"

using namespace mfem;
using mfem::internal::tensor;
using mfem::internal::dual;

class LambdaOperator : public Operator
{
public:
    LambdaOperator(int size,
                   std::function<void(const Vector&, Vector&)> mult_f) :
        Operator(size),
        mult_f(mult_f)
    {}

    void Mult(const Vector& X, Vector& Y) const
    {
        mult_f(X, Y);
    }

    std::function<void(const Vector&, Vector&)> mult_f;
};

class ADOperator : public Operator
{
public:
    ADOperator(int size) : Operator(size) {}
    virtual void GradientMult(const Vector &dX, Vector &Y) const = 0;
    virtual void AdjointMult(const Vector &L, Vector &Y) const = 0;
};

class PLaplacianGradientOperator : public Operator
{
public:
    PLaplacianGradientOperator(ADOperator &op) :
        Operator(op.Height()), op(op) {}

    void Mult(const Vector &x, Vector &y) const override
    {
        op.GradientMult(x, y);
    }

    ADOperator &op;
};

class PLaplacianAdjointOperator : public Operator
{
public:
    PLaplacianAdjointOperator(ADOperator &op) :
        Operator(op.Height()), op(op) {}

    void Mult(const Vector &x, Vector &y) const override
    {
        op.AdjointMult(x, y);
    }

    ADOperator &op;
};

class PLaplacianOperator : public ADOperator
{
public:
    PLaplacianOperator(ParFiniteElementSpace &fes, ParGridFunction &u) :
        ADOperator(fes.GetTrueVSize()),
        mesh(fes.GetParMesh()),
        fes(fes),
        u(u.ParFESpace()),
        l(u.ParFESpace()),
        ir(const_cast<IntegrationRule &>(IntRules.Get(
                                             Element::QUADRILATERAL,
                                             2 * mesh->GetNodes()->FESpace()->GetOrder(0) + 1)))
    {
        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 1;
        fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        du.SetSpace(u.FESpace());
        x_lvec.SetSize(fes.GetProlongationMatrix()->Height());
        x_lvec = 0.0;
        y_lvec.SetSize(fes.GetProlongationMatrix()->Height());
        du_tvec.SetSize(fes.GetProlongationMatrix()->Width());
        l_tvec.SetSize(fes.GetProlongationMatrix()->Width());

        gradient = new PLaplacianGradientOperator(*this);
        adjoint = new PLaplacianAdjointOperator(*this);
    }

    void SetVolumeForcing(ParGridFunction &forcing)
    {
        volume_force = &forcing;
    }

    // F(u, g) = (pow(norm(grad_u), 0.5 * (p - 2)) * grad u, grad phi) - (g, phi)
    void Mult(const Vector &x, Vector &y) const override
    {
        // T -> L
        fes.GetProlongationMatrix()->Mult(x, u);

        // L -> Q

        // [vdim, dim, num_qp, num_el]
        auto grad_u_qp = gradient_wrt_x(u, ir);

        // Q -> Q
        auto plap = [](tensor<double, 2> grad_u)
        {
            // grad_u := [dudx dudy]
            const int p = 4;
            return (pow(norm(grad_u), 0.5 * (p - 2)) * grad_u);
        };

        auto f_grad_u_qp = forall(plap, ir.GetNPoints() * mesh->GetNE(),
                                  grad_u_qp);

        // Layout of f_grad_u_qp_flat has to be [vdim, dim, num_qp, num_el]
        Vector f_grad_u_qp_flat((double *)f_grad_u_qp.GetData(),
                                1 * 2 * ir.GetNPoints() *
                                mesh->GetNE());

        Vector f_grad_u_grad_phi = integrate_basis_gradient(f_grad_u_qp_flat,
                                   fes,
                                   ir);

        // - (g, phi)
        auto g_qp = interpolate(*volume_force, ir);
        Vector g_phi = integrate_basis(g_qp, fes, ir);
        f_grad_u_grad_phi -= g_phi;

        // L -> T
        fes.GetProlongationMatrix()->MultTranspose(f_grad_u_grad_phi, y);

        y.SetSubVector(ess_tdof_list, 0.0);
    }

    // df/du
    Operator &GetGradient(const Vector &x) const override
    {
        // T -> L
        fes.GetProlongationMatrix()->Mult(x, state_lvec);
        return *gradient;
    }

    // dX: current iterate
    // Y: dR/dU * dX
    void GradientMult(const Vector &X, Vector &Y) const override
    {
        // apply essential bcs
        du_tvec = X;
        du_tvec.SetSubVector(ess_tdof_list, 0.0);

        du.SetFromTrueDofs(du_tvec);
        u = state_lvec;

        auto grad_u_qp = gradient_wrt_x(u, ir);
        auto grad_du_qp = gradient_wrt_x(du, ir);

        const int N = mesh->GetNE() * ir.GetNPoints();

        auto plap2 = [](tensor<double, 2> &grad_u)
        {
            const int p = 4;
            return (pow(norm(grad_u), 0.5 * (p - 2)) * grad_u);
        };

        auto flux_qp = forall([&, plap2](tensor<double,2> grad_u,
                                         tensor<double,2> grad_du)
        {
            return fwddiff(+plap2)(grad_u, grad_du);
        }, N, grad_u_qp, grad_du_qp);

        // has to be [vdim, dim, num_qp, num_el]
        Vector flux_qp_flat((double *)flux_qp.GetData(), 1 * 2 * N);

        Vector y = integrate_basis_gradient(flux_qp_flat, fes, ir);

        // L-vector to T-vector
        fes.GetProlongationMatrix()->MultTranspose(y, Y);

        // Re-assign the essential degrees of freedom on the final output vector.
        for (int i = 0; i < ess_tdof_list.Size(); i++)
        {
            Y[ess_tdof_list[i]] = X[ess_tdof_list[i]];
        }
    }

    // (dF/dU)^t
    Operator &GetAdjoint(const Vector &u) const
    {
        // T -> L
        fes.GetProlongationMatrix()->Mult(u, state_lvec);
        return *adjoint;
    }

    // dL: current iterate of adjoint state
    // Y: (dF/dU)^t * dL
    void AdjointMult(const Vector &dL, Vector &Y) const override
    {
        // apply essential bcs
        l_tvec = dL;
        l_tvec.SetSubVector(ess_tdof_list, 0.0);

        l.SetFromTrueDofs(l_tvec);
        u = state_lvec;

        auto grad_u_qp = gradient_wrt_x(u, ir);
        // L * B^t -> B * L
        auto grad_l_qp = gradient_wrt_x(l, ir);

        const int N = mesh->GetNE() * ir.GetNPoints();

        auto plap2 = [](tensor<double, 2> &grad_u, tensor<double, 2> &flux)
        {
            const int p = 4;
            flux = (pow(norm(grad_u), 0.5 * (p - 2)) * grad_u);
        };

        auto ev_action_qp = forall([&, plap2](tensor<double,2> grad_u,
                                              tensor<double,2> grad_l)
        {
            tensor<double, 2> unused_output{};
            tensor<double, 2> dgrad_u{};
            // autodiff == reverse mode
            __enzyme_autodiff<tensor<double, 2>>(+plap2, &grad_u, &dgrad_u, &unused_output,
                                                 &grad_l);
            return dgrad_u;
        }, N, grad_u_qp, grad_l_qp);

        // has to be [vdim, dim, num_qp, num_el]
        Vector ev_action_qp_flat((double *)ev_action_qp.GetData(), 1 * 2 * N);
        Vector y = integrate_basis_gradient(ev_action_qp_flat, fes, ir);

        // L-vector to T-vector
        fes.GetProlongationMatrix()->MultTranspose(y, Y);

        // Re-assign the essential degrees of freedom on the final output vector.
        for (int i = 0; i < ess_tdof_list.Size(); i++)
        {
            Y[ess_tdof_list[i]] = dL[ess_tdof_list[i]];
        }
    }

    // Compute adjoint state of the primal state u
    Vector ComputeAdjointState(const ParGridFunction &u)
    {
        Vector l_tdof(u.ParFESpace()->GetTrueVSize());
        l_tdof = 0.0; // ?
        l_tdof.SetSubVector(ess_tdof_list, 0.0);

        // Get adjoint load
        auto rhs_tdof = ComputeDQoIDU(u);
        rhs_tdof.SetSubVector(ess_tdof_list, 0.0);

        // Get Jacobian
        auto u_tdof = u.GetTrueDofs();
        Operator &J = GetAdjoint(*u_tdof);

        std::ofstream myfile("adjoint.txt");
        J.PrintMatlab(myfile);
        myfile.close();

        Operator &G = GetGradient(*u_tdof);
        std::ofstream myfile2("jacobian.txt");
        G.PrintMatlab(myfile2);
        myfile2.close();

        GMRESSolver gmres(MPI_COMM_WORLD);
        gmres.SetRelTol(1e-12);
        gmres.SetMaxIter(2000);
        gmres.SetPrintLevel(0);
        gmres.SetOperator(J);

        gmres.Mult(rhs_tdof, l_tdof);

        delete u_tdof;
        return l_tdof;
    }

    Vector ComputeDfDpTv(const ParGridFunction &v)
    {
        const int N = mesh->GetNE() * ir.GetNPoints();
        Vector dfdpTv(volume_force->ParFESpace()->GetTrueVSize());

        LambdaOperator K(dfdpTv.Size(), [&](const Vector& v, Vector& dfdpTv)
        {
            Vector vv(v.Size());
            // apply essential bcs
            vv = v;
            vv.SetSubVector(ess_tdof_list, 0.0);

            GridFunction v_gf(volume_force->FESpace());
            v_gf.SetFromTrueDofs(vv);

            auto g_qp = interpolate(*volume_force, ir);
            auto v_qp = interpolate(v_gf, ir);

            auto fg = [](double &g, double &f)
            {
                f = -g;
            };

            auto ev_action_qp = forall([&, fg](double g, double v)
            {
                double unused_output = 0.0;
                double dg = 0.0;
                __enzyme_autodiff<double>(+fg, &g, &dg, &unused_output, &v);
                return dg;
            }, N, g_qp, v_qp);

            Vector ev_action_qp_flat((double *)ev_action_qp.GetData(), N);
            Vector y = integrate_basis(ev_action_qp_flat, fes, ir);

            // L-vector to T-vector
            fes.GetProlongationMatrix()->MultTranspose(y, dfdpTv);

            // Re-assign the essential degrees of freedom on the final output vector.
            for (int i = 0; i < ess_tdof_list.Size(); i++)
            {
                dfdpTv[ess_tdof_list[i]] = v[ess_tdof_list[i]];
            }
        });

        auto v_tdof = v.GetTrueDofs();
        K.Mult(*v_tdof, dfdpTv);

        std::ofstream myfile("dfdp.txt");
        K.PrintMatlab(myfile);
        myfile.close();

        delete v_tdof;
        return dfdpTv;
    }

    double ComputeQoI(const ParGridFunction &u_gf)
    {
        const int N = mesh->GetNE() * ir.GetNPoints();

        auto u_qp = interpolate(u_gf, ir);

        auto qoi = [](double u)
        {
            return 0.5 * pow(u, 2.0);
        };

        auto qoi_qp = forall(qoi, 2 * N, u_qp);
        Vector qoi_qp_flat((double *)qoi_qp.GetData(), 2 * N);

        L2_FECollection l2_0(0, mesh->Dimension());
        ParFiniteElementSpace l2_0_fes(mesh, &l2_0);

        auto qoi_value = integrate_basis(qoi_qp_flat, l2_0_fes, ir);

        return qoi_value.Sum();
    }

    Vector ComputeDQoIDU(const ParGridFunction &u_gf)
    {
        const int N = mesh->GetNE() * ir.GetNPoints();

        auto u_qp = interpolate(u_gf, ir);
        Vector du_qp(u_qp);
        du_qp = 1.0;

        auto qoi = [](double &u)
        {
            return 0.5 * pow(u, 2.0);
        };

        auto dqoidu_qp = forall([&, qoi](double u, double du)
        {
            return fwddiff(+qoi)(u, du);
        }, N, u_qp, du_qp);

        Vector dqoidu_qp_flat((double *)dqoidu_qp.GetData(), N);

        auto dqoidu_qp_value = integrate_basis(dqoidu_qp_flat, *u_gf.FESpace(), ir);

        Vector Y(u_gf.ParFESpace()->GetTrueVSize());
        u_gf.ParFESpace()->GetProlongationMatrix()->MultTranspose(dqoidu_qp_value, Y);

        return Y;
    }

    ~PLaplacianOperator()
    {
    }

    ParMesh *mesh;
    ParFiniteElementSpace &fes;
    mutable ParGridFunction u, du, l, *volume_force = nullptr;
    IntegrationRule &ir;
    Array<int> ess_tdof_list;
    mutable Vector x_lvec, y_lvec, state_lvec, du_tvec, l_tvec;

    PLaplacianGradientOperator *gradient = nullptr;
    PLaplacianAdjointOperator *adjoint = nullptr;

    bool enable_constant_du = false;
};

void run_problem6()
{
    const int dim = 2;

    Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL, false,
                                      2.0*M_PI,
                                      2.0*M_PI);
    mesh.EnsureNodes();

    ParMesh pmesh(MPI_COMM_WORLD, mesh);

    auto bdr_attributes = pmesh.bdr_attributes;

    Array<int> ess_attr(bdr_attributes.Max());
    ess_attr = 1;

    IntegrationRule ir = IntRules.Get(Element::QUADRILATERAL,
                                      2 * pmesh.GetNodes()->FESpace()->GetOrder(0) + 1);

    H1_FECollection h1fec(2);
    ParFiniteElementSpace h1fes(&pmesh, &h1fec);

    ParBilinearForm M(&h1fes);
    auto mass_integrator = new MassIntegrator;
    mass_integrator->SetIntegrationRule(IntRules.Get(
                                            Element::QUADRILATERAL,
                                            2 * pmesh.GetNodes()->FESpace()->GetOrder(0) + 1));
    M.AddDomainIntegrator(mass_integrator);
    M.Assemble();
    M.EliminateEssentialBC(ess_attr);
    M.Finalize();
    auto Mmat = M.ParallelAssemble();

    std::ofstream myfile("mass.txt");
    Mmat->PrintMatlab(myfile);
    myfile.close();

    ParGridFunction u(&h1fes), g(&h1fes);

    PLaplacianOperator plap(h1fes, u);

    auto coef_g = FunctionCoefficient([](const Vector &x)
    {
        return sin(x(0)) * sin(x(1));
    });
    g.ProjectCoefficient(coef_g);
    plap.SetVolumeForcing(g);

    GMRESSolver gmres(MPI_COMM_WORLD);
    gmres.iterative_mode = false;
    gmres.SetRelTol(1e-8);
    gmres.SetMaxIter(10000);
    gmres.SetPrintLevel(0);

    NewtonSolver newton(MPI_COMM_WORLD);
    newton.SetPreconditioner(gmres);
    newton.SetOperator(plap);
    newton.SetRelTol(1e-8);
    newton.SetAbsTol(1e-12);
    newton.SetMaxIter(100);
    newton.SetPrintLevel(0);

    Vector zero;
    u.Randomize(1234);

    ConstantCoefficient zero_coeff(0.0);
    u.ProjectBdrCoefficient(zero_coeff, bdr_attributes);

    Vector *u_tdof = u.GetTrueDofs();
    newton.Mult(zero, *u_tdof);

    u.SetFromTrueDofs(*u_tdof);

    std::cout << "\nComputing adjoint state\n";
    auto adjoint_state_tdof = plap.ComputeAdjointState(u);

    ParGridFunction adjoint_state(&h1fes);
    adjoint_state.SetFromTrueDofs(adjoint_state_tdof);

    Vector dfdpTv = plap.ComputeDfDpTv(adjoint_state);

    Vector dqoidp(dfdpTv);
    dqoidp.Neg();
    // adjoint_state.SetFromTrueDofs(dqoidp);
    // dqoidp.Print();

    // FD test
    {
        std::cout << "FD TEST DQoIDu\n";

        auto eval_f = [&](double h)
        {
            Vector dqoidu(g.Size());
            for (int i = 0; i < u.Size(); i++)
            {
                // Assign perturbation to input
                u(i) += h;
                dqoidu(i) = plap.ComputeQoI(u);

                // Revert perturbation
                u(i) -= h;
            }

            return dqoidu;
        };

        double h = 1e-8;
        Vector fx = eval_f(0.0);
        Vector fxph = eval_f(h);
        fxph -= fx;
        fxph /= h;

        auto dqoidu = plap.ComputeDQoIDU(u);

        fxph -= dqoidu;
        std::cout << "|DQoIDU - FD_DQoIDU|_l2 = " << fxph.Norml2() << "\n";
    }

    // FD test
    {
        std::cout << "FD TEST DfDp*v\n";

        double h = 1e-8;
        Vector fx(dfdpTv), fxph(dfdpTv);

        plap.Mult(*u_tdof, fx);

        adjoint_state *= h;
        g += adjoint_state;
        plap.Mult(*u_tdof, fxph);
        g -= adjoint_state;
        adjoint_state /= h;

        fxph -= fx;
        fxph /= h;

        fxph -= dfdpTv;
        std::cout << "|DfDp*v - FD_DfDp*v|_l2 = " << fxph.Norml2() << "\n";
    }

    // FD test
    {
        std::cout << "FD TEST DQoIDp (total derivative)\n";

        auto eval_f = [&](double h)
        {
            Vector dqoidp(g.Size());
            for (int i = 0; i < u.Size(); i++)
            {
                // Assign perturbation to input
                g(i) += h;

                // u.Randomize(1234);
                // u.ProjectBdrCoefficient(zero_coeff, ess_attr);
                // u.GetTrueDofs(*u_tdof);
                newton.Mult(zero, *u_tdof);
                u.SetFromTrueDofs(*u_tdof);
                dqoidp(i) = plap.ComputeQoI(u);

                // Revert perturbation
                g(i) -= h;
            }

            return dqoidp;
        };

        double h = 1e-6;
        Vector fx = eval_f(0.0);
        Vector fxph = eval_f(h);
        fxph -= fx;
        fxph /= h;

        fxph.SetSubVector(plap.ess_tdof_list, 0.0);

        Vector dqoidp(g.Size());
        dqoidp = dfdpTv;
        dqoidp.Neg();

        // fxph.Print(out, fxph.Size());
        // dqoidp.Print(out, dqoidp.Size());

        fxph -= dqoidp;
        std::cout << "|DQoIDp - FD_DQoIDp|_l2 = " << fxph.Norml2() << "\n";
    }

    // auto coef_f = FunctionCoefficient([](const Vector &x)
    // {
    //    return 2.0;
    // });
    // u.ProjectCoefficient(coef_f);

    // auto qoi = plap.ComputeQoI(u);
    // std::cout << "QoI = " << qoi << std::endl;

    // auto dqoidu = plap.ComputeDQoIDU(u);
    // std::cout << "DQoIDU = " << std::endl;
    // // dqoidu.Print(std::cout, dqoidu.Size());

    // ParLinearForm u_lf(&h1fes);
    // u_lf.AddDomainIntegrator(new DomainLFIntegrator(coef_f));
    // u_lf.Assemble();
    // Vector* u_lf_tdofs = u_lf.ParallelAssemble();
    // // u_lf_tdofs->Print(std::cout, u_lf_tdofs->Size());

    // *u_lf_tdofs -= dqoidu;
    // std::cout << "||DQoIDU - EXACT||_L2 = " << u_lf_tdofs->Norml2() << std::endl;

    char vishost[] = "128.15.198.77";
    int  visport   = 19916;
    {
        socketstream sol_sock(vishost, visport);
        sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
        sol_sock.precision(8);
        sol_sock << "solution\n" << pmesh << u << std::flush;
    }
    {
        socketstream sol_sock(vishost, visport);
        sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
        sol_sock.precision(8);
        sol_sock << "solution\n" << pmesh << adjoint_state << std::flush;
    }
}

void run_problem6();
void run_problem7();

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    int problem_type = 0;

    OptionsParser args(argc, argv);
    args.AddOption(&problem_type, "-p", "--problem",
                   "Problem to run");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(mfem::out);
        }
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(mfem::out);
    }

    if (problem_type == 6)
    {
        run_problem6();
    }

    return 0;
}
