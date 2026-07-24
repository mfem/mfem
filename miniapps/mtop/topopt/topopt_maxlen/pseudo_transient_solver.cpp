#include "pseudo_transient_solver.hpp"

namespace mfem
{

void LinearEvolutionOperator::Mult(const Vector &x, Vector &y) const
{
    MFEM_VERIFY(b != nullptr, "LinearEvolutionOperator: source is not set");
    K->Mult(x, z);
    z.Neg();
    z += *b;
    Minv->Mult(z, y);
}

void PseudoTransientSolver::MarchToSteadyState(LinearEvolutionOperator &evol,
                                                Vector &u) const
{
    MFEM_VERIFY(ode, "PseudoTransientSolver: ode solver is not set");
    ode->Init(evol);

    const real_t eps = 1e-12;
    Vector u_prev(u.Size());

    iter_count = 0;
    real_t t = 0.0;
    real_t rate = 2.0 * tol;

    while (rate > tol && t < final_t)
    {
        u_prev = u;
        real_t u_prev_norm = ParNormlp(u_prev, 2.0, comm);

        real_t step = dt;
        ode->Step(u, t, step);
        iter_count++;

        u_prev -= u;
        rate = ParNormlp(u_prev, 2.0, comm) / (step * (u_prev_norm + eps));
    }
}

void PseudoTransientSolver::Mult(const Vector &x, Vector &y) const
{
    MFEM_VERIFY(rhs != nullptr, "PseudoTransientSolver: rhs is not set");
    MFEM_VERIFY(Minv != nullptr, "PseudoTransientSolver: Minv is not set");

    LinearEvolutionOperator evol(*K, *Minv);
    evol.SetSource(*rhs);

    y = x;
    MarchToSteadyState(evol, y);
}

void PseudoTransientSolver::MultTranspose(const Vector &x, Vector &y) const
{
    MFEM_VERIFY(adj_rhs != nullptr, "PseudoTransientSolver: adjoint rhs is not set");
    MFEM_VERIFY(Minv != nullptr, "PseudoTransientSolver: Minv is not set");

    TransposeOperator Kt(*K);
    LinearEvolutionOperator evol(Kt, *Minv);
    evol.SetSource(*adj_rhs);

    y = x;
    MarchToSteadyState(evol, y);
}

MaterialThicknessSolver::MaterialThicknessSolver(ParFiniteElementSpace &design_fes_,
    ParFiniteElementSpace &eval_fes_, VectorCoefficient &v_cf_, bool pa_)
    : Operator(eval_fes_.GetTrueVSize(), design_fes_.GetTrueVSize()),
    design_fes(&design_fes_), sol_fes(&eval_fes_), pa(pa_), v_cf(v_cf_)
{
    design_mesh = design_fes->GetParMesh();
    eval_mesh   = sol_fes->GetParMesh();

    MFEM_VERIFY(design_mesh && eval_mesh,
                "MaterialThicknessSolver: design_fes or eval_fes has no mesh");
    same_mesh = (design_mesh == eval_mesh);

    src_fes = new ParFiniteElementSpace(eval_mesh, design_fes->FEColl());
    rho_p.SetSpace(src_fes);
    rho_p = 0.0;

    rho_a.SetSpace(sol_fes);
    rho_a = 0.0;

    constexpr real_t alpha = 1.0;
    K = new ParBilinearForm(sol_fes);
    K->AddDomainIntegrator(new ConvectionIntegrator(v_cf, alpha));
    K->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(v_cf, alpha));
    K->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(v_cf, alpha));

    Assemble();

    pt_solver = new PseudoTransientSolver(*Kopt, sol_fes->GetComm());
    pt_solver->SetODESolver(std::make_unique<RK4Solver>());
}

MaterialThicknessSolver::~MaterialThicknessSolver()
{
    delete pt_solver;
    delete K;
    delete src_fes;
}

void MaterialThicknessSolver::Assemble()
{
    if(pa) 
    {
        K->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        K->Assemble();
        Kopt.Reset(K, false);
    }
    else
    {
        K->Assemble();
        K->Finalize();
        Kopt.Reset(K->ParallelAssemble());
    }

    ParMixedBilinearForm Nmix(src_fes, sol_fes);
    Nmix.AddDomainIntegrator(new MixedScalarMassIntegrator());
    Nmix.Assemble();
    Nmix.Finalize();
    Nopt.Reset(Nmix.ParallelAssemble());
}

void MaterialThicknessSolver::AssembleLinearSolver()
{
    MFEM_VERIFY(!pa, "MaterialThicknessSolver: AssembleLinearSolver requires full assembly (pa = false)");

    fwd_prec = std::make_unique<BlockILU>(*Kopt, sol_fes->GetFE(0)->GetDof());
    fwd_gmres = std::make_unique<GMRESSolver>(sol_fes->GetComm());
    fwd_gmres->SetOperator(*Kopt);
    fwd_gmres->SetPreconditioner(*fwd_prec);
    fwd_gmres->SetRelTol(1e-12);
    fwd_gmres->SetAbsTol(1e-14);
    fwd_gmres->SetMaxIter(2000);
    fwd_gmres->SetKDim(50);

    Kt = std::make_unique<TransposeOperator>(*Kopt);
    KoptT.reset(Kopt.As<HypreParMatrix>()->Transpose());

    adj_prec = std::make_unique<BlockILU>(*KoptT, sol_fes->GetFE(0)->GetDof());
    adj_gmres = std::make_unique<GMRESSolver>(sol_fes->GetComm());
    adj_gmres->SetOperator(*Kt);
    adj_gmres->SetPreconditioner(*adj_prec);
    adj_gmres->SetRelTol(1e-12);
    adj_gmres->SetAbsTol(1e-14);
    adj_gmres->SetMaxIter(2000);
    adj_gmres->SetKDim(50);
}

void MaterialThicknessSolver::BuildRhs(const Vector &design_tv,
                                        Vector &rhs_tv) const
{
    ParGridFunction rf_tmp(design_fes);
    rf_tmp.SetFromTrueDofs(design_tv);

    if (same_mesh)
    {
        rho_p.ProjectGridFunction(rf_tmp);
    }
    else
    {
        ParSubMesh::Transfer(rf_tmp, rho_p);
    }

    Vector rho_p_tv(src_fes->GetTrueVSize());
    rho_p.GetTrueDofs(rho_p_tv);

    rhs_tv.SetSize(sol_fes->GetTrueVSize());
    Nopt->Mult(rho_p_tv, rhs_tv);
}

void MaterialThicknessSolver::Mult(const Vector &x, Vector &y) const
{
    MFEM_VERIFY(fwd_gmres, "MaterialThicknessSolver: fwd_gmres is not set");

    Vector rhs_tv;
    BuildRhs(x, rhs_tv);

    y.SetSize(Height());
    y = 0.0;
    fwd_gmres->Mult(rhs_tv, y);
}

void MaterialThicknessSolver::MultTranspose(const Vector &x, Vector &y) const
{
    MFEM_VERIFY(adj_gmres, "MaterialThicknessSolver: adj_gmres is not set");

    Vector lambda(sol_fes->GetTrueVSize());
    lambda = 0.0;
    adj_gmres->Mult(x, lambda);

    Vector sens_eval(src_fes->GetTrueVSize());
    Nopt->MultTranspose(lambda, sens_eval);

    y.SetSize(Width());
    if (same_mesh)
    {
        y = sens_eval;
    }
    else
    {
        ParGridFunction sens_eval_gf(src_fes); sens_eval_gf.SetFromTrueDofs(sens_eval);
        ParGridFunction y_gf(design_fes);
        ParSubMesh::Transfer(sens_eval_gf, y_gf);
        y_gf.GetTrueDofs(y);
    }
}

void MaterialThicknessSolver::FSolve()
{
    MFEM_VERIFY(fwd_rhs.Size() == design_fes->GetTrueVSize(),
                "MaterialThicknessSolver: rhs is not set (or wrong size)");

    Vector rhs_tv;
    BuildRhs(fwd_rhs, rhs_tv);
    pt_solver->SetRhs(rhs_tv);

    Vector ra_tv(sol_fes->GetTrueVSize());
    rho_a.GetTrueDofs(ra_tv);

    Vector ra_new;
    pt_solver->Mult(ra_tv, ra_new);
    rho_a.SetFromTrueDofs(ra_new);
}

void MaterialThicknessSolver::ASolve()
{
    MFEM_VERIFY(adj_rhs.Size() == sol_fes->GetTrueVSize(),
                "MaterialThicknessSolver: adjoint rhs is not set (or wrong size)");
    pt_solver->SetAdjointRhs(adj_rhs);

    Vector lam_init_tv(sol_fes->GetTrueVSize());
    lam_init_tv = 0.0;
    pt_solver->MultTranspose(lam_init_tv, lambda_tv);

    Vector sens_eval(src_fes->GetTrueVSize());
    Nopt->MultTranspose(lambda_tv, sens_eval);

    sens_tv.SetSize(design_fes->GetTrueVSize());
    if (same_mesh)
    {
        sens_tv = sens_eval;
    }
    else
    {
        ParGridFunction sens_eval_gf(src_fes); sens_eval_gf.SetFromTrueDofs(sens_eval);
        ParGridFunction sens_gf(design_fes);
        ParSubMesh::Transfer(sens_eval_gf, sens_gf);
        sens_gf.GetTrueDofs(sens_tv);
    }
}

} // namespace mfem
