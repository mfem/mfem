#include "mfem.hpp"
#include "vec_conv_integrator.hpp"

using namespace std;
using namespace mfem;

struct OptionSet
{
    double rey;
} opt_;

class NavierStokesOperator : public Operator
{
public:
    ParMesh *pmesh_;
    Array<ParFiniteElementSpace *> fes_;
    Array<int> ess_bdr_attr_;
    Array<int> ess_tdof_list_;

    Array<int> block_offsets_;
    Array<int> block_trueOffsets_;
    BlockVector rhs, trueRhs;

    ParNonlinearForm *N;
    ParBilinearForm *sform;
    ParLinearForm *fform;
    ParBilinearForm *mpform;
    ParMixedBilinearForm *dform;

    HypreParMatrix *S;
    HypreParMatrix *Mp;
    HypreParMatrix *D;
    HypreParMatrix *G;
    mutable HypreParMatrix *NjacS;

    BlockOperator *jac;
    BlockOperator *lin;

    HypreSolver *invS;
    HypreSolver *invMp;
    BlockDiagonalPreconditioner *stokesprec;
    IterativeSolver *jac_solver;
    NewtonSolver newton_solver;

    NavierStokesOperator(Array<ParFiniteElementSpace *> &fes) : Operator(fes[0]->TrueVSize() + fes[1]->TrueVSize()),
                                                                pmesh_(fes[0]->GetParMesh()), fes_(fes),
                                                                ess_bdr_attr_(pmesh_->bdr_attributes.Max())
    {
        fes_[0]->GetEssentialTrueDofs(ess_bdr_attr_, ess_tdof_list_);

        block_offsets_.SetSize(3);
        block_offsets_[0] = 0;
        block_offsets_[1] = fes[0]->GetVSize();
        block_offsets_[2] = fes[1]->GetVSize();
        block_offsets_.PartialSum();

        block_trueOffsets_.SetSize(3);
        block_trueOffsets_[0] = 0;
        block_trueOffsets_[1] = fes[0]->TrueVSize();
        block_trueOffsets_[2] = fes[1]->TrueVSize();
        block_trueOffsets_.PartialSum();

        rhs.Update(block_offsets_);
        trueRhs.Update(block_trueOffsets_);
    }

    void Init(const BlockVector &x)
    {
        N = new ParNonlinearForm(fes_[0]);
        N->AddDomainIntegrator(new VectorConvectionNLFIntegrator);
        N->SetEssentialTrueDofs(ess_tdof_list_);

        sform = new ParBilinearForm(fes_[0]);
        ConstantCoefficient kin_visc(1.0 / opt_.rey);
        sform->AddDomainIntegrator(new VectorDiffusionIntegrator(kin_visc));
        sform->Assemble();
        sform->EliminateEssentialBC(ess_bdr_attr_, x.GetBlock(0), rhs.GetBlock(0));
        sform->Finalize();
        S = sform->ParallelAssemble();

        dform = new ParMixedBilinearForm(fes_[0], fes_[1]);
        dform->AddDomainIntegrator(new VectorDivergenceIntegrator);
        dform->Assemble();
        dform->EliminateTrialDofs(ess_bdr_attr_, x.GetBlock(0), rhs.GetBlock(1));
        dform->Finalize();
        D = dform->ParallelAssemble();

        G = D->Transpose();
        (*G) *= -1.0;

        jac = new BlockOperator(block_trueOffsets_);
        jac->SetBlock(0, 0, S);
        jac->SetBlock(0, 1, G);
        jac->SetBlock(1, 0, D);

        lin = new BlockOperator(block_trueOffsets_);
        lin->SetBlock(0, 0, S);
        lin->SetBlock(0, 1, G);
        lin->SetBlock(1, 0, D);

        mpform = new ParBilinearForm(fes_[1]);
        mpform->AddDomainIntegrator(new MassIntegrator);
        mpform->Assemble();
        mpform->Finalize();
        Mp = mpform->ParallelAssemble();

        invS = new HypreBoomerAMG(*S);
        static_cast<HypreBoomerAMG *>(invS)->SetPrintLevel(0);
        invS->iterative_mode = false;

        invMp = new HypreBoomerAMG(*Mp);
        static_cast<HypreBoomerAMG *>(invMp)->SetPrintLevel(0);
        invMp->iterative_mode = false;

        stokesprec = new BlockDiagonalPreconditioner(block_trueOffsets_);
        stokesprec->SetDiagonalBlock(0, invS);
        stokesprec->SetDiagonalBlock(1, invMp);

        fes_[0]->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(0),
                                                        trueRhs.GetBlock(0));
        fes_[1]->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(1),
                                                        trueRhs.GetBlock(1));

        jac_solver = new GMRESSolver(MPI_COMM_WORLD);
        jac_solver->iterative_mode = false;
        jac_solver->SetAbsTol(0.0);
        jac_solver->SetRelTol(1e-4);
        static_cast<GMRESSolver *>(jac_solver)->SetKDim(100);
        jac_solver->SetMaxIter(1000);
        jac_solver->SetOperator(*jac);
        jac_solver->SetPreconditioner(*stokesprec);
        jac_solver->SetPrintLevel(2);

        newton_solver.iterative_mode = true;
        newton_solver.SetSolver(*jac_solver);
        newton_solver.SetOperator(*this);
        newton_solver.SetPrintLevel(1);
        newton_solver.SetAbsTol(0.0);
        newton_solver.SetRelTol(1e-7);
        newton_solver.SetMaxIter(15);
    }

    virtual void Mult(const Vector &x, Vector &y) const
    {
        Vector tmp(block_trueOffsets_[1]);
        Vector vel_in(x.GetData(), block_trueOffsets_[1]);
        Vector vel_out(y.GetData(), block_trueOffsets_[1]);

        // Apply linear BlockOperator
        lin->Mult(x, y);

        // Apply nonlinear action to velocity
        N->Mult(vel_in, tmp);
        vel_out += tmp;
    }

    virtual Operator &GetGradient(const Vector &x) const
    {
        Vector u(x.GetData(), block_trueOffsets_[1]);

        delete NjacS;

        hypre_ParCSRMatrix *NjacS_wrap;
        hypre_ParcsrAdd(1.0, *static_cast<HypreParMatrix *>(&(N->GetGradient(u))), 1.0,
                        *S, &NjacS_wrap);
        NjacS = new HypreParMatrix(NjacS_wrap);

        HypreParMatrix *NjacS_e = NjacS->EliminateRowsCols(ess_tdof_list_);
        delete NjacS_e;

        jac->SetBlock(0, 0, NjacS);

        // invS->SetOperator(*NjacS);

        return *jac;
    }
};

class TDNavierStokesOperator : public TimeDependentOperator
{
private:
    NavierStokesOperator *nso_;

public:
    TDNavierStokesOperator(NavierStokesOperator *nso) : TimeDependentOperator(nso->fes_[0]->GetTrueVSize()), nso_(nso)
    {
    }

    void ImplicitSolve(const double dt,
                       const Vector &x, Vector &dx_dt)
    {
    }
};

void vel_ldc(const Vector &x, Vector &u)
{
    double yi = x(1);

    if (yi > 1.0 - 1e-8)
    {
        u(0) = 1.0;
    }
    else
    {
        u(0) = 0.0;
    }
    u(1) = 0.0;
}

int main(int argc, char *argv[])
{
    MPI_Session mpi(argc, argv);

    int num_procs, myid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int serial_ref_levels = 0;
    int order = 2;
    double dt = 1e-5;
    double t_final = 2 * dt;

    opt_.rey = 1.0;

    const char *mesh_file = "../../data/inline-quad.mesh";

    int vel_order = order;
    int pres_order = order - 1;

    Mesh *mesh = new Mesh(mesh_file);
    int dim = mesh->Dimension();

    for (int l = 0; l < serial_ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    FiniteElementCollection *vel_fec = new H1_FECollection(vel_order, dim);
    FiniteElementCollection *pres_fec = new H1_FECollection(pres_order);

    ParFiniteElementSpace *vel_fes = new ParFiniteElementSpace(pmesh, vel_fec, dim);
    ParFiniteElementSpace *pres_fes = new ParFiniteElementSpace(pmesh, pres_fec);

    Array<ParFiniteElementSpace *> fes(2);
    fes[0] = vel_fes;
    fes[1] = pres_fes;

    int fes_size0 = fes[0]->GlobalVSize();
    int fes_size1 = fes[1]->GlobalVSize();

    if (myid == 0)
    {
        cout << "Velocity #DOFs: " << fes_size0 << endl;
        cout << "Pressure #DOFs: " << fes_size1 << endl;
    }

    Array<int> block_offsets;
    Array<int> block_trueOffsets;
    block_offsets.SetSize(3);
    block_offsets[0] = 0;
    block_offsets[1] = fes[0]->GetVSize();
    block_offsets[2] = fes[1]->GetVSize();
    block_offsets.PartialSum();

    block_trueOffsets.SetSize(3);
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = fes[0]->TrueVSize();
    block_trueOffsets[2] = fes[1]->TrueVSize();
    block_trueOffsets.PartialSum();

    ODESolver *ode_solver = new BackwardEulerSolver;

    Array<int> ess_bdr_attr(pmesh->bdr_attributes.Max());

    BlockVector x(block_offsets), trueX(block_trueOffsets);

    VectorFunctionCoefficient ldccoeff(dim, vel_ldc);
    ParGridFunction vel_gf;
    vel_gf.MakeRef(fes[0], x.GetBlock(0));
    vel_gf.ProjectBdrCoefficient(ldccoeff, ess_bdr_attr);

    fes[0]->GetRestrictionMatrix()->Mult(x.GetBlock(0), trueX.GetBlock(0));
    fes[1]->GetRestrictionMatrix()->Mult(x.GetBlock(1), trueX.GetBlock(1));

    NavierStokesOperator nso(fes);
    TDNavierStokesOperator td_nso(&nso);
    ode_solver->Init(td_nso);

    double t = 0.0;
    bool last_step = false;

    for (int ti = 1; !last_step; ti++)
    {
        if (t + dt >= t_final - dt / 2)
        {
            last_step = true;
        }

        cout << "Timestep: " << t << endl;
        ode_solver->Step(trueX, t, dt);

        vel_gf.Distribute(trueX.GetBlock(0));

        t += dt;
    }

    return 0;
}
