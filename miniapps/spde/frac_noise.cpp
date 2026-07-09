#include "frac_noise.hpp"

using namespace mfem;
using namespace std;

FracRandomFieldGenerator::FracRandomFieldGenerator(ParMesh &pmesh_,
                                                   const int par_ref_levels_,
                                                   const int order_,
                                                   real_t sigma_, real_t s_,
                                                   int smoother_applications_)
    : pmesh(pmesh_), par_ref_levels(par_ref_levels_),
      order(order_), sigma(sigma_), s(s_),
      smoother_applications(smoother_applications_)
{
    MFEM_VERIFY(smoother_applications >= 1,
                "smoother_applications must be at least 1.");

    fec.reset(new H1_FECollection(order, pmesh.Dimension()));
    ParFiniteElementSpace *coarse_fespace = new ParFiniteElementSpace(&pmesh, fec.get());

    fespaces.reset(new ParFiniteElementSpaceHierarchy(&pmesh, coarse_fespace,
                                                      false, true)); // take ownership of coarse space

    for(int l = 0; l < par_ref_levels; l++)
    {
        fespaces->AddUniformlyRefinedLevel(1, Ordering::byVDIM);
    }

    // set the prolongation operators for the multigrid
    const int nlevels = fespaces->GetNumLevels();
    prolongations.SetSize(nlevels - 1);
    for (int level = 0; level < nlevels - 1; ++level)
    {
        prolongations[level] = fespaces->GetProlongationAtLevel(level);
    }

    //set the operators and the smoothers for the multigrid
    operators.SetSize(nlevels);
    smoothers.SetSize(nlevels);

    const Array<int> ess_tdofs;

    for (int level = 0; level < nlevels; ++level)
    {
        // mass and diffusion operators on each level
        ParFiniteElementSpace &fespace = fespaces->GetFESpaceAtLevel(level);
        unique_ptr<ParBilinearForm> bf(new ParBilinearForm(&fespace));
        ConstantCoefficient one(1.0);
        ConstantCoefficient mass_coef(sigma*sigma);

        bf->AddDomainIntegrator(new MassIntegrator(mass_coef));
        bf->AddDomainIntegrator(new DiffusionIntegrator(one));
        bf->Assemble();
        bf->Finalize();
        HypreParMatrix *mat= bf->ParallelAssemble();
        operators[level] = mat;

        // Jacobi smoother on each level
        Vector diag(fespace.GetTrueVSize());
        Vector tmp(fespace.GetTrueVSize());
        mat->GetDiag(diag);
        if(false)
        {
            tmp=1.0;
            mat->AbsMult(tmp, diag); // diag = |A|*1
        }
        else{
            for(int i=0; i<diag.Size(); i++)
            {
                diag[i]=1.0/sqrt(diag[i]);
            }
            mat->Mult(diag, tmp);
            real_t omega=InnerProduct(fespace.GetComm(), diag, tmp);
            mat->GetDiag(diag); diag *= omega;
        }

        OperatorJacobiSmoother *jac = new OperatorJacobiSmoother(diag,
                                                                 ess_tdofs,
                                                                 1.0 /*weight*/);
        jac->iterative_mode = false;
        jac->Setup(diag);
        smoothers[level]=jac;
    }
};

FracRandomFieldGenerator::~FracRandomFieldGenerator()
{
    for(int i = 0; i < operators.Size(); i++){delete operators[i];}
    for(int i = 0; i < smoothers.Size(); i++){delete smoothers[i];}
}

real_t FracRandomFieldGenerator::GetLevelH(int level) const
{
    real_t h_min, h_max, kappa_min, kappa_max;
    ParFiniteElementSpace &fes = fespaces->GetFESpaceAtLevel(level);
    fes.GetParMesh()->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
    return h_max;
}

void FracRandomFieldGenerator::ApplySmootherRepeated(const Solver &smoother,
                                                     const Vector &x,
                                                     Vector &y) const
{
    y.SetSize(x.Size());
    if(smoother_applications == 1)
    {
        smoother.Mult(x, y);
        return;
    }

    Vector src(x);
    Vector dst(y.Size());
    for(int i = 0; i < smoother_applications; i++)
    {
        smoother.Mult(src, dst);
        if(i + 1 < smoother_applications)
        {
            src = dst;
        }
    }
    y = dst;
}

void FracRandomFieldGenerator::ApplySmootherTransposeRepeated(
    const Solver &smoother, const Vector &x, Vector &y) const
{
    y.SetSize(x.Size());
    if(smoother_applications == 1)
    {
        smoother.MultTranspose(x, y);
        return;
    }

    Vector src(x);
    Vector dst(y.Size());
    for(int i = 0; i < smoother_applications; i++)
    {
        smoother.MultTranspose(src, dst);
        if(i + 1 < smoother_applications)
        {
            src = dst;
        }
    }
    y = dst;
}

void FracRandomFieldGenerator::SolveCoarseLevel(const Vector &rhs0,
                                                Solver &smoother0,
                                                Operator &operator0,
                                                Vector &u0) const
{
    SymmetrizedSmoother ms(&smoother0, &operator0);
    ApplySmootherRepeated(ms, rhs0, u0);
}

void FracRandomFieldGenerator::ApplyCoarseLevel(const Vector &u0, real_t h0,
                                                Vector &v0) const
{
    const real_t mu = pow(sigma*sigma + 1.0/(h0*h0), s);
    v0.Set(mu, u0);
}

void FracRandomFieldGenerator::Mult(const Vector &x, Vector &y) const
{

    const int nlevels = fespaces->GetNumLevels();
    const int finest_size =
        fespaces->GetFinestFESpace().GetTrueVSize();
    MFEM_VERIFY(x.Size() == finest_size,
                "Input vector size does not match finest FE space true size.");
    y.SetSize(finest_size);

    SymmetrizedSmoother ms(smoothers[nlevels-1], operators[nlevels-1]);
    vector<Vector*> u;
    vector<Vector*> v;

    int myrank;
    MPI_Comm_rank(pmesh.GetComm(), &myrank);


    for(int level = 0; level < nlevels; level++)
    {
        Vector *w = new Vector(); w->SetSize( fespaces->GetFESpaceAtLevel(level).GetTrueVSize() );
        u.push_back(w);

        w = new Vector(); w->SetSize( fespaces->GetFESpaceAtLevel(level).GetTrueVSize() );
        v.push_back(w);

        if(0==myrank)  {cout << "Level " << level <<
            " size: " << fespaces->GetFESpaceAtLevel(level).GlobalTrueVSize()<<
                " loc:" << fespaces->GetFESpaceAtLevel(level).GetTrueVSize() << endl;

                cout<<" smoother size:"<< smoothers[level]->Height()
                    << " operator size:" << operators[level]->Height() << endl;

                if(level<nlevels-1)
                {
                    cout<<" prolong size:"<< prolongations[level]->Height()
                        << " x " << prolongations[level]->Width() << endl;
                }
        }

    }


    if(nlevels == 1)
    {
        SolveCoarseLevel(x, *smoothers[0], *operators[0], *(u[0]));
    }
    else
    {
        ApplySmootherRepeated(ms, x, *(u[nlevels -1]));
    }

    Vector  tva, tvb, tvc;
    tva.SetSize(x.Size()); tva=x;
    tvb.SetSize(y.Size());
    tvc.SetSize(y.Size());

    for(int level = nlevels - 2; level >=0; level--)
    {
         if(1==myrank)  {cout << "Level " << level <<
                " size: " << fespaces->GetFESpaceAtLevel(level).GlobalTrueVSize() << endl;}

        ApplySmootherRepeated(*smoothers[level+1], tva, tvb);
        operators[level+1]->Mult(tvb, tvc);
        add(-1.0, tvc, 1.0, tva, tvb);
        tva.SetSize(prolongations[level]->Width());

        if(1==myrank)  {cout << "Level " << level <<
                " restrict from size: " << tvb.Size() <<
                " to size: " << tva.Size() << endl;}

        prolongations[level]->MultTranspose(tvb, tva);


        if(level == 0)
        {
            SolveCoarseLevel(tva, *smoothers[level], *operators[level],
                             *(u[level]));
        }
        else
        {
            ms.SetSmoother(*smoothers[level]);
            ms.SetOperator(*operators[level]);
            //smoothing
            ApplySmootherRepeated(ms, tva, *(u[level]));
        }


        tvb.SetSize( tva.Size() );
        tvc.SetSize( tva.Size() );
    }


    // compute h on each level
    Array<real_t> h; h.SetSize(nlevels);
    {
        real_t h_min, h_max, kappa_min, kappa_max;
        for(int level =0; level < nlevels; level++)
        {
            ParFiniteElementSpace &fes = fespaces->GetFESpaceAtLevel(level);
            fes.GetParMesh()->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
            h[level]=h_max;
            if(0==myrank)
            {
                cout << " Level " << level << " h_min: " << h_min <<
                    " h_max: " << h_max << endl;
            }
        }
    }

    // coarse to fine loop
    real_t mu;
    ApplyCoarseLevel(*(u[0]), h[0], *(v[0]));

    if(0==myrank){cout << " Level " << 0 <<" h="<<h[0]<<std::endl;}

    for(int level =1; level < nlevels; level++)
    {
        mu = sigma*sigma + 1.0/(h[level]*h[level]);
        mu = pow(mu, s);

        if(0==myrank){cout << " Level " << level <<" h="<<h[level]<<" mu="<<mu<<std::endl;}

        tva.SetSize( v[level]->Size() );
        tvb.SetSize( v[level]->Size() );
        tvc.SetSize( v[level]->Size() );

        prolongations[level-1]->Mult(*(v[level-1]), tva );
        operators[level]->Mult(tva, tvb);
        ApplySmootherTransposeRepeated(*smoothers[level], tvb, tvc);
        tva.Add(-1.0, tvc);


        v[level]->Set(mu, *(u[level]) );
        v[level]->Add(1.0, tva);
    }

    // final scaling
    mu=pow(sigma*sigma +1.0/(h[0]*h[0]),s);
    y.Set(1.0/mu, *(v[nlevels -1]));


    for(int level = 0; level < nlevels; level++)
    {
        delete u[level];
        delete v[level];
    }
}

FracRandomFieldGeneratorSPDE::FracRandomFieldGeneratorSPDE(
    ParMesh &pmesh_, const int par_ref_levels_, const int order_,
    real_t sigma_, real_t s_, int smoother_applications_)
    : FracRandomFieldGenerator(pmesh_, par_ref_levels_, order_, sigma_, s_,
                               smoother_applications_)
{
    const int dim = GetCoarsestFESpace()->GetParMesh()->Dimension();
    const real_t exponent = 1.0 - s;
    const real_t nu = 2.0*exponent - dim/2.0;
    MFEM_VERIFY(nu > 0.0, "Coarse SPDE solve requires positive nu.");

    const real_t corr_len = sigma > 0.0 ? sqrt(2.0*nu)/sigma : sqrt(2.0*nu);
    coarse_spde_solver.reset(
        new spde::SPDESolver(nu, coarse_bc, GetCoarsestFESpace(),
                             corr_len, corr_len, corr_len));
    coarse_spde_solver->SetPrintLevel(0);
}

void FracRandomFieldGeneratorSPDE::SolveCoarseLevel(const Vector &rhs0,
                                                    Solver &smoother0,
                                                    Operator &operator0,
                                                    Vector &u0) const
{
    MFEM_VERIFY(coarse_spde_solver, "Coarse SPDE solver is not initialized.");

    ParFiniteElementSpace *coarse_fes = GetCoarsestFESpace();
    ParLinearForm rhs(coarse_fes);
    rhs = 0.0;

    const Operator *P = coarse_fes->GetProlongationMatrix();
    if (P)
    {
        P->Mult(rhs0, rhs);
    }
    else
    {
        MFEM_VERIFY(rhs.Size() == rhs0.Size(),
                    "Cannot map true vector to coarse linear form.");
        rhs = rhs0;
    }

    ParGridFunction coarse_solution(coarse_fes);
    coarse_solution = 0.0;
    coarse_spde_solver->Solve(rhs, coarse_solution);
    coarse_solution.GetTrueDofs(u0);
}

void FracRandomFieldGeneratorSPDE::ApplyCoarseLevel(const Vector &u0,
                                                    real_t h0,
                                                    Vector &v0) const
{
    v0 = u0;
    v0 *= pow(sigma*sigma + 1.0/(h0*h0), s);
}
