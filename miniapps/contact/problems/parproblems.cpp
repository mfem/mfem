#include "parproblems.hpp"


void ParElasticityProblem::Setup()
{
    int dim = pmesh->Dimension();
    fec = new H1_FECollection(order,dim);
    fes = new ParFiniteElementSpace(pmesh,fec,dim,Ordering::byVDIM);
    ndofs = fes->GetTrueVSize();
    gndofs = fes->GlobalTrueVSize();
    pmesh->SetNodalFESpace(fes);
    if (pmesh->bdr_attributes.Size())
    {
        ess_bdr.SetSize(pmesh->bdr_attributes.Max());
    }
    ess_bdr = 0;
    fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
    
    // Solution GridFunction
    x.SetSpace(fes);  x = 0.0;

    // RHS
    b.Update(fes); b.Assemble();

    // Elasticity operator
    lambda.SetSize(pmesh->attributes.Max()); lambda = 57.6923076923;
    mu.SetSize(pmesh->attributes.Max()); mu = 38.4615384615;

    PWConstCoefficient lambda_cf(lambda);
    PWConstCoefficient mu_cf(mu);

    a = new ParBilinearForm(fes);
    a->AddDomainIntegrator(new ElasticityIntegrator(lambda_cf,mu_cf));
    a->Assemble();
    a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);
}


ParContactProblem::ParContactProblem(ParElasticityProblem * prob1_, ParElasticityProblem * prob2_)
: prob1(prob1_), prob2(prob2_)
{

}