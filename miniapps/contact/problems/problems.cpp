#include "problems.hpp"


void ElasticityProblem::Setup()
{
    int dim = mesh->Dimension();
    fec = new H1_FECollection(order,dim);
    fes = new FiniteElementSpace(mesh,fec,dim,Ordering::byVDIM);
    ndofs = fes->GetTrueVSize();
    mesh->SetNodalFESpace(fes);
    if (mesh->bdr_attributes.Size())
    {
        ess_bdr.SetSize(mesh->bdr_attributes.Max());
    }
    ess_bdr = 0;
    fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
    
    // Solution GridFunction
    x.SetSpace(fes);  x = 0.0;

    // RHS
    b.Update(fes); b.Assemble();

    // Elasticity operator
    lambda.SetSize(mesh->attributes.Max()); lambda = 57.6923076923;
    mu.SetSize(mesh->attributes.Max()); mu = 38.4615384615;

    PWConstCoefficient lambda_cf(lambda);
    PWConstCoefficient mu_cf(mu);

    a = new BilinearForm(fes);
    a->AddDomainIntegrator(new ElasticityIntegrator(lambda_cf,mu_cf));
    a->Assemble();
    a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);
}

ContactProblem::ContactProblem(ElasticityProblem * prob1_, ElasticityProblem * prob2_)
: prob1(prob1_), prob2(prob2_)
{

}