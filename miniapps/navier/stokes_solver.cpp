#include "stokes_solver.hpp"

namespace mfem {

StokesOperator::StokesOperator(ParFiniteElementSpace &vel_fes,
                               ParFiniteElementSpace &pres_fes):
    Operator(vel_fes.GetTrueVSize()+pres_fes.GetTrueVSize()),
    vfes(vel_fes),
    pfes(pres_fes),
    offsets({0, vel_fes.GetTrueVSize(), pres_fes.GetTrueVSize()}),
    intrules(0, Quadrature1D::GaussLobatto),
    zero_coeff(0.0)
{
    if (vel_fes.GetParMesh()->bdr_attributes.Size() > 0)
    {
        vel_ess_bdr.SetSize(vel_fes.GetParMesh()->bdr_attributes.Max());
        vel_ess_bdr = 0.0;
        pres_ess_bdr.SetSize(vel_fes.GetParMesh()->bdr_attributes.Max());
        pres_ess_bdr = 0.0;
    }

    vfes.GetEssentialTrueDofs(vel_ess_bdr, vel_ess_tdofs);
    pfes.GetEssentialTrueDofs(pres_ess_bdr, pres_ess_tdofs);

    offsets.PartialSum();


    vel_bc_gf.reset(new ParGridFunction(&vfes));
    *vel_bc_gf = 0.0; //set the velocity grid function to zero

    pres_bc_gf.reset(new ParGridFunction(&pfes));
    *pres_bc_gf = 0.0; //set the pressure grid function to zero

    // The nonlinear convective integrators use over-integration (dealiasing) as
    // a stabilization mechanism.
    ir_nl = intrules.Get(vfes.GetFE(0)->GetGeomType(),
                         (int)(ceil(1.5 * 2*(vel_fes.GetOrder(0)+1) - 3)));

    ir = intrules.Get(vfes.GetFE(0)->GetGeomType(),
                      (int)(2*(vel_fes.GetOrder(0)+1) - 3));

    ir_face = intrules.Get(vfes.GetFaceElement(0)->GetGeomType(),
                           (int)(2*(vel_fes.GetOrder(0)+1) - 3));

    b11_form=nullptr;
    b22_form=nullptr;
    b12_form=nullptr;
    b21_form=nullptr;

}


void StokesOperator::SetVelBC(std::vector<VelDirichletBC>& vvbc)
{
    for(auto vbc=vvbc.begin();vbc!=vvbc.end();vbc++)
    {
        for (int i = 0; i < vbc->second->Size(); i++)
        {
            if (*(vbc->second)[i] == 1)
            {
                vel_ess_bdr[i] = 1;
            }
        }
    }

    vfes.GetEssentialTrueDofs(vel_ess_bdr, vel_ess_tdofs);
}

void StokesOperator::SetPressBC(std::vector<PresDirichletBC>& vpbc)
{
    for(auto pbc=vpbc.begin();pbc!=vpbc.end();pbc++)
    {
        for(int i=0;i<pbc->second->Size();i++){
            if (*(pbc->second)[i] == 1)
            {
                vel_ess_bdr[i] = 1;
            }
        }
    }

    pfes.GetEssentialTrueDofs(pres_ess_bdr, pres_ess_tdofs);
}

void StokesOperator::Mult(const Vector &x, Vector &y) const
{

}

void StokesOperator::Setup()
{
    BilinearFormIntegrator *integrator;

    delete b11_form;
    b11_form=new ParBilinearForm(&vfes);
    integrator=new ElasticityIntegrator(zero_coeff,*viscosity);
    integrator->SetIntRule(&ir);
    b11_form->AddDomainIntegrator(integrator);

    delete b12_form;
    b12_form=new ParMixedBilinearForm(&pfes,&vfes);
    integrator=new VectorDivergenceIntegrator();
    integrator->SetIntRule(&ir);
    b12_form->AddDomainIntegrator(integrator);

    delete b21_form;
    b21_form=new ParMixedBilinearForm(&vfes,&pfes);
    integrator=new GradientIntegrator();
    integrator->SetIntRule(&ir);
    b21_form->AddDomainIntegrator(integrator);

    if (matrix_free)
    {
        b11_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        b12_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        b21_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    }
}

}
