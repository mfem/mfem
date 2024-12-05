#include "stokessolver.hpp"

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
    *vel_bc_gf = 0.0;

    pres_bc_gf.reset(new ParGridFunction(&pfes));
    *pres_bc_gf = 0.0;

    // The nonlinear convective integrators use over-integration (dealiasing) as
    // a stabilization mechanism.
    ir_nl = intrules.Get(vfes.GetFE(0)->GetGeomType(),
                         (int)(ceil(1.5 * 2*(vel_fes.GetOrder(0)+1) - 3)));

    ir = intrules.Get(vfes.GetFE(0)->GetGeomType(),
                      (int)(2*(vel_fes.GetOrder(0)+1) - 3));

    ir_face = intrules.Get(vfes.GetFaceElement(0)->GetGeomType(),
                           (int)(2*(vel_fes.GetOrder(0)+1) - 3));

}

void StokesOperator::SetVelBC(std::vector<VelDirichletBC>& vvbc)
{
    for(auto &vbc : vvbc){
        for (int i = 0; i < vbc.second->Size(); i++)
        {
            if (*(vbc.second)[i] == 1)
            {
                vel_ess_bdr[i] = 1;
            }
        }
    }
}

void StokesOperator::SetPressBC(std::vector<PresDirichletBC>& vpbc)
{
    for(auto &pbc : vpbc){
        for(int i=0;i<pbc.second->Size();i++){
            if (*(pbc.second)[i] == 1)
            {
                vel_ess_bdr[i] = 1;
            }
        }
    }
}


}
