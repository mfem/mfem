#ifndef STOKESSOLVER_H
#define STOKESSOLVER_H

#define STOKES_VERSION 0.1

#include "mfem.hpp"


namespace mfem {

using VelDirichletBC = std::pair<VectorCoefficient *, Array<int> *>;
using PresDirichletBC = std::pair<Coefficient *, Array<int> *>;

class StokesOperator:public Operator
{
public:
StokesOperator(ParFiniteElementSpace &vel_fes,
                   ParFiniteElementSpace &pres_fes);

void SetVelBC(std::vector<VelDirichletBC>& vvbc);
void SetPressBC(std::vector<PresDirichletBC>& vpbc);


void Mult(const Vector &x, Vector &y) const override;

const Array<int>& GetOffsets() const;

ParFiniteElementSpace &vfes;
ParFiniteElementSpace &pfes;

ParGridFunction &kinematic_viscosity;

std::unique_ptr<ParGridFunction> vel_bc_gf, pres_bc_gf;

Array<int> vel_ess_bdr;
Array<int> pres_ess_bdr;

Array<int> vel_ess_tdofs;
Array<int> pres_ess_tdofs;

bool matrix_free;

Array<int> offsets;

IntegrationRules intrules;
IntegrationRule ir; //general integraion rule
IntegrationRule ir_nl; //non-linear integration rule
IntegrationRule ir_face; //face integration rule

ConstantCoefficient zero_coeff;

};

class StokesSolver
{
public:
    StokesSolver();
};


}

#endif // STOKESSOLVER_H
