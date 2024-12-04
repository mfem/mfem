#ifndef STOKESSOLVER_H
#define STOKESSOLVER_H

#define STOKES_VERSION 0.1

#include "mfem.hpp"


namespace mfem {

class StokesOperator:public Operator
{
public:
StokesOperator(ParFiniteElementSpace &vel_fes,
                   ParFiniteElementSpace &pres_fes);

void Mult(const Vector &x, Vector &y) const override;

const Array<int>& GetOffsets() const;

ParFiniteElementSpace &vfes;
ParFiniteElementSpace &pfes;

ParGridFunction &kinematic_viscosity;

Array<int> vel_ess_bdr;
Array<int> pres_ess_bdr;

Array<int> vel_ess_tdofs;
Array<int> pres_ess_tdofs;

bool matrix_free;

Array<int> offsets;

IntegrationRules intrules;

ConstantCoefficient zero_coeff;

};

class StokesSolver
{
public:
    StokesSolver();
};


}

#endif // STOKESSOLVER_H
