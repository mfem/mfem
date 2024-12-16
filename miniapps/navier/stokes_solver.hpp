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

    virtual
    void Mult(const Vector &x, Vector &y) const override;

    const Array<int>& GetOffsets() const
    {
        return offsets;
    }

    void Setup();
    void Assemble();


private:
    ParFiniteElementSpace &vfes;
    ParFiniteElementSpace &pfes;

// ParGridFunction &kinematic_viscosity;

    std::unique_ptr<ParGridFunction> vel_bc_gf;
    std::unique_ptr<ParGridFunction> pres_bc_gf;

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

    std::unique_ptr<Coefficient> viscosity;

    ParBilinearForm *b11_form; //velocity
    ParBilinearForm *b22_form; //pressure
    ParMixedBilinearForm *b12_form; //mixed (velocity,pressure)
    ParMixedBilinearForm *b21_form; //mized (pressure,velocity)

    BlockOperator* A;

};


}

#endif // STOKESSOLVER_H
