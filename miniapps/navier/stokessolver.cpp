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

}


}
