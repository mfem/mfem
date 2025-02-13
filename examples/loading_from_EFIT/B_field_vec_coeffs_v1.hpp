#include "mfem.hpp"
#include <iostream>
#include <functional>

using namespace std;
using namespace mfem;

/// @brief Returns f(u(x)) where u is a scalar GridFunction and f:R → R
class BTorFromFGridFunctionCoefficient : public Coefficient
{
private:
    const GridFunction *gf;

public:
    int counter = 0;
    FindPointsGSLIB finder;

    BTorFromFGridFunctionCoefficient()
    {
    }

    BTorFromFGridFunctionCoefficient(const GridFunction *gf)
        : gf(gf)
    {
        gf->FESpace()->GetMesh()->EnsureNodes();
        finder.Setup(*gf->FESpace()->GetMesh());
    }

    real_t Eval(ElementTransformation &T,
                const IntegrationPoint &ip) override
    {
        // get r, z coordinates
        Vector x;
        T.Transform(ip, x);
        real_t r = x[0];
        counter++;
        Vector interp_val(1);
        finder.Interpolate(x, *gf, interp_val, 0);
        return interp_val[0] / (1e-10 + r);
    }
};
