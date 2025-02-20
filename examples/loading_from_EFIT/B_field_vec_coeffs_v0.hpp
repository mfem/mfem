#include "mfem.hpp"
#include <iostream>
#include <functional>

using namespace std;
using namespace mfem;

/// @brief Input $f$ and return $f/r$
class BTorFOverRGridFunctionCoefficient : public GridFunctionCoefficient
{
public:
    BTorFOverRGridFunctionCoefficient() : GridFunctionCoefficient() {}

    BTorFOverRGridFunctionCoefficient(const GridFunction *gf) : GridFunctionCoefficient(gf)
    {
    }

    real_t Eval(ElementTransformation &T,
                const IntegrationPoint &ip) override
    {
        // get r, z coordinates
        Vector x;
        T.Transform(ip, x);
        real_t r = x[0];
        return GridFunctionCoefficient::Eval(T, ip) / (1e-10 + r);
    }
};
