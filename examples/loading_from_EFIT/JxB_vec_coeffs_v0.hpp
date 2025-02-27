#include "mfem.hpp"
#include <iostream>
#include <functional>

using namespace std;
using namespace mfem;

/// @brief Input $f$ and return $f/r$
class JPerpRVectorGridFunctionCoefficient : public VectorGridFunctionCoefficient
{
public:
    JPerpRVectorGridFunctionCoefficient() : VectorGridFunctionCoefficient() {}

    JPerpRVectorGridFunctionCoefficient(const GridFunction *gf) : VectorGridFunctionCoefficient(gf)
    {
    }

    void Eval(Vector &V, ElementTransformation &T,
              const IntegrationPoint &ip) override
    {
        // get r, z coordinates
        Vector x;
        T.Transform(ip, x);
        real_t r = x[0];

        VectorGridFunctionCoefficient::Eval(V, T, ip);
        V *= r;
    }
};