#include "mfem.hpp"
#include <iostream>
#include <functional>

using namespace std;
using namespace mfem;

class FindPointsGSLIBOneByOne : public FindPointsGSLIB {
public:
    FindPointsGSLIBOneByOne() : FindPointsGSLIB() {
    }

    void InterpolateOneByOne(const Vector &point_pos,
                             const GridFunction &field_in,
                             Vector &field_out,
                             int point_pos_ordering = Ordering::byNODES) {
        FindPoints(point_pos, point_pos_ordering);
        // gsl_mfem_elem (element number) and gsl_mfem_ref (this is the integration point location)
        int element_number = gsl_mfem_elem[0];
        IntegrationPoint ip;
        ip.Set2(gsl_mfem_ref.GetData());
        field_in.GetVectorValue(element_number, ip, field_out);
    }
};

/// @brief Returns f(u(x)) where u is a scalar GridFunction and f:R → R
class BTorFromFGridFunctionCoefficient : public Coefficient {
private:
    const GridFunction *gf;

public:
    int counter = 0;
    FindPointsGSLIBOneByOne finder;

    BTorFromFGridFunctionCoefficient() : Coefficient(), gf(nullptr) {
    }

    BTorFromFGridFunctionCoefficient(const GridFunction *gf)
        : Coefficient(), gf(gf) {
        gf->FESpace()->GetMesh()->EnsureNodes();
        finder.Setup(*gf->FESpace()->GetMesh());
    }

    real_t Eval(ElementTransformation &T,
                const IntegrationPoint &ip) override {
        // get r, z coordinates
        Vector x;
        T.Transform(ip, x);
        real_t r = x[0];
        counter++;
        Vector interp_val(1);
        finder.InterpolateOneByOne(x, *gf, interp_val, 0);
        return interp_val[0] / (1e-10 + r);
    }
};
