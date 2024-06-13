#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

int enzyme_width;
int enzyme_dupv;
int enzyme_dupnoneedv;

// #define metricskew
#define metric2

real_t mu1(double *data)
{
    real_t normT = data[0]*data[0] + data[1]*data[1] + data[2]*data[2] + data[3]*data[3];
    return normT;
}

real_t mu2(double *data)
{
    real_t normT = data[0]*data[0] + data[1]*data[1] + data[2]*data[2] + data[3]*data[3];
    real_t detT = data[0]*data[3] - data[1]*data[2];
    return 0.5 *normT/detT - 1.0;
}

real_t muskew(double *data)
{
    // similar to TMOP_Metric_skew2D, assumes that Jtr=Identity so target sin(theta) = 1.0;
    real_t col1[2] = {data[0], data[1]};
    real_t col2[2] = {data[2], data[3]};
    real_t norm_col1 = std::pow(col1[0]*col1[0] + col1[1]*col1[1], 0.5);
    real_t norm_col2 = std::pow(col2[0]*col2[0] + col2[1]*col2[1], 0.5);
    real_t norm_prod = norm_col1 * norm_col2;
    real_t detT = data[0]*data[3] - data[1]*data[2];
    const real_t cos_Jpr = (col1[0]*col2[0] + col1[1]*col2[1]) / norm_prod,
                 sin_Jpr = fabs(detT) / norm_prod;

    real_t value = 0.5*(1.0 - sin_Jpr);
    return value;
}

// (det(T) - 1.0)^2
real_t mu55(double *data)
{
    double det = data[0] * data[3] - data[1] * data[2];
    return std::pow((det-1),2.0);
}


real_t mu(double *data)
{
    // mu1
    #ifdef metric1
        return mu1(data);
    #endif
    #ifdef metric2
        return mu2(data);
    #endif
    #ifdef metric55
        return mu55(data);
    #endif
    #ifdef metricskew
        return muskew(data);
    #endif
    return 0.0;
}

void TMOPGrad(double*in, double* out)
{
    __enzyme_autodiff<void>((void*)mu, in, out);
}

template<size_t width>
void hessian(double* in, double* identity, double* outtmp, DenseTensor &outputs) {
    for (size_t batch=0; batch<width*width; batch+=width) {
        __enzyme_fwddiff<void>((void*)TMOPGrad,
                                enzyme_width, width,
                                enzyme_dupv, sizeof(double)*width, in, &identity[batch * width],
                                enzyme_dupnoneedv, sizeof(double)*width, outtmp, outputs(batch).GetData()
                            );
    }
}

int main() {
    const int dim = 2;
    DenseMatrix T(dim);

    // this is shadow memory where the derivates are activated/deactivated
    DenseMatrix dT(dim);
    dT = 0.0;

    // this is the "gradient" it's size is size(T)
    DenseMatrix dmudT(dim);
    // This is the gradient from the TMOP metric class.
    DenseMatrix dmudT_tmop(dim);
    DenseTensor ddmuddT(dim, dim, dim*dim);

    Vector Tvec(T.GetData(), dim*dim);
    Tvec.Randomize(0);

    TMOP_QualityMetric *metric = NULL;

    // TMOP_Metric_055 metric;
    #ifdef metric1
        metric = new TMOP_Metric_001();
        MFEM_VERIFY(dim == 2, "This metric is only for 2D");
        std::cout << "Metric 1\n";
    #endif
    #ifdef metric2
        metric = new TMOP_Metric_002();
        MFEM_VERIFY(dim == 2, "This metric is only for 2D");
        std::cout << "Metric 2\n";
    #endif
    #ifdef metric55
        metric = new TMOP_Metric_055();
        MFEM_VERIFY(dim == 2, "This metric is only for 2D");
        std::cout << "Metric 55\n";
    #endif
    #ifdef metricskew
        metric = new TMOP_Metric_skew2D();
        MFEM_VERIFY(dim == 2, "This metric is only for 2D");
        std::cout << "Metric skew 2D\n";
    #endif

    DenseMatrix Iden(dim);
    Iden = 0.0;
    Iden(0, 0) = 1.0;
    Iden(1, 1) = 1.0;
    metric->SetTargetJacobian(Iden);

    std::cout << "Metric value = " << metric->EvalW(T) << " " << mu(T.GetData()) << std::endl;

#ifndef metricskew
    metric->EvalP(T, dmudT_tmop);
#endif

    TMOPGrad(T.GetData(), dmudT.GetData());

    DenseMatrix diff(dim);
    diff = dmudT;
    diff -= dmudT_tmop;

    std::cout << "==========================\n";
    std::cout << "T = " << std::endl;
    T.Print();
    std::cout << "==========================\n";

    std::cout << "TMOP derivative = " << std::endl;
    dmudT_tmop.Print();
    std::cout << "==========================\n";

    std::cout << "Enzyme derivative = " << std::endl;
    dmudT.Print();
    std::cout << "==========================\n";

    std::cout << "Difference in gradient = " << diff.FNorm2() << std::endl;
    std::cout << "==========================\n";


    // Second derivatives;
    DenseTensor ddT_TMOP(dim, dim, dim*dim);
#ifndef metricskew
    metric->ComputeH(T, ddT_TMOP);
#endif

    std::cout << "TMOP Hessian = " << std::endl;
    for (int i = 0; i < dim*dim; i++) {
        DenseMatrix ddt = ddT_TMOP(i);
        ddt.Print();
    }
    std::cout << "==========================\n";

    int matsize = dim*dim;
    double* identity = new double[matsize*matsize];
    for (size_t i=0; i<matsize; i++) {
        for (size_t j=0; j<matsize; j++) {
            identity[i*matsize+j] = (i == j) ? 1.0 : 0.0;
        }
    }

    Vector outtmp(matsize);
    outtmp = 0.0;

    DenseTensor outputs(dim, dim, dim*dim);
    outputs = 0.0;

    std::cout << "Enzyme Hessian = " << std::endl;
    hessian<dim*dim>(T.GetData(), identity, outtmp.GetData(), outputs);
    for (int i = 0; i < dim*dim; i++)
    {
        outputs(i).Print();
    }

    double ddt_diff_norm = 0.0;
    for (int i = 0; i < dim*dim; i++)
    {
        DenseMatrix temp = outputs(i);
        temp -= ddT_TMOP(i);
        ddt_diff_norm += temp.FNorm2();
    }

    std::cout << "==========================\n";
    std::cout << "Difference in Hessian = " << ddt_diff_norm << std::endl;
    std::cout << "==========================\n";

}
