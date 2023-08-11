#include "custom_bilinteg.hpp"

namespace mfem
{

/**
 * @brief Get the integration rule for the given finite element and transformation.
 *
 * @param fe The finite element for which to get the integration rule.
 * @param T The element transformation.
 * @return const IntegrationRule& The integration rule.
 */
const IntegrationRule& VectorConvectionIntegrator::GetRule(const FiniteElement &fe,
        ElementTransformation &T)
{
    const int order = 2 * fe.GetOrder() + T.OrderGrad(&fe);
    return IntRules.Get(fe.GetGeomType(), order);
}

/**
 * @brief Assemble the element matrix for the VectorConvectionIntegrator.
 *
 * @param el The finite element.
 * @param Trans The element transformation.
 * @param elmat The element matrix to be assembled.
 */
void VectorConvectionIntegrator::AssembleElementMatrix(const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{
    const int dof = el.GetDof();
    dim = el.GetDim();

    elmat.SetSize(dim * dof);
    dshape.SetSize(dof, dim);
    adjJ.SetSize(dim);
    shape.SetSize(dof);
    vec1.SetSize(dim);
    vec2.SetSize(dim);
    vec3.SetSize(dof);
    pelmat.SetSize(dof);
    DenseMatrix pelmat_T(dof);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &GetRule(el, Trans);
    }

    W->Eval(W_ir, Trans, *ir);

    elmat = 0.0;
    pelmat_T = 0.0;

    // Calculate constant values outside the loop
    const double alpha_weight = alpha * ir->IntPoint(0).weight;
    Trans.SetIntPoint(&ir->IntPoint(0));
    CalcAdjugate(Trans.Jacobian(), adjJ);
    const double q_const = alpha_weight * Trans.Weight();

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcDShape(ip, dshape);
        el.CalcShape(ip, shape);

        Trans.SetIntPoint(&ip);
        CalcAdjugate(Trans.Jacobian(), adjJ);
        W_ir.GetColumnReference(i, vec1);     // tmp = W

        const double q = alpha * ip.weight; // q = alpha*weight   || q = weight
        adjJ.Mult(vec1, vec2);               // element Transformation J^{-1} |J|
        vec2 *= q;

        dshape.Mult(vec2, vec3);           // (w . grad u)           q ( alpha J^{-1} |J| w dPhi )
        MultVWt(shape, vec3, pelmat);      // (w . grad u,v)         q ( alpha J^{-1} |J| w dPhi Phi^T)

        if (SkewSym)
        {
            pelmat_T.Transpose(pelmat);
        }

        for (int k = 0; k < dim; k++)
        {
            if (SkewSym)
            {
                elmat.AddMatrix(.5, pelmat, dof * k, dof * k);
                elmat.AddMatrix(-.5, pelmat_T, dof * k, dof * k);
            }
            else
            {
                elmat.AddMatrix(pelmat, dof * k, dof * k);
            }
        }
    }
}

/**
 * @brief Get the integration rule for the VectorGradCoefficientIntegrator.
 *
 * @param fe The finite element for which to get the integration rule.
 * @param T The element transformation.
 * @return const IntegrationRule& The integration rule.
 */
const IntegrationRule& VectorGradCoefficientIntegrator::GetRule(const FiniteElement &fe,
        ElementTransformation &T)
{
    return VectorConvectionIntegrator::GetRule(fe, T);
}

/**
 * @brief Assemble the element matrix for the VectorGradCoefficientIntegrator.
 *
 * @param el The finite element.
 * @param Trans The element transformation.
 * @param elmat The element matrix to be assembled.
 */
void VectorGradCoefficientIntegrator::AssembleElementMatrix(const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{
    const int dof = el.GetDof();
    dim = el.GetDim();

    shape.SetSize(dof);
    elmat.SetSize(dof * dim);
    pelmat.SetSize(dof);
    gradW.SetSize(dim);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &GetRule(el, Trans);
    }

    elmat = 0.0;
    // compute gradient (with respect to the physical element)
    W_gf = W->GetGridFunction();

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        el.CalcShape(ip, shape);

        W_gf->GetVectorGradient(Trans, gradW);

        MultVVt(shape, pelmat);
        const double q = alpha * ip.weight * Trans.Weight();

        for (int ii = 0; ii < dim; ii++)
        {
            for (int jj = 0; jj < dim; jj++)
            {
                elmat.AddMatrix(q * gradW(ii, jj), pelmat, ii * dof, jj * dof);
            }
        }
    }
}

}  // namespace mfem