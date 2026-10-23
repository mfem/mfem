//
//  fe_skgrad.hpp
//  mfem
//
//  Created by psaber on 6/20/25.
//

#ifndef fe_curl_hpp
#define fe_curl_hpp

#include <stdio.h>
#include "fe_base.hpp"
#include "fe_h1.hpp"
#include "fe_nd.hpp"
#include "fe_l2.hpp"
#include "fe_pyramid.hpp"

namespace mfem
{

class HCurl_PentatopeElement : public VectorFiniteElement
{
    
    static const double  tk1[10][4], tk2[14][4];
    
#ifndef MFEM_THREAD_SAFE
    mutable Vector shape_x, shape_y, shape_z, shape_t, shape_l;
    mutable Vector dshape_x, dshape_y, dshape_z, dshape_t, dshape_l;
    mutable DenseTensor u;
    mutable DenseMatrix Curlu;
#endif
    Array<int> dof2tk1;
    Array<int> dof2tk2;
    DenseMatrixInverse Ti;
    DenseMatrix T_inv;
    
    CURL_PentDofTransformation doftrans;

    
public:
    HCurl_PentatopeElement(const int p);
    virtual void CalcVShape(const IntegrationPoint &ip,
                            DenseMatrix &shape) const override;
    virtual void CalcVShape(ElementTransformation &Trans,
                            DenseMatrix &shape) const override
    { CalcVShape_Hcurl(Trans, shape); }
    virtual void CalcCurlShape(const IntegrationPoint &ip,
                                  DenseMatrix &Curlshape) const;
    virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                       DenseMatrix &I) const override
    {mfem_error("GetLocalInterpolaton error");}
    virtual void GetLocalRestriction(ElementTransformation &Trans,
                                     DenseMatrix &R) const override
    {          mfem_error("GetLocalRestriction error");}
    virtual void GetTransferMatrix(const FiniteElement &fe,
                                   ElementTransformation &Trans,
                                   DenseMatrix &I) const override
    {          mfem_error("GetTransferMatrix error");}
    const StatelessDofTransformation *GetDofTransformation() const override
    { return &doftrans; }
    using FiniteElement::Project;
    virtual void Project(VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const override;
    virtual void ProjectMatrixCoefficient(
                                          MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const override
    {          mfem_error("ProjectMatrixCoefficient error");}
    virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                         DenseMatrix &I) const override 
    {          mfem_error("Project error");}
    virtual void ProjectDivSkew(const FiniteElement &fe,
                                ElementTransformation &Trans,
                                DenseMatrix &DivSkew);
};

} // end of namespace
#endif /* fe_curl_hpp */
