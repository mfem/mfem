//
//  fe_skgrad.hpp
//  mfem
//
//  Created by psaber on 6/20/25.
//

#ifndef fe_skgrad_hpp
#define fe_skgrad_hpp

#include <stdio.h>
#include "fe_base.hpp"
#include "fe_h1.hpp"
#include "fe_nd.hpp"
#include "fe_l2.hpp"
#include "fe_pyramid.hpp"

namespace mfem
{

class HSkwGrad_PentatopeElement : public VectorFiniteElement
{
    static const double tk[40], c;
    
#ifndef MFEM_THREAD_SAFE
    mutable Vector shape_x, shape_y, shape_z, shape_t, shape_l;
    mutable Vector dshape_x, dshape_y, dshape_z, dshape_t, dshape_l;
    mutable DenseMatrix u;
    mutable DenseMatrix SkwGradu;
#endif
    Array<int> dof2tk;
    DenseMatrixInverse Ti;
    
    ND_PentDofTransformation doftrans;

    
public:
    HSkwGrad_PentatopeElement(const int p);
    virtual void CalcVShape(const IntegrationPoint &ip,
                            DenseMatrix &shape) const override;
    virtual void CalcVShape(ElementTransformation &Trans,
                            DenseMatrix &shape) const override
    { CalcVShape_ND(Trans, shape); }
    virtual void CalcSkwGradShape(const IntegrationPoint &ip,
                                  DenseMatrix &SkwGradshape) const;
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
                         ElementTransformation &Trans, Vector &dofs) const override
    { Project_ND(tk, dof2tk, vc, Trans, dofs); }
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
#endif /* fe_skgrad_hpp */
