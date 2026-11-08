#ifndef MFEM_STOKES_INTEGRATORS_HPP
#define MFEM_STOKES_INTEGRATORS_HPP

#include "mfem.hpp"
#include <memory>
#include <utility>

namespace mfem::stokes
{

int HessianIndex(int dimension, int row, int column);

namespace detail
{

/// Local uniform polynomial coordinate degree, or -1 for unsupported geometry.
int StokesGeometryOrder(const Mesh &mesh);
/// Polynomial degree of a supported P_g/Q_g isoparametric transformation.
int StokesGeometryOrder(const ElementTransformation &transformation);
/// Automatic geometry-aware rule, or the unchanged positive user override.
int StokesIntegrationOrder(int solution_order, int geometry_order,
                           int dimension, int requested = -1);

/** @brief Element-local, geometry-preserving workspace for physical Hessians.

    Prepare borrows the source until the next Prepare or destruction. It owns
    an equivalent H1 nodal representation of the original polynomial map,
    including discontinuous periodic L2 coordinates. No mesh nodes/topology
    change. Prepare once outside the quadrature loop; synchronize points before
    evaluating derivatives. Coefficients must use the original transformation.
    The full chain rule retains curvature at every scale, without an absolute
    cutoff on the coordinate Hessian. Instances are not copyable. */
class StokesGeometryContext
{
   ElementTransformation *source = nullptr;
   std::unique_ptr<H1_FECollection> collection;
   IsoparametricTransformation geometry;
   int order = 0, dimension = 0;
public:
   StokesGeometryContext() = default;
   StokesGeometryContext(const StokesGeometryContext &) = delete;
   StokesGeometryContext &operator=(const StokesGeometryContext &) = delete;
   void Prepare(ElementTransformation &transformation);
   void SetIntPoint(const IntegrationPoint &point);
   int GeometryOrder() const { return order; }
   ElementTransformation &HessianTransformation() { return geometry; }
   void CalcPhysHessian(const FiniteElement &element, DenseMatrix &hessian);
};

}

/** @brief Evaluate the strong symmetric-gradient momentum operator on a basis.

    Supports polynomial high-order H1/L2 coordinate maps without linearizing
    geometry. The workspace overload reuses a map prepared for this element. */
void EvaluateMomentumBasis(const FiniteElement &element,
                           ElementTransformation &transformation,
                           Coefficient &viscosity,
                           VectorCoefficient &viscosity_gradient,
                           DenseMatrix &residual);
void EvaluateMomentumBasis(const FiniteElement &element,
                           ElementTransformation &transformation,
                           Coefficient &viscosity,
                           VectorCoefficient &viscosity_gradient,
                           DenseMatrix &residual,
                           detail::StokesGeometryContext &geometry);

template <typename Base> class RuleBilinearIntegrator : public Base
{
   int order;
public:
   template <typename... Args>
   RuleBilinearIntegrator(int order_, Args&&... args)
      : Base(std::forward<Args>(args)...), order(order_) { }

   void AssembleElementMatrix(const FiniteElement &element,
                              ElementTransformation &transformation,
                              DenseMatrix &matrix) override
   {
      this->SetIntRule(&IntRules.Get(element.GetGeomType(), order));
      Base::AssembleElementMatrix(element, transformation, matrix);
   }

   void AssembleElementMatrix2(const FiniteElement &trial,
                               const FiniteElement &test,
                               ElementTransformation &transformation,
                               DenseMatrix &matrix) override
   {
      this->SetIntRule(&IntRules.Get(trial.GetGeomType(), order));
      Base::AssembleElementMatrix2(trial, test, transformation, matrix);
   }
};

template <typename Base> class RuleLinearIntegrator : public Base
{
   int order;
public:
   template <typename... Args>
   RuleLinearIntegrator(int order_, Args&&... args)
      : Base(std::forward<Args>(args)...), order(order_) { }

   void AssembleRHSElementVect(const FiniteElement &element,
                              ElementTransformation &transformation,
                              Vector &vector) override
   {
      this->SetIntRule(&IntRules.Get(element.GetGeomType(), order));
      Base::AssembleRHSElementVect(element, transformation, vector);
   }
};

class PSPGSymGradIntegrator : public BilinearFormIntegrator
{
public:
   PSPGSymGradIntegrator(Coefficient &viscosity,
                         VectorCoefficient &viscosity_gradient,
                         Coefficient &tau, int integration_order = -1);

   void AssembleElementMatrix2(const FiniteElement &trial,
                               const FiniteElement &test,
                               ElementTransformation &transformation,
                               DenseMatrix &matrix) override;

private:
   Coefficient &viscosity;
   VectorCoefficient &viscosity_gradient;
   Coefficient &tau;
   int integration_order;
};

class GLSVelocityIntegrator : public BilinearFormIntegrator
{
   Coefficient &viscosity;
   VectorCoefficient &viscosity_gradient;
   Coefficient &tau;
   int integration_order;
public:
   GLSVelocityIntegrator(Coefficient &viscosity,
                         VectorCoefficient &viscosity_gradient,
                         Coefficient &tau, int integration_order = -1);
   void AssembleElementMatrix(const FiniteElement &element,
                              ElementTransformation &transformation,
                              DenseMatrix &matrix) override;
};

class GLSForceIntegrator : public LinearFormIntegrator
{
   Coefficient &viscosity;
   VectorCoefficient &viscosity_gradient;
   Coefficient &tau;
   VectorCoefficient &force;
   int integration_order;
public:
   GLSForceIntegrator(Coefficient &viscosity,
                      VectorCoefficient &viscosity_gradient,
                      Coefficient &tau, VectorCoefficient &force,
                      int integration_order = -1);
   void AssembleRHSElementVect(const FiniteElement &element,
                              ElementTransformation &transformation,
                              Vector &vector) override;
};

class PressureCIPIntegrator : public BilinearFormIntegrator
{
   Coefficient &element_size;
   Coefficient &element_viscosity;
   real_t gamma;
   int integration_order;
public:
   PressureCIPIntegrator(Coefficient &element_size,
                         Coefficient &element_viscosity,
                         real_t gamma, int integration_order = -1);
   void AssembleFaceMatrix(const FiniteElement &first,
                           const FiniteElement &second,
                           FaceElementTransformations &transformation,
                           DenseMatrix &matrix) override;
};

}
#endif
