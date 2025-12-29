/// Templated AD (block) nonlinear form integrators definitions
#pragma once
#include "mfem.hpp"
#include "ad_native.hpp"

namespace mfem
{
// A helper struct to pass the loop index as a template parameter
template<std::size_t N>
struct __loop_index
{
   static const constexpr size_t value = N;
   constexpr operator std::size_t() const { return N; }
};
// loop over indeces at compile time
template <class F, std::size_t... Is>
void _constexpr_for(F func, std::index_sequence<Is...>)
{
   (func(__loop_index<Is> {}), ...);
}
// @brief ADQuadEvalMode is an enumeration for the evaluation modes of the ADEnergy class.
// For example, if you want to evaluate the value and gradient of the function, you can use
// constexpr auto mode = ADEval::VALUE | ADEval::GRAD;
enum class ADEval
{
   QVALUE  = 1 << 0, // u(T, ip) (quadrature value)
   VALUE   = 1 << 1, // u(T, ip)
   GRAD    = 1 << 2, // grad u(T, ip)
   DIV     = 1 << 3, // div u(T, ip) (not yet implemented)
   CURL    = 1 << 4, // curl u(T, ip) (not yet implemented)
   Hessian = 1 << 5, // D^2 u(T, ip) (not yet implemented)

   VECTOR  = 1 << 6, // vector-valued scalar FE
   VECFE   = 1 << 7, // vector-valued vector FE (not yet implemented)
   NUMOPT  = 1 << 8, // number of options. If change options, change this value to last
};

constexpr ADEval operator|(ADEval a, ADEval b)
{
   return static_cast<ADEval>(static_cast<int>(a) | static_cast<int>(b));
}
constexpr ADEval operator&(ADEval a, ADEval b)
{
   return static_cast<ADEval>(static_cast<int>(a) & static_cast<int>(b));
}
inline constexpr ADEval operator~(ADEval mode)
{
   return static_cast<ADEval>(~static_cast<int>(mode));
}
inline constexpr bool hasFlag(ADEval mode, ADEval flag)
{
   return (mode & flag) == flag;
}

template <ADEval mode>
constexpr bool isValidADEval()
{
   constexpr auto INVALID = ADEval::Hessian;
   if constexpr (static_cast<int>(mode & INVALID) != 0) { return false; }
   if constexpr (hasFlag(mode, ADEval::QVALUE))
   {
      // QVALUE cannot be combined with other modes except VECTOR
      return static_cast<int>(mode & (~(ADEval::QVALUE | ADEval::VECTOR))) == 0;
   }
   return true;
}

template <ADEval... modes>
class ADBlockNonlinearFormIntegrator;

template <ADEval mode>
class ADNonlinearFormIntegrator : public NonlinearFormIntegrator
{
   static_assert(isValidADEval<mode>(),
              "ADNonlinearFormIntegrator: Invalid ADEval mode");
protected:
   ADFunction &f;

private:
   int vdim;
   Vector x, jac;
   DenseMatrix H, Hx;

   // only if ADEvalInput::VECTOR. Each column corresponds to a vector component
   DenseMatrix xmat, jacMat, Hs, Hxsub;
   DenseMatrix elfun_matview, elvectmat, partelmat;

   DenseMatrix allshapes; // all shapes, [?shape, ?dshape]
   Vector shape, shape1, shape2;
   DenseMatrix vshape, vshape1, vshape2;
   DenseMatrix gshape, gshape1, gshape2;
   Vector divshape, divshape1, divshape2;
   DenseMatrix curlshape, curlshape1, curlshape2;
   Vector nor;
   // DenseMatrix d2shape, d2shape1, d2shape2; // for hessian. Not implemented yet.
public:
   ADNonlinearFormIntegrator(ADFunction &f, IntegrationRule *ir = nullptr)
      : NonlinearFormIntegrator(ir), f(f), vdim(1) {}
   ADNonlinearFormIntegrator(ADFunction &f, int vdim,
                             IntegrationRule *ir = nullptr)
      : NonlinearFormIntegrator(ir), f(f), vdim(vdim) {}

   const IntegrationRule* GetDefaultIntegrationRule(
      const FiniteElement& trial_fe, const FiniteElement& test_fe,
      const ElementTransformation& trans) const override
   {
      int order = std::max(trial_fe.GetOrder(), test_fe.GetOrder());
      return &IntRules.Get(trans.GetGeometryType(), order*2 + 2);
   }

   /// Compute the local energy
   real_t GetElementEnergy(const FiniteElement &el,
                           ElementTransformation &Tr,
                           const Vector &elfun) override;

   /// Perform the local action of the NonlinearFormIntegrator
   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override;

   /// Assemble the local gradient matrix
   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override;

   /// @brief Perform the local action of the NonlinearFormIntegrator resulting
   /// from a face integral term.
   void AssembleFaceVector(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, Vector &elvect) override;


   /// @brief Assemble the local action of the gradient of the
   /// NonlinearFormIntegrator resulting from a face integral term.
   void AssembleFaceGrad(const FiniteElement &el1,
                         const FiniteElement &el2,
                         FaceElementTransformations &Tr,
                         const Vector &elfun, DenseMatrix &elmat) override;


protected:

   // Initialize shapes to [?value_shapes, ?grad_shapes]
   // and make value_shapes and grad_shapes reference to
   // allshapes.
   inline int InitInputShapes(const FiniteElement &el,
                                     ElementTransformation &Tr,
                                     DenseMatrix &shapes);

   // Calculate parameter, shape, dshape at the given integration point
   inline void CalcInputShapes(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const IntegrationPoint &ip,
                                      DenseMatrix &shapes);
   template <ADEval... modes>
   friend class ADBlockNonlinearFormIntegrator;
private:
};

template <ADEval... modes>
class ADBlockNonlinearFormIntegrator : public BlockNonlinearFormIntegrator
{
public:
   const IntegrationRule *IntRule = nullptr;

protected:
   constexpr static size_t numSpaces = sizeof...(modes);
   static constexpr std::array<ADEval, sizeof...(modes)> modes_arr = {modes...};
   ADFunction &f;
   const IntegrationRule* GetIntegrationRule(
      const FiniteElement& trial_fe, const FiniteElement& test_fe,
      const ElementTransformation& trans) const;

   /** @brief Returns an integration rule based on the arguments and
              internal state. (Version for identical trial_fe and test_fe)

       @see GetIntegrationRule(const FiniteElement*, const FiniteElement*,
            const ElementTransformation*)
   */
   const IntegrationRule* GetIntegrationRule(
      const FiniteElement& el,
      const ElementTransformation& trans) const;

private:
   Array<int> vdim;
   Vector x, jac;
   std::vector<Vector> xvar, jacVar;
   DenseMatrix H;
   DenseMatrix Hsub;
   DenseMatrix Hx;
   DenseMatrix Hxsub;

   // only if ADEvalInput::VECTOR. Each column corresponds to a vector component
   std::vector<DenseMatrix> xmat, jacVarMat, Hs;
   std::vector<DenseMatrix> elfun_matview, elvectmat, partelmat;

   std::vector<DenseMatrix> allshapes; // all shapes, [?shape, ?dshape]
   std::vector<Vector> shape, shape1, shape2;
   std::vector<DenseMatrix> vshape, vshape1, vshape2;
   std::vector<DenseMatrix> gshape, gshape1, gshape2;
   std::vector<Vector> divshape, divshape1, divshape2;
   std::vector<DenseMatrix> curlshape, curlgshape1, curlgshape2;
   Vector nor;
   // DenseMatrix d2shape, d2shape1, d2shape2; // for hessian. Not implemented yet.
public:
   ADBlockNonlinearFormIntegrator(ADFunction &f,
                                  const IntegrationRule *ir = nullptr)
      : IntRule(ir), f(f), vdim(numSpaces)
      , allshapes(numSpaces)
      , xvar(numSpaces), jacVar(numSpaces)
      , Hx(numSpaces)
      , xmat(numSpaces), jacVarMat(numSpaces)
      , Hs(numSpaces), Hxsub(numSpaces)
      , elfun_matview(numSpaces), elvectmat(numSpaces)
      , partelmat(numSpaces)
      , shape(numSpaces), shape1(numSpaces), shape2(numSpaces)
      , vshape(numSpaces), vshape1(numSpaces), vshape2(numSpaces)
      , gshape(numSpaces), gshape1(numSpaces), gshape2(numSpaces)
      , divshape(numSpaces), divshape1(numSpaces), divshape2(numSpaces)
      , curlshape(numSpaces), curlgshape1(numSpaces), curlgshape2(numSpaces)
   { vdim = 1; }

   ADBlockNonlinearFormIntegrator(ADFunction &f, std::initializer_list<int> vdim,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator(f, ir), vdim(vdim)
   {}

   ADBlockNonlinearFormIntegrator(ADFunction &f, const Array<int> &vdim,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator(f, ir)
   { this->vdim = vdim; }

   virtual void SetIntRule(const IntegrationRule *ir)
   { IntRule = ir; }

   /** @brief Prescribe a fixed IntegrationRule to use. Sets the NURBS patch
              integration rule to null.

       @see SetIntRule(const IntegrationRule*)
   */
   void SetIntegrationRule(const IntegrationRule &ir) { SetIntRule(&ir); }

   /** @brief Directly return the IntRule pointer (possibly null) without
       checking for NURBS patch rules or falling back on a default. */
   const IntegrationRule *GetIntRule() const { return IntRule; }

   /** @brief Equivalent to GetIntRule, but retained for backward
       compatibility with applications. */
   const IntegrationRule *GetIntegrationRule() const { return GetIntRule(); }


   /// Compute the local energy
   real_t GetElementEnergy(const Array<const FiniteElement *> &el,
                           ElementTransformation &Tr,
                           const Array<const Vector*> &elfun) override;

   /// Perform the local action of the NonlinearFormIntegrator
   void AssembleElementVector(const Array<const FiniteElement *>&el,
                              ElementTransformation &Tr,
                              const Array<const Vector *>&elfun,
                              const Array<Vector *>&elvect) override;

   /// Assemble the local gradient matrix
   void AssembleElementGrad(const Array<const FiniteElement *>&el,
                            ElementTransformation &Tr,
                            const Array<const Vector *>&elfun,
                            const Array2D<DenseMatrix *>&elmat) override;

   /// @brief Perform the local action of the NonlinearFormIntegrator resulting
   /// from a face integral term.
   void AssembleFaceVector(const Array<const FiniteElement *>&el1,
                           const Array<const FiniteElement *>&el2,
                           FaceElementTransformations &Tr,
                           const Array<const Vector *>&elfun,
                           const Array<Vector *>&elvect) override;


   /// @brief Assemble the local action of the gradient of the
   /// NonlinearFormIntegrator resulting from a face integral term.
   void AssembleFaceGrad(const Array<const FiniteElement *>&el1,
                         const Array<const FiniteElement *>&el2,
                         FaceElementTransformations &Tr,
                         const Array<const Vector *>&elfun,
                         const Array2D<DenseMatrix *>&elmat) override;


protected:

   const IntegrationRule* GetIntegrationRule(
      const Array<const FiniteElement *>& trial_fe,
      const Array<const FiniteElement *>& test_fe,
      const ElementTransformation& trans) const
   {
      if (IntRule) { return IntRule; }
      return GetDefaultIntegrationRule(trial_fe, test_fe, trans);
   }

   const IntegrationRule* GetIntegrationRule(
      const Array<const FiniteElement *>& el,
      const ElementTransformation& trans) const
   {
      if (IntRule) { return IntRule; }
      return GetDefaultIntegrationRule(el, el, trans);
   }

   virtual const IntegrationRule* GetDefaultIntegrationRule(
      const Array<const FiniteElement *>& trial_fe,
      const Array<const FiniteElement *>& test_fe,
      const ElementTransformation& trans) const
   {
      int order = 0;
      for (int i=0; i<trial_fe.Size(); i++)
      {
         order = std::max(order, trial_fe[i]->GetOrder());
      }
      for (int i=0; i<test_fe.Size(); i++)
      {
         order = std::max(order, test_fe[i]->GetOrder());
      }
      return &IntRules.Get(trans.GetGeometryType(), order*2 + 2);
   }

   std::array<int, sizeof...(modes)> InitInputShapes(
                                    const Array<const FiniteElement *>& el,
                                    ElementTransformation &Tr,
                                    std::vector<DenseMatrix> &shapes);

   void CalcInputShapes(
      const Array<const FiniteElement *>& el,
      ElementTransformation &Tr,
      const IntegrationPoint &ip,
      std::vector<DenseMatrix> &allshapes);

private:
};
} // namespace mfem

