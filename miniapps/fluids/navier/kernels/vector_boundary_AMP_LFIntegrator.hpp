#include <mfem.hpp>

namespace mfem
{
   /// Class for boundary integration $ L(v) = ( f \cdot t, v \cdot t ) $ in 2D
   /// $f=(f_1,\dots,f_n)$ and $v =(v_1,\dots,v_n)$.
   /// f = (g_s + \alpha u_s)
   class VectorBoundaryTangentialAMPLFIntegrator : public LinearFormIntegrator
   {
      Vector shape, tangent, Qvec, Uvec;
      VectorCoefficient *VG; // Coefficient storing sigmaS_dot_n  //< NOT OWNED
      VectorCoefficient *VU; // Coefficient storing u_s           //< NOT OWNED
      Coefficient *Q;        // Coefficient storing alpha         //< NOT OWNED

   public:
      /// Constructs a boundary integrator with a given Coefficient QG
      VectorBoundaryTangentialAMPLFIntegrator(VectorCoefficient *sigmaS_dot_n, VectorCoefficient *uS, Coefficient *alpha)
          : VG(sigmaS_dot_n), VU(uS), Q(alpha) {}

      void AssembleRHSElementVect(const FiniteElement &el,
                                  ElementTransformation &Tr,
                                  Vector &elvect) override;

      using LinearFormIntegrator::AssembleRHSElementVect;
   };

   /// Class for boundary integration $ L(v) = ( f \cdot n, v \cdot n ) $ in 2D
   /// $f=(f_1,\dots,f_n)$ and $v =(v_1,\dots,v_n)$.
   /// f = (g_s + \alpha u_s)
   class VectorBoundaryNormalAMPLFIntegrator : public LinearFormIntegrator
   {
      Vector shape, normal, Qvec, Uvec;
      VectorCoefficient *VG; // Coefficient storing sigmaS_dot_n  //< NOT OWNED
      VectorCoefficient *VU; // Coefficient storing u_s           //< NOT OWNED
      Coefficient *Q;        // Coefficient storing alpha         //< NOT OWNED

   public:
      /// Constructs a boundary integrator with a given Coefficient QG
      VectorBoundaryNormalAMPLFIntegrator(VectorCoefficient *sigmaS_dot_n, VectorCoefficient *uS, Coefficient *alpha)
          : VG(sigmaS_dot_n), VU(uS), Q(alpha) {}

      void AssembleRHSElementVect(const FiniteElement &el,
                                  ElementTransformation &Tr,
                                  Vector &elvect) override;

      using LinearFormIntegrator::AssembleRHSElementVect;
   };
}