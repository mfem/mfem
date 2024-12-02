#include "mfem.hpp"

namespace mfem
{


/**
 * @brief Extract orthogornal vector from B not belonging to A.
 *
 * @param A sub-subspace
 * @param B subspace
 * @param n a unit vector in B orthogornal to column space of A.
 */
void CalcOrtho(const DenseMatrix &A, const DenseMatrix &B, Vector &n);

class ManifoldCoord
{
   // attributes
private:
   ElementTransformation *curr_el;
   mutable DenseMatrix mani_vec_state;
   mutable DenseMatrix phys_vec_state;
   mutable Vector normal_comp;
   mutable Vector phys_vec;
protected:
public:
   const int dim;
   const int sdim;

   // methods
private:
protected:
public:
   ManifoldCoord(const int dim, const int sdim):dim(dim), sdim(sdim) {}

   /**
    * @brief Convert manifold state to physical state
    *
    * @param el Target element
    * @param state  Current state value
    * @param phys_state Current Physical state value
    */
   void convertElemState(ElementTransformation &Tr,
                         const int nrScalar, const int nrVector,
                         const Vector &state, Vector &phys_state);

   /**
    * @brief Convert left and right states to physical states
    *
    * In physical space, scalar spaces are the same as manifold state
    * However, vector states are translated in local coordinates
    * Basic conversion is v -> Jv
    * To incoporate discontinuity of local coordinates along the interface,
    * we convert state from one element (left) to another element (right) by
    * v -> J1v -> J1v + (n1 dot Jv) (n2 - n1)
    * where J1, n1 are Jacobian and normal vector from one element,
    * and n2 is the normal vector from another element.
    *
    * @param Tr Interface transformation
    * @param nrScalar The number of scalar states
    * @param nrVector The number of vector states
    * @param stateL Input left state
    * @param stateR Input right state
    * @param normalL **outward** normal from the left element
    * @param normalR **inward** normal from the right element
    * @param stateL_L **left** state in the __left__ coordinate system
    * @param stateR_L **right** state in the __left__ coordinate system
    * @param stateL_R **left** state in the __right__ coordinate system
    * @param stateR_R **right** state in the __right__ coordinate system
    */
   void convertFaceState(FaceElementTransformations &Tr,
                         const int nrScalar, const int nrVector,
                         const Vector &stateL, const Vector &stateR,
                         Vector &normalL, Vector &normalR,
                         Vector &stateL_L, Vector &stateR_L,
                         Vector &stateL_R, Vector &stateR_R);

};

class ManifoldFlux : public FluxFunction
{
   // attributes
private:
   FluxFunction &org_flux;
   ManifoldCoord &coord;
   int nrScalar;
   int nrVector;
   mutable Vector phys_state;
protected:
public:
   const int sdim;

   // methods
private:
protected:
public:
   ManifoldFlux(FluxFunction &flux, ManifoldCoord &coord, int nrScalar)
      : FluxFunction(flux.num_equations, flux.dim), org_flux(flux),
        coord(coord), nrScalar(nrScalar), sdim(coord.sdim)
   {
      nrVector = (org_flux.num_equations - nrScalar)/sdim;
      phys_state.SetSize(nrScalar + nrVector*sdim);
   }

   /**
    * @brief Compute physical flux from manifold state
    *
    * @param state manifold state
    * @param Tr local element transformation
    * @param flux physical state
    * @return maximum characteristic speed
    */
   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override final;
   real_t ComputeNormalFluxes(const Vector &stateL,
                              const Vector &stateR,
                              FaceElementTransformations &Tr,
                              Vector &normalL, Vector &normalR,
                              Vector &stateL_L, Vector &stateR_L,
                              Vector &fluxL_L, Vector &fluxR_L,
                              Vector &stateL_R, Vector &stateR_R,
                              Vector &fluxL_R, Vector &fluxR_R) const;
};


class ManifoldNumericalFlux : public RiemannSolver
{
   // attributes
private:
protected:
   const ManifoldFlux &maniflux;
   mutable Vector fluxL_L, fluxR_L, fluxL_R, fluxR_R;
   mutable Vector stateL_L, stateR_L, stateL_R, stateR_R;
   mutable Vector normalL, normalR;
public:

   // methods
private:
protected:
public:
   ManifoldNumericalFlux(const ManifoldFlux &flux):RiemannSolver(flux),
      maniflux(flux)
   {
      fluxL_L.SetSize(maniflux.num_equations);
      fluxR_L.SetSize(maniflux.num_equations);
      fluxL_R.SetSize(maniflux.num_equations);
      fluxR_R.SetSize(maniflux.num_equations);
      stateL_L.SetSize(maniflux.num_equations);
      stateR_L.SetSize(maniflux.num_equations);
      stateL_R.SetSize(maniflux.num_equations);
      stateR_R.SetSize(maniflux.num_equations);
      normalL.SetSize(maniflux.sdim);
      normalR.SetSize(maniflux.sdim);
   }
   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const final { MFEM_ABORT("Use the other Eval function") };
   virtual real_t Eval(const Vector &stateL, const Vector &stateR,
                       FaceElementTransformations &Tr,
                       const Vector &fluxLN, const Vector &fluxRN,
                       const real_t max_char_speed, Vector &hatFL, Vector &hatFR) const = 0;

};

class ManifoldRusanovFlux : public ManifoldNumericalFlux
{
   // attributes
private:
protected:
public:

   // methods
private:
protected:
public:
   ManifoldRusanovFlux(const ManifoldFlux &flux):ManifoldNumericalFlux(flux) {}
   virtual real_t Eval(const Vector &stateL, const Vector &stateR,
                       FaceElementTransformations &Tr,
                       const Vector &fluxLN, const Vector &fluxRN,
                       const real_t max_char_speed, Vector &hatFL, Vector &hatFR) const override
   {
#ifdef MFEM_THREAD_SAFE
      Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
      const real_t maxE = maniflux.ComputeNormalFluxes(stateL, stateR, Tr,
                                                       normalL, normalR,
                                                       stateL_L, stateR_L, fluxL_L, fluxR_L,
                                                       stateL_R, stateR_R, fluxL_R, fluxR_R);
      // here, std::sqrt(nor*nor) is multiplied to match the scale with fluxN
      const real_t scaledMaxE = maxE*std::sqrt(normalL*normalL);
      for (int i=0; i<maniflux.num_equations; i++)
      {
         hatFL[i] = 0.5*(scaledMaxE*(stateL_L[i] - stateR_L[i]) +
                         (fluxL_L[i] + fluxR_L[i]));
         hatFR[i] = 0.5*(scaledMaxE*(stateL_R[i] - stateR_R[i]) +
                         (fluxL_R[i] + fluxR_R[i]));
      }
      return maxE;
   }
};

class ManifoldHyperbolicFormIntegrator : public NonlinearFormIntegrator
{
   // attributes
private:
protected:
public:

   // methods
private:
protected:
public:

};

} // end of namespace mfem
