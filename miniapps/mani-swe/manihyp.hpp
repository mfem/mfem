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
   void convertElemState(ElementTransformation &el,
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
    * That is,
    *     stateL
    *
    * @param el 
    * @param nrScalar 
    * @param nrVector 
    * @param stateL 
    * @param stateR 
    * @param normalL 
    * @param normalR 
    * @param stateL_L 
    * @param stateR_L 
    * @param stateL_R 
    * @param stateR_R 
    */
   void convertFaceState(FaceElementTransformations &el,
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
protected:
public:

   // methods
private:
protected:
public:
   ManifoldFlux(FluxFunction &flux, ManifoldCoord &coord, int nrScalar)
      : FluxFunction(flux.num_equations, flux.dim), org_flux(flux),
        coord(coord), nrScalar(nrScalar)
   {
      nrVector = (org_flux.num_equations - nrScalar)/coord.sdim;
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
