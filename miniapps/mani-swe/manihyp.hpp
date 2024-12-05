#include "mfem.hpp"

namespace mfem
{

void sphere(const Vector &x, Vector &y, const real_t r=1.0);

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
                         const Vector &state, Vector &phys_state) const;

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
                         Vector &stateL_R, Vector &stateR_R) const;

};

class ManifoldFlux : public FluxFunction
{
   // attributes
private:
   FluxFunction &org_flux;
   const ManifoldCoord &coord;
   int nrScalar;
   int nrVector;
   mutable Vector phys_state;
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
      phys_state.SetSize(nrScalar + nrVector*coord.sdim);
   }

   const ManifoldCoord &GetCoordinate() const {return coord;}

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

   real_t ComputeFluxDotN(const Vector &state, const Vector &normal,
                          ElementTransformation &Tr,
                          Vector &fluxDotN) const override final
   {
      MFEM_ABORT("Use ComputeNormalFluxes.");
   }

   real_t ComputeNormalFluxes(const Vector &stateL,
                              const Vector &stateR,
                              FaceElementTransformations &Tr,
                              Vector &normalL, Vector &normalR,
                              Vector &stateL_L, Vector &stateR_L,
                              Vector &fluxL_L, Vector &fluxR_L,
                              Vector &stateL_R, Vector &stateR_R,
                              Vector &fluxL_R, Vector &fluxR_R) const;

   int GetNumScalars() const { return nrScalar; }
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
      normalL.SetSize(maniflux.GetCoordinate().sdim);
      normalR.SetSize(maniflux.GetCoordinate().sdim);
   }
   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const final { MFEM_ABORT("Use the other Eval function") };
   virtual real_t Eval(const Vector &stateL, const Vector &stateR,
                       FaceElementTransformations &Tr,
                       Vector &hatFL, Vector &hatFR) const = 0;

   const ManifoldFlux &GetManifoldFluxFunction() const {return maniflux;}
   const ManifoldCoord &GetCoordinate() const {return maniflux.GetCoordinate();}

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
                       Vector &hatFL, Vector &hatFR) const override
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
   const ManifoldNumericalFlux &numFlux;
   const ManifoldFlux &maniFlux;
   const ManifoldCoord &coord;
   real_t max_char_speed=0.0;
   Vector state, phys_state;
   Vector stateL, stateR;
   Vector phys_stateL, phys_stateR;
   Vector phys_hatFL, phys_hatFR;
   Vector hatFL, hatFR;
   Vector shape;
   Vector shape1, shape2;
   // DenseMatrix adjJ;
   DenseMatrix dshape;
   DenseMatrix gshape, vector_gshape;
   DenseMatrix hess_shape;
   DenseMatrix Hess;
   DenseTensor HessMat;
   DenseMatrix gradJ;
   Vector x_nodes;
   DenseMatrix phys_flux;
   DenseMatrix phys_flux_scalars, phys_flux_vectors;
   const IntegrationRule *intrule;
   Array<int> hess_map;
public:

   // methods
private:
   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         const ElementTransformation &Trans);
   const IntegrationRule &GetRule(const FiniteElement &el1,
                                  const FiniteElement &el2,
                                  const FaceElementTransformations &Trans);
protected:
public:
   /**
    * @brief Integrator of (F(u), grad v) - <\hat{F}(u), [v]> with given numerical flux.
    * numerical flux both implements F(u) and numerical flux \hat{F}(u).
    *
    * @param flux Numerical flux
    * @param ir Optionally chosen integration rule
    */
   ManifoldHyperbolicFormIntegrator(const ManifoldNumericalFlux &flux,
                                    const IntegrationRule *ir=nullptr);

   // Compute (F(u), grad v)
   void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override;

   // Compute -<\hat{F}(u), [v]> with given numerical flux
   void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect) override;
   // Get maximum characteristic speed for each processor.
   // For parallel assembly, you need to use MPI_Allreduce to synchronize.
   real_t GetMaxCharSpeed() { return max_char_speed; }

   // Set max_char_speed to 0
   void ResetMaxCharSpeed() { max_char_speed=0.0;}

};

class ManifoldStateCoefficient : public VectorCoefficient
{
   // attributes
private:
   VectorCoefficient &phys_cf;
   Vector phys_state;
   DenseMatrix mani_vecs, phys_vecs;
   const int nrScalar, nrVector, dim, sdim;
protected:
public:

   // methods
private:
protected:
public:
   ManifoldStateCoefficient(VectorCoefficient &phys_cf,
                            const int nrScalar, const int nrVector, const int dim)
      :VectorCoefficient(nrScalar + nrVector*dim), phys_cf(phys_cf),
       phys_state(phys_cf.GetVDim()), nrScalar(nrScalar), nrVector(nrVector), dim(dim),
       sdim((phys_cf.GetVDim()-nrScalar)/nrVector)
   {}
   virtual void Eval(Vector &mani_state, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      phys_cf.Eval(phys_state, T, ip);
      for (int i=0; i<nrScalar; i++)
      {
         mani_state[i] = phys_state[i];
      }
      const DenseMatrix& invJ = T.InverseJacobian();
      mani_vecs.UseExternalData(mani_state.GetData() + nrScalar, dim, nrVector);
      phys_vecs.UseExternalData(phys_state.GetData() + nrScalar, sdim, nrVector);
      Mult(invJ, phys_vecs, mani_vecs);
   }

};

class ManifoldPhysVectorCoefficient : public VectorCoefficient
{
   // attributes
private:
   GridFunction &gf;
   const int vid, dim, sdim;
   Vector val;
   Vector val_view;
protected:
public:

   // methods
private:
protected:
public:
   ManifoldPhysVectorCoefficient(GridFunction &gf,
                                 const int vid, const int dim, const int sdim)
      :VectorCoefficient(sdim), gf(gf), vid(vid), dim(dim), sdim(sdim)
   {
      val.SetSize(gf.VectorDim());
   }
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      gf.GetVectorValue(T, ip, val);
      val_view.SetDataAndSize(val.GetData() + vid, dim);
      T.Jacobian().Mult(val_view, V);
   }

};


} // end of namespace mfem
