/* block ZZ estimator: 
 * generalized BlockZZEstimator that takes two blocks of solutions 
 * and compute the errors
 *
 * Author: QT
 * */

#include "mfem.hpp"
using namespace std;
using namespace mfem;

/** @brief The BlockZZEstimator class implements the Zienkiewicz-Zhu
    error estimation procedure for two blocks of the solution.

    The required BilinearFormIntegrator must implement the methods
    ComputeElementFlux() and ComputeFluxEnergy().
 */
class BlockZZEstimator : public ErrorEstimator
{
protected:
   long current_sequence;
   Vector error_estimates;
   double total_error;
   /* ratio of errors between two solutions
    * ratio should be positive but does not have to be between 0 and 1*/
   double ratio; 
   int flux_averaging;

   BilinearFormIntegrator *integ1; //< Not owned.
   GridFunction *solution1; //< Not owned.
   BilinearFormIntegrator *integ2; ///< Not owned.
   GridFunction *solution2; //< Not owned.

   FiniteElementSpace *flux_space1, *flux_space2; 
   bool own_flux_fes; //< Ownership flag for flux_space.

   // Check if the mesh of the solution was modified.
   // Mesh of two solutions should be identical
   bool MeshIsModified()
   {
      long mesh_sequence = solution1->FESpace()->GetMesh()->GetSequence();
      long mesh_tmp = solution2->FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      MFEM_ASSERT(mesh_sequence == mesh_tmp, "inconsistent meshes in solutions");
      return (mesh_sequence > current_sequence);
   }

   // Compute the element error estimates.
   void ComputeEstimates();

public:
   /** @brief Construct a new BlockZZEstimator object.
       @param integ    This BilinearFormIntegrator must implement the methods
                       ComputeElementFlux() and ComputeFluxEnergy().
       @param sol      The solution field whose error is to be estimated.
       @param flux_fes The BlockZZEstimator assumes ownership of this
                       FiniteElementSpace and will call its Update() method when
                       needed. */
   BlockZZEstimator(BilinearFormIntegrator &integ1, GridFunction &sol1,
                     BilinearFormIntegrator &integ2, GridFunction &sol2,
                    FiniteElementSpace *flux_fes1, FiniteElementSpace *flux_fes2)
      : current_sequence(-1),
        total_error(),
        ratio(1.),
        flux_averaging(0),
        integ1(&integ1),solution1(&sol1),
        integ2(&integ2),solution2(&sol2),
        flux_space1(flux_fes1),flux_space2(flux_fes2),
        own_flux_fes(true)
   { }

   /** @brief Construct a new BlockZZEstimator object.
       @param integ    This BilinearFormIntegrator must implement the methods
                       ComputeElementFlux() and ComputeFluxEnergy().
       @param sol      The solution field whose error is to be estimated.
       @param flux_fes The BlockZZEstimator does NOT assume ownership of
                       this FiniteElementSpace; will call its Update() method
                       when needed. */
   BlockZZEstimator(BilinearFormIntegrator &integ1, GridFunction &sol1,
                    BilinearFormIntegrator &integ2, GridFunction &sol2,
                    FiniteElementSpace &flux_fes1, FiniteElementSpace &flux_fes2)
      : current_sequence(-1),
        total_error(),
        ratio(1.),
        flux_averaging(0),
        integ1(&integ1),solution1(&sol1),
        integ2(&integ2),solution2(&sol2),
        flux_space1(&flux_fes1),flux_space2(&flux_fes2),
        own_flux_fes(false)
   { }

   /** @brief Set the way the flux is averaged (smoothed) across elements.

       When @a fa is zero (default), averaging is performed globally. When @a fa
       is non-zero, the flux averaging is performed locally for each mesh
       attribute, i.e. the flux is not averaged across interfaces between
       different mesh attributes. */
   void SetFluxAveraging(int fa) { flux_averaging = fa; }

   // Return the total error from the last error estimate.
   double GetTotalError() const { return total_error; }

   void SetErrorRatio(double ra) 
   { 
       MFEM_ASSERT(ra > 0.0, "error ratio should be positive!");
       ratio = ra; 
   }

   // Get a Vector with all element errors.
   virtual const Vector &GetLocalErrors()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /// Reset the error estimator.
   virtual void Reset() { current_sequence = -1; }

   /** @brief Destroy a BlockZZEstimator object. Destroys, if owned, the
       FiniteElementSpace, flux_space. */
   virtual ~BlockZZEstimator()
   {
      if (own_flux_fes) 
      { 
          delete flux_space1; 
          delete flux_space2; 
      }
   }
};

void BlockZZEstimator::ComputeEstimates()
{
   flux_space1->Update(false);
   flux_space2->Update(false);
   // In parallel, 'flux' can be a GridFunction, as long as 'flux_space' is a
   // ParFiniteElementSpace and 'solution' is a ParGridFunction.
   GridFunction flux1(flux_space1), flux2(flux_space2);

   double err_tmp;
   Vector estimates_tmp;

   total_error = ZZErrorEstimator(*integ1, *solution1, flux1, error_estimates,
                                  NULL,
                                  flux_averaging);

       err_tmp = ZZErrorEstimator(*integ2, *solution2, flux2, estimates_tmp,
                                  NULL,
                                  flux_averaging);

   //cout <<"size1="<<error_estimates.Size()<<" size2="<<estimates_tmp.Size()<<endl;
   error_estimates.Add(ratio, estimates_tmp);
   total_error+=(ratio*err_tmp);

   current_sequence = solution1->FESpace()->GetMesh()->GetSequence();
   long mesh_tmp = solution2->FESpace()->GetMesh()->GetSequence();
   MFEM_ASSERT(current_sequence == mesh_tmp, "different meshes in solutions!!");
}
