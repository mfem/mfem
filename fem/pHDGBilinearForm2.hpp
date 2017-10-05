// Implementation of class ParHDGBilinearForm2
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo

#ifndef MFEM_PHDGBILINEARFORM2
#define MFEM_PHDGBILINEARFORM2

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>
#include "../linalg/hypre.hpp"
#include "pfespace.hpp"
#include "pgridfunc.hpp"
#include "HDGBilinearForm2.hpp"

namespace mfem
{

/** Class for ParHDGBilinearForm2. */
class ParHDGBilinearForm2 : public HDGBilinearForm2
{
protected:
   /// FE spaces on which the form lives.
   ParFiniteElementSpace *pfes1, *pfes2;

   void ParallelAssemble(const ParGridFunction *F,
                         const double memA = 0.0, const double memB = 0.0,
                         int skip_zeros = 1);

public:
   /// Creates bilinear form associated with FE space *f.
   ParHDGBilinearForm2(ParFiniteElementSpace *pf1, ParFiniteElementSpace *pf2) 
      : HDGBilinearForm2 (pf1, pf2), pfes1(pf1), pfes2(pf2) { }
   
   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   HypreParMatrix *ParallelAssembleSC() { return ParallelAssembleSC(mat); }
   /// Return the matrix m assembled on the true dofs, i.e. P^t A P
   HypreParMatrix *ParallelAssembleSC(SparseMatrix *m);
   
   /// Returns the rhs vector assembled on the true dofs
   HypreParVector *ParallelVectorSC();

   /// Assembles the Schur complement
   void AssembleSC(const ParGridFunction *F,
                   const double memA = 0.0, const double memB = 0.0,
                   int skip_zeros = 1);
   
   /// Computes face based integrators for shared faces
   void compute_face_integrals_shared(const int elem, 
                                      const int edge,
                                      const int sf,
                                      const bool is_reconstruction,
                                      DenseMatrix *A_local,
                                      DenseMatrix *B_local,
                                      DenseMatrix *C_local,
                                      DenseMatrix *D_local);

   /// Reconstructs u from the facet unknowns
   void Reconstruct(const ParGridFunction *F,
                    const ParGridFunction *ubar, 
                    ParGridFunction *u);

   /// Updates the spaces
   virtual void Update(FiniteElementSpace *nfes1 = NULL, FiniteElementSpace *nfes2 = NULL);

   /// Returns the FE space associated with the BilinearForm.
   ParFiniteElementSpace *ParFE1Space() const { return pfes1; }
   ParFiniteElementSpace *ParFE2Space() const { return pfes2; }

   /// Destroys bilinear form.
   virtual ~ParHDGBilinearForm2() { }
};

}

#endif

#endif
