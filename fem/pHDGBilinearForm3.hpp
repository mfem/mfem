// Implementation of class ParHDGBilinearForm3
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo

#ifndef MFEM_PHDG_BILINFORM
#define MFEM_PHDG_BILINFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>
#include "../linalg/hypre.hpp"
#include "pfespace.hpp"
#include "pgridfunc.hpp"
#include "HDGBilinearForm3.hpp"

namespace mfem
{

/** Class for ParHDGBilinearForm3. */
class ParHDGBilinearForm3 : public HDGBilinearForm3
{
protected:
   /// FE spaces on which the form lives.
   ParFiniteElementSpace *pfes1, *pfes2, *pfes3;

   void ParallelAssemble(const ParGridFunction *R,
                         const ParGridFunction *F,
                         const ParGridFunction sol,
                         const double memA, const double memB,
                         int skip_zeros);
public:
   /// Creates bilinear form associated with FE space *pf1, pf2, pf3.
   ParHDGBilinearForm3(ParFiniteElementSpace *pf1, ParFiniteElementSpace *pf2, ParFiniteElementSpace *pf3)
      : HDGBilinearForm3 (pf1, pf2, pf3), pfes1(pf1), pfes2(pf2), pfes3(pf3) { }

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   HypreParMatrix *ParallelAssembleSC() { return ParallelAssembleSC(mat); }
   /// Returns the matrix m assembled on the true dofs, i.e. P^t A P
   HypreParMatrix *ParallelAssembleSC(SparseMatrix *m);

   /// Returns the rhs vector assembled on the true dofs
   HypreParVector *ParallelVectorSC();

   /// Assembles the Schur complement
   void AssembleSC(const ParGridFunction *R,
                   const ParGridFunction *F,
                   Array<int> &bdr_attr_is_ess,
                   const ParGridFunction sol,
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

   /// Eliminates boundary conditions from the local 
   /// matrices and modifies the right hand side vector
   void Eliminate_BC(const Array<int> &vdofs_e1, 
                     const int ndof_u, const int ndof_q,
                     const ParGridFunction &sol, 
                     Vector *rhs_RF, Vector *rhs_L,
                     DenseMatrix *B_local, DenseMatrix *C_local, DenseMatrix *D_local);

   /// Reconstructs u and q from the facet unknowns
   void Reconstruct(const ParGridFunction *R, const ParGridFunction *F,
                    ParGridFunction *ubar,
                    ParGridFunction *q, ParGridFunction *u);

   /// Updates the spaces
   virtual void Update(FiniteElementSpace *nfes1 = NULL, FiniteElementSpace *nfes2 = NULL, FiniteElementSpace *nfes3 = NULL);

   /// Returns the FE space associated with the BilinearForm.
   ParFiniteElementSpace *ParFE1Space() const { return pfes1; }
   ParFiniteElementSpace *ParFE2Space() const { return pfes2; }
   ParFiniteElementSpace *ParFE3Space() const { return pfes3; }

   /// Destroys bilinear form.
   virtual ~ParHDGBilinearForm3() { }
};

}

#endif

#endif
