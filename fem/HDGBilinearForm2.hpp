// Implementation of class HDGBilinearForm2
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo

#ifndef MFEM_HDGBILINEARFORM2
#define MFEM_HDGBILINEARFORM2

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "fespace.hpp"
#include "gridfunc.hpp"
#include "linearform.hpp"
#include "bilininteg.hpp"
#include "staticcond.hpp"
#include "hybridization.hpp"

namespace mfem
{

/**
   Class for HDG bilinear forms

   a0(u,v)       a1(u,mu)
   a2(lambda,u)  a3(lambda,mu)

   where u,v and lambda,mu are defined on different spaces. Both spaces
   should be defined on the same mesh. The matrices are
      [A0  A1
       A2  A3]

   such that

   a0(u,v)       = V^t A0 U
   a1(u,mu)      = M^t A1 U
   a2(lambda,u)  = U^t A2 L
   a3(lambda,mu)  = M^t A3 L

   where U, V, M and Lambda are the vectors representing the function u, v, mu
   and lambda, respectively.

   The linear system is

   [ A  B ] [  U   ]   [ R ]
   [ C  D ] [ Ubar ] = [ F ]

   Eliminating U we find the global system

   S Ubar = G

   where S = - C A^{-1} B + D and G = -C A^{-1} R + F.

   Having solved this system for lambda, we can compute u from

   U = A^{-1} (R - B Ubar)

   For HDG computations the negative inverse of the A0 can be computed by esily,
   since it is block-diagonal.
*/
class HDGBilinearForm2
{
protected:
   /// FE spaces on which the form lives.
   FiniteElementSpace *fes1, *fes2;

   /// Sparse matrix to be assembled
   SparseMatrix *mat;

   /// Right hand side vector to be assembled.
   Vector *rhs_SC;

   /// Table that contains the faces for all elements
   Table *el_to_face;

   /// List that separates the interior edges from the boundary edges
   Array<int> Edge_to_Be;

   /// Vectors to store A and B, the corresponding offsets and the number
   /// of elements on which A and B will be stroed
   Array<int> A_offsets, B_offsets;
   double *A_data, *B_data;
   int elements_A, elements_B;

   /// HDG Integrators
   Array<BilinearFormIntegrator*> hdg_dbfi;
   Array<BilinearFormIntegrator*> hdg_bbfi;

   /// Dense matrices to be used for computing the integrals
   DenseMatrix elemmat1, elemmat2, elemmat3, elemmat4;

   // may be used in the construction of derived classes
   HDGBilinearForm2()
   {
      fes1 = NULL; fes2 = NULL;
      mat = NULL;
      rhs_SC = NULL;
      el_to_face = NULL;
      A_data = NULL; B_data = NULL;
      elements_A = elements_B = 0;
   }

public:
   /// Creates bilinear form associated with FE spaces *_fes1 and _fes2.
   HDGBilinearForm2(FiniteElementSpace *_fes1, FiniteElementSpace *_fes2);

   // Arrays of the HDG domain integrators
   Array<BilinearFormIntegrator*> *GetHDG_DBFI() { return &hdg_dbfi; }

   // Arrays of the HDG face integrators
   Array<BilinearFormIntegrator*> *GetHDG_BBFI() { return &hdg_bbfi; }

   /// Returns reference to a_{ij}
   virtual double &Elem(int i, int j);

   /// Returns constant reference to a_{ij}
   virtual const double &Elem(int i, int j) const;

   /// Finalizes the matrix
   virtual void Finalize(int skip_zeros = 1);

   /// Returns a reference to the sparse matrix
   const SparseMatrix *SpMatSC() const
   {
      MFEM_VERIFY(mat, "mat is NULL and can't be dereferenced");
      return mat;
   }

   SparseMatrix *SpMatSC()
   {
      MFEM_VERIFY(mat, "mat is NULL and can't be dereferenced");
      return mat;
   }

   /// Returns a reference to the right hand side vector
   const Vector *VectorSC() const
   {
      return rhs_SC;
   }

   Vector *VectorSC()
   {
      return rhs_SC;
   }

   /// Adds new HDG Integrators.
   void AddHDGDomainIntegrator(BilinearFormIntegrator *bfi);
   void AddHDGBdrIntegrator(BilinearFormIntegrator *bfi);

   /// Allocates the vectors for the part of A and B that will be stored
   void Allocate(const double memA = 0.0, const double memB = 0.0);

   /// Assembles the Schur complement
   void AssembleSC(const Vector rhs_F,
                   const double memA = 0.0, const double memB = 0.0,
                   int skip_zeros = 1);

   /// Computes domain based integrators
   void compute_domain_integrals(const int elem, DenseMatrix *A_local);

   /// Computes face based integrators
   void compute_face_integrals(const int elem,
                               const int edge,
                               const int isboundary,
                               const bool is_reconstruction,
                               DenseMatrix *A_local,
                               DenseMatrix *B_local,
                               DenseMatrix *C_local,
                               DenseMatrix *D_local);

   /// Reconstructs u from the facet unknowns
   void Reconstruct(const GridFunction *F, const GridFunction *ubar,
                    GridFunction *u);

   /// Updates the spaces
   virtual void Update(FiniteElementSpace *nfes1 = NULL,
                       FiniteElementSpace *nfes2 = NULL);

   /// Returns the FE space associated with the BilinearForm.
   FiniteElementSpace *GetFES1() { return fes1; }
   FiniteElementSpace *GetFES2() { return fes2; }

   /// Destroys bilinear form.
   virtual ~HDGBilinearForm2();
};



}

#endif
