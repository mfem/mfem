// Implementation of class HDGBilinearForm3
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo

#ifndef MFEM_HDG_BILINFORM
#define MFEM_HDG_BILINFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "fespace.hpp"
#include "gridfunc.hpp"
#include "linearform.hpp"
#include "bilininteg.hpp"

namespace mfem
{
/**
   Class for HDG bilinear forms
   
   a0(q, v)    a1(u, v)    b1(lambda, v)
   a1(w, q)    a3(u, w)    b2(lambda, w)
   b1(q, mu)   b2(u, mu)    d(mu, lambda)
   
   where q,v are defined on fes1, u,w are defined on fes2, lambda, mu are defined on fes3.
   All 3 spaces should be defined on the same mesh.
   
   Using the notations Q, V, U, W, LAMBDA and MU for the vectors representing the
   functions q, v, u, w, lambda and mu, respectively, the forms can be written as

   a0(q,v)         = V^t A0 Q,
   a1(u,v)         = V^t A1 U,
   a2(UBAR,v)   = V^t B1 LAMBDA,
   a3(u,w)         = W^t A3 U,
   a4(UBAR,w)   = W^t B2 LAMBDA,
   a5(VBAR,\lambda) = LAMBDA^t D MU,

   and the linear system as

   [ A0  A1  B1 ] [ Q ]        [ R ]
   [ A1' A3  B2 ] [ U ]      = [ F ]
   [ B1' B2' D  ] [ LAMBDA ]   [ L ]

   Using additional notations

       [ A0  A1 ]     [ B1 ]                     [ R ]      [ Q ]
   A = [ A1' A3 ] B = [ B2 ] C = [ B1' B2'] RF = [ F ] QU = [ U ]

   we can reformulate the problem:

   [ A   B ] [   QU   ] = [ RF ]
   [ C   D ] [ LAMBDA ]   [ L  ]

   Eliminating QU we find the global system

   S LAMBDA = G

   where S = - C A^{-1} B + D and G = -C A^{-1} RF + L.
   Having solved this system for lambda, we can compute u from

   QU = A^{-1} (RF - B LAMBDA)

   For HDG computations the negative of the inverse of the top left 2 x 2 block matrix
   can be computed element-wise due to the fact that all matrices are
   containing integrals the only on a single element (or element boundary).
   Using this the problem can be reduced to the facet unknowns.
*/
class HDGBilinearForm3
{
protected:
   /// FE spaces on which the form lives.
   FiniteElementSpace *fes1, *fes2, *fes3;

   /// Sparse matrix to be assembled.
   SparseMatrix *mat;

   /// Right hand side vector to be assembled.
   Vector *rhs_SC;

   /// Table that contains the faces for all elements
   Table *el_to_face;

   /// List that separates the interior edges from the boundary edges
   /// and the list of essential boundary conditions
   Array<int> Edge_to_Be, ess_dofs;

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
   HDGBilinearForm3()
   {
      fes1 = NULL; fes2 = NULL; fes3 = NULL;
      mat = NULL;
      rhs_SC = NULL;
      el_to_face = NULL;
      A_data = NULL; B_data = NULL;
      elements_A = elements_B = 0;
   }

public:
   /// Creates bilinear form associated with FE spaces *f1, f2, f3
   HDGBilinearForm3(FiniteElementSpace *f1, FiniteElementSpace *f2, FiniteElementSpace *f3);

   // Arrays of the HDG domain integrators
   Array<BilinearFormIntegrator*> *GetHDG_DBFI() { return &hdg_dbfi; }

   // Arrays of the HDG face integrators
   Array<BilinearFormIntegrator*> *GetHDG_BBFI() { return &hdg_bbfi; }

   /// Returns reference to a_{ij}.
   virtual double &Elem(int i, int j);

   /// Returns constant reference to a_{ij}.
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
   void Allocate(Array<int> &bdr_attr_is_ess, const double memA = 0.0, const double memB = 0.0);

   /// Assembles the Schur complement
   void AssembleSC(const Vector rhs_R, const Vector rhs_F,
                   Array<int> &bdr_attr_is_ess,
                   Vector &sol,
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

   /// Eliminates boundary conditions from the local 
   /// matrices and modifies the right hand side vector
   void Eliminate_BC(const Array<int> &vdofs_e1, 
                     const int ndof_u, const int ndof_q,
                     const Vector &sol, 
                     Vector *rhs_RF, Vector *rhs_L,
                     DenseMatrix *B_local, DenseMatrix *C_local, DenseMatrix *D_local);

   /// Reconstructs u and q from the facet unknowns
   void Reconstruct(const GridFunction *R, const GridFunction *F, 
                    Vector &sol, 
                    GridFunction *q, GridFunction *u);

   /// Updates the spaces
   virtual void Update(FiniteElementSpace *nfes1 = NULL, FiniteElementSpace *nfes2 = NULL, FiniteElementSpace *nfes3 = NULL);

   /// Returns the FE space associated with the BilinearForm.
   FiniteElementSpace *GetFES1() { return fes1; }
   FiniteElementSpace *GetFES2() { return fes2; }
   FiniteElementSpace *GetFES3() { return fes3; }

   /// Destroys bilinear form.
   virtual ~HDGBilinearForm3();
};



}

#endif
