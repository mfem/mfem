// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"

//TODO: Currently, I claim and initialize all variables/DenseMatrix/Vector
//      within the member function. By following the convention, it seems
//      that we should put them as protected/private members in the
//      integrator classes.
namespace mfem
{
/** Class for integrating the bilinear form for the mass matrix.

    a(sigma, tau) := Re(sigma, tau)_K:= int_K tr(sigma^t tau) dK,

    for tensor FE spaces, where
    		[ sigma_11 ... sigma_1d ]
    sigma = [    :     ...   :      ],
     	 	[ sigma_d1 ... sigma_dd ]

    and

            [ tau_11 ... tau_1d ]
    tau   = [    :   ...   :    ].
     	 	[ tau_d1 ... tau_dd ]

    Here, d is the dimension (dim), and sigma_ij and tau_ij are defined
    by scalar FE through the standard DG space. The resulting local
    element matrix is square, of size <tt> tdim*dof </tt>,
    where \c tdim is the vector dimension of the size dimxdim
    and \c dof is the local degrees of freedom. In particular,
    both tensor can be vectorized as

    vec(sigma):=(sigma_11, ... sigma_1d,sima_21,...,sima_2d,sigma_d1 ... sigma_dd),

    and

    tau(sigma):=(tau11, ... tau_1d,sima_21,...,tau_2d,tau_d1 ... tau_dd). As result,

    the bilinear form can also be defined as,

     a(sigma, tau) := Re(vec(sigma), vec(tau))_K.

*/
class TensorMassIntegrator : public BilinearFormIntegrator
{
protected:
	double Re; // Reynolds number


private:
    Vector shape;
    DenseMatrix partelmat;
    int ndof, dim, tdim;
    double weight;

public:
	TensorMassIntegrator(const double &_Re):Re(_Re){}
	using BilinearFormIntegrator::AssembleElementMatrix;
	virtual void AssembleElementMatrix(const FiniteElement &el,
            						   ElementTransformation &Trans,
									   DenseMatrix &elmat);
};


/** Class for integrating the bilinear form for the stiffness matrix
 *  of the diffusive term.

    a(u, tau) := (u, div tau)_K,

    for vector FE spaces, where

	u	  = [u_1, ... , u_d],

    and a tensor FE spaces

            [ tau_11 ... tau_1d ]
    tau   = [    :   ...   :    ].
     	 	[ tau_d1 ... tau_dd ]

    Here, d is the dimension (dim), and u_i and tau_ij are defined
    by scalar FE through the standard DG space. The resulting local
    element matrix is rectangle, of size <tt> tdim*dof x vdim*dof </tt>,
    where \c vdim is the vector dimemsion of the size dim,
    \c tdim is the vector dimension of the size dimxdim,
    and \c dof is the local degrees of freedom. Further,
    the divergence is defined as

    (div tau)_i := \partial_j tau_{ij}

*/
class MixedVectorDivTensorIntegrator : public BilinearFormIntegrator
{
protected:
	int trial_ndof, test_ndof, dim, vdim, tdim;
	double weight;

private:
	Vector shape;
	DenseMatrix gshape, dshape, Jadj, partelmat;

public:
	MixedVectorDivTensorIntegrator(){}

	using BilinearFormIntegrator::AssembleElementMatrix2;
    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                        const FiniteElement &test_fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &elmat);
};

//
/** Class for integrating the bilinear form for the stiffness matrix
 *  of the convective term.

    a(u, tau) := lambda( u \otimes Q, grad v )_K

    for vector FE spaces, where

	u	  = [u_1, ... , u_d],

    and,

    v   = [v_1, ... , v_d],

    and the given velocity field Q=[Q_1, ... ,Q_d].

    Here, d is the dimension (dim), and u_i and v_i are defined
    by scalar FE through the standard DG space. The resulting local
    element matrix is square, of size <tt> vdim*dof </tt>,
    where \c vdim is the vector dimemsion of the size dim,
    and \c dof is the local degrees of freedom. Further,
    the gradient is defined as

    (grad v)_ij := \partial_j v_i

*/
class VectorGradVectorIntegrator: public BilinearFormIntegrator
{
protected:
	int ndof, dim, vdim;
	VectorCoefficient *Q;
	double lambda, weight;

private:
	Vector shape, evalQ, gshape_Q;
	DenseMatrix gshape, dshape, Jadj;

public:
	VectorGradVectorIntegrator(VectorCoefficient &_Q, const double _lambda):
		Q(&_Q), lambda(_lambda) { }
	using BilinearFormIntegrator::AssembleElementMatrix;
	virtual void AssembleElementMatrix(const FiniteElement &el,
            						   ElementTransformation &Trans,
									   DenseMatrix &elmat);
};


/** Class for integrating the bilinear form for the vector average-jump matrix.

    a(u, tau) := lambda< {u}, [tau]n >_F

    for vector FE spaces, where

	u	  = [u_1, ... , u_d],

    and a tensor FE spaces

            [ tau_11 ... tau_1d ]
    tau   = [    :   ...   :    ].
     	 	[ tau_d1 ... tau_dd ]

    Here, d is the dimension (dim), and u_i and tau_ij are defined
    by scalar FE through the standard DG space. The resulting local
    element matrix is rectangle, of size <tt> tdim*dof x vdim*dof </tt>,
    where \c vdim is the vector dimemsion of the size dim,
    \c tdim is the vector dimension of the size dimxdim,
    and \c dof is the local degrees of freedom. Further, the average
    and jump operators are defined as:

    {.} := 1/2( (.)^- + (.)^+ ),

    and

    [.] := ( (.)^- - (.)^+ ).

*/
class DGVectorAvgNormalJumpIntegration:  public BilinearFormIntegrator
{
protected:
	int dim, vdim, tdim;
	int tr_ndof1, te_ndof1, tr_ndof2, te_ndof2, tr_ndofs, te_ndofs;
	double lambda;
	double weight;

private:
	Vector nor, tr_s1, tr_s2, te_s1, te_s2;
	DenseMatrix A11, A12, A21, A22;

public:
	DGVectorAvgNormalJumpIntegration(const double _lambda):lambda(_lambda){}
	void AssembleFaceMatrix(const FiniteElement &tr_fe1,
	                        const FiniteElement &tr_fe2,
	                        const FiniteElement &te_fe1,
	                        const FiniteElement &te_fe2,
	                        FaceElementTransformations &T,
	                        DenseMatrix &elmat);
};


/** Class for integrating the bilinear form for the vector jump-jump matrix.

    a(u, v) := lambda< (kappa/h)[u], [v] >_F, F \in interiror faces

    for vector FE spaces, where

	u	  = [u_1, ... , u_d],

    and,

    v   = [v_1, ... , v_d],

    Here, d is the dimension (dim), and u_i and v_i are defined
    by scalar FE through the standard DG space. The resulting local
    element matrix is square, of size <tt> vdim*dof </tt>,
    where \c vdim is the vector dimemsion of the size dim,
    and \c dof is the local degrees of freedom (on face).  Further,
    the jump operator is defined as:

    [.] := ( (.)^- - (.)^+ ).

*/
class DGVectorNormalJumpIntegrator : public BilinearFormIntegrator
{

private:
	int dim, vdim, ndof1, ndof2, ndofs;
	double kappa, lambda;
	double h, weight, stab_weight;

private:
	Vector shape1, shape2;
	DenseMatrix A11, A12, A21, A22;

public:
	DGVectorNormalJumpIntegrator(const double _kappa, const double _lambda) : kappa(_kappa), lambda(_lambda){}

	   using BilinearFormIntegrator::AssembleFaceMatrix;
	   virtual void AssembleFaceMatrix( const FiniteElement &el1,
	                                    const FiniteElement &el2,
	                                    FaceElementTransformations &Trans,
	                                    DenseMatrix &elmat);
};


/** Class for integrating the bilinear form for the upwind matrix.

    a(u, v) := lambda< (Q \cdot n) u^* \otimes n, [v] \otimes n >_F,
    F \in interiror faces \cup Neumann boundaries (or outflow boundaries)

    for vector FE spaces, where

	u	  = [u_1, ... , u_d],

    and,

    v   = [v_1, ... , v_d],

    Here, d is the dimension (dim), and u_i and v_i are defined
    by scalar FE through the standard DG space. The resulting local
    element matrix is square, of size <tt> vdim*dof </tt>,
    where \c vdim is the vector dimemsion of the size dim,
    and \c dof is the local degrees of freedom (on face).  Further,
	the upwind state is defined as:

	        | u^- if Q \cdot n >= 0,
	u^* := {  u^+ if Q \cdot n < 0,
            | u^- if F \in Neumann boundaries (outflow boundaries),

    and jump operator is defined as

            | ( (.)^- - (.)^+ ) on interior faces
    [.] := {
            | (.)^- on boundary faces
*/
class DGVectorUpwindJumpIntegrator : public BilinearFormIntegrator
{

private:
	int dim, vdim, ndof1, ndof2, ndofs;
	VectorCoefficient *Q;
	double lambda;
	double weight, inner_prod;

private:
	Vector shape1, shape2, evalQ, nor;
	DenseMatrix A11, A12, A21, A22;

public:
	DGVectorUpwindJumpIntegrator(VectorCoefficient &_Q, const double _lambda) : Q(&_Q), lambda(_lambda) { }

	using BilinearFormIntegrator::AssembleFaceMatrix;
	virtual void AssembleFaceMatrix( const FiniteElement &el1,
									const FiniteElement &el2,
									FaceElementTransformations &Trans,
									DenseMatrix &elmat);
};


/** Class for integrating the bilinear form for the scalar average-jump matrix.

    a(p, v) := < {p}, [v] \cdot n >_F,
    F \in interiror faces \cup Dirichlet boundaries

    for scalar FE space p, and vector FE space

    v   = [v_1, ... , v_d],

    Here, d is the dimension (dim), and u_i and v_i are defined
    by scalar FE through the standard DG space. The resulting local
    element matrix is rectangle, of size <tt> vdim*dof x dof </tt>,
    where \c vdim is the vector dimemsion of the size dim,
    and \c dof is the local degrees of freedom (on face).  Further,
	the average and jump operators are defined as:

    {.} := 1/2( (.)^- + (.)^+ ),

    and

            | ( (.)^- - (.)^+ ) on interior faces,
    [.] := {
            | (.)^- on boundary faces.

	This class is copied from "DG Stokes example #3215".
*/
class DGAvgNormalJumpIntegrator : public BilinearFormIntegrator
{
private:
	int dim, vdim;
	int tr_ndof1, te_ndof1, tr_ndof2, te_ndof2, tr_ndofs, te_ndofs;
	double weight;

private:
	Vector nor, tr_s1, tr_s2, te_s1, te_s2;
	DenseMatrix A11, A12, A21, A22;

public:
   DGAvgNormalJumpIntegrator(){};

   void AssembleFaceMatrix(const FiniteElement &tr_fe1,
                           const FiniteElement &tr_fe2,
                           const FiniteElement &te_fe1,
                           const FiniteElement &te_fe2,
                           FaceElementTransformations &T,
                           DenseMatrix &elmat);
};


/** Class for integrating the linear form for the Dirichlet boundary condition
 *  where test function is a tensor.

    b(tau) := < Q, tau n >_F, for F \in Dirichlet boundaries.

    the test functions are from tensor FE space where

            [ tau_11 ... tau_1d ]
    tau   = [    :   ...   :    ].
     	 	[ tau_d1 ... tau_dd ]

    Here, d is the dimension (dim), and tau_ij are defined
    by scalar FE through the standard DG space. The resulting local
    element vector is of size <tt> tdim*dof </tt>,
    where \c tdim is the vector dimension of the size dimxdim,
    and \c dof is the local degrees of freedom (on face).

*/
class TensorDGDirichletLFIntegrator : public LinearFormIntegrator
{
protected:
	int dim, vdim, tdim, ndof;
	VectorCoefficient *Q;
	double lambda, weight;

private:
	Vector nor, evalQ, shape;

public:
   TensorDGDirichletLFIntegrator(VectorCoefficient &_Q, const double _lambda)
      : Q(&_Q), lambda(_lambda) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
};


/** Class for integrating the linear form for the Dirichlet boundary condition
 *  where test function is a Vector.

    b(v) := lambda< (kappa/h) uD \otimes n, v \otimes n >_F,
    		or
    		lambda< (Q \cdot n) uD \otimes n, v \otimes n >_F,
    for F \in Dirichlet boundaries.

    the test functions are from vector FE space where

	v   = [v_1, ... , v_d].

    Here, d is the dimension (dim), and tau_ij are defined
    by scalar FE through the standard DG space. The resulting local
    element vector is of size <tt> vdim*dof </tt>,
    where \c vdim is the vector dimension of the size dim,
    and \c dof is the local degrees of freedom (on face).

*/
class VectorDGDirichletLFIntegrator : public LinearFormIntegrator
{
protected:
	int dim, vdim, ndof;
	VectorCoefficient *uD;
	VectorCoefficient *Q;
	double lambda, kappa;
	double weight, inner_prod;

private:
	Vector shape, evaluD, evalQ, nor;

public:
   VectorDGDirichletLFIntegrator(VectorCoefficient &u, const double _lambda,
                                 const double _kappa)
      : uD(&u), lambda(_lambda), kappa(_kappa), Q(NULL) { }

   VectorDGDirichletLFIntegrator(VectorCoefficient &_Q, VectorCoefficient &u, const double _lambda)
      : uD(&u), lambda(_lambda), kappa(0.0), Q(&_Q) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/** Class for integrating the linear form for the Neumann boundary condition
 *  where test function is a Vector.

    b(v) := lambda< Q, v \otimes n >_F,
    for F \in Neumann boundaries.

    the test functions are from vector FE space where

	v   = [v_1, ... , v_d].

    Here, d is the dimension (dim), and tau_ij are defined
    by scalar FE through the standard DG space. The resulting local
    element vector is of size <tt> vdim*dof </tt>,
    where \c vdim is the vector dimension of the size dim,
    and \c dof is the local degrees of freedom (on face).

    Note that in Oseen or Stokes equations the Q is defined as
    sigma - pI in LDG methods, where sigma is the gradient of velocity
    ,p is the pressure, and I is the identity matrix.



*/
class VectorDGNeumannLFIntegrator : public LinearFormIntegrator
{
protected:
	int dim, vdim, tdim, ndof;
	VectorCoefficient *Q;
	double lambda;
	double weight;

private:
	Vector shape, evalQ, nor;

public:
   VectorDGNeumannLFIntegrator(VectorCoefficient &_Q, const double _lambda)
      : Q(&_Q), lambda(_lambda) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};


/** Class for integrating the linear form for the Dirichlet boundary condition
 *  where test function is a scalar.

    b(q) := lambda< Q \cdot n, q >_F,
    for F \in Dirichlet boundaries.

    the test functions are scalar FE space q (the standard DG space).

	The resulting local element vector is of size <tt> dof </tt>,
    where \c dof is the local degrees of freedom (on face).

	This class is copied from "DG Stokes example #3215".
*/
class BoundaryNormalLFIntegrator_mod : public LinearFormIntegrator
{
protected:
   int dim, ndof;
   VectorCoefficient &Q;
   int oa, ob;
   double lambda;
   double weight;

private:
   Vector shape, nor, evalQ;

public:
   /// Constructs a boundary integrator with a given Coefficient QG
   BoundaryNormalLFIntegrator_mod(VectorCoefficient &QG, const double _lambda, int a = 1, int b = 1)
      : Q(QG), lambda(_lambda), oa(a), ob(b) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect); // added this

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// Testing Section ///
// TODO: put integrators used to perform the test in this section.

} // end of name space "mfem"

