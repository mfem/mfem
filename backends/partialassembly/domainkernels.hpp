// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// This file contains operator-based bilinear form integrators used
// with BilinearFormOperator.


#ifndef MFEM_DOMAINKERNELS
#define MFEM_DOMAINKERNELS

#include "../../general/array.hpp"
#include "tensor.hpp"
#include "dgpabilininteg.hpp"
#include "tensorialfunctions.hpp"
#include "partialassemblykernel.hpp"

namespace mfem
{

namespace pa
{

/////////////////////////////////////////////////
//                                             //
//                                             //
//               DOMAIN KERNEL                 //
//                                             //
//                                             //
/////////////////////////////////////////////////

  //////////////////////////////
 // Available Domain Kernels //
//////////////////////////////
/**
*  simple CPU implementation
*/
template <typename  Equation, PAOp OpName, typename Vector>
class DomainMult;

/**
*  Kernel using the Tensor class
*/
template <typename  Equation, PAOp OpName, typename Vector>
class TensorDomainMult;

/**
*  A class that implement the BtDB partial assembly Kernel
*/
template <typename Equation, typename Vector>
class DomainMult<Equation,PAOp::BtDB,Vector>: private Equation
{
public:
   static const int dimD = 2;
   typedef Tensor<dimD,double> DTensor;
   typedef DenseMatrix Tensor2d;

protected:
   mfem::FiniteElementSpace *fes;
   Tensor2d shape1d;
   DTensor D;

public:
   template <typename Args>
   DomainMult(mfem::FiniteElementSpace* _fes, int order, const Args& args)
   : fes(_fes), shape1d(GetNDofs1d(fes),GetNQuads1d(order)), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d);
   }

   void InitD(const int dim, const int quads, const int nb_elts){
      this->D.setSize(quads,nb_elts);
   }

   const DTensor& getD() const
   {
      return D;
   }

   template <typename Args>
   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const Tensor<2>& J, const Args& args)
   {
      double res = 0.0;
      this->evalD(res, Tr, ip, J, args);
      this->D(k,e) = res;
   }

protected:
   /**
   *  The domain Kernels for BtDB in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U) const;
   void Mult2d(const Vector &V, Vector &U) const;
   void Mult3d(const Vector &V, Vector &U) const;  

};


/**
*  A class that implement a GtDG partial assembly Kernel
*/
template <typename Equation, typename Vector>
class DomainMult<Equation,PAOp::GtDG,Vector>: private Equation
{
public:
   static const int dimD = 4;
   typedef Tensor<dimD,double> DTensor;
   typedef DenseMatrix Tensor2d;

protected:
   mfem::FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   template <typename Args>
   DomainMult(mfem::FiniteElementSpace* _fes, int order, const Args& args)
   : fes(_fes), shape1d(GetNDofs1d(fes),GetNQuads1d(order)),
   dshape1d(GetNDofs1d(fes),GetNQuads1d(order)), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   void InitD(const int dim, const int quads, const int nb_elts){
      this->D.setSize(dim,dim,quads,nb_elts);
   }

   const DTensor& getD() const
   {
      return D;
   }

   template <typename Args>
   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const Tensor<2>& J, const Args& args)
   {
      Tensor<2> res(dim,dim);
      this->evalD(res, Tr, ip, J, args);
      for (int i = 0; i < dim; ++i)
      {
         for (int j = 0; j < dim; ++j)
         {
            this->D(i,j,k,e) = res(i,j);
         }
      }
   }

protected:
   /**
   *  The domain Kernels for GtDG in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U) const;
   void Mult2d(const Vector &V, Vector &U) const;
   void Mult3d(const Vector &V, Vector &U) const;

};


/**
*  A class that implement a BtDG partial assembly Kernel
*/
template <typename Equation, typename Vector>
class DomainMult<Equation,PAOp::BtDG,Vector>: private Equation
{
public:
   static const int dimD = 3;
   typedef Tensor<dimD,double> DTensor;
   typedef DenseMatrix Tensor2d;

protected:
   mfem::FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   template <typename Args>
   DomainMult(mfem::FiniteElementSpace* _fes, int order, const Args& args)
   : fes(_fes), shape1d(GetNDofs1d(fes),GetNQuads1d(order)),
   dshape1d(GetNDofs1d(fes),GetNQuads1d(order)), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   const DTensor& getD() const
   {
      return D;
   }

   void InitD(const int dim, const int quads, const int nb_elts){
      this->D.setSize(dim,quads,nb_elts);
   }

   template <typename Args>
   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const Tensor<2>& J, const Args& args)
   {
      Tensor<1> res(dim);
      this->evalD(res, Tr, ip, J, args);
      for (int i = 0; i < dim; ++i)
      {
         this->D(i,k,e) = res(i);
      }
   }

protected:
   /**
   *  The domain Kernels for BtDG in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U) const;
   void Mult2d(const Vector &V, Vector &U) const;
   void Mult3d(const Vector &V, Vector &U) const;

};

/**
*  A class that implement a GtDB partial assembly Kernel
*/
template <typename Equation, typename Vector>
class DomainMult<Equation,PAOp::GtDB,Vector>: private Equation
{
public:
   static const int dimD = 3;
   typedef Tensor<dimD,double> DTensor;
   typedef DenseMatrix Tensor2d;

protected:
   mfem::FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   template <typename Args>
   DomainMult(mfem::FiniteElementSpace* _fes, int order, const Args& args)
   : fes(_fes), shape1d(GetNDofs1d(fes),GetNQuads1d(order)),
   dshape1d(GetNDofs1d(fes),GetNQuads1d(order)), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   void InitD(const int dim, const int quads, const int nb_elts){
      this->D.setSize(dim,quads,nb_elts);
   }

   const DTensor& getD() const
   {
      return D;
   }

   template <typename Args>
   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const Tensor<2>& J, const Args& args)
   {
      Tensor<1> res(dim);
      this->evalD(res, Tr, ip, J, args);
      for (int i = 0; i < dim; ++i)
      {
         this->D(i,k,e) = res(i);
      }
   }

protected:
   /**
   *  The domain Kernels for GtDB in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U) const;
   void Mult2d(const Vector &V, Vector &U) const;
   void Mult3d(const Vector &V, Vector &U) const;

};

/////////////////////////////
//  TensorDomainMult

/**
*  A class that implement the BtDB partial assembly Kernel
*/
template <typename Equation, typename Vector>
class TensorDomainMult<Equation,PAOp::BtDB,Vector>: private Equation
{
public:
   static const int dimD = 2;
   typedef Tensor<dimD> DTensor;
   typedef Tensor<2> Tensor2d;

protected:
   mfem::FiniteElementSpace *fes;
   Tensor2d shape1d;
   DTensor D;

public:
   template <typename Args>
   TensorDomainMult(mfem::FiniteElementSpace* _fes, int order, const Args& args)
   : fes(_fes), shape1d(GetNDofs1d(fes),GetNQuads1d(order)), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d);
   }

   void InitD(const int dim, const int quads, const int nb_elts){
      this->D.setSize(quads,nb_elts);
   }

   const DTensor& getD() const
   {
      return D;
   }

   template <typename Args>
   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const Tensor<2>& J, const Args& args)
   {
      double res = 0.0;
      this->evalD(res, Tr, ip, J, args);
      this->D(k,e) = res;
   }

protected:
   /**
   *  The domain Kernels for BtDB in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U) const;
   void Mult2d(const Vector &V, Vector &U) const;
   void Mult3d(const Vector &V, Vector &U) const;  

};


/**
*  A class that implement a GtDG partial assembly Kernel
*/
template <typename Equation, typename Vector>
class TensorDomainMult<Equation,PAOp::GtDG,Vector>: private Equation
{
public:
   static const int dimD = 4;
   typedef Tensor<dimD> DTensor;
   typedef Tensor<2> Tensor2d;

protected:
   mfem::FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   template <typename Args>
   TensorDomainMult(mfem::FiniteElementSpace* _fes, int order, const Args& args)
   : fes(_fes), shape1d(GetNDofs1d(fes),GetNQuads1d(order)),
   dshape1d(GetNDofs1d(fes),GetNQuads1d(order)), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   void InitD(const int dim, const int quads, const int nb_elts){
      this->D.setSize(dim,dim,quads,nb_elts);
   }

   const DTensor& getD() const
   {
      return D;
   }

   template <typename Args>
   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const Tensor<2>& J, const Args& args)
   {
      Tensor<2> res(dim,dim);
      this->evalD(res, Tr, ip, J, args);
      for (int i = 0; i < dim; ++i)
      {
         for (int j = 0; j < dim; ++j)
         {
            this->D(i,j,k,e) = res(i,j);
         }
      }
   }

protected:
   /**
   *  The domain Kernels for GtDG in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U) const;
   void Mult2d(const Vector &V, Vector &U) const;
   void Mult3d(const Vector &V, Vector &U) const;

};


/**
*  A class that implement a BtDG partial assembly Kernel
*/
template <typename Equation, typename Vector>
class TensorDomainMult<Equation,PAOp::BtDG,Vector>: private Equation
{
public:
   static const int dimD = 3;
   typedef Tensor<dimD> DTensor;
   typedef Tensor<2> Tensor2d;

protected:
   mfem::FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   template <typename Args>
   TensorDomainMult(mfem::FiniteElementSpace* _fes, int order, const Args& args)
   : fes(_fes), shape1d(GetNDofs1d(fes),GetNQuads1d(order)),
   dshape1d(GetNDofs1d(fes),GetNQuads1d(order)), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   const DTensor& getD() const
   {
      return D;
   }

   void InitD(const int dim, const int quads, const int nb_elts){
      this->D.setSize(dim,quads,nb_elts);
   }

   template <typename Args>
   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const Tensor<2>& J, const Args& args)
   {
      Tensor<1> res(dim);
      this->evalD(res, Tr, ip, J, args);
      for (int i = 0; i < dim; ++i)
      {
         this->D(i,k,e) = res(i);
      }
   }

protected:
   /**
   *  The domain Kernels for BtDG in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U) const;
   void Mult2d(const Vector &V, Vector &U) const;
   void Mult3d(const Vector &V, Vector &U) const;

};

/**
*  A class that implement a GtDB partial assembly Kernel
*/
template <typename Equation, typename Vector>
class TensorDomainMult<Equation,PAOp::GtDB,Vector>: private Equation
{
public:
   static const int dimD = 3;
   typedef Tensor<dimD> DTensor;
   typedef Tensor<2> Tensor2d;

protected:
   mfem::FiniteElementSpace *fes;
   Tensor2d shape1d, dshape1d;
   DTensor D;

public:
   template <typename Args>
   TensorDomainMult(mfem::FiniteElementSpace* _fes, int order, const Args& args)
   : fes(_fes), shape1d(GetNDofs1d(fes),GetNQuads1d(order)),
   dshape1d(GetNDofs1d(fes),GetNQuads1d(order)), D()
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   void InitD(const int dim, const int quads, const int nb_elts){
      this->D.setSize(dim,quads,nb_elts);
   }

   const DTensor& getD() const
   {
      return D;
   }

   template <typename Args>
   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const Tensor<2>& J, const Args& args)
   {
      Tensor<1> res(dim);
      this->evalD(res, Tr, ip, J, args);
      for (int i = 0; i < dim; ++i)
      {
         this->D(i,k,e) = res(i);
      }
   }

protected:
   /**
   *  The domain Kernels for GtDB in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U) const;
   void Mult2d(const Vector &V, Vector &U) const;
   void Mult3d(const Vector &V, Vector &U) const;

};

      //////////////////////////////////////
     ///                                ///
    ///                                ///
   /// IMPLEMENTATION OF THE KERNELS  ///
  ///                                ///
 ///                                ///
//////////////////////////////////////

template <typename Equation, typename Vector>
void TensorDomainMult<Equation,PAOp::BtDB,Vector>::Mult1d(const Vector &U, Vector &V) const
{
   const int dim     = 1;
   const int dofs1d  = shape1d.Height();
   const int dofs    = dofs1d;
   const int quads1d = shape1d.Width();
   const int quads   = quads1d;
   Tensor<dim> BT(quads1d), DBT(quads1d), BDBT(dofs1d);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<dim> T(U.GetData() + e*dofs, dofs1d);
      Tensor<dim>       R(V.GetData() + e*dofs, dofs1d);
      Tensor<dim>    De(D.getData() + e*quads, quads1d);
      contract(shape1d,T,BT);
      cWiseMult(De,BT,DBT);
      contractT(shape1d,DBT,BDBT);
      R+=BDBT;
   }
}

template <typename Equation, typename Vector>
void TensorDomainMult<Equation,PAOp::BtDB,Vector>::Mult2d(const Vector &U, Vector &V) const
{
   const int dim     = 2;
   const int dofs1d  = shape1d.Height();
   const int dofs    = dofs1d * dofs1d;
   const int quads1d = shape1d.Width();
   const int quads   = quads1d * quads1d;
   Tensor<dim> BT(dofs1d,quads1d), BBT(quads1d,quads1d),
               DBT(quads1d,quads1d),BDBT(quads1d,dofs1d),
               BBDBT(dofs1d,dofs1d);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<dim> T(U.GetData() + e*dofs, dofs1d, dofs1d);
      Tensor<dim>       R(V.GetData() + e*dofs, dofs1d, dofs1d);
      Tensor<dim>    De(D.getData() + e*quads, quads1d, quads1d);
      contract(shape1d,T,BT);
      contract(shape1d,BT,BBT);
      cWiseMult(De,BBT,DBT);
      contractT(shape1d,DBT,BDBT);
      contractT(shape1d,BDBT,BBDBT);
      R+=BBDBT;
   }
}

template <typename Equation, typename Vector>
void TensorDomainMult<Equation,PAOp::BtDB,Vector>::Mult3d(const Vector &U, Vector &V) const
{
   const int dim     = 3;
   const int dofs1d  = shape1d.Height();
   const int dofs    = dofs1d * dofs1d * dofs1d;
   const int quads1d = shape1d.Width();
   const int quads   = quads1d * quads1d * quads1d;
   Tensor<dim> BT(dofs1d,dofs1d,quads1d), BBT(dofs1d,quads1d,quads1d), BBBT(quads1d,quads1d,quads1d),
               DBT(quads1d,quads1d,quads1d), BDBT(quads1d,quads1d,dofs1d), BBDBT(quads1d,dofs1d,dofs1d),
               BBBDBT(dofs1d,dofs1d,dofs1d);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<dim> T(U.GetData() + e*dofs, dofs1d, dofs1d, dofs1d);
      Tensor<dim>       R(V.GetData() + e*dofs, dofs1d, dofs1d, dofs1d);
      Tensor<dim>    De(D.getData() + e*quads, quads1d, quads1d, quads1d);
      contract(shape1d,T,BT);
      contract(shape1d,BT,BBT);
      contract(shape1d,BBT,BBBT);
      cWiseMult(De,BBBT,DBT);
      contractT(shape1d,DBT,BDBT);
      contractT(shape1d,BDBT,BBDBT);
      contractT(shape1d,BBDBT,BBBDBT);
      R+=BBBDBT;
   }
}

template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::BtDB,Vector>::Mult1d(const Vector &V, Vector &U) const
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   mfem::Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const mfem::Vector Vmat = V.GetVectorView(offset, dofs1d);
      mfem::Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = dshape_j1_k1 * V_i1
      shape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k) { data_q[k] *= D(k,e); }

      // Q_k1 = dshape_j1_k1 * Q_k1
      shape1d.AddMult(Q, Umat);
   }
}

// template <typename Equation, typename Vector>
// void DomainMult<Equation,PAOp::BtDB,Vector>::Mult1d(const Vector &V, Vector &U) const
// {
   
// }

template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::BtDB,Vector>::Mult2d(const Vector &V, Vector &U) const
{
   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d*quads1d;

   DenseMatrix QQ(quads1d, quads1d);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat = V.GetMatrixView(offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1 -- contract in x direction
      // QQ_0_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, shape1d, QQ);

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ.GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k) { data_qq[k] *= D(k,e); }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ, DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }
}

// template <typename Equation, typename Vector>
// void DomainMult<Equation,PAOp::BtDB,Vector>::Mult2d(const Vector &U, Vector &V) const
// {
//    const int dofs1d = shape1d.Height();
//    const int dofs   = dofs1d * dofs1d;
//    const int quads1d = shape1d.Width();
//    for (int e = 0; e < fes->GetNE(); e++)
//    {
//       const Tensor<2> Ut(U.GetData() + e*dofs, dofs1d, dofs1d);
//       Tensor<2> Vt(V.GetData() + e*dofs, dofs1d, dofs1d);
//       Tensor<2> tmp1(dofs1d,quads1d), tmp2(quads1d,quads1d), tmp3(quads1d,dofs1d);
//       // Ap = A * p
//       for (int k1 = 0; k1 < quads1d; ++k1)
//       {
//          for (int i2 = 0; i2 < dofs1d; ++i2)
//          {
//             tmp1(i2,k1) = 0.0;
//             for (int i1 = 0; i1 < dofs1d/4*4; i1+=4)
//             {
//                tmp1(i2,k1) += shape1d(i1,k1) * Ut(i1,i2) + shape1d(i1+1,k1) * Ut(i1+1,i2) + shape1d(i1+2,k1) * Ut(i1+2,i2) + shape1d(i1+3,k1) * Ut(i1+3,i2);
//             }
//             for (int i1 = dofs1d/4*4; i1 < dofs1d; ++i1)
//             {
//                tmp1(i2,k1) += shape1d(i1,k1) * Ut(i1,i2);
//             }
//          }
//       }
//       for (int k2 = 0; k2 < quads1d; ++k2)
//       {
//          for (int k1 = 0; k1 < quads1d; ++k1)
//          {
//             tmp2(k1,k2) = 0.0;
//             for (int i2 = 0; i2 < dofs1d/4*4; i2+=4)
//             {
//                tmp2(k1,k2) += shape1d(i2,k2) * tmp1(i2,k1) + shape1d(i2+1,k2) * tmp1(i2+1,k1) + shape1d(i2+2,k2) * tmp1(i2+2,k1) + shape1d(i2+3,k2) * tmp1(i2+3,k1);
//             }
//             for (int i2 = dofs1d/4*4; i2 < dofs1d; ++i2)
//             {
//                tmp2(k1,k2) += shape1d(i2,k2) * tmp1(i2,k1);
//             }
//          }
//       }
//       for (int k2 = 0, k = 0; k2 < quads1d; ++k2)
//       {
//          for (int k1 = 0; k1 < quads1d; ++k1, ++k)
//          {
//             tmp2(k1,k2) = D(k,e) * tmp2(k1,k2);
//          }
//       }
//       for (int j1 = 0; j1 < dofs1d; ++j1)
//       {
//          for (int k2 = 0; k2 < quads1d; ++k2)
//          {
//             tmp3(k2,j1) = 0.0;
//             for (int k1 = 0; k1 < quads1d/4*4; k1+=4)
//             {
//                tmp3(k2,j1) += shape1d(j1,k1) * tmp2(k1,k2) + shape1d(j1,k1+1) * tmp2(k1+1,k2) + shape1d(j1,k1+2) * tmp2(k1+2,k2) + shape1d(j1,k1+3) * tmp2(k1+3,k2);
//             }
//             for (int k1 = quads1d/4*4; k1 < quads1d; ++k1)
//             {
//                tmp3(k2,j1) += shape1d(j1,k1) * tmp2(k1,k2);
//             }
//          }
//       }
//       for (int j2 = 0; j2 < dofs1d; ++j2)
//       {
//          for (int j1 = 0; j1 < dofs1d; ++j1)
//          {
//             Vt(j1,j2) = 0.0;
//             for (int k2 = 0; k2 < quads1d/4*4; k2+=4)
//             {
//                Vt(j1,j2) += shape1d(j2,k2) * tmp3(k2,j1) + shape1d(j2,k2+1) * tmp3(k2+1,j1) + shape1d(j2,k2+2) * tmp3(k2+2,j1) + shape1d(j2,k2+3) * tmp3(k2+3,j1);
//             }
//             for (int k2 = quads1d/4*4; k2 < quads1d; ++k2)
//             {
//                Vt(j1,j2) += shape1d(j2,k2) * tmp3(k2,j1);
//             }
//          }
//       }
//    }
// }

template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::BtDB,Vector>::Mult3d(const Vector &V, Vector &U) const
{
   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   mfem::Vector Q(quads1d);
   DenseMatrix QQ(quads1d, quads1d);
   DenseTensor QQQ(quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat = V.GetTensorView(offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

      // QQQ_k1_k2_k3 = shape_j1_k1 * shape_j2_k2  * shape_j3_k3  * Vmat_j1_j2_j3
      QQQ = 0.;
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2) += Q(k1) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ(k1, k2, k3) += QQ(k1, k2) * shape1d(j3, k3);
               }
      }

      // QQQ_k1_k2_k3 = Dmat_k1_k2_k3 * QQQ_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      double *data_qqq = QQQ.GetData(0);
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k) { data_qqq[k] *= D(k,e); }

      // Apply transpose of the first operator that takes V -> QQQ -- QQQ -> U
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         QQ = 0.;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            Q = 0.;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1) += QQQ(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2) += Q(i1) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) += shape1d(i3, k3) * QQ(i1, i2);
               }
      }

      // increment offset
      offset += dofs;
   }  
}


template <typename Equation, typename Vector>
void TensorDomainMult<Equation,PAOp::GtDG,Vector>::Mult1d(const Vector &U, Vector &V) const
{
   const int dim     = 1;
   const int dofs1d  = shape1d.Height();
   const int dofs    = dofs1d;
   const int quads1d = shape1d.Width();
   const int quads   = quads1d;
   Tensor<dim> GT(quads1d), DGT(quads1d), GDGT(dofs1d);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<dim> T(U.GetData() + e*dofs, dofs1d);
      Tensor<dim>       R(V.GetData() + e*dofs, dofs1d);
      Tensor<dim>    De(D.getData() + e*dim*dim*quads, quads1d);
      contract(dshape1d,T,GT);
      cWiseMult(De,GT,DGT);
      contractT(dshape1d,DGT,GDGT);
      R+=GDGT;
   }
}

template <typename Equation, typename Vector>
void TensorDomainMult<Equation,PAOp::GtDG,Vector>::Mult2d(const Vector &U, Vector &V) const
{
   const int dim     = 2;
   const int dofs1d  = shape1d.Height();
   const int dofs    = dofs1d * dofs1d;
   const int quads1d = shape1d.Width();
   const int quads   = quads1d * quads1d;
   Tensor<dim> BT(dofs1d,quads1d), GT(dofs1d,quads1d),
               BGT(quads1d,quads1d), GBT(quads1d,quads1d),
               D0GT(quads1d,quads1d), D1GT(quads1d,quads1d),
               BDGT(quads1d,dofs1d), GDGT(quads1d,dofs1d),
               GBDGT(dofs1d,dofs1d), BGDGT(dofs1d,dofs1d);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<dim> T(U.GetData() + e*dofs, dofs1d, dofs1d);
      Tensor<dim>       R(V.GetData() + e*dofs, dofs1d, dofs1d);
      Tensor<dim+2>    De(D.getData() + e*dim*dim*quads, dim, dim, quads1d, quads1d);
      contract(dshape1d,T,GT);
      contract(shape1d,GT,BGT);
      contract(shape1d,T,BT);
      contract(dshape1d,BT,GBT);
      cWiseMult(De,BGT,GBT,D0GT,D1GT);
      contractT(dshape1d,D0GT,GDGT);
      contractT(shape1d,GDGT,BGDGT);
      R+=BGDGT;
      contractT(shape1d,D1GT,BDGT);
      contractT(dshape1d,BDGT,GBDGT);
      R+=GBDGT;
   }
}

template <typename Equation, typename Vector>
void TensorDomainMult<Equation,PAOp::GtDG,Vector>::Mult3d(const Vector &U, Vector &V) const
{
   const int dim     = 3;
   const int dofs1d  = shape1d.Height();
   const int dofs    = dofs1d * dofs1d * dofs1d;
   const int quads1d = shape1d.Width();
   const int quads   = quads1d * quads1d * quads1d;
   Tensor<dim> BT(dofs1d,dofs1d,quads1d), GT(dofs1d,dofs1d,quads1d),
               BGT(dofs1d,quads1d,quads1d), GBT(dofs1d,quads1d,quads1d), BBT(dofs1d,quads1d,quads1d),
               BBGT(quads1d,quads1d,quads1d), BGBT(quads1d,quads1d,quads1d), GBBT(quads1d,quads1d,quads1d),
               D0GT(quads1d,quads1d,quads1d), D1GT(quads1d,quads1d,quads1d), D2GT(quads1d,quads1d,quads1d),
               BD1GT(quads1d,quads1d,dofs1d), BD2GT(quads1d,quads1d,dofs1d), GDGT(quads1d,quads1d,dofs1d),
               GBDGT(quads1d,dofs1d,dofs1d), BGDGT(quads1d,dofs1d,dofs1d), BBDGT(quads1d,dofs1d,dofs1d),
               BGBDGT(dofs1d,dofs1d,dofs1d), BBGDGT(dofs1d,dofs1d,dofs1d), GBBDGT(dofs1d,dofs1d,dofs1d);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<dim> T(U.GetData() + e*dofs, dofs1d, dofs1d, dofs1d);
      Tensor<dim>       R(V.GetData() + e*dofs, dofs1d, dofs1d, dofs1d);
      Tensor<dim+2>    De(D.getData() + e*dim*dim*quads, dim, dim, quads1d, quads1d, quads1d);
      contract(shape1d,T,BT);
      contract(dshape1d,T,GT);
      contract(shape1d,GT,BGT);
      contract(dshape1d,BT,GBT);
      contract(shape1d,BT,BBT);
      contract(shape1d,BGT,BBGT);
      contract(shape1d,GBT,BGBT);
      contract(dshape1d,BBT,GBBT);
      cWiseMult(De,BBGT,BGBT,GBBT,D0GT,D1GT,D2GT);
      contractT(dshape1d,D0GT,GDGT);
      contractT(shape1d,GDGT,BGDGT);
      contractT(shape1d,BGDGT,BBGDGT);
      R+=BBGDGT;
      contractT(shape1d,D1GT,BD1GT);
      contractT(dshape1d,BD1GT,GBDGT);
      contractT(shape1d,GBDGT,BGBDGT);
      R+=BGBDGT;
      contractT(shape1d,D2GT,BD2GT);
      contractT(shape1d,BD2GT,BBDGT);
      contractT(dshape1d,BBDGT,GBBDGT);
      R+=GBBDGT;
   }
}

template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::GtDG,Vector>::Mult1d(const Vector &V, Vector &U) const
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   mfem::Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const mfem::Vector Vmat = V.GetVectorView(offset, dofs1d);
      mfem::Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = dshape_j1_k1 * V_i1
      dshape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         data_q[k] *= D(0,0,k,e);
      }

      // Q_k1 = dshape_j1_k1 * Q_k1
      dshape1d.AddMult(Q, Umat);
   }   
}

template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::GtDG,Vector>::Mult2d(const Vector &V, Vector &U) const
{
   const int dim = 2;
   const int terms = dim*dim;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d;

   DenseTensor QQ(quads1d, quads1d, dim);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat = V.GetMatrixView(offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      // DQ_j2_k1   = E_j1_j2  * dshape_j1_k1 -- contract in x direction
      // QQ_0_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, dshape1d, DQ);
      MultAtB(DQ, shape1d, QQ(0));

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1  -- contract in x direction
      // QQ_1_k1_k2 = DQ_j2_k1 * dshape_j2_k2 -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, dshape1d, QQ(1));

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ(0).GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         // int ind00[] = {0,0,k,e};
         const double D00 = D(0,0,k,e);
         // int ind10[] = {1,0,k,e};
         const double D10 = D(1,0,k,e);
         // int ind01[] = {0,1,k,e};
         const double D01 = D(0,1,k,e);
         // int ind11[] = {1,1,k,e};
         const double D11 = D(1,1,k,e);

         const double q0 = data_qq[0*quads + k];
         const double q1 = data_qq[1*quads + k];

         data_qq[0*quads + k] = D00 * q0 + D01 * q1;
         data_qq[1*quads + k] = D10 * q0 + D11 * q1;
      }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ(0), DQ);
      AddMultABt(dshape1d, DQ, Umat);

      // DQ_i2_k1   = dshape_i2_k2 * QQ_1_k1_k2
      // U_i1_i2   += shape_i1_k1  * DQ_i2_k1
      MultABt(dshape1d, QQ(1), DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }   
}


template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::GtDG,Vector>::Mult3d(const Vector &V, Vector &U) const
{
   const int dim = 3;
   const int terms = dim*dim;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   DenseMatrix Q(quads1d, dim);
   DenseTensor QQ(quads1d, quads1d, dim);

   mfem::Array<double> QQQmem(quads1d * quads1d * quads1d * dim);
   double *data_qqq = QQQmem.GetData();
   DenseTensor QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat = V.GetTensorView(offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

      // QQQ_0_k1_k2_k3 = dshape_j1_k1 * shape_j2_k2  * shape_j3_k3  * Vmat_j1_j2_j3
      // QQQ_1_k1_k2_k3 = shape_j1_k1  * dshape_j2_k2 * shape_j3_k3  * Vmat_j1_j2_j3
      // QQQ_2_k1_k2_k3 = shape_j1_k1  * shape_j2_k2  * dshape_j3_k3 * Vmat_j1_j2_j3
      QQQ0 = 0.; QQQ1 = 0.; QQQ2 = 0.;
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1, 0) += Vmat(j1, j2, j3) * dshape1d(j1, k1);
                  Q(k1, 1) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2, 0) += Q(k1, 0) * shape1d(j2, k2);
                  QQ(k1, k2, 1) += Q(k1, 1) * dshape1d(j2, k2);
                  QQ(k1, k2, 2) += Q(k1, 1) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * shape1d(j3, k3);
                  QQQ1(k1, k2, k3) += QQ(k1, k2, 1) * shape1d(j3, k3);
                  QQQ2(k1, k2, k3) += QQ(k1, k2, 2) * dshape1d(j3, k3);
               }
      }

      // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         // int ind00[] = {0,0,k,e};
         const double D00 = D(0,0,k,e);
         // int ind10[] = {1,0,k,e};
         const double D10 = D(1,0,k,e);
         // int ind20[] = {2,0,k,e};
         const double D20 = D(2,0,k,e);
         // int ind01[] = {0,1,k,e};
         const double D01 = D(0,1,k,e);
         // int ind11[] = {1,1,k,e};
         const double D11 = D(1,1,k,e);
         // int ind21[] = {2,1,k,e};
         const double D21 = D(2,1,k,e);
         // int ind02[] = {0,2,k,e};
         const double D02 = D(0,2,k,e);
         // int ind12[] = {1,2,k,e};
         const double D12 = D(1,2,k,e);
         // int ind22[] = {2,2,k,e};
         const double D22 = D(2,2,k,e);

         const double q0 = data_qqq[0*quads + k];
         const double q1 = data_qqq[1*quads + k];
         const double q2 = data_qqq[2*quads + k];

         data_qqq[0*quads + k] = D00 * q0 + D01 * q1 + D02 * q2;
         data_qqq[1*quads + k] = D10 * q0 + D11 * q1 + D12 * q2;
         data_qqq[2*quads + k] = D20 * q0 + D21 * q1 + D22 * q2;
      }

      // Apply transpose of the first operator that takes V -> QQQd -- QQQd -> U
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         QQ = 0.;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            Q = 0.;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1, 0) += QQQ0(k1, k2, k3) * dshape1d(i1, k1);
                  Q(i1, 1) += QQQ1(k1, k2, k3) * shape1d(i1, k1);
                  Q(i1, 2) += QQQ2(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2, 0) += Q(i1, 0) * shape1d(i2, k2);
                  QQ(i1, i2, 1) += Q(i1, 1) * dshape1d(i2, k2);
                  QQ(i1, i2, 2) += Q(i1, 2) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) +=
                     QQ(i1, i2, 0) * shape1d(i3, k3) +
                     QQ(i1, i2, 1) * shape1d(i3, k3) +
                     QQ(i1, i2, 2) * dshape1d(i3, k3);
               }
      }

      // increment offset
      offset += dofs;
   }
}

template <typename Equation, typename Vector>
void TensorDomainMult<Equation,PAOp::BtDG,Vector>::Mult1d(const Vector &U, Vector &V) const
{
   const int dim     = 1;
   const int dofs1d  = shape1d.Height();
   const int dofs    = dofs1d;
   const int quads1d = shape1d.Width();
   const int quads   = quads1d;
   Tensor<dim> BT(quads1d), GT(quads1d), DGT(quads1d), BDGT(dofs1d);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<dim> T(U.GetData() + e*dofs, dofs1d);
      Tensor<dim> R(V.GetData() + e*dofs, dofs1d);
      Tensor<dim> De(D.getData() + e*quads, quads1d);
      contract(dshape1d,T,GT);
      cWiseMult(De,GT,DGT);
      contractT(shape1d,DGT,BDGT);
      R+=BDGT;
   }
}

template <typename Equation, typename Vector>
void TensorDomainMult<Equation,PAOp::BtDG,Vector>::Mult2d(const Vector &U, Vector &V) const
{
   const int dim     = 2;
   const int dofs1d  = shape1d.Height();
   const int dofs    = dofs1d * dofs1d;
   const int quads1d = shape1d.Width();
   const int quads   = quads1d * quads1d;
   Tensor<dim> BT(dofs1d,quads1d), GT(dofs1d,quads1d),
            BGT(quads1d,quads1d), GBT(quads1d,quads1d),
            DGT(quads1d,quads1d),
            BDGT(quads1d,dofs1d), BBDGT(dofs1d,dofs1d);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<dim> T(U.GetData() + e*dofs, dofs1d, dofs1d);
      Tensor<dim>       R(V.GetData() + e*dofs, dofs1d, dofs1d);
      Tensor<dim+1>    De(D.getData() + e*dim*quads, dim, quads1d, quads1d);
      contract(shape1d,T,BT);
      contract(dshape1d,T,GT);
      contract(shape1d,GT,BGT);
      contract(dshape1d,BT,GBT);
      cWiseMult(De,BGT,GBT,DGT);
      contractT(shape1d,DGT,BDGT);
      contractT(shape1d,BDGT,BBDGT);
      R+=BBDGT;
   }
}

template <typename Equation, typename Vector>
void TensorDomainMult<Equation,PAOp::BtDG,Vector>::Mult3d(const Vector &U, Vector &V) const
{
   const int dim     = 3;
   const int dofs1d  = shape1d.Height();
   const int dofs    = dofs1d * dofs1d * dofs1d;
   const int quads1d = shape1d.Width();
   const int quads   = quads1d * quads1d * quads1d;
   Tensor<dim> BT(dofs1d,dofs1d,quads1d), GT(dofs1d,dofs1d,quads1d),
            BGT(dofs1d,quads1d,quads1d), BBT(dofs1d,quads1d,quads1d), GBT(dofs1d,quads1d,quads1d),
            BBGT(quads1d,quads1d,quads1d), GBBT(quads1d,quads1d,quads1d), BGBT(quads1d,quads1d,quads1d),
            DGT(quads1d,quads1d,quads1d),
            BDGT(quads1d,quads1d,dofs1d), BBDGT(quads1d,dofs1d,dofs1d), BBBDGT(dofs1d,dofs1d,dofs1d);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<dim> T(U.GetData() + e*dofs, dofs1d, dofs1d, dofs1d);
      Tensor<dim> R(V.GetData() + e*dofs, dofs1d, dofs1d, dofs1d);
      Tensor<dim+1> De(D.getData() + e*dim*quads,dim,quads1d,quads1d,quads1d);
      contract(shape1d,T,BT);
      contract(dshape1d,T,GT);
      contract(shape1d,GT,BGT);
      contract(shape1d,BGT,BBGT);
      contract(shape1d,BT,BBT);
      contract(dshape1d,BBT,GBBT);
      contract(dshape1d,BT,GBT);
      contract(shape1d,GBT,BGBT);
      cWiseMult(De,BBGT,BGBT,GBBT,DGT);
      contractT(shape1d,DGT,BDGT);
      contractT(shape1d,BDGT,BBDGT);
      contractT(shape1d,BBDGT,BBBDGT);
      R+=BBBDGT;
   }
}

template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::BtDG,Vector>::Mult1d(const Vector &V, Vector &U) const
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   mfem::Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const mfem::Vector Vmat = V.GetVectorView(offset, dofs1d);
      mfem::Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = dshape_j1_k1 * V_i1
      dshape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         data_q[k] *= D(0,k,e);
      }

      // Q_k1 = dshape_j1_k1 * Q_k1
      shape1d.AddMult(Q, Umat);
   }   
}

template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::BtDG,Vector>::Mult2d(const Vector &V, Vector &U) const
{
   const int dim = 2;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d;

   DenseTensor QQ(quads1d, quads1d, dim);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat = V.GetMatrixView(offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      // DQ_j2_k1   = E_j1_j2  * dshape_j1_k1 -- contract in x direction
      // QQ_0_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, dshape1d, DQ);
      MultAtB(DQ, shape1d, QQ(0));

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1  -- contract in x direction
      // QQ_1_k1_k2 = DQ_j2_k1 * dshape_j2_k2 -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, dshape1d, QQ(1));

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ(0).GetData();
      for (int k = 0; k < quads; ++k)
      {
         // int ind0[] = {0,k,e};
         const double D0 = D(0,k,e);
         // int ind1[] = {1,k,e};
         const double D1 = D(1,k,e);

         const double q0 = data_qq[0*quads + k];
         const double q1 = data_qq[1*quads + k];

         data_qq[0*quads + k] = D0 * q0 + D1 * q1;
      }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ(0), DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }
}


template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::BtDG,Vector>::Mult3d(const Vector &V, Vector &U) const
{
   const int dim = 3;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   DenseMatrix Q(quads1d, dim);
   DenseTensor QQ(quads1d, quads1d, dim);

   mfem::Array<double> QQQmem(quads1d * quads1d * quads1d * dim);
   double *data_qqq = QQQmem.GetData();
   DenseTensor QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat = V.GetTensorView(offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

      // QQQ_0_k1_k2_k3 = dshape_j1_k1 * shape_j2_k2  * shape_j3_k3  * Vmat_j1_j2_j3
      // QQQ_1_k1_k2_k3 = shape_j1_k1  * dshape_j2_k2 * shape_j3_k3  * Vmat_j1_j2_j3
      // QQQ_2_k1_k2_k3 = shape_j1_k1  * shape_j2_k2  * dshape_j3_k3 * Vmat_j1_j2_j3
      QQQ0 = 0.; QQQ1 = 0.; QQQ2 = 0.;
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1, 0) += Vmat(j1, j2, j3) * dshape1d(j1, k1);
                  Q(k1, 1) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2, 0) += Q(k1, 0) * shape1d(j2, k2);
                  QQ(k1, k2, 1) += Q(k1, 1) * dshape1d(j2, k2);
                  QQ(k1, k2, 2) += Q(k1, 1) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * shape1d(j3, k3);
                  QQQ1(k1, k2, k3) += QQ(k1, k2, 1) * shape1d(j3, k3);
                  QQQ2(k1, k2, k3) += QQ(k1, k2, 2) * dshape1d(j3, k3);
               }
      }

      // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         // int ind0[] = {0,k,e};
         const double D0 = D(0,k,e);
         // int ind1[] = {1,k,e};
         const double D1 = D(1,k,e);
         // int ind2[] = {2,k,e};
         const double D2 = D(2,k,e);

         const double q0 = data_qqq[0*quads + k];
         const double q1 = data_qqq[1*quads + k];
         const double q2 = data_qqq[2*quads + k];

         data_qqq[0*quads + k] = D0 * q0 + D1 * q1 + D2 * q2;
      }

      // Apply transpose of the first operator that takes V -> QQQd -- QQQd -> U
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         QQ = 0.;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            Q = 0.;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1, 0) += QQQ0(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2, 0) += Q(i1, 0) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) += QQ(i1, i2, 0) * shape1d(i3, k3);
               }
      }

      // increment offset
      offset += dofs;
   }
}

//TODO GtDB for TensorDomainMult

template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::GtDB,Vector>::Mult1d(const Vector &V, Vector &U) const
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;

   mfem::Vector Q(quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const mfem::Vector Vmat(V.GetData() + offset, dofs1d);
      Vector Umat(U.GetData() + offset, dofs1d);

      // Q_k1 = shape_j1_k1 * V_i1
      shape1d.MultTranspose(Vmat, Q);

      double *data_q = Q.GetData();
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         data_q[k] *= D(0,k,e);
      }

      // Q_k1 = dshape_j1_k1 * Q_k1
      dshape1d.AddMult(Q, Umat);
   }    
}

template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::GtDB,Vector>::Mult2d(const Vector &V, Vector &U) const
{
   const int dim = 2;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d;

   DenseTensor QQ(quads1d, quads1d, dim);
   DenseMatrix DQ(dofs1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseMatrix Vmat(V.GetData() + offset, dofs1d, dofs1d);
      DenseMatrix Umat(U.GetData() + offset, dofs1d, dofs1d);

      //TODO One QQ should be enough
      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1 -- contract in x direction
      // QQ_k1_k2 = DQ_j2_k1 * shape_j2_k2  -- contract in y direction
      MultAtB(Vmat, shape1d, DQ);
      MultAtB(DQ, shape1d, QQ(0));

      // DQ_j2_k1   = E_j1_j2  * shape_j1_k1  -- contract in x direction
      // QQ_1_k1_k2 = DQ_j2_k1 * dshape_j2_k2 -- contract in y direction
      //Can be optimized since this is the same data
      //MultAtB(Vmat, shape1d, DQ);
      //MultAtB(DQ, shape1d, QQ(1));

      // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
      // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
      double *data_qq = QQ(0).GetData();
      //const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         // const double D0 = data_d[terms*k + 0];
         // const double D1 = data_d[terms*k + 1];
         // int ind0[] = {0,k,e};
         const double D0 = D(0,k,e);
         // int ind1[] = {1,k,e};
         const double D1 = D(1,k,e);
         const double q = data_qq[0*quads + k];

         data_qq[0*quads + k] = D0 * q;
         data_qq[1*quads + k] = D1 * q;
      }

      // DQ_i2_k1   = shape_i2_k2  * QQ_0_k1_k2
      // U_i1_i2   += dshape_i1_k1 * DQ_i2_k1
      MultABt(shape1d, QQ(0), DQ);
      AddMultABt(dshape1d, DQ, Umat);

      // DQ_i2_k1   = dshape_i2_k2 * QQ_1_k1_k2
      // U_i1_i2   += shape_i1_k1  * DQ_i2_k1
      MultABt(dshape1d, QQ(1), DQ);
      AddMultABt(shape1d, DQ, Umat);

      // increment offset
      offset += dofs;
   }    
}

template <typename Equation, typename Vector>
void DomainMult<Equation,PAOp::GtDB,Vector>::Mult3d(const Vector &V, Vector &U) const
{
   const int dim = 3;
   const int terms = dim*(dim+1)/2;

   const FiniteElement *fe = fes->GetFE(0);
   const int dofs   = fe->GetDof();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads  = quads1d * quads1d * quads1d;

   DenseMatrix Q(quads1d, dim);
   DenseTensor QQ(quads1d, quads1d, dim);

   mfem::Array<double> QQQmem(quads1d * quads1d * quads1d * dim);
   double *data_qqq = QQQmem.GetData();
   DenseTensor QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
   DenseTensor QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);

   int offset = 0;
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const DenseTensor Vmat(V.GetData() + offset, dofs1d, dofs1d, dofs1d);
      DenseTensor Umat(U.GetData() + offset, dofs1d, dofs1d, dofs1d);

      // TODO One QQQ should be enough
      // QQQ_0_k1_k2_k3 = shape_j1_k1 * shape_j2_k2 * shape_j3_k3 * Vmat_j1_j2_j3
      // QQQ_1_k1_k2_k3 = shape_j1_k1 * shape_j2_k2 * shape_j3_k3 * Vmat_j1_j2_j3
      // QQQ_2_k1_k2_k3 = shape_j1_k1 * shape_j2_k2 * shape_j3_k3 * Vmat_j1_j2_j3
      QQQ0 = 0.; QQQ1 = 0.; QQQ2 = 0.;
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         QQ = 0.;
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1, 0) += Vmat(j1, j2, j3) * shape1d(j1, k1);
               }
            }
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQ(k1, k2, 0) += Q(k1, 0) * shape1d(j2, k2);
               }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
            for (int k2 = 0; k2 < quads1d; ++k2)
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * shape1d(j3, k3);
               }
      }

      //TODO insert the three QQQ only here
      // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
      // NOTE: (k1, k2, k3) = q -- 1d quad point index
      // const double *data_d = D.GetElmtData(e);
      for (int k = 0; k < quads; ++k)
      {
         // int ind0[] = {0,k,e};
         const double D0 = D(0,k,e);
         // int ind1[] = {1,k,e};
         const double D1 = D(1,k,e);
         // int ind2[] = {2,k,e};
         const double D2 = D(2,k,e);

         const double q0 = data_qqq[0*quads + k];

         data_qqq[0*quads + k] = D0 * q0;
         data_qqq[1*quads + k] = D1 * q0;
         data_qqq[2*quads + k] = D2 * q0;
      }

      // Apply transpose of the first operator that takes V -> QQQd -- QQQd -> U
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         QQ = 0.;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            Q = 0.;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Q(i1, 0) += QQQ0(k1, k2, k3) * dshape1d(i1, k1);
                  Q(i1, 1) += QQQ1(k1, k2, k3) * shape1d(i1, k1);
                  Q(i1, 2) += QQQ2(k1, k2, k3) * shape1d(i1, k1);
               }
            }
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  QQ(i1, i2, 0) += Q(i1, 0) * shape1d(i2, k2);
                  QQ(i1, i2, 1) += Q(i1, 1) * dshape1d(i2, k2);
                  QQ(i1, i2, 2) += Q(i1, 2) * shape1d(i2, k2);
               }
         }
         for (int i3 = 0; i3 < dofs1d; ++i3)
            for (int i2 = 0; i2 < dofs1d; ++i2)
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1, i2, i3) +=
                     QQ(i1, i2, 0) * shape1d(i3, k3) +
                     QQ(i1, i2, 1) * shape1d(i3, k3) +
                     QQ(i1, i2, 2) * dshape1d(i3, k3);
               }
      }

      // increment offset
      offset += dofs;
   }   
}

}

}
#endif //MFEM_DOMAINKERNELS