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


#ifndef MFEM_FACEKERNELS
#define MFEM_FACEKERNELS

#include "dalg.hpp"
#include "dgfacefunctions.hpp"
#include "partialassemblykernel.hpp"

namespace mfem
{

/**
*  This class contains the permutation functions to apply before applying the external flux kernels.
*/
class Permutation{
public:
   /**
   *  A structure that stores the indirection and permutation on dofs for an external flux on a face.
   */
   struct PermIndir{
      int indirection;
      int permutation;
   };
   /**
   *  KData contains 
   */
   typedef Tensor<2,PermIndir> KData;
   typedef Tensor<3,double> Tensor3d;
   typedef Tensor<4,double> Tensor4d;

   /**
   *  Permutation for 2D hex meshes.
   */
   static void Permutation2d(int face_id, int nbe, int dofs1d, KData& kernel_data, const Tensor3d& T0,
                      Tensor3d& T0p);

   /**
   *  Permutation for 3D hex meshes.
   */
   static void Permutation3d(int face_id, int nbe, int dofs1d, KData& kernel_data, const Tensor4d& T0,
                      Tensor4d& T0p);
};

/////////////////////////////////////////////////
//                                             //
//                                             //
//                 FACE KERNEL                 //
//                                             //
//                                             //
/////////////////////////////////////////////////

  ////////////////////////////
 // Available Face Kernels //
////////////////////////////

/**
*  The Operator selector class
*/
// template <typename Equation, PAOp Op>
// class FacePAK;


template<typename Equation, PAOp Op>
class FaceMult;

  ///////////////////
 // BtDB Operator //
///////////////////

/**
*  A face kernel to compute BtDB
*/
template<typename Equation>
class FaceMult<Equation,BtDB>
: private Equation, Permutation
{

public:
   static const int dimD = 3;
   typedef Tensor<dimD,double> DTensor;
   typedef Tensor<2> Tensor2d;
   typedef Tensor<3,double> Tensor3d;

protected:
   FiniteElementSpace *fes;
   DTensor Dint, Dext;
   KData kernel_data;// Data needed by the Kernel
   Tensor2d shape1d, dshape1d;
   Tensor2d shape0d0, shape0d1, dshape0d0, dshape0d1;

public:
   template <typename Args>
   FaceMult(FiniteElementSpace* _fes, int order, const Args& args)
   : fes(_fes), Dint(), Dext(), kernel_data(),
   shape1d(fes->GetNDofs1d(),fes->GetNQuads1d(order)),
   dshape1d(fes->GetNDofs1d(),fes->GetNQuads1d(order)),
   shape0d0(fes->GetNDofs1d(),1),
   shape0d1(fes->GetNDofs1d(),1),
   dshape0d0(fes->GetNDofs1d(),1),
   dshape0d1(fes->GetNDofs1d(),1)
   {
      // Store the two 0d shape functions and gradients
      // in x = 0.0
      ComputeBasis0d(fes->GetFE(0), 0.0, shape0d0, dshape0d0);
      // in x = 1.0
      ComputeBasis0d(fes->GetFE(0), 1.0, shape0d1, dshape0d1);
      // Store the 1d shape functions and gradients
      ComputeBasis1d(fes->GetFE(0), order, shape1d, dshape1d);
   }

   void init(const int dim, const int quads, const int nb_elts, const int nb_faces_elt)
   {
      kernel_data.setSize(nb_elts,nb_faces_elt);
      Dint.setSize(quads,nb_elts,nb_faces_elt);
      Dext.setSize(quads,nb_elts,nb_faces_elt);
   }

   void initFaceData(const int& dim,
                     const int& ind_elt1, const int& face_id1, const int& nb_rot1, int& perm1,
                     const int& ind_elt2, const int& face_id2, const int& nb_rot2, int& perm2)
   {
      GetPermutation(dim,face_id1,face_id2,nb_rot2,perm1,perm2);
      // Initialization of indirection and permutation identification
      this->kernel_data(ind_elt2,face_id2).indirection = ind_elt1;
      this->kernel_data(ind_elt2,face_id2).permutation = perm1;
      this->kernel_data(ind_elt1,face_id1).indirection = ind_elt2;
      this->kernel_data(ind_elt1,face_id1).permutation = perm2;
   }

   void initBoundaryFaceData(const int& ind_elt, const int& face_id)
   {
      this->kernel_data(ind_elt,face_id).indirection = -1;
      this->kernel_data(ind_elt,face_id).permutation = 0;
   }

   template <typename Args>
   void evalEq(const int dim, const int k1, const int k2, const Vector& normal,
               const int ind_elt1, const int face_id1,
               const int ind_elt2, const int face_id2, FaceElementTransformations* face_tr,
               const IntegrationPoint & ip1, const IntegrationPoint & ip2,
               const Args& args)
   {
      //res'i''j' is the value from element 'j' to element 'i'
      double res11, res21, res22, res12;
      this->evalFaceD(res11,res21,res22,res12,face_tr,normal,ip1,ip2,args);
      Dint(k1, ind_elt1, face_id1) = res11;
      Dext(k2, ind_elt2, face_id2) = res21;
      Dint(k2, ind_elt2, face_id2) = res22;
      Dext(k1, ind_elt1, face_id1) = res12;
   }

   template <typename Args>
   void evalEq(const int dim, const int k1, const int k2, const Vector& normal,
               const int ind_elt1, const int face_id1,
               const int ind_elt2, const int face_id2, FaceElementTransformations* face_tr,
               const IntegrationPoint & ip1, const IntegrationPoint & ip2,
               const Tensor<2>& Jac1, const Tensor<2>& Jac2,
               const Args& args)
   {
      //res'i''j' is the value from element 'j' to element 'i'
      double res11, res21, res22, res12;
      this->evalFaceD(res11,res21,res22,res12,face_tr,normal,ip1,ip2,Jac1,Jac2,args);
      Dint(k1, ind_elt1, face_id1) = res11;
      Dext(k2, ind_elt2, face_id2) = res21;
      Dint(k2, ind_elt2, face_id2) = res22;
      Dext(k1, ind_elt1, face_id1) = res12;
   }

   /**
   *  Computes internal fluxes in 2D
   */
   void EvalInt2D(const Vector& U, Vector& V)
   {
      // North Faces
      int face_id = 2;
      MultIntY2D(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // South Faces
      face_id = 0;
      MultIntY2D(fes,shape1d,shape0d0,Dint,face_id,U,V);
      // East Faces
      face_id = 1;
      MultIntX2D(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // West Faces
      face_id = 3;
      MultIntX2D(fes,shape1d,shape0d0,Dint,face_id,U,V);
   }

   /**
   *  Computes external fluxes in 2D
   */
   void EvalExt2D(const Vector& U, Vector& V)
   {
      // North Faces
      int face_id_test = 2;
      MultExtY2D(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // South Faces
      face_id_test = 0;
      MultExtY2D(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);
      // East Faces
      face_id_test = 1;
      MultExtX2D(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // West Faces
      face_id_test = 3;
      MultExtX2D(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);
   }

   /**
   *  Computes internal fluxes in 3D
   */
   void EvalInt3D(const Vector& U, Vector& V)
   {
      // Bottom Faces
      int face_id = 0;
      MultIntZ3D(fes,shape1d,shape0d0,Dint,face_id,U,V);
      // Top Faces
      face_id = 5;
      MultIntZ3D(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // North Faces
      face_id = 3;
      MultIntY3D(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // South Faces
      face_id = 1;
      MultIntY3D(fes,shape1d,shape0d0,Dint,face_id,U,V);
      // East Faces
      face_id = 2;
      MultIntX3D(fes,shape1d,shape0d1,Dint,face_id,U,V);
      // West Faces
      face_id = 4;
      MultIntX3D(fes,shape1d,shape0d0,Dint,face_id,U,V);
   }

   /**
   *  Computes external fluxes in 3D
   */
   void EvalExt3D(const Vector& U, Vector& V)
   {
      // Bottom Faces
      int face_id_test = 0;
      MultExtZ3D(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);
      // Top Faces
      face_id_test = 5;
      MultExtZ3D(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // North Faces
      face_id_test = 3;
      MultExtY3D(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // South Faces
      face_id_test = 1;
      MultExtY3D(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);
      // East Faces
      face_id_test = 2;
      MultExtX3D(fes,shape1d,shape0d0,shape0d1,kernel_data,Dext,face_id_test,U,V);
      // West Faces
      face_id_test = 4;
      MultExtX3D(fes,shape1d,shape0d1,shape0d0,kernel_data,Dext,face_id_test,U,V);
   }

private:
   static void MultIntX2D(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultIntY2D(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultExtX2D(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void MultExtY2D(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void MultIntX3D(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultIntY3D(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultIntZ3D(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& Dint, int face_id, const Vector& U, Vector& V);
   static void MultExtX3D(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void MultExtY3D(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
   static void MultExtZ3D(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dtest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V);
};



void Permutation::Permutation2d(int face_id, int nbe, int dofs1d, KData& kernel_data, const Tensor3d& T0,
                     Tensor3d& T0p)
{
   for (int e = 0; e < nbe; ++e)
   {
      const int trial = kernel_data(e,face_id).indirection;
      const int permutation = kernel_data(e,face_id).permutation;
      if(trial!=-1)
      {
         switch(permutation)
         {
         case 0:
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  T0p(i1,i2,e) = T0(i1,i2,trial);
               }
            }
            break;
         case 1:
            for (int i2 = 0, j1 = dofs1d-1; i2 < dofs1d; ++i2, --j1)
            {
               for (int i1 = 0, j2 = 0; i1 < dofs1d; ++i1, ++j2)
               {
                  T0p(i1,i2,e) = T0(j1,j2,trial);
               }
            }
            break;
         case 2:
            for (int i2 = 0, j2 = dofs1d-1; i2 < dofs1d; ++i2, --j2)
            {
               for (int i1 = 0, j1 = dofs1d-1; i1 < dofs1d; ++i1, --j1)
               {
                  T0p(i1,i2,e) = T0(j1,j2,trial);
               }
            }
            break;
         case 3:
            for (int i2 = 0, j1 = 0; i2 < dofs1d; ++i2, ++j1)
            {
               for (int i1 = 0, j2 = dofs1d-1; i1 < dofs1d; ++i1, --j2)
               {
                  T0p(i1,i2,e) = T0(j1,j2,trial);
               }
            }
            break;
         default:
            mfem_error("This permutation id does not exist in 2D");
         }
      }else{
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               T0p(i1,i2,e) = 0.0;
            }
         }
      }
   }
}

void Permutation::Permutation3d(int face_id, int nbe, int dofs1d, KData& kernel_data, const Tensor4d& T0,
                     Tensor4d& T0p)
{
   const double* U = T0.getData();
   int elt, ii, jj, kk;
   const int step_elt = dofs1d*dofs1d*dofs1d;
   for (int e = 0; e < nbe; ++e)
   {
      const int trial = kernel_data(e,face_id).indirection;
      const int permutation = kernel_data(e,face_id).permutation;
      if (trial!=-1)
      {
         elt = trial*step_elt;
         IntMatrix P(3,3);
         GetChangeOfBasis(permutation, P);
         // cout << "decoding P" << endl;
         // cout << P(0,0) << ", " << P(0,1) << ", " << P(0,2) << endl;
         // cout << P(1,0) << ", " << P(1,1) << ", " << P(1,2) << endl;
         // cout << P(2,0) << ", " << P(2,1) << ", " << P(2,2) << endl;
         int begin_ii = (P(0,0)==-1)*(dofs1d-1) + (P(1,0)==-1)*(dofs1d*dofs1d-1) + (P(2,0)==-1)*(dofs1d*dofs1d*dofs1d-1);
         int begin_jj = (P(0,1)==-1)*(dofs1d-1) + (P(1,1)==-1)*(dofs1d*dofs1d-1) + (P(2,1)==-1)*(dofs1d*dofs1d*dofs1d-1);
         int begin_kk = (P(0,2)==-1)*(dofs1d-1) + (P(1,2)==-1)*(dofs1d*dofs1d-1) + (P(2,2)==-1)*(dofs1d*dofs1d*dofs1d-1);
         int step_ii  = P(0,0) + P(1,0)*dofs1d + P(2,0)*dofs1d*dofs1d;
         int step_jj  = P(0,1) + P(1,1)*dofs1d + P(2,1)*dofs1d*dofs1d;
         int step_kk  = P(0,2) + P(1,2)*dofs1d + P(2,2)*dofs1d*dofs1d;
         kk = begin_kk;
         for (int k = 0; k < dofs1d; ++k)
         {
            jj = begin_jj;
            for (int j = 0; j < dofs1d; ++j)
            {
               ii = begin_ii;
               for (int i = 0; i < dofs1d; ++i)
               {
                  T0p(i,j,k,e) = U[ elt + ii + jj + kk ];
                  ii += step_ii;
               }
               jj += step_jj;
            }
            kk += step_kk;
         }
      }
      else
      {
         for (int k = 0; k < dofs1d; ++k)
         {
            for (int j = 0; j < dofs1d; ++j)
            {
               for (int i = 0; i < dofs1d; ++i)
               {
                  T0p(i,j,k,e) = 0.0;
               }
            }
         }
      }
   }
}

    //////////////////////////////////////
   ///                                ///
  ///   FACE KERNELS IMPLEMENTATION  ///
 ///                                ///
//////////////////////////////////////

template <typename Equation>
void FaceMult<Equation,PAOp::BtDB>::MultIntX2D(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   DTensor T0(U.GetData(),dofs1d,dofs1d,nbe);
   DTensor R(V.GetData(),dofs1d,dofs1d,nbe);
   Tensor<1> T1(dofs1d),T2(quads1d),T3(dofs1d);
   for (int e = 0; e < nbe; ++e)
   {
      // T1.zero();
      for (int i2 = 0; i2 < dofs1d; ++i2)
      {
         T1(i2) = 0.0;
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T1(i2) += B0d(i1,0) * T0(i1,i2,e);
         }
      }
      // T2.zero();
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         T2(k2) = 0.0;
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T2(k2) += B(i2,k2) * T1(i2);
         }
      }
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         T2(k2) = D(k2,e,face_id) * T2(k2);
      }
      // T3.zero();
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         T3(j2) = 0.0;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            T3(j2) += B(j2,k2) * T2(k2);
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            R(j1,j2,e) += B0d(j1,0) * T3(j2);
         }
      }
   }
}

template <typename Equation>
void FaceMult<Equation,PAOp::BtDB>::MultIntY2D(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   DTensor T0(U.GetData(),dofs1d,dofs1d,nbe);
   DTensor R(V.GetData(),dofs1d,dofs1d,nbe);
   Tensor<1> T1(dofs1d),T2(quads1d),T3(dofs1d);
   //T1_i1 = B0d^i2 U_i1i2
   for (int e = 0; e < nbe; ++e)
   {
      // T1.zero();
      for (int i1 = 0; i1 < dofs1d; ++i1)
      {
         T1(i1) = 0.0;
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T1(i1) += B0d(i2,0) * T0(i1,i2,e);
         }
      }
      // T2.zero();
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         T2(k1) = 0.0;
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T2(k1) += B(i1,k1) * T1(i1);
         }
      }
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         T2(k1) = D(k1,e,face_id) * T2(k1);
      }
      // T3.zero();
      for (int j1 = 0; j1 < dofs1d; ++j1)
      {
         T3(j1) = 0.0;
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            T3(j1) += B(j1,k1) * T2(k1);
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            R(j1,j2,e) += B0d(j2,0) * T3(j1);
         }
      }
   }
}

template <typename Equation>
void FaceMult<Equation,PAOp::BtDB>::MultExtX2D(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dTest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   Tensor3d T0(U.GetData(),dofs1d,dofs1d,nbe);
   Tensor3d R(V.GetData(),dofs1d,dofs1d,nbe);
   Tensor<1,double> T1(dofs1d),T2(quads1d),T3(dofs1d);
   // Indirections
   Tensor3d T0p(dofs1d,dofs1d,nbe);
   Permutation2d(face_id,nbe,dofs1d,kernel_data,T0,T0p);
   //T1_i2 = B0d^i1 U_i1i2
   for (int e = 0; e < nbe; ++e)
   {
      // T1.zero();
      for (int i2 = 0; i2 < dofs1d; ++i2)
      {
         T1(i2) = 0.0;
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T1(i2) += B0dTrial(i1,0) * T0p(i1,i2,e);
         }
      }
      // T2.zero();
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         T2(k2) = 0.0;
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T2(k2) += B(i2,k2) * T1(i2);
         }
      }
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         T2(k2) = D(k2,e,face_id) * T2(k2);
      }
      // T3.zero();
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         T3(j2) = 0.0;
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            T3(j2) += B(j2,k2) * T2(k2);
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            R(j1,j2,e) += B0dTest(j1,0) * T3(j2);
         }
      }
   }
}

template <typename Equation>
void FaceMult<Equation,PAOp::BtDB>::MultExtY2D(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dTest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   DTensor T0(U.GetData(),dofs1d,dofs1d,nbe);
   DTensor R(V.GetData(),dofs1d,dofs1d,nbe);
   Tensor<1,double> T1(dofs1d),T2(quads1d),T3(dofs1d);
   // Indirections
   DTensor T0p(dofs1d,dofs1d,nbe);
   Permutation2d(face_id,nbe,dofs1d,kernel_data,T0,T0p);
   //T1_i1 = B0d^i2 U_i1i2
   for (int e = 0; e < nbe; ++e)
   {
      // T1.zero();
      for (int i1 = 0; i1 < dofs1d; ++i1)
      {
         T1(i1) = 0.0;
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T1(i1) += B0dTrial(i2,0) * T0p(i1,i2,e);
         }
      }
      // T2.zero();
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         T2(k1) = 0.0;
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T2(k1) += B(i1,k1) * T1(i1);
         }
      }
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         T2(k1) = D(k1,e,face_id) * T2(k1);
      }
      // T3.zero();
      for (int j1 = 0; j1 < dofs1d; ++j1)
      {
         T3(j1) = 0.0;
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            T3(j1) += B(j1,k1) * T2(k1);
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            R(j1,j2,e) += B0dTest(j2,0) * T3(j1);
         }
      }
   }
}

template <typename Equation>
void FaceMult<Equation,PAOp::BtDB>::MultIntX3D(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   Tensor<4> T0(U.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   Tensor<4> R(V.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   //TODO: Check size
   Tensor<2> T1(dofs1d,dofs1d),T2(dofs1d,quads1d),T3(quads1d,quads1d),T4(quads1d,dofs1d),T5(dofs1d,dofs1d);
   for (int e = 0; e < nbe; ++e)
   {
      for (int i3 = 0; i3 < dofs1d; ++i3)
      {
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T1(i2,i3) = 0.0;
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               T1(i2,i3) += B0d(i1,0) * T0(i1,i2,i3,e);
            }
         }
      }
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         for (int i3 = 0; i3 < dofs1d; ++i3)
         {
            T2(i3,k2) = 0.0;
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               T2(i3,k2) += B(i2,k2) * T1(i2,i3);
            }
         }
      }
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            T3(k2,k3) = 0.0;
            for (int i3 = 0; i3 < dofs1d; ++i3)
            {
               T3(k2,k3) += B(i3,k3) * T2(i3,k2);
            }
         }
      }
      for (int k3 = 0, k = 0; k3 < quads1d; ++k3)
      {
         for (int k2 = 0; k2 < quads1d; ++k2, ++k)
         {
            T3(k2,k3) = D(k,e,face_id) * T3(k2,k3);
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int k3 = 0; k3 < quads1d; ++k3)
         {
            T4(k3,j2) = 0.0;
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               T4(k3,j2) += B(j2,k2) * T3(k2,k3);
            }
         }
      }
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            T5(j2,j3) = 0.0;
            for (int k3 = 0; k3 < quads1d; ++k3)
            {
               T5(j2,j3) += B(j3,k3) * T4(k3,j2);
            }
         }
      }
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               R(j1,j2,j3,e) += B0d(j1,0) * T5(j2,j3);
            }
         }
      }
   }
}

template <typename Equation>
void FaceMult<Equation,PAOp::BtDB>::MultIntY3D(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   Tensor<4> T0(U.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   Tensor<4> R(V.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   //TODO: Check size
   Tensor<2> T1(dofs1d,dofs1d),T2(dofs1d,quads1d),T3(quads1d,quads1d),T4(quads1d,dofs1d),T5(dofs1d,dofs1d);
   for (int e = 0; e < nbe; ++e)
   {
      for (int i3 = 0; i3 < dofs1d; ++i3)
      {
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T1(i1,i3) = 0.0;
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               T1(i1,i3) += B0d(i2,0) * T0(i1,i2,i3,e);
            }
         }
      }
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         for (int i3 = 0; i3 < dofs1d; ++i3)
         {
            T2(i3,k1) = 0.0;
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               T2(i3,k1) += B(i1,k1) * T1(i1,i3);
            }
         }
      }
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            T3(k1,k3) = 0.0;
            for (int i3 = 0; i3 < dofs1d; ++i3)
            {
               T3(k1,k3) += B(i3,k3) * T2(i3,k1);
            }
         }
      }
      for (int k3 = 0, k = 0; k3 < quads1d; ++k3)
      {
         for (int k1 = 0; k1 < quads1d; ++k1, ++k)
         {
            T3(k1,k3) = D(k,e,face_id) * T3(k1,k3);
         }
      }
      for (int j1 = 0; j1 < dofs1d; ++j1)
      {
         for (int k3 = 0; k3 < quads1d; ++k3)
         {
            T4(k3,j1) = 0.0;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               T4(k3,j1) += B(j1,k1) * T3(k1,k3);
            }
         }
      }
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            T5(j1,j3) = 0.0;
            for (int k3 = 0; k3 < quads1d; ++k3)
            {
               T5(j1,j3) += B(j3,k3) * T4(k3,j1);
            }
         }
      }
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               R(j1,j2,j3,e) += B0d(j2,0) * T5(j1,j3);
            }
         }
      }
   }
}

template <typename Equation>
void FaceMult<Equation,PAOp::BtDB>::MultIntZ3D(FiniteElementSpace* fes, Tensor2d& B, Tensor2d& B0d,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   Tensor<4> T0(U.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   Tensor<4> R(V.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   //TODO: Check size
   Tensor<2> T1(dofs1d,dofs1d),T2(dofs1d,quads1d),T3(quads1d,quads1d),T4(quads1d,dofs1d),T5(dofs1d,dofs1d);
   for (int e = 0; e < nbe; ++e)
   {
      for (int i2 = 0; i2 < dofs1d; ++i2)
      {
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T1(i1,i2) = 0.0;
            for (int i3 = 0; i3 < dofs1d; ++i3)
            {
               T1(i1,i2) += B0d(i3,0) * T0(i1,i2,i3,e);
            }
         }
      }
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T2(i2,k1) = 0.0;
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               T2(i2,k1) += B(i1,k1) * T1(i1,i2);
            }
         }
      }
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            T3(k1,k2) = 0.0;
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               T3(k1,k2) += B(i2,k2) * T2(i2,k1);
            }
         }
      }
      for (int k2 = 0, k = 0; k2 < quads1d; ++k2)
      {
         for (int k1 = 0; k1 < quads1d; ++k1, ++k)
         {
            T3(k1,k2) = D(k,e,face_id) * T3(k1,k2);
         }
      }
      for (int j1 = 0; j1 < dofs1d; ++j1)
      {
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            T4(k2,j1) = 0.0;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               T4(k2,j1) += B(j1,k1) * T3(k1,k2);
            }
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            T5(j1,j2) = 0.0;
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               T5(j1,j2) += B(j2,k2) * T4(k2,j1);
            }
         }
      }
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               R(j1,j2,j3,e) += B0d(j3,0) * T5(j1,j2);
            }
         }
      }
   }
}

template <typename Equation>
void FaceMult<Equation,PAOp::BtDB>::MultExtX3D(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dTest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   Tensor<4> T0p(U.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   Tensor<4> R(V.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   Tensor<2> T1(dofs1d,dofs1d),T2(dofs1d,quads1d),T3(quads1d,quads1d),T4(quads1d,dofs1d),T5(dofs1d,dofs1d);
   Tensor<4> T0(dofs1d,dofs1d,dofs1d,nbe);
   Permutation3d(face_id,nbe,dofs1d,kernel_data,T0p,T0);
   for (int e = 0; e < nbe; ++e)
   {
      for (int i3 = 0; i3 < dofs1d; ++i3)
      {
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T1(i2,i3) = 0.0;
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               T1(i2,i3) += B0dTrial(i1,0) * T0(i1,i2,i3,e);
            }
         }
      }
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         for (int i3 = 0; i3 < dofs1d; ++i3)
         {
            T2(i3,k2) = 0.0;
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               T2(i3,k2) += B(i2,k2) * T1(i2,i3);
            }
         }
      }
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            T3(k2,k3) = 0.0;
            for (int i3 = 0; i3 < dofs1d; ++i3)
            {
               T3(k2,k3) += B(i3,k3) * T2(i3,k2);
            }
         }
      }
      for (int k3 = 0, k = 0; k3 < quads1d; ++k3)
      {
         for (int k2 = 0; k2 < quads1d; ++k2, ++k)
         {
            T3(k2,k3) = D(k,e,face_id) * T3(k2,k3);
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int k3 = 0; k3 < quads1d; ++k3)
         {
            T4(k3,j2) = 0.0;
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               T4(k3,j2) += B(j2,k2) * T3(k2,k3);
            }
         }
      }
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            T5(j2,j3) = 0.0;
            for (int k3 = 0; k3 < quads1d; ++k3)
            {
               T5(j2,j3) += B(j3,k3) * T4(k3,j2);
            }
         }
      }
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               R(j1,j2,j3,e) += B0dTest(j1,0) * T5(j2,j3);
            }
         }
      }
   }
}

template <typename Equation>
void FaceMult<Equation,PAOp::BtDB>::MultExtY3D(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dTest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   Tensor<4> T0p(U.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   Tensor<4> R(V.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   Tensor<2> T1(dofs1d,dofs1d),T2(dofs1d,quads1d),T3(quads1d,quads1d),T4(quads1d,dofs1d),T5(dofs1d,dofs1d);
   Tensor<4> T0(dofs1d,dofs1d,dofs1d,nbe);
   Permutation3d(face_id,nbe,dofs1d,kernel_data,T0p,T0);
   for (int e = 0; e < nbe; ++e)
   {
      for (int i3 = 0; i3 < dofs1d; ++i3)
      {
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T1(i1,i3) = 0.0;
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               T1(i1,i3) += B0dTrial(i2,0) * T0(i1,i2,i3,e);
            }
         }
      }
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         for (int i3 = 0; i3 < dofs1d; ++i3)
         {
            T2(i3,k1) = 0.0;
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               T2(i3,k1) += B(i1,k1) * T1(i1,i3);
            }
         }
      }
      for (int k3 = 0; k3 < quads1d; ++k3)
      {
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            T3(k1,k3) = 0.0;
            for (int i3 = 0; i3 < dofs1d; ++i3)
            {
               T3(k1,k3) += B(i3,k3) * T2(i3,k1);
            }
         }
      }
      for (int k3 = 0, k = 0; k3 < quads1d; ++k3)
      {
         for (int k1 = 0; k1 < quads1d; ++k1, ++k)
         {
            T3(k1,k3) = D(k,e,face_id) * T3(k1,k3);
         }
      }
      for (int j1 = 0; j1 < dofs1d; ++j1)
      {
         for (int k3 = 0; k3 < quads1d; ++k3)
         {
            T4(k3,j1) = 0.0;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               T4(k3,j1) += B(j1,k1) * T3(k1,k3);
            }
         }
      }
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            T5(j1,j3) = 0.0;
            for (int k3 = 0; k3 < quads1d; ++k3)
            {
               T5(j1,j3) += B(j3,k3) * T4(k3,j1);
            }
         }
      }
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               R(j1,j2,j3,e) += B0dTest(j2,0) * T5(j1,j3);
            }
         }
      }
   }
}

template <typename Equation>
void FaceMult<Equation,PAOp::BtDB>::MultExtZ3D(FiniteElementSpace* fes, Tensor2d& B,
                        Tensor2d& B0dTrial, Tensor2d& B0dTest, KData& kernel_data,
                        DTensor& D, int face_id, const Vector& U, Vector& V)
{
   // nunber of elements
   const int nbe = fes->GetNE();
   // number of degrees of freedom in 1d (assumes that i1=i2=i3)
   const int dofs1d = B.Height();
   // number of quadrature points
   const int quads1d = B.Width();
   Tensor<4> T0p(U.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   Tensor<4> R(V.GetData(),dofs1d,dofs1d,dofs1d,nbe);
   Tensor<2> T1(dofs1d,dofs1d),T2(dofs1d,quads1d),T3(quads1d,quads1d),T4(quads1d,dofs1d),T5(dofs1d,dofs1d);
   Tensor<4> T0(dofs1d,dofs1d,dofs1d,nbe);
   Permutation3d(face_id,nbe,dofs1d,kernel_data,T0p,T0);
   for (int e = 0; e < nbe; ++e)
   {
      for (int i2 = 0; i2 < dofs1d; ++i2)
      {
         for (int i1 = 0; i1 < dofs1d; ++i1)
         {
            T1(i1,i2) = 0.0;
            for (int i3 = 0; i3 < dofs1d; ++i3)
            {
               T1(i1,i2) += B0dTrial(i3,0) * T0(i1,i2,i3,e);
            }
         }
      }
      for (int k1 = 0; k1 < quads1d; ++k1)
      {
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            T2(i2,k1) = 0.0;
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               T2(i2,k1) += B(i1,k1) * T1(i1,i2);
            }
         }
      }
      for (int k2 = 0; k2 < quads1d; ++k2)
      {
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            T3(k1,k2) = 0.0;
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               T3(k1,k2) += B(i2,k2) * T2(i2,k1);
            }
         }
      }
      for (int k2 = 0, k = 0; k2 < quads1d; ++k2)
      {
         for (int k1 = 0; k1 < quads1d; ++k1, ++k)
         {
            T3(k1,k2) = D(k,e,face_id) * T3(k1,k2);
         }
      }
      for (int j1 = 0; j1 < dofs1d; ++j1)
      {
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            T4(k2,j1) = 0.0;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               T4(k2,j1) += B(j1,k1) * T3(k1,k2);
            }
         }
      }
      for (int j2 = 0; j2 < dofs1d; ++j2)
      {
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            T5(j1,j2) = 0.0;
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               T5(j1,j2) += B(j2,k2) * T4(k2,j1);
            }
         }
      }
      for (int j3 = 0; j3 < dofs1d; ++j3)
      {
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               R(j1,j2,j3,e) += B0dTest(j3,0) * T5(j1,j2);
            }
         }
      }
   }
}

}


#endif //MFEM_FACEKERNELS