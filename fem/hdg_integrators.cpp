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

// Implementation of Bilinear Form Integrators

#include "fem.hpp"
#include <cmath>
#include <algorithm>
#include "hdg_integrators.hpp"

using namespace std;

namespace mfem
{
void HDGDomainIntegratorAdvection::AssembleElementMatrix(
   const FiniteElement &fe_u,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int ndof_u = fe_u.GetDof();
   int dim  = fe_u.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   bool square = (dim == spaceDim);

   Vector vec1; // for the convection integral
   vec2.SetSize(dim);
   BdFidxT.SetSize(ndof_u);

   dshape.SetSize (ndof_u, dim); // for nabla \tilde{u}
   gshape.SetSize (ndof_u, dim); // for nabla u
   Jadj.SetSize (dim);  // for the Jacobian
   shapeu.SetSize (ndof_u);    // shape of u

   // setting the sizes of the local element matrices
   elmat.SetSize(ndof_u, ndof_u);

   // setting the order of integration
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * fe_u.GetOrder() + 1;
      ir = &IntRules.Get(fe_u.GetGeomType(), order);
   }

   elmat = 0.0;

   // evaluate the advection vector at all integration point
   avec->Eval(Adv_ir, Trans, *ir);

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // shape functions
      fe_u.CalcDShape (ip, dshape);
      fe_u.CalcShape (ip, shapeu);

      // calculate the Adjugate of the Jacobian
      Trans.SetIntPoint (&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);

      double w = Trans.Weight();
      w = ip.weight / (square ? w : w*w*w);
      // AdjugateJacobian = / adj(J),         if J is square
      //                    \ adj(J^t.J).J^t, otherwise

      // Calculate the gradient of the function of the physical element
      Mult (dshape, Jadj, gshape);

      // get the advection at the current integration point
      Adv_ir.GetColumnReference(i, vec1);
      vec1 *= ip.weight; // so it will be (cu, nabla v)

      // compute -(cu, nabla v)
      Jadj.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);
      AddMultVWt(shapeu, BdFidxT, elmat);

      double massw = Trans.Weight() * ip.weight;

      if (mass_coeff)
      {
         massw *= mass_coeff->Eval(Trans, ip);
      }
      AddMult_a_VVt(massw, shapeu, elmat);

   }
}
//---------------------------------------------------------------------
void HDGFaceIntegratorAdvection::AssembleFaceMatrixOneElement1and1FES(
   const FiniteElement &fe_u,
   const FiniteElement &face_fe,
   FaceElementTransformations &Trans,
   const int elem1or2,
   const bool onlyB,
   DenseMatrix &elmat1,
   DenseMatrix &elmat2,
   DenseMatrix &elmat3,
   DenseMatrix &elmat4)
{
   int dim, ndof, ndof_face;
   double w;

   dim = fe_u.GetDim();
   ndof_face = face_fe.GetDof();

   shape_face.SetSize(ndof_face);

   normal.SetSize(dim);
   normalJ.SetSize(dim);
   invJ.SetSize(dim);
   adv.SetSize(dim);

   ndof = fe_u.GetDof();

   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_normal.SetSize(ndof);

   elmat1.SetSize(ndof);
   elmat2.SetSize(ndof, ndof_face);
   elmat3.SetSize(ndof_face, ndof);
   elmat4.SetSize(ndof_face);

   elmat1 = 0.0;
   elmat2 = 0.0;
   elmat3 = 0.0;
   elmat4 = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order
      int order;
      order = 2*fe_u.GetOrder();

      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip; // element integration point

      Trans.Face->SetIntPoint(&ip);
      face_fe.CalcShape(ip, shape_face);

      if (dim == 1)
      {
         normal(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), normal);
      }

      Trans.Loc1.Transform(ip, eip);
      Trans.Elem1->SetIntPoint(&eip);

      avec->Eval(adv, *Trans.Elem1, eip);
      double an = adv * normal;
      double an_L = an;

      double zeta_R = 0.0, zeta_L = 0.0, zeta = 0.0;
      if (an < 0.0)
      {
         zeta_L = 1.0;
      }

      if (elem1or2 == 1)
      {
         zeta = zeta_L;
      }
      else
      {
         Trans.Loc2.Transform(ip, eip);
         Trans.Elem2->SetIntPoint(&eip);

         avec->Eval(adv, *Trans.Elem2, eip);
         an = adv * normal;
         an *= -1.;

         zeta_R = 1.0 - zeta_L;
         zeta = zeta_R;
      }

      fe_u.CalcShape(eip, shape);

      w = ip.weight;

      for (int i = 0; i < ndof; i++)
      {
         for (int j = 0; j < ndof; j++)
         {
            // - < 1, [zeta a.n u v] >
            elmat1(i, j) -= w * zeta * an * shape(i) * shape(j);
         }

         for (int j = 0; j < ndof_face; j++)
         {
            if (!onlyB)
            {
               // - < ubar, [(1-zeta) a.n v] >
               elmat3(j, i) -= w * an * (1.-zeta) * shape(i) * shape_face(j);
            }

            // + < ubar, [zeta a.n v] >
            elmat2(i, j) += w * zeta * an * shape(i) * shape_face(j);

         }
      }
      if (!onlyB)
      {

         for (int i = 0; i < ndof_face; i++)
            for (int j = 0; j < ndof_face; j++)
            {
               // - < 1, [zeta a.n ubar vbar] > + < 1, [(1-zeta) a.n ubar vbar >_{\Gamma_N}
               if (Trans.Elem2No >= 0)
               {
                  if (elem1or2 == 1)
                  {
                     elmat4(i, j) += -w * zeta_L * an_L * shape_face(i) * shape_face(j);
                  }
                  else
                  {
                     elmat4(i, j) += - w * (1.0 - zeta_L) * (-an_L) * shape_face(i) * shape_face(j);
                  }
               }
               else
               {
                  elmat4(i, j) += -w * zeta_L * an * shape_face(i) * shape_face(j)
                                  + w * (1.0 - zeta_L) * an * shape_face(i) * shape_face(j);
               }
            }
      }

   }
}

//---------------------------------------------------------------------
void HDGInflowLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("Not implemented \n");
}

void HDGInflowLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &face_S, FaceElementTransformations &Trans,
   Vector &favect)
{
   int dim, ndof_face;
   double w, uin;

   dim = face_S.GetDim(); // This is face dimension which is 1 less than
   dim += 1;              // space dimension so add 1 to face dim to
   // get the space dim.
   n_L.SetSize(dim);
   Vector adv(dim);

   ndof_face = face_S.GetDof();

   shape_f.SetSize(ndof_face);
   favect.SetSize(ndof_face);
   favect = 0.0;

   if (Trans.Elem2No >= 0)
   {
      // Interior face, do nothing
   }
   else
   {
      // Boundary face
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int order = 2 * face_S.GetOrder();
         if (face_S.GetMapType() == FiniteElement::VALUE)
         {
            order += Trans.Face->OrderW();
         }

         ir = &IntRules.Get(Trans.FaceGeom, order);
      }

      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         face_S.CalcShape(ip, shape_f);

         IntegrationPoint eip_L;
         Trans.Loc1.Transform(ip, eip_L);
         Trans.Face->SetIntPoint(&ip);

         avec->Eval(adv, *Trans.Elem1, eip_L);
         uin = u_in->Eval(*Trans.Elem1, eip_L);

         if (dim == 1)
         {
            n_L(0) = 2*eip_L.x - 1.0;
         }
         else
         {
            CalcOrtho(Trans.Face->Jacobian(), n_L);
         }

         double an_L = adv * n_L;

         double zeta_L = 0.0;
         if (an_L < 0.0)
         {
            zeta_L = 1.0;
         }

         w = ip.weight;

         double gg = -uin * an_L * zeta_L;
         for (int i = 0; i < ndof_face; i++)
         {
            favect(i) += w * gg * shape_f(i);
         }
      }
   }
}
//---------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////////////////////////
void HDGDomainIntegratorDiffusion::AssembleElementMatrix2FES(
   const FiniteElement &fe_q,
   const FiniteElement &fe_u,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   // get the number of degrees of freedoms and the dimension of the problem
   int ndof_u = fe_u.GetDof();
   int ndof_q = fe_q.GetDof();
   int dim  = fe_q.GetDim();
   double norm;

   int vdim = dim ;

   // set the vector and matrix sizes
   dshape.SetSize (ndof_u, dim); // for nabla u_reference
   gshape.SetSize (ndof_u, dim); // for nabla u
   Jadj.SetSize (dim);  // the Jacobian
   divshape.SetSize (vdim*ndof_u);  // divergence of q
   shape.SetSize (ndof_q);  // shape of q (and u)

   // for vector diffusion the matrix is built up from partial matrices
   partelmat.SetSize(ndof_q);

   DenseMatrix local_A11, local_A12, local_A21;

   // setting the sizes of the local element matrices
   local_A11.SetSize(dim*ndof_q, dim*ndof_q);
   local_A12.SetSize(vdim*ndof_q, ndof_u);
   local_A21.SetSize(ndof_u, vdim*ndof_q);

   elmat.SetSize(dim*ndof_q + ndof_u);

   local_A11 = 0.0;
   local_A12 = 0.0;
   local_A21 = 0.0;
   elmat  = 0.0;

   // setting the order of integration
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order1 = 2 * fe_q.GetOrder();
      int order2 = 2 * fe_q.GetOrder() + Trans.OrderW();
      int order = max(order1, order2);
      ir = &IntRules.Get(fe_u.GetGeomType(), order);
   }

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // compute the shape and the gradient values on the reference element
      fe_u.CalcDShape (ip, dshape);
      fe_q.CalcShape (ip, shape);

      // calculate the adjugate of the Jacobian
      Trans.SetIntPoint (&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);

      // Calculate the gradient of the function of the physical element
      Mult (dshape, Jadj, gshape);

      // the weight is the product of the integral weight and the
      // determinant of the Jacobian
      norm = ip.weight * Trans.Weight();
      MultVVt(shape, partelmat);

      double c = ip.weight;

      // transform the the matrix to divergence vector
      gshape.GradToDiv (divshape);

      // mulitply by 1.0/nu
      partelmat *= 1.0/nu->Eval(Trans, ip);

      shape *= c;
      // compute the (u, \div v) term
      AddMultVWt (shape, divshape, local_A21);

      // assemble -(q, v) from the partial matrices
      partelmat *= norm*(-1.0);
      for (int k = 0; k < vdim; k++)
      {
    	  local_A11.AddMatrix(partelmat, ndof_q*k, ndof_q*k);
      }

   }

   local_A12.Transpose(local_A21);

   int block_size1 = dim*ndof_q;
   int block_size2 = ndof_u;

   elmat.CopyMN(local_A11, 0, 0);
   elmat.CopyMN(local_A12, 0, block_size1);
   elmat.CopyMN(local_A21, block_size1, 0);

//   for (int i = 0; i<block_size1; i++)
//   {
//      for (int j = 0; j<block_size1; j++)
//      {
//         elmat(i,j) = local_A11(i,j);
//      }
//      for (int j = 0; j<block_size2; j++)
//      {
//         elmat(i,j+block_size1) = local_A12(i,j);
//      }
//   }
//
//   for (int i = 0; i<block_size2; i++)
//   {
//      for (int j = 0; j<block_size1; j++)
//      {
//         elmat(i+block_size1,j) = local_A21(i,j);
//      }
//   }
}


void HDGFaceIntegratorDiffusion::AssembleFaceMatrixOneElement2and1FES(
   const FiniteElement &fe_q,
   const FiniteElement &fe_u,
   const FiniteElement &face_fe,
   FaceElementTransformations &Trans,
   const int elem1or2,
   const bool onlyB,
   DenseMatrix &elmat1,
   DenseMatrix &elmat2,
   DenseMatrix &elmat3,
   DenseMatrix &elmat4)
{
   // Get DoF from faces and the dimension
   int ndof_face = face_fe.GetDof();
   int ndof_q, ndof_u;
   int dim = fe_q.GetDim();
   int vdim = dim;
   int order;

   DenseMatrix shape1_n_mtx;

   // set the dofs for u and q
   ndof_u = fe_u.GetDof();
   ndof_q = fe_q.GetDof();

   DenseMatrix local_B1, local_A22, local_B2, local_C1, local_C2, local_D;

   // set the shape functions, the normal and the advection
   shapeu.SetSize(ndof_u);
   shapeq.SetSize(ndof_q);
   shape_face.SetSize(ndof_face);
   normal.SetSize(dim);

   // set the proper size for the matrices
   local_B1.SetSize(vdim*ndof_q, ndof_face);
   local_B1 = 0.0;
   local_A22.SetSize(ndof_u, ndof_u);
   local_A22 = 0.0;
   local_B2.SetSize(ndof_u, ndof_face);
   local_B2 = 0.0;
   local_C1.SetSize(vdim*ndof_q, ndof_face);
   local_C1 = 0.0;
   local_C2.SetSize(ndof_u, ndof_face);
   local_C2 = 0.0;
   local_D.SetSize(ndof_face, ndof_face);
   local_D = 0.0;

   int sub_block_size1 = vdim*ndof_q;
   int sub_block_size2 = ndof_u;

   int block_size1 = sub_block_size1 + sub_block_size2;
   int block_size2 = ndof_face;

   elmat1.SetSize(block_size1);
   elmat1 = 0.0;
   elmat2.SetSize(block_size1, block_size2);
   elmat2 = 0.0;
   elmat3.SetSize(block_size2, block_size1);
   elmat3 = 0.0;
   elmat4.SetSize(block_size2);
   elmat4 = 0.0;


   shape1_n_mtx.SetSize(ndof_q,dim);
   shape_dot_n.SetSize(ndof_q,dim);

   // set the order of integration
   // using the fact that q and u has the same order!
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      order = 2*max(max(fe_q.GetOrder(), fe_u.GetOrder()), face_fe.GetOrder());
      order += 2;

      // IntegrationRule depends on the Geometry of the face (pont, line, triangle, rectangular)
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip; // integration point on the element

      // Trace finite element shape function
      Trans.Face->SetIntPoint(&ip);
      face_fe.CalcShape(ip, shape_face);

      // calculate the normal at the integration point
      if (dim == 1)
      {
         normal(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), normal);
      }

      if (elem1or2 == 1)
      {
         // Side 1 finite element shape function
         Trans.Loc1.Transform(ip, eip);
      }
      else
      {
         // Side 2 finite element shape function
         Trans.Loc2.Transform(ip, eip);
      }

      fe_u.CalcShape(eip, shapeu);
      fe_q.CalcShape(eip, shapeq);
      MultVWt(shapeq, normal, shape_dot_n) ;


      // set the coefficients for the different terms
      // if the normal is involved Trans.Face->Weight() is not required
      double w1 = ip.weight*(-1.0);

      if (elem1or2 == 2)
      {
         w1 *=-1.0;
      }

      double w2 = tauD*Trans.Face->Weight()* ip.weight;

      double w3 = -w2;

      // local_B1 = < \lambda,\nu v\cdot n>
      for (int i = 0; i < vdim; i++)
         for (int k = 0; k < ndof_q; k++)
            for (int j = 0; j < ndof_face; j++)
            {
            	local_B1(i*ndof_q + k, j) += shape_face(j) * shape_dot_n(k,i) * w1;
            }

      // local_A22 =  < \tau u, w>
      // local_B2= -< tau \lambda, w>
      // local_C2 = -< tau \lambda, w>
      for (int i = 0; i < ndof_u; i++)
      {
         for (int j = 0; j < ndof_u; j++)
         {
        	 local_A22(i, j) += w2 * shapeu(i) * shapeu(j);
         }

         for (int j = 0; j < ndof_face; j++)
         {
        	 local_B2(i, j) += w3 * shapeu(i) * shape_face(j);
         }
      }

      if (!onlyB)
      {
         // local_D = < \tau \lambda, \mu>

         AddMult_a_VVt(w2, shape_face, local_D);
      }
   }

   local_C1.Transpose(local_B1);
   local_C2.Transpose(local_B2);

   elmat1.CopyMN(local_A22, sub_block_size1, sub_block_size1);

   elmat2.CopyMN(local_B1, 0, 0);
   elmat2.CopyMN(local_B2, sub_block_size1, 0);

   elmat3.CopyMN(local_C1, 0, 0);
   elmat3.CopyMN(local_C2, 0, sub_block_size1);

//   for (int i = 0; i<sub_block_size2; i++)
//   {
//      for (int j = 0; j<sub_block_size2; j++)
//      {
//         elmat1(i+sub_block_size1,j+sub_block_size1) = local_A22(i,j);
//      }
//   }
//
//   for (int i = 0; i<block_size2; i++)
//   {
//      for (int j = 0; j<sub_block_size1; j++)
//      {
//         elmat2(j,i) = local_B1(j,i);
//
//         elmat3(i,j) = local_C1(i,j);
//      }
//      for (int j = 0; j<sub_block_size2; j++)
//      {
//         elmat2(j+sub_block_size1,i) = local_B2(j,i);
//
//         elmat3(i,j+sub_block_size1) = local_C2(i,j);
//      }
//   }

   elmat4 = local_D;
}


}
;
