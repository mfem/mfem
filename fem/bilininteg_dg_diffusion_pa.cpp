
// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "restriction.hpp"

/* 
   Integrator for the DG form:

    - < {(Q grad(u)).n}, [v] > + sigma < [u], {(Q grad(v)).n} >  + kappa < {h^{-1} Q} [u], [v] >

*/

namespace mfem
{

void DGDiffusionIntegrator::SetupPA(const FiniteElementSpace &fes, FaceType type)
{
   std::cout << "%--------  SetupPA  ------------------------------------------ "<< std::endl;

   nf = fes.GetNFbyType(type);
   if (nf==0) { return; }
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el =
      *fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0));
   FaceElementTransformations &Trans0 =
      *fes.GetMesh()->GetFaceElementTransformations(0);
   const IntegrationRule *ir = IntRule?
                               IntRule:
                               &GetRule(el.GetGeomType(), el.GetOrder(), Trans0);
   const int nq = ir->GetNPoints();
   auto weights = ir->GetWeights();
   dim = mesh->Dimension();
   facegeom = mesh->GetFaceGeometricFactors(
                  *ir,
                  FaceGeometricFactors::DETERMINANTS |
                  FaceGeometricFactors::NORMALS, type);

   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt; 

   auto detJ = Reshape(facegeom->detJ.Read(), nq, nf); // assumes conforming mesh

   coeff_data_1_old.SetSize( 4 * nq * nf, Device::GetMemoryType());
   coeff_data_2_old.SetSize( 2 * nq * nf, Device::GetMemoryType());
   coeff_data_3_old.SetSize( 2 * nq * nf, Device::GetMemoryType());
   coeff_data_1_old = 0.0;
   coeff_data_2_old = 0.0;
   coeff_data_3_old = 0.0;
   
   int NS = 2;

   // Input
   /*
   const bool const_Q = Q->Size() == 1;
   auto Qreshaped =
      const_Q ? Reshape(Q->Read(), 1,1) : Reshape(Q->Read(), Q1D,NF);
   */
   auto wgts = weights.Read();

   // convert Q to a vector
   Vector Qcoeff;
   if (Q==nullptr)
   {
      // Default value
      Qcoeff.SetSize(1);
      Qcoeff(0) = 1.0;
   }
   else if (ConstantCoefficient *c_Q = dynamic_cast<ConstantCoefficient*>(Q))
   {
      mfem_error("not yet implemented.");
      // Constant Coefficient
      Qcoeff.SetSize(1);
      Qcoeff(0) = c_Q->constant;
   }
   else if (QuadratureFunctionCoefficient* c_Q =
               dynamic_cast<QuadratureFunctionCoefficient*>(Q))
   {
      mfem_error("not yet implemented.");
   }
   else
   {
      mfem_error("not yet implemented.");
   }

   face_2_elem_volumes.SetSize( 2 * nf, Device::GetMemoryType());

   // Get element sizes on either side of each face
   auto f2ev = Reshape(face_2_elem_volumes.ReadWrite(), 2, nf);

   int f_ind = 0;
   // Loop over all faces
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      int e0,e1;
      int inf0, inf1;
      fes.GetMesh()->GetFaceElements(f, &e0, &e1);
      fes.GetMesh()->GetFaceInfos(f, &inf0, &inf1);

      bool int_type_match = (type==FaceType::Interior && (e1>=0 || (e1<0 && inf1>=0)));
      bool bdy_type_match = (type==FaceType::Boundary && e1<0 && inf1<0);

      if ( int_type_match )
      {
         mesh->GetFaceElements(f, &e0, &e1);
         f2ev(0,f_ind) = mesh->GetElementVolume(e0);
         f2ev(1,f_ind) = mesh->GetElementVolume(e1);
         f_ind++;
      }
      else if ( bdy_type_match )
      {
         mesh->GetFaceElements(f, &e0, &e1);
         f2ev(0,f_ind) = mesh->GetElementVolume(e0);
         f2ev(1,f_ind) = -1.0; // Not a real element
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Incorrect number of faces.");

   coeff_data_1.SetSize( 4 * nq * nf, Device::GetMemoryType());
   coeff_data_2.SetSize( 2 * nq * nf, Device::GetMemoryType());
   coeff_data_3.SetSize( 2 * nq * nf, Device::GetMemoryType());
   coeff_data_1 = 0.0;
   coeff_data_2 = 0.0;
   coeff_data_3 = 0.0;

   auto norm = Reshape(facegeom->normal.Read(), nq, dim, nf);

   auto op1 = Reshape(coeff_data_1.Write(), nq, NS, NS, nf);
   auto op2 = Reshape(coeff_data_2.Write(), nq, NS, nf);
   auto op3 = Reshape(coeff_data_3.Write(), nq, NS, nf);
   
   f_ind = 0;
   for (int face_num = 0; face_num < fes.GetNF(); ++face_num)
   {
      int e0,e1;
      int inf0, inf1;
      // Get the two elements associated with the current face
      fes.GetMesh()->GetFaceElements(face_num, &e0, &e1);
      fes.GetMesh()->GetFaceInfos(face_num, &inf0, &inf1);
      //int face_id = inf0 / 64; //I don't know what 64 is all about 
      // Act if type matches the kind of face f is

      FaceElementTransformations &Trans =
         *fes.GetMesh()->GetFaceElementTransformations(face_num);

      bool int_type_match = (type==FaceType::Interior && (e1>=0 || (e1<0 && inf1>=0)));
      bool bdy_type_match = (type==FaceType::Boundary && e1<0 && inf1<0);

      if( int_type_match || bdy_type_match )
      {
         const FiniteElement &el1 =
         *fes.GetTraceElement(e0, fes.GetMesh()->GetFaceBaseGeometry(f_ind));
         const FiniteElement &el2 =
         *fes.GetTraceElement(e1, fes.GetMesh()->GetFaceBaseGeometry(f_ind));

         int ndof1, ndof2, ndofs;
         bool kappa_is_nonzero = (kappa != 0.);
         double w, wq = 0.0;

         const int dim = fes.GetMesh()->SpaceDimension();
         ndof1 = el1.GetDof();
         nor.SetSize(dim);
         nh.SetSize(dim);
         ni.SetSize(dim);
         adjJ.SetSize(dim);
         if (MQ)
         {
            mq.SetSize(dim);
         }
         shape1.SetSize(ndof1);
         dshape1.SetSize(ndof1, dim);
         dshape1dn.SetSize(ndof1);
         if (int_type_match)
         {
            ndof2 = el2.GetDof();
            shape2.SetSize(ndof2);
            dshape2.SetSize(ndof2, dim);
            dshape2dn.SetSize(ndof2);
         }
         else
         {
            ndof2 = 0;
         }

         ndofs = ndof1 + ndof2;

         const IntegrationRule *ir = IntRule;
         if (ir == NULL)
         {
            // a simple choice for the integration order; is this OK?
            int order;
            if (int_type_match)
            {
               order = 2*std::max(el1.GetOrder(), el2.GetOrder());
            }
            else
            {
               order = 2*el1.GetOrder();
            }
            ir = &IntRules.Get(Trans.GetGeometryType(), order);
         }
         else
         {
            std::cout << "ir is sometimes defined?" << std::endl;
            std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
            //exit(1);
         }
         for (int p = 0; p < ir->GetNPoints(); p++)
         {

            const IntegrationPoint &ip = ir->IntPoint(p);
            // Set the integration point in the face and the neighboring elements
            Trans.SetAllIntPoints(&ip);
            // Access the neighboring elements' integration points
            // Note: eip2 will only contain valid data if Elem2 exists
            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();
            if (dim == 1)
            {
               nor(0) = 2*eip1.x - 1.0;
            }
            else
            {
               CalcOrtho(Trans.Jacobian(), nor);
            }
            el1.CalcShape (eip1,  shape1);
            el1.CalcDShape(eip1, dshape1);
            //w = ip.weight/Trans.Elem1->Weight();
            w = ip.weight;///Trans.Elem1->Weight();
            if (int_type_match)
            {
               w /= 2;
            }
            if (!MQ)
            {
               if (Q)
               {
                  w *= Q->Eval(*Trans.Elem1, eip1);
               }
               ni.Set(w, nor);
            }
            else
            {
               nh.Set(w, nor);
               MQ->Eval(mq, *Trans.Elem1, eip1);
               mq.MultTranspose(nh, ni);
            }
            CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
            adjJ.Mult(ni, nh);
            dshape1.Mult(nh, dshape1dn);
            if (kappa_is_nonzero)
            {
               wq = ni * nor;
            }

            op1(p,0,0,f_ind) =  beta*w*detJ(p,f_ind);
            op1(p,1,0,f_ind) = - beta*w*detJ(p,f_ind); 
            const double normx = norm(p,0,f_ind);
            const double normy = norm(p,1,f_ind);
            const double normz = ( dim == 3 ) ? norm(p,2,f_ind) : 0.0;
            const double mag_norm = sqrt(normx*normx + normy*normy + normz*normz);
            const double h0 = 1.0/mag_norm;
            op3(p,0,f_ind) = -kappa*w/h0;
            op2(p,0,f_ind) = -sigma*w*detJ(p,f_ind);

            if (int_type_match)
            {
               el2.CalcShape(eip2, shape2);
               el2.CalcDShape(eip2, dshape2);
               double t2w = 1.0;//Trans.Elem2->Weight();
               double ipw = ip.weight;
               w = ipw/2/t2w;
               if (!MQ)
               {
                  if (Q)
                  {
                     w *= Q->Eval(*Trans.Elem2, eip2);
                  }
                  ni.Set(w, nor);
               }
               else
               {
                  nh.Set(w, nor);
                  MQ->Eval(mq, *Trans.Elem2, eip2);
                  mq.MultTranspose(nh, ni);
               }
               CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
               adjJ.Mult(ni, nh);
               if (kappa_is_nonzero)
               {
                  wq += ni * nor;
               }
               const double h1 = 1.0/mag_norm;

               op1(p,0,1,f_ind) =  beta*w*detJ(p,f_ind);
               op1(p,1,1,f_ind) = - beta*w*detJ(p,f_ind);

               op2(p,1,f_ind) =  sigma*w*detJ(p,f_ind);
               op2(p,0,f_ind) = -sigma*w*detJ(p,f_ind);

               op3(p,0,f_ind) = kappa*w*(1.0/h0+1.0/h1)/2.0;
               op3(p,1,f_ind) = kappa*w*(1.0/h0+1.0/h1)/2.0;
            }
         }
         f_ind++;
      }
   }
}

void DGDiffusionIntegrator::AssemblePAInteriorFaces(const FiniteElementSpace& fes)
{
   SetupPA(fes, FaceType::Interior);
}

void DGDiffusionIntegrator::AssemblePABoundaryFaces(const FiniteElementSpace& fes)
{
   
   SetupPA(fes, FaceType::Boundary);
}

// PA DGDiffusion Apply 2D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0> static
void PADGDiffusionApply2D(const int NF,
                          const Array<double> &b,
                          const Array<double> &bt,
                          const Vector &bf,
                          const Vector &gf,
                          const Vector &_op1,
                          const Vector &_op2,
                          const Vector &_op3,
                          const Vector &_x,
                          Vector &_y,
                          const int d1d = 0,
                          const int q1d = 0)
{
   /*
   std::cout << " PA x" << std::endl;
   _x.Print(std::cout,1);
   std::cout << " end PA x" << std::endl;
   */

   // vdim is confusing as it seems to be used differently based on the context
   const int VDIM = 1;
   const int NS = 2; // number of values per face (2 for double-values faces)
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   // the following variables are evaluated at compile time
   //constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
   constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

   auto B = Reshape(b.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op1 = Reshape(_op1.Read(), Q1D, NS, NS, NF);
   auto op2 = Reshape(_op2.Read(), Q1D, NS, NF);
   auto op3 = Reshape(_op3.Read(), Q1D, NS, NF);
   auto x = Reshape(_x.Read(), D1D, VDIM, NS, NF, 2);
   auto y = Reshape(_y.ReadWrite(), D1D, VDIM, NS, NF, 2);

   // Loop over all faces
   MFEM_FORALL(f, NF,
   {
      // 1. Interpolate u and du/dn along the face
      double Bu0[max_Q1D][VDIM];
      double Bu1[max_Q1D][VDIM];      
      double BGu0[max_Q1D][VDIM];
      double BGu1[max_Q1D][VDIM];      

      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; c++)
         {
            Bu0[q][c] = 0;
            Bu1[q][c] = 0;
            BGu0[q][c] = 0;
            BGu1[q][c] = 0;
         }
         for (int d = 0; d < D1D; ++d)
         {
            const double b = B(q,d);
            for (int c = 0; c < VDIM; c++)
            {
               Bu0[q][c] += b*x(d,c,0,f,0);
               Bu1[q][c] += b*x(d,c,1,f,0);
               BGu0[q][c] += b*x(d,c,0,f,1);
               BGu1[q][c] += b*x(d,c,1,f,1);
            }
         }
      }

      // 2. Form numerical fluxes
      double D1[max_Q1D][VDIM];
      double D0[max_Q1D][VDIM];
      double D1jumpu[max_Q1D][VDIM];
      double D0jumpu[max_Q1D][VDIM];

      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; c++)
         {
            const double jump_u = Bu1[q][c] - Bu0[q][c];
            const double jump_Gu = BGu1[q][c] - BGu0[q][c];
            D0[q][c] = op1(q,0,0,f)*jump_Gu + op3(q,0,f)*jump_u;
            D1[q][c] = op1(q,1,0,f)*jump_Gu + op3(q,1,f)*jump_u;
            D0jumpu[q][c] = op2(q,0,f)*jump_u;
            D1jumpu[q][c] = op2(q,1,f)*jump_u;
         }
      }

      // 3. Integrate along faces
      for (int d = 0; d < D1D; ++d)
      {
         for (int c = 0; c < VDIM; c++)
         {
            double BD0 = 0.0;
            double BD1 = 0.0;
            double BD0jumpu = 0.0;
            double BD1jumpu = 0.0;
            for (int q = 0; q < Q1D; ++q)
            {
               const double b = Bt(d,q);
               BD0 += b*D0[q][c];
               BD1 += b*D1[q][c];
               BD0jumpu += b*D0jumpu[q][c];
               BD1jumpu += b*D1jumpu[q][c];
            }
            y(d,c,0,f,0) =  BD0;
            y(d,c,1,f,0) =  BD1;
            y(d,c,0,f,1) =  BD0jumpu;
            y(d,c,1,f,1) =  BD1jumpu;
         }
      }
   });
/*
   std::cout << " PA y" << std::endl;
   _y.Print(std::cout,1);
   std::cout << " end PA y" << std::endl;
*/

}

// PA DGDiffusion Apply 3D kernel for Gauss-Lobatto/Bernstein
template<int T_D1D = 0, int T_Q1D = 0> static
void PADGDiffusionApply3D(const int NF,
                          const Array<double> &b,
                          const Array<double> &bt,
                          const Vector &bf,
                          const Vector &gf,
                          const Vector &_op1,
                          const Vector &_op2,
                          const Vector &_op3,
                          const Vector &_x,
                          Vector &_y,
                          const int d1d = 0,
                          const int q1d = 0)
{
   std::cout << " PA x 3D" << std::endl;
   _x.Print(std::cout,1);
   std::cout << " end PA x 3D" << std::endl;

   // vdim is confusing as it seems to be used differently based on the context
   const int VDIM = 1;
   const int NS = 2; // number of values per face (2 for double-values faces)
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   // the following variables are evaluated at compile time
   constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
   constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

   auto B = Reshape(b.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto op1 = Reshape(_op1.Read(), Q1D, Q1D, NS, NS, NF);
   auto op2 = Reshape(_op2.Read(), Q1D, Q1D, NS, NF);
   auto op3 = Reshape(_op3.Read(), Q1D, Q1D, NS, NF);
   auto x = Reshape(_x.Read(), D1D, D1D, VDIM, NS, NF, 2);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, VDIM, NS, NF, 2);

   // Loop over all faces
   MFEM_FORALL(f, NF,
   {
      // 1. Interpolate u and du/dn along the face
      double Bu0[max_Q1D][max_D1D][VDIM];
      double Bu1[max_Q1D][max_D1D][VDIM];
      double BGu0[max_Q1D][max_D1D][VDIM];
      double BGu1[max_Q1D][max_D1D][VDIM];
      for (int q = 0; q < Q1D; ++q)
      {
         for (int d2 = 0; d2 < D1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {

               Bu0[q][d2][c] = 0;
               Bu1[q][d2][c] = 0;
               BGu0[q][d2][c] = 0;
               BGu1[q][d2][c] = 0;
            }      
            for (int d1 = 0; d1 < D1D; ++d1)
            {
               const double b = B(q,d1);
               for (int c = 0; c < VDIM; c++)
               {
                  Bu0[q][d2][c] += b*x(d1,d2,c,0,f,0);
                  Bu1[q][d2][c] += b*x(d1,d2,c,1,f,0);
                  BGu0[q][d2][c] += b*x(d1,d2,c,1,f,1);
                  BGu1[q][d2][c] += b*x(d1,d2,c,1,f,1);
               }
            }
         }
      }

      double BBu0[max_Q1D][max_Q1D][VDIM];
      double BBu1[max_Q1D][max_Q1D][VDIM];
      double BBGu0[max_Q1D][max_Q1D][VDIM];
      double BBGu1[max_Q1D][max_Q1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int q2 = 0; q2 < Q1D; q2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BBu0[q1][q2][c] = 0;
               BBu1[q1][q2][c] = 0;
               BBGu0[q1][q2][c] = 0;
               BBGu1[q1][q2][c] = 0;
            }
            for (int d2 = 0; d2 < D1D; ++d2)
            {
               const double b = B(q2,d2);
               for (int c = 0; c < VDIM; c++)
               {
                  BBu0[q1][q2][c] += b*Bu0[q1][d2][c];
                  BBu1[q1][q2][c] += b*Bu1[q1][d2][c];
                  BBGu0[q1][q2][c] += b*BGu0[q1][d2][c];
                  BBGu1[q1][q2][c] += b*BGu1[q1][d2][c];
               }
            }
         }
      }

      // 2. Form numerical fluxes
      double D1[max_Q1D][max_Q1D][VDIM];
      double D0[max_Q1D][max_Q1D][VDIM];
      double D1jumpu[max_Q1D][max_Q1D][VDIM];
      double D0jumpu[max_Q1D][max_Q1D][VDIM];

      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int q2 = 0; q2 < Q1D; ++q2)
         {
            for (int c = 0; c < VDIM; c++)
            {
               // need to have different op2 and op3 for each side, then use n
               const double jump_u = BBu1[q1][q2][c] - BBu0[q1][q2][c];
               const double jump_Gu = BBGu1[q1][q2][c] - BBGu0[q1][q2][c];
               D0[q1][q2][c] = op1(q1,q2,0,0,f)*jump_Gu + op3(q1,q2,0,f)*jump_u;
               D1[q1][q2][c] = op1(q1,q2,1,0,f)*jump_Gu + op3(q1,q2,1,f)*jump_u;
               D0jumpu[q1][q2][c] = op2(q1,q2,0,f)*jump_u;
               D1jumpu[q1][q2][c] = op2(q1,q2,1,f)*jump_u;
            }
         }
      }
      
      // 3. Contraction with B^T evaluation B^T:(G*D*B:u) and B^T:(D*B:Gu)   
      double BD1[max_Q1D][max_D1D][VDIM];
      double BD0[max_Q1D][max_D1D][VDIM];
      double BD1jumpu[max_Q1D][max_D1D][VDIM];
      double BD0jumpu[max_Q1D][max_D1D][VDIM];
      for (int q1 = 0; q1 < Q1D; ++q1)
      {
         for (int d = 0; d < D1D; ++d)
         {
            for (int c = 0; c < VDIM; c++)
            {
               BD0[q1][d][c] = 0;
               BD1[q1][d][c] = 0;
               BD0jumpu[q1][d][c] = 0;
               BD1jumpu[q1][d][c] = 0;
            }
            for (int q2 = 0; q2 < Q1D; ++q2)
            {
               const double b = Bt(d,q2);
               for (int c = 0; c < VDIM; c++)
               {
                  BD0[q1][d][c] += b*D0[q1][q2][c];
                  BD1[q1][d][c] += b*D1[q1][q2][c];
                  BD0jumpu[q1][d][c] += b*D0jumpu[q1][q2][c];
                  BD1jumpu[q1][d][c] += b*D1jumpu[q1][q2][c];
               }
            }
         }
      }

      for (int d1 = 0; d1 < D1D; ++d1)
      {
         for (int d2 = 0; d2 < D1D; ++d2)
         {
            for (int c = 0; c < VDIM; c++)
            {
               double BBD0 = 0;
               double BBD1 = 0;
               double BBD0jumpu = 0;
               double BBD1jumpu = 0;
               for (int q = 0; q < Q1D; ++q)
               {
                  const double b = Bt(d1,q);
                  BBD0 += b*D0[q][d2][c];
                  BBD1 += b*D1[q][d2][c];
                  BBD0jumpu += b*D0jumpu[q][d2][c];
                  BBD1jumpu += b*D1jumpu[q][d2][c];
               }
               y(d1,d2,c,0,f,0) = BBD0;
               y(d1,d2,c,1,f,0) = BBD1;
               y(d1,d2,c,0,f,1) = BBD0jumpu;
               y(d1,d2,c,1,f,1) = BBD1jumpu;
            }
         }
      }
   });

/*
   std::cout << " PA y 3D" << std::endl;
   _y.Print(std::cout,1);
   std::cout << " end PA y 3D" << std::endl;
*/

}

static void PADGDiffusionApply(const int dim,
                           const int D1D,
                           const int Q1D,
                           const int NF,
                           const Array<double> &B,
                           const Array<double> &Bt,
                           const Vector &Bf,
                           const Vector &Gf,
                           const Vector &_op1,
                           const Vector &_op2,
                           const Vector &_op3,   
                           const Vector &x,
                           Vector &y)                           
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {  
         /*
         case 0x22: return PADGDiffusionApply2D<2,2>(NF,B,Bt,op,x,y);
         case 0x33: return PADGDiffusionApply2D<3,3>(NF,B,Bt,op,x,y);
         case 0x44: return PADGDiffusionApply2D<4,4>(NF,B,Bt,op,x,y);
         case 0x55: return PADGDiffusionApply2D<5,5>(NF,B,Bt,op,x,y);
         case 0x66: return PADGDiffusionApply2D<6,6>(NF,B,Bt,op,x,y);
         case 0x77: return PADGDiffusionApply2D<7,7>(NF,B,Bt,op,x,y);
         case 0x88: return PADGDiffusionApply2D<8,8>(NF,B,Bt,op,x,y);
         case 0x99: return PADGDiffusionApply2D<9,9>(NF,B,Bt,op,x,y);
         */
         default:   return PADGDiffusionApply2D(NF,B,Bt,Bf,Gf,_op1,_op2,_op3,x,y,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         /*
         case 0x23: return SmemPADGDiffusionApply3D<2,3,1>(NF,B,Bt,op,x,y);
         case 0x34: return SmemPADGDiffusionApply3D<3,4,2>(NF,B,Bt,op,x,y);
         case 0x45: return SmemPADGDiffusionApply3D<4,5,2>(NF,B,Bt,op,x,y);
         case 0x56: return SmemPADGDiffusionApply3D<5,6,1>(NF,B,Bt,op,x,y);
         case 0x67: return SmemPADGDiffusionApply3D<6,7,1>(NF,B,Bt,op,x,y);
         case 0x78: return SmemPADGDiffusionApply3D<7,8,1>(NF,B,Bt,op,x,y);
         case 0x89: return SmemPADGDiffusionApply3D<8,9,1>(NF,B,Bt,op,x,y);
         */
         default:   return PADGDiffusionApply3D(NF,B,Bt,Bf,Gf,_op1,_op2,_op3,x,y,D1D,Q1D);
      }
   }
   MFEM_ABORT("PADGDiffusionApply not implemented for dim.");
}

// PA DGDiffusionIntegrator Apply kernel
void DGDiffusionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   PADGDiffusionApply(dim, dofs1D, quad1D, nf,
                  maps->B, maps->Bt,
                  bf, gf,
                  coeff_data_1,coeff_data_2,coeff_data_3,
                  x, y);
}

void DGDiffusionIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   MFEM_ABORT("DGDiffusionIntegrator::AddMultTransposePA not yet implemented");
}

} // namespace mfem