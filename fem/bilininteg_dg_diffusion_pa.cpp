
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

// PA DG DGDiffusion Integrator
static void PADGDiffusionSetup2D(const int Q1D,
                             const int D1D,
                             const int NF,
                             const Array<double> &weights,
                             const Array<double> &g,
                             const Array<double> &b,
                             const Vector &jac,
                             const Vector &det_jac,
                             const Vector &nor,
                             const Vector &Q,
                             const Vector &rho,
                             const Vector &vel,
                             const Vector &face_2_elem_volumes,
                             const double sigma,
                             const double kappa,
                             const double beta,
                             Vector &op1,
                             Vector &op2,
                             Vector &op3)
{
   //std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto B = Reshape(b.Read(), Q1D, D1D);
   const int VDIM = 2; // why ? 
   //auto jac = Reshape(jac.Read(), Q1D, NF); // assumes conforming mesh
   auto detJ = Reshape(det_jac.Read(), Q1D, NF); // assumes conforming mesh
   auto norm = Reshape(nor.Read(), Q1D, VDIM, NF);
   auto f2ev = Reshape(face_2_elem_volumes.Read(), 2, NF);

   // Input
   const bool const_Q = Q.Size() == 1;
   auto Qreshaped =
      const_Q ? Reshape(Q.Read(), 1,1) : Reshape(Q.Read(), Q1D,NF);
   auto wgts = weights.Read();

   // Output
   auto op_data_ptr1 = Reshape(op1.Write(), Q1D, 2, 2, NF);
   auto op_data_ptr2 = Reshape(op2.Write(), Q1D, 2, NF);
   auto op_data_ptr3 = Reshape(op3.Write(), Q1D, 2, NF);

   MFEM_FORALL(f, NF, // can be optimized with Q1D thread for NF blocks
   {
      for (int q = 0; q < Q1D; ++q)
      {
         const double normx = norm(q,0,f);
         const double normy = norm(q,1,f);
         //const double normz = norm(q,0,f) + norm(q,1,f); //todo: fix for 
         const double mag_norm = sqrt(normx*normx + normy*normy); //todo: fix for 
         const double Q = const_Q ? Qreshaped(0,0) : Qreshaped(q,f);
         const double w = wgts[q]*Q*detJ(q,f);
         // Need to correct the scaling of w to account for d/dn, etc..
         double w_o_detJ = w/detJ(q,f);

/*
         ////std::cout << "mag_norm  = " << mag_norm << std::endl;
         ////std::cout << "detJ(q,f) = " << detJ(q,f) << std::endl;
         ////std::cout << "wgts[q] = " << wgts[q] << std::endl;
         ////std::cout << "w = " << w << std::endl;
         ////std::cout << "w_o_detJ = " << w_o_detJ << std::endl;
         */
         //double n = normx+normy;
         
         if( f2ev(1,f) == -1.0 ) // Need a more standard way to detect bdr faces
         {
            // Boundary face
            // data for 1st term    - < {(Q grad(u)).n}, [v] >
            op_data_ptr1(q,0,0,f) = beta*w_o_detJ;
            op_data_ptr1(q,1,0,f) = -beta*w_o_detJ;
            op_data_ptr1(q,0,1,f) =  0.0;
            op_data_ptr1(q,1,1,f) =  0.0;
            // data for 2nd term    + sigma < [u], {(Q grad(v)).n} > 
            op_data_ptr2(q,0,f) =   -w_o_detJ*sigma;
            op_data_ptr2(q,1,f) =   0.0;
            // data for 3rd term    + kappa < {h^{-1} Q} [u], [v] >
            const double h0 = detJ(q,f)/mag_norm;
            const double h1 = detJ(q,f)/mag_norm;
            op_data_ptr3(q,0,f) =   -w*kappa/h0;
            op_data_ptr3(q,1,f) =   0.0;
            //////std::cout << "f2ev(1,f) = " << f2ev(1,f)  << std::endl;
            //////std::cout << "op3 = " << op_data_ptr3(q,0,f) <<" "<< op_data_ptr3(q,1,f) << std::endl;
         }
         else
         {
            // Interior face
            // data for 1st term    - < {(Q grad(u)).n}, [v] >
            op_data_ptr1(q,0,0,f) = beta*w_o_detJ/2.0;
            op_data_ptr1(q,1,0,f) = -beta*w_o_detJ/2.0;
            op_data_ptr1(q,0,1,f) = beta*w_o_detJ/2.0;
            op_data_ptr1(q,1,1,f) = -beta*w_o_detJ/2.0;
            // data for 2nd term    + sigma < [u], {(Q grad(v)).n} > 
            op_data_ptr2(q,0,f) =   -w_o_detJ*sigma/2.0;
            op_data_ptr2(q,1,f) =   w_o_detJ*sigma/2.0;
            // data for 3rd term    + kappa < {h^{-1} Q} [u], [v] >
            const double h0 = detJ(q,f)/mag_norm;
            const double h1 = detJ(q,f)/mag_norm;
            op_data_ptr3(q,0,f) =   -w*kappa*(1.0/h0+1.0/h1)/2.0;
            op_data_ptr3(q,1,f) =   w*kappa*(1.0/h0+1.0/h1)/2.0;
            //////std::cout << "f2ev(1,f) = " << f2ev(1,f)  << std::endl;
            //////std::cout << "op3 = " << op_data_ptr3(q,0,f) <<" "<< op_data_ptr3(q,1,f) << std::endl;
         }
      }
   });
}

static void PADGDiffusionSetup3D(const int Q1D,
                             const int D1D,
                             const int NF,
                             const Array<double> &weights,
                             const Array<double> &g,
                             const Array<double> &b,
                             const Vector &jac,
                             const Vector &det_jac,
                             const Vector &nor,
                             const Vector &Q,
                             const Vector &rho,
                             const Vector &vel,
                             const Vector &face_2_elem_volumes,
                             const double sigma,
                             const double kappa,
                             const double beta,
                             Vector &op1,
                             Vector &op2,
                             Vector &op3)
{
   mfem_error("not yet implemented.");
}

static void PADGDiffusionSetup(const int dim,
                           const int D1D,
                           const int Q1D,
                           const int NF,
                           const Array<double> &weights,
                           const Array<double> &b,
                           const Array<double> &g,
                           const Vector &bf,
                           const Vector &gf,
                           const Vector &jac,
                           const Vector &det_jac,
                           const Vector &nor,
                           const Vector &Q,
                           const Vector &rho,
                           const Vector &u,
                           const Vector &face_2_elem_volumes,
                           const double sigma,
                           const double kappa,
                           const double beta,
                           Vector &op1,
                           Vector &op2,
                           Vector &op3)
{

   //std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADGDiffusionSetup"); }
   else if (dim == 2)
   {
      PADGDiffusionSetup2D(Q1D, D1D, NF, weights, g, b, jac, det_jac, nor, Q, rho, u, face_2_elem_volumes, sigma, kappa, beta, op1, op2, op3);
   }
   else if (dim == 3)
   {
      PADGDiffusionSetup3D(Q1D, D1D, NF, weights, g, b, jac, det_jac, nor, Q, rho, u, face_2_elem_volumes, sigma, kappa, beta, op1, op2, op3);
   }
   else
   {
      MFEM_ABORT("dim > 3 not supported in PADGDiffusionSetup");     
   }
}

void DGDiffusionIntegrator::SetupPA(const FiniteElementSpace &fes, FaceType type)
{
   //std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

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
   dim = mesh->Dimension();
   facegeom = mesh->GetFaceGeometricFactors(
                  *ir,
                  FaceGeometricFactors::DETERMINANTS |
                  FaceGeometricFactors::NORMALS, type);


/*
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

      w = ip.weight/Trans.Elem1->Weight();

*/

   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt; 

   // Initialize face restriction operators
   bf.SetSize(dofs1D); 
   gf.SetSize(dofs1D); 

   // Evaluate shape function defined on [0,1] on the x=0 face in 1d
   IntegrationPoint zero;
   double zeropt[1] = {0};
   zero.Set(zeropt,1);
   const FiniteElement &elv = *fes.GetFE(0);
   elv.Calc1DShape(zero, bf, gf);
   //  {df/dn}(0) = -{df/dx}(0)   
   gf *= -1.0;

   coeff_data_1_old.SetSize( 4 * nq * nf, Device::GetMemoryType());
   coeff_data_2_old.SetSize( 2 * nq * nf, Device::GetMemoryType());
   coeff_data_3_old.SetSize( 2 * nq * nf, Device::GetMemoryType());
   coeff_data_1_old = 0.0;
   coeff_data_2_old = 0.0;
   coeff_data_3_old = 0.0;

   //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   
   int NS = 2;
   dudn_stencil.SetSize( 2 * dofs1D * nq * nf, Device::GetMemoryType());
   auto dudn = Reshape(dudn_stencil.Write(), dofs1D, nq, NS, nf);

   //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

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

   //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   
   face_2_elem_volumes.SetSize( 2 * nf, Device::GetMemoryType());

   // Get element sizes on either side of each face
   auto f2ev = Reshape(face_2_elem_volumes.ReadWrite(), 2, nf);

   int f_ind = 0;
   // Loop over all faces
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

      int e0,e1;
      int inf0, inf1;
      // Get the two elements associated with the current face
      fes.GetMesh()->GetFaceElements(f, &e0, &e1);
      fes.GetMesh()->GetFaceInfos(f, &inf0, &inf1);
      //int face_id = inf0 / 64; //I don't know what 64 is all about 
      // Act if type matches the kind of face f is

      ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

      bool int_type_match = (type==FaceType::Interior && (e1>=0 || (e1<0 && inf1>=0)));
      bool bdy_type_match = (type==FaceType::Boundary && e1<0 && inf1<0);

      ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

      if ( int_type_match )
      {
         ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
         ////std::cout << "f = " << f  << std::endl;
         ////std::cout << "e0 = " << e0 << std::endl;
         ////std::cout << "e1 = " << e1  << std::endl;
         mesh->GetFaceElements(f, &e0, &e1);
         f2ev(0,f_ind) = mesh->GetElementVolume(e0);
         f2ev(1,f_ind) = mesh->GetElementVolume(e1);
         f_ind++;
         //exit(1);
      }
      else if ( bdy_type_match )
      {
         ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
         ////std::cout << "f = " << f  << std::endl;
         ////std::cout << "e0 = " << e0 << std::endl;
         ////std::cout << "e1 = " << e1  << std::endl;
         mesh->GetFaceElements(f, &e0, &e1);
         f2ev(0,f_ind) = mesh->GetElementVolume(e0);
         f2ev(1,f_ind) = -1.0; // Not a real element
         f_ind++;
      }




   }
   MFEM_VERIFY(f_ind==nf, "Incorrect number of faces.");


   // Delete me
   double beta = 1.0; 

   //std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

   PADGDiffusionSetup(dim, dofs1D, quad1D, nf, ir->GetWeights(), 
                        maps->B, maps->Bt,
                        bf, gf,
                        facegeom->J, 
                        facegeom->detJ, 
                        facegeom->normal,
                        Qcoeff, r, vel,
                        face_2_elem_volumes,
                        sigma, kappa, beta,
                        coeff_data_1_old,coeff_data_2_old,coeff_data_3_old);


   //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   auto op1test = Reshape(coeff_data_1_old.Read(), nq, NS, NS, nf);
   auto op2test = Reshape(coeff_data_2_old.Read(), nq, NS, nf);
   auto op3test = Reshape(coeff_data_3_old.Read(), nq, NS, nf);

   double error1 = 0;
   double error2 = 0;
   double error3 = 0;

   coeff_data_1.SetSize( 4 * nq * nf, Device::GetMemoryType());
   coeff_data_2.SetSize( 2 * nq * nf, Device::GetMemoryType());
   coeff_data_3.SetSize( 2 * nq * nf, Device::GetMemoryType());
   coeff_data_1 = 0.0;
   coeff_data_2 = 0.0;
   coeff_data_3 = 0.0;

   auto op1 = Reshape(coeff_data_1.Write(), nq, NS, NS, nf);
   auto op2 = Reshape(coeff_data_2.Write(), nq, NS, nf);
   auto op3 = Reshape(coeff_data_3.Write(), nq, NS, nf);

   f_ind = 0;
   for (int face_num = 0; face_num < fes.GetNF(); ++face_num)
   {
      ////std::cout << "---------------------------------------------------------" << std::endl;

      ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
      int e0,e1;
      int inf0, inf1;
      // Get the two elements associated with the current face
      fes.GetMesh()->GetFaceElements(face_num, &e0, &e1);
      fes.GetMesh()->GetFaceInfos(face_num, &inf0, &inf1);
      //int face_id = inf0 / 64; //I don't know what 64 is all about 
      // Act if type matches the kind of face f is

      FaceElementTransformations &Trans =
         *fes.GetMesh()->GetFaceElementTransformations(face_num);

      ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

      bool int_type_match = (type==FaceType::Interior && (e1>=0 || (e1<0 && inf1>=0)));
      bool bdy_type_match = (type==FaceType::Boundary && e1<0 && inf1<0);
/*
      ////std::cout << "int_type_match  = " << int_type_match << std::endl;
      ////std::cout << "bdy_type_match  = " << bdy_type_match << std::endl;
*/
      if( int_type_match || bdy_type_match )
      {
         /*
         ////std::cout << " f_ind = " << f_ind << std::endl;
         ////std::cout << " nf = " << nf << std::endl;
         ////std::cout << " nq = " << nq << std::endl;
         */

         /*
         if( (!int_type_match) && (!bdy_type_match) )
         {
            ////std::cout << "e0 = " << e0  <<  std::endl;
            ////std::cout << "e1 = " << e1  <<  std::endl;
            ////std::cout << "inf0 = " << inf0 <<  std::endl;
            ////std::cout << "inf1 = " << inf1 <<  std::endl;
            ////std::cout << "   type==FaceType::Interior = " << bool(type==FaceType::Interior) <<  std::endl;
            ////std::cout << "   type==FaceType::Boundary = " << bool(type==FaceType::Boundary) <<  std::endl;
            exit(1);
         }
         */
         ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

         const FiniteElement &el1 =
         *fes.GetTraceElement(e0, fes.GetMesh()->GetFaceBaseGeometry(f_ind));
         const FiniteElement &el2 =
         *fes.GetTraceElement(e1, fes.GetMesh()->GetFaceBaseGeometry(f_ind));

         const FiniteElement &elf1 = *fes.GetFE(e0);
         const FiniteElement &elf2 = *fes.GetFE(e1);

         int dim, ndof1, ndof2, ndofs;
         bool kappa_is_nonzero = (kappa != 0.);
         double w, wq = 0.0;

         dim = 2;

         //////std::cout << " dim = " << dim << std::endl;
         ndof1 = el1.GetDof();
         //////std::cout << " ndof1 = " << ndof1 << std::endl;

         nor.SetSize(dim);
         nh.SetSize(dim);
         ni.SetSize(dim);
         adjJ.SetSize(dim);
         if (MQ)
         {
            mq.SetSize(dim);
         }

         //////std::cout << " e1 = " << e1 << std::endl;

         shape1.SetSize(ndof1);
         dshape1.SetSize(ndof1, dim);
         dshape1dn.SetSize(ndof1);
         if (int_type_match)
         {
            /*
            if (Trans.Elem2No < 0)
               ////std::cout << " uhhh... Trans.Elem2No =  "<< Trans.Elem2No << "   " << __LINE__ << std::endl;
            */
            ndof2 = el2.GetDof();
 //           ////std::cout << " ndof2 = " << ndof2 << std::endl;
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

         //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

         // assemble: < {(Q \nabla u).n},[v] >      --> elmat
         //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
         for (int p = 0; p < ir->GetNPoints(); p++)
         {
            /*
            ////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
            ////std::cout << " op1(p,0,0,f_ind) = " << p <<" "<< 0 <<" "<< 0 <<" "<< f_ind <<" "<< op1(p,0,0,f_ind) << std::endl;
            ////std::cout << " op1(p,1,0,f_ind) = " << p <<" "<< 1 <<" "<< 0 <<" "<< f_ind <<" "<< op1(p,1,0,f_ind) << std::endl;
            ////std::cout << " op1(p,0,1,f_ind) = " << p <<" "<< 0 <<" "<< 1 <<" "<< f_ind <<" "<< op1(p,0,1,f_ind) << std::endl;
            ////std::cout << " op1(p,1,1,f_ind) = " << p <<" "<< 1 <<" "<< 1 <<" "<< f_ind <<" "<< op1(p,1,1,f_ind) << std::endl;

            */

            ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
            const IntegrationPoint &ip = ir->IntPoint(p);

            // Set the integration point in the face and the neighboring elements
            Trans.SetAllIntPoints(&ip);

            // Access the neighboring elements' integration points
            // Note: eip2 will only contain valid data if Elem2 exists
            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

            ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

            //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

            if (dim == 1)
            {
               nor(0) = 2*eip1.x - 1.0;
            }
            else
            {
               CalcOrtho(Trans.Jacobian(), nor);
            }

            //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;


            el1.CalcShape (eip1,  shape1);
            el1.CalcDShape(eip1, dshape1);

            ////std::cout << "ip index (p) = " << p << std::endl;
/*

            ////std::cout << "shape1 " << std::endl;
            shape1.Print();
            ////std::cout << "dshape1 " << std::endl;
            dshape1.Print();
*/
            //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

            elf1.Calc1DShape(zero, bf, gf);

            gf *= -1.0;
            for( int i = 0 ; i < dofs1D ; i++ )
               dudn(i,p,1,f_ind) = gf(i);

            //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

            w = ip.weight/Trans.Elem1->Weight();

            //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

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

            //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

            //////std::cout << " Trans.Elem1->Weight() = " << Trans.Elem1->Weight() << std::endl;

            //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
            
            //////std::cout << " ip.weight = " << ip.weight << std::endl;

            dshape1.Mult(nh, dshape1dn);
            //////std::cout << " dshape1dn = " << std::endl;
            //dshape1dn.Print();

            ////std::cout << " adjJ = " << std::endl;
            //adjJ.Print();

            ////std::cout << " w = " << w << std::endl;

            ////std::cout << " nor = " << std::endl;
            //nor.Print();

            ////std::cout << " ni = " << std::endl;
            //ni.Print();

            ////std::cout << " nh = " << std::endl;
            //nh.Print();

            if (kappa_is_nonzero)
            {
               wq = ni * nor;
               ////std::cout << " wq = " << wq << std::endl;
               ////std::cout << " ni = "  << std::endl;
               //ni.Print();
               ////std::cout << " nor = "  << std::endl;
               //nor.Print();
            }

            double dnh = ni * nor;
            op1(p,0,0,f_ind) =  beta*dnh;
            op1(p,1,0,f_ind) = - beta*dnh;

            ////std::cout << " dnh = " << dnh << std::endl;
/*
            ////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
            ////std::cout << " op1(p,0,0,f_ind) = " << p <<" "<< 0 <<" "<< 0 <<" "<< f_ind <<" "<< op1(p,0,0,f_ind) << std::endl;
            ////std::cout << " op1(p,1,0,f_ind) = " << p <<" "<< 1 <<" "<< 0 <<" "<< f_ind <<" "<< op1(p,1,0,f_ind) << std::endl;
            ////std::cout << " op1(p,0,1,f_ind) = " << p <<" "<< 0 <<" "<< 1 <<" "<< f_ind <<" "<< op1(p,0,1,f_ind) << std::endl;
            ////std::cout << " op1(p,1,1,f_ind) = " << p <<" "<< 1 <<" "<< 1 <<" "<< f_ind <<" "<< op1(p,1,1,f_ind) << std::endl;
*/

            //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

            if (int_type_match)
            {
               ////std::cout << " interior match..." << std::endl;

               //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

               el2.CalcShape(eip2, shape2);

               //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

               el2.CalcDShape(eip2, dshape2);

               //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

               elf2.Calc1DShape(zero, bf, gf);
               gf *= -1.0;
               for( int i = 0 ; i < dofs1D ; i++ )
                  dudn(i,p,1,f_ind) = gf(i);

               //el2.CalcShape(eip2, shape2);

               ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

               //el2.CalcDShape(eip2, dshape2);

               //elf2.Calc1DShape()
               //dudn()

               //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

               double t2w = Trans.Elem2->Weight();

               //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

               double ipw = ip.weight;

               //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

               w = ipw/2/t2w;

               //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

               if (!MQ)
               {
                  if (Q)
                  {
                     //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
                     w *= Q->Eval(*Trans.Elem2, eip2);
                  }
                  //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
                  ni.Set(w, nor);
               }
               else
               {
                  //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
                  nh.Set(w, nor);
                  MQ->Eval(mq, *Trans.Elem2, eip2);
                  mq.MultTranspose(nh, ni);
               }

               //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
               CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);

               //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

               adjJ.Mult(ni, nh);

               if (kappa_is_nonzero)
               {
                  wq += ni * nor;
                  ////std::cout << " wq = " << wq << std::endl;
                  ////std::cout << " ni = "  << std::endl;
                  //ni.Print();
                  ////std::cout << " nor = "  << std::endl;
                  //nor.Print();
                  ////std::cout << " wq*kappa = " << wq*kappa << std::endl;
               }

               ////std::cout << " adjJ = " << std::endl;
               //adjJ.Print();

               ////std::cout << " w = " << w << std::endl;

               ////std::cout << " nor = " << std::endl;
               //nor.Print();

               ////std::cout << " ni = " << std::endl;
               //ni.Print();

               ////std::cout << " nh = " << std::endl;
               //nh.Print();

               double dnh = ni * nor;
               op1(p,0,1,f_ind) =  beta*dnh;
               op1(p,1,1,f_ind) = - beta*dnh;


               ////std::cout << " dnh = " << dnh << std::endl;
/*
               ////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
               ////std::cout << " op1(p,0,0,f_ind) = " << p <<" "<< 0 <<" "<< 0 <<" "<< f_ind <<" "<< op1(p,0,0,f_ind) << std::endl;
               ////std::cout << " op1(p,1,0,f_ind) = " << p <<" "<< 1 <<" "<< 0 <<" "<< f_ind <<" "<< op1(p,1,0,f_ind) << std::endl;
               ////std::cout << " op1(p,0,1,f_ind) = " << p <<" "<< 0 <<" "<< 1 <<" "<< f_ind <<" "<< op1(p,0,1,f_ind) << std::endl;
               ////std::cout << " op1(p,1,1,f_ind) = " << p <<" "<< 1 <<" "<< 1 <<" "<< f_ind <<" "<< op1(p,1,1,f_ind) << std::endl;
*/
               op2(p,1,f_ind) = dnh*sigma;

               ////std::cout << " op2(p,1,f_ind) = " << p <<" "<< 1 <<" "<< f_ind <<" "<< op2(p,1,f_ind) << std::endl;

               ////std::cout << " wq*kappa = " << wq*kappa << std::endl;
               op3(p,1,f_ind) = wq*kappa;
            }
            ////std::cout << " wq*kappa = " << wq*kappa << std::endl;
            op3(p,0,f_ind) = -wq*kappa;

            op2(p,0,f_ind) = -dnh*sigma;

/*
            double bsign = 1;
            double ssign = 1;
            double ksign = 1;

            op1(p,0,0,f_ind) = bsign*op1(p,0,0,f_ind);
            op1(p,1,0,f_ind) = bsign*op1(p,1,0,f_ind);
            op1(p,0,1,f_ind) = bsign*op1(p,0,1,f_ind);
            op1(p,1,1,f_ind) = bsign*op1(p,1,1,f_ind);
            op2(p,0,f_ind) = ssign*op2(p,0,f_ind);
            op2(p,1,f_ind) = ssign*op2(p,1,f_ind);
            op3(p,0,f_ind) = ksign*op3(p,0,f_ind);
            op3(p,1,f_ind) = ksign*op3(p,1,f_ind);
            */

            ////std::cout << " op2(p,0,f_ind) = " << p <<" "<< 0 <<" "<< f_ind <<" "<< op2(p,0,f_ind) << std::endl;


            //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
            /*
            if( f2ev(1,f) == -1.0 ) // Need a more standard way to detect bdr faces
            {
               // Boundary face
               // data for 1st term    - < {(Q grad(u)).n}, [v] >
               op1(q,0,0,f) = beta*w_o_detJ;
               op1(q,1,0,f) = -beta*w_o_detJ;
               op1(q,0,1,f) =  0.0;
               op1(q,1,1,f) =  0.0;
               // data for 2nd term    + sigma < [u], {(Q grad(v)).n} > 
               op2(q,0,f) =   -w_o_detJ*sigma;
               op2(q,1,f) =   0.0;
               // data for 3rd term    + kappa < {h^{-1} Q} [u], [v] >
               const double h0 = detJ(q,f)/mag_norm;
               const double h1 = detJ(q,f)/mag_norm;
               op3(q,0,f) =   -w*kappa/h0;
               op3(q,1,f) =   0.0;
            }
            else
            {
               // Interior face
               // data for 1st term    - < {(Q grad(u)).n}, [v] >
               op1(q,0,0,f) = w_o_detJ/2.0;
               op1(q,1,0,f) = -w_o_detJ/2.0;
               op1(q,0,1,f) = w_o_detJ/2.0;
               op1(q,1,1,f) = -w_o_detJ/2.0;
               // data for 2nd term    + sigma < [u], {(Q grad(v)).n} > 
               op2(q,0,f) =   -w_o_detJ*sigma/2.0;
               op2(q,1,f) =   w_o_detJ*sigma/2.0;
               // data for 3rd term    + kappa < {h^{-1} Q} [u], [v] >
               const double h0 = detJ(q,f)/mag_norm;
               const double h1 = detJ(q,f)/mag_norm;
               op3(q,0,f) =   -wq*kappa*(1.0/h0+1.0/h1)/2.0;
               op3(q,1,f) =   wq*kappa*(1.0/h0+1.0/h1)/2.0;
            }
            */
            ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
         }

         ////std::cout << " pa cell " << std::endl;         
         //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

         f_ind++;
      }


/*      
      //std::cout << "p side1 side2 f2 op12test op1 sanity check ----------" << std::endl;
      double err = 0;
      for( int f2 = 0; f2 <= f_ind-1 ; f2++ )
      for( int iq = 0; iq < nq ; iq++ )
      for( int side2 = 0; side2 < 2 ; side2++ )
         for( int side1 = 0; side1 < 2 ; side1++ )
            {
               double err1 = std::abs( op1test(iq,side1,side2,f2) - op1(iq,side1,side2,f2) ); 
               err = std::max( err,err1);
               if( (err1 > 1.0e-14 ) || (f2 == f_ind-1))
               {
                  //std::cout << iq <<" "<< side1 <<" "<< side2 <<" "<< f2  ;
                  //std::cout <<" "<< op1test(iq,side1,side2,f2) <<" "<< op1(iq,side1,side2,f2)
                           <<" "<< err << std::endl;
               }
            }


      //std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

      if( err > 1.0e-14 )
      {
         exit(1);
      }
      
      //std::cout << "p side1 f2 op1test op2 sanity check ----------" << std::endl;
      err = 0;
      for( int f2 = 0; f2 <= f_ind-1 ; f2++ )
      for( int iq = 0; iq < nq ; iq++ )
      for( int side1 = 0; side1 < 2 ; side1++ )
         {
            double err1 = std::abs( op2test(iq,side1,f2) - op2(iq,side1,f2) ); 
            err = std::max( err,err1);
            if( (err1 > 1.0e-14 ) || (f2 == f_ind-1))
            {
               //std::cout << iq <<" "<< side1 <<" "<< f2  ;
               //std::cout <<" "<< op2test(iq,side1,f2) <<" "<< op2(iq,side1,f2)
                        << std::endl;
            }
         }


      //std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
      if( err > 1.0e-14 )
      {
         exit(1);
      }
      */
      //exit(1);
   }
   ////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   
   //////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   /*
   ////std::cout << "f iq side1 op2test(iq,side1,f) op2(iq,side1,f) op3test(iq,side1,f) op3(iq,side1,f) "<< std::endl;
   for( int f = 0; f < nf ; f++ )
      for( int iq = 0; iq < nq ; iq++ )
         for( int side1 = 0; side1 < 2 ; side1++ )
         {
            double err2 = std::abs( op2test(iq,side1,f) - op2(iq,side1,f) ); 
            error2 += err2;
            double err3 = std::abs( op3test(iq,side1,f) - op3(iq,side1,f) ); 
            error3 += err3;
            if( err2 + err3 > 1.0e-14 )
            {
               ////std::cout << f <<" "<< iq <<" "<< side1 <<" " ;
               ////std::cout <<" "<< op2test(iq,side1,f) <<" "<< op2(iq,side1,f)
                        <<" "<< op3test(iq,side1,f) <<" "<< op3(iq,side1,f)
                        << std::endl;
            }
         }

   ////std::cout << "iq side1 side2 f op1test(iq,side1,f) op1(iq,side1,f) err1"<< std::endl;
   for( int f = 0; f < nf ; f++ )
   {
      for( int side2 = 0; side2 < 2 ; side2++ )
         for( int side1 = 0; side1 < 2 ; side1++ )
            for( int iq = 0; iq < nq ; iq++ )
            {
               double err1 = std::abs( op1test(iq,side1,side2,f) - op1(iq,side1,side2,f) ); 
               error1 += err1;
               if( err1 > 1.0e-14 )
               {
                  ////std::cout << iq <<" "<< side1 <<" "<< side2 <<" "<< f  ;
                  ////std::cout <<" "<< op1test(iq,side1,side2,f) <<" "<< op1(iq,side1,side2,f)
                            <<" "<< err1 
                           << std::endl;
               }
            }
   }

   ////std::cout << "error1 = " << error1 << std::endl;
   ////std::cout << "error2 = " << error2 << std::endl;
   ////std::cout << "error3 = " << error3 << std::endl;
   ////std::cout << error3  << std::endl;
   exit(1);
   */
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
   //std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

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
   auto Bf = Reshape(bf.Read(), D1D);
   auto Gf = Reshape(gf.Read(), D1D);
   auto op1 = Reshape(_op1.Read(), Q1D, NS, NS, NF);
   auto op2 = Reshape(_op2.Read(), Q1D, NS, NF);
   auto op3 = Reshape(_op3.Read(), Q1D, NS, NF);
   auto x = Reshape(_x.Read(), D1D, D1D, VDIM, NS, NF, 2);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, VDIM, NS, NF, 2);

   // Optimization: Pick "max" D1Dbf/D1Dgf if B0/G0 is sparse
   /*
   int short_D1Dbf = 0;
   for(int i = 0; i < Q1D ; i++)
      if( abs(Bf[i]) > 1.0e-17  )
         short_D1Dbf = i+1;
   int short_D1Dgf = 0;
   for(int i = 0; i < Q1D ; i++)
      if( abs(Gf[i]) > 1.0e-17  )
         short_D1Dgf = i+1;
         */

//std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

   // Loop over all faces
   MFEM_FORALL(f, NF,
   {
      // 1. Evaluation of solution and normal derivative on the faces
      double u0[max_D1D][VDIM] = {0};
      double u1[max_D1D][VDIM] = {0};
      double Gu0[max_D1D][VDIM] = {0};
      double Gu1[max_D1D][VDIM] = {0};

      //std::cout << "q g x Gu0[d][c]"  << std::endl;

      for (int d = 0; d < D1D; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            /*
            u0[d][c] += x(0,d,c,0,f,0);
            Gu0[d][c] += x(0,d,c,0,f,1);
            u1[d][c] += x(0,d,c,1,f,0);          
            Gu1[d][c] += x(0,d,c,1,f,1);
            */
/*
            u0[d][c] += x(0,d,c,0,f,0);
            u1[d][c] += x(0,d,c,1,f,0);          
            Gu0[d][c] += x(0,d,c,0,f,1);
            Gu1[d][c] += x(0,d,c,1,f,1);
            */
            //for (int q = 0; q < short_D1Dbf; q++)

         
            for (int q = 0; q < D1D; q++)
            {
               // Evaluate u on the face from each side
               const double b = Bf[q];
               u0[d][c] += b*x(q,d,c,0,f,0);
               u1[d][c] += b*x(q,d,c,1,f,0);               
            }
            //for (int q = 0; q < short_D1Dgf; q++)

            for (int q = 0; q < D1D; q++)
            {  
               // Evaluate du/dn on the face from each side
               // Uses a stencil inside 
               const double g = Gf[q];
               Gu0[d][c] += g*x(q,d,c,0,f,0);
               Gu1[d][c] += g*x(q,d,c,1,f,0);
               //std::cout << q << " " << d << " " << f << " " << g << " " << x(q,d,c,1,f,0) << Gu1[d][c] << std::endl;

            }

         }
         //std::cout << d << " " << f << " " <<  Gu1[d][0] << std::endl;
      }

      // 2. Interpolate u and du/dn along the face
      //    Bu = B:u, and Gu = G:u    
      double Bu0[max_Q1D][VDIM] = {0};
      double Bu1[max_Q1D][VDIM] = {0};      
      double BGu0[max_Q1D][VDIM] = {0};
      double BGu1[max_Q1D][VDIM] = {0};      

      for (int q = 0; q < Q1D; ++q)
      {
         for (int d = 0; d < D1D; ++d)
         {
            const double b = B(q,d);
            for (int c = 0; c < VDIM; c++)
            {
               Bu0[q][c] += b*u0[d][c];
               Bu1[q][c] += b*u1[d][c];
               BGu0[q][c] += b*Gu0[d][c];
               BGu1[q][c] += b*Gu1[d][c];
            }
         }
      }

      // 3. Form numerical fluxes
      double D1[max_Q1D][VDIM] = {0};
      double D0[max_Q1D][VDIM] = {0};
      double D1jumpu[max_Q1D][VDIM] = {0};
      double D0jumpu[max_Q1D][VDIM] = {0};

      for (int q = 0; q < Q1D; ++q)
      {
         for (int c = 0; c < VDIM; c++)
         {
            // need to have different op2 and op3 for each side, then use n
            const double jump_u = Bu1[q][c] - Bu0[q][c];
            const double jump_Gu = BGu1[q][c] - BGu0[q][c];
            D0[q][c] = op1(q,0,0,f)*jump_Gu + op3(q,0,f)*jump_u;
            D1[q][c] = op1(q,1,0,f)*jump_Gu + op3(q,1,f)*jump_u;
            D0jumpu[q][c] = op2(q,0,f)*jump_u;
            D1jumpu[q][c] = op2(q,1,f)*jump_u;
         }
      }
      
      // 4. Contraction with B^T evaluation B^T:(G*D*B:u) and B^T:(D*B:Gu)   
      double BD1[max_D1D][VDIM] = {0};
      double BD0[max_D1D][VDIM] = {0};
      double BD1jumpu[max_D1D][VDIM] = {0};
      double BD0jumpu[max_D1D][VDIM] = {0};

      for (int d = 0; d < D1D; ++d)
      {
         for (int q = 0; q < Q1D; ++q)
         {
            const double b = Bt(d,q);
            for (int c = 0; c < VDIM; c++)
            {
               BD0[d][c] += b*D0[q][c];
               BD1[d][c] += b*D1[q][c];
               BD0jumpu[d][c] += b*D0jumpu[q][c];
               BD1jumpu[d][c] += b*D1jumpu[q][c];
            }
         }
      }

      // 5. Add to y      
      for (int c = 0; c < VDIM; c++)
      {
         for (int d = 0; d < D1D; ++d)
         {
//            for (int q = 0; q < short_D1Dbf ; q++)
            for (int q = 0; q < D1D ; q++)
            {
               const double b = Bf[q];
               y(q,d,c,0,f,0) +=  b*BD0[d][c];
               y(q,d,c,1,f,0) +=  b*BD1[d][c];
            }
//            for (int q = 0; q < short_D1Dgf ; q++)
            for (int q = 0; q < D1D ; q++)
            {
               const double g = Gf[q];
               y(q,d,c,0,f,0) +=  g*BD0jumpu[d][c];
               y(q,d,c,1,f,0) +=  g*BD1jumpu[d][c];
            }
         }
      }
   });// done with the loop over all faces

   //exit(1);
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
   MFEM_ABORT("PADGDiffusionApply3D not implemented.");
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
   //std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

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

/*
static void PADGDiffusionApplyTranspose(const int dim,
                                    const int D1D,
                                    const int Q1D,
                                    const int NF,
                                    const Array<double> &B,
                                    const Array<double> &Bt,
                                    const Array<double> &G,
                                    const Array<double> &Gt,
                                    const Vector &op,
                                    const Vector &x,
                                    Vector &y)
{
   //////////std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;
   //////std::cout << "TODO: Correct this for DG diffusion" << std::endl;

   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         default: return PADGDiffusionApplyTranspose2D(NF,B,Bt,op,x,y,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {

         default: return PADGDiffusionApplyTranspose3D(NF,B,Bt,op,x,y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}
*/

// PA DGDiffusionIntegrator Apply kernel
void DGDiffusionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   //std::cout << __LINE__ << " in " << __FUNCTION__ << " in " << __FILE__ << std::endl;

   PADGDiffusionApply(dim, dofs1D, quad1D, nf,
                  maps->B, maps->Bt,
                  bf, gf,
                  coeff_data_1,coeff_data_2,coeff_data_3,
                  x, y);
}

void DGDiffusionIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   MFEM_ABORT("DGDiffusionIntegrator::AddMultTransposePA not yet implemented");
   /*
   PADGDiffusionApplyTranspose( ... );
                           */
}

} // namespace mfem