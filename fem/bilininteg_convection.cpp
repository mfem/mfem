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

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"

using namespace std;

namespace mfem
{

static const IntegrationRule &DefaultGetRule(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe)
{
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }
   else
   {
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }
   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

// PA Convection Integrator

// PA Convection Assemble kernel
void ConvectionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assuming the same element type
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation *T = mesh->GetElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &DefaultGetRule(el, el);
   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::COORDINATES |
                                    GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(dim*ne*nq, Device::GetMemoryType());

   if(dim==2)
   {
     const int NE = ne;
     const int NQ = nq;
     auto w = ir->GetWeights().Read();
     auto J = Reshape(geom->J.Read(), NQ,2,2,NE);
     auto v = Reshape(pa_data.Write(), 2, NQ, NE);
     const double COEFF = -1.0;

    MFEM_FORALL(e, NE,
    {
      for(int q=0; q<NQ; ++q)
      {
        const double J11 = J(q,0,0,e);
        const double J21 = J(q,1,0,e);
        const double J12 = J(q,0,1,e);
        const double J22 = J(q,1,1,e);

        const double w_coeff = w[q] * COEFF;

        double cx = 1.0;
        double cy = 5.0;
        v(0,q,e) =    w_coeff*(cx * J22 - cy * J12);
        v(1,q,e) =  - w_coeff*(cx * J21 - cy * J11);
      }
    });

   }//dim = 2

   if(dim==3)
   {
     const int NE = ne;
     const int NQ = nq;
     auto w = ir->GetWeights().Read();
     auto J = Reshape(geom->J.Read(), NQ,3,3,NE);
     auto v = Reshape(pa_data.Write(), 3, NQ, NE);
     const double COEFF = -1.0;

    MFEM_FORALL(e, NE,
    {
      for(int q=0; q<NQ; ++q) {

        const double J11 = J(q,0,0,e);
        const double J12 = J(q,0,1,e);
        const double J13 = J(q,0,2,e);

        const double J21 = J(q,1,0,e);
        const double J22 = J(q,1,1,e);
        const double J23 = J(q,1,2,e);

        const double J31 = J(q,2,0,e);
        const double J32 = J(q,2,1,e);
        const double J33 = J(q,2,2,e);

        const double A11 = (J22 * J33) - (J23 * J32);
        const double A12 = (J23 * J31) - (J21 * J33);
        const double A13 = (J21 * J32) - (J22 * J31);
        const double A21 = (J13 * J32) - (J12 * J33);
        const double A22 = (J11 * J33) - (J13 * J31);
        const double A23 = (J12 * J31) - (J11 * J32);
        const double A31 = (J12 * J23) - (J13 * J22);
        const double A32 = (J13 * J21) - (J11 * J23);
        const double A33 = (J11 * J22) - (J12 * J21);

        const double w_coeff = w[q] * COEFF;

        double cx = 1.0;
        double cy = 5.0;
        double cz = 2.5;
        v(0,q,e) = w_coeff*(cx*A11 + cy*A12 + cz*A13);
        v(1,q,e) = w_coeff*(cx*A21 + cy*A22 + cz*A23);
        v(2,q,e) = w_coeff*(cx*A31 + cy*A32 + cz*A33);
      }

    });

   }//dim = 3

}

// PA Convection Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PAConvectionApply2D(const int NE,
                        const Array<double> &b,
                        const Array<double> &g,
                        const Array<double> &bt,
                        const Array<double> &gt,
                        const Vector &_op,
                        const Vector &_x,
                        Vector &_y,
                        const int d1d = 0,
                        const int q1d = 0)
{
   const int DIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto D = Reshape(_op.Read(), DIM, Q1D, Q1D, NE);
   auto xloc = Reshape(_x.Read(), D1D, D1D, NE);
   auto yloc = Reshape(_y.ReadWrite(), D1D, D1D, NE);

   MFEM_FORALL(e, NE,
   {     
     const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
     const int Q1D = T_Q1D ? T_Q1D : q1d;
     
     // the following variables are evaluated at compile time
     constexpr int iDIM     = 2; 
     constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
     constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
     
     double U[iDIM][max_D1D][max_Q1D];
     for(int j1=0; j1<Q1D; ++j1) {
       for(int i2=0; i2<D1D; ++i2) {

         double dot0=0.0; double dot1=0.0;
         for(int i1=0; i1<D1D; ++i1){
           dot0 += G(j1,i1)*xloc(i1, i2, e);
           dot1 += B(j1,i1)*xloc(i1, i2, e);
         }
         U[0][i2][j1] = dot0;
         U[1][i2][j1] = dot1;
       }
     }

     double W[iDIM][max_Q1D][max_Q1D];
     for(int j1=0; j1<Q1D; ++j1) {
       for(int i2=0; i2<Q1D; ++i2) {

         double dot0=0.0; double dot1=0.0;
         for(int i1=0; i1<D1D; ++i1){
           dot0 += B(j1,i1)*U[0][i1][i2];
           dot1 += G(j1,i1)*U[1][i1][i2];
         }
         W[0][i2][j1] = dot0;
         W[1][i2][j1] = dot1;
       }
     }

     double Z[max_Q1D][max_Q1D];
     for(int k2=0; k2<Q1D; ++k2) {
       for(int k1=0; k1<Q1D; ++k1) {

         double dot(0.0);
         for(int c=0; c<2; ++c) {
           dot += D(c, k1, k2, e) * W[c][k1][k2];
         }
         Z[k1][k2] = dot;
       }
     }

     //Transposed contraction onward
     double Q[max_Q1D][max_D1D];
     for(int j1=0; j1<D1D; ++j1) {
       for(int i2=0; i2<Q1D; ++i2) {

         double dot(0.0);
         for(int i1=0; i1<Q1D; ++i1) {
           dot += Bt(j1, i1)*Z[i1][i2];
         }
         Q[i2][j1] = dot;
       }
     }


     //double R[MAX_D1D][MAX_D1D];
     for(int j1=0; j1<D1D; ++j1) {
       for(int i2=0; i2<D1D; ++i2) {

         double dot(0.0);
         for(int i1=0; i1<Q1D; ++i1) {
           dot += Bt(j1, i1)*Q[i1][i2];
         }
         //R[i2][j1] = dot;
         yloc(i2,j1,e) += dot;
       }
     }

   });


}


// PA Convection Apply 3D kernel
template<const int T_D1D = 0,
         const int T_Q1D = 0> static
void PAConvectionApply3D(const int NE,
                        const Array<double> &b,
                        const Array<double> &g,
                        const Array<double> &bt,
                        const Array<double> &gt,
                        const Vector &_op,
                        const Vector &_x,
                        Vector &_y,
                        int d1d = 0, int q1d = 0)
{
   const int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto D = Reshape(_op.Read(), DIM, Q1D,Q1D, Q1D, NE);
   auto xloc = Reshape(_x.Read(), D1D, D1D, D1D, NE);
   auto yloc = Reshape(_y.ReadWrite(), D1D, D1D, D1D, NE);

   MFEM_FORALL(e, NE,
   {

     //qpt x dof x dof
     double BX[MAX_D1D][MAX_D1D][MAX_Q1D];
     double GX[MAX_D1D][MAX_D1D][MAX_Q1D];

     for(int j1=0; j1<Q1D; ++j1) {
       for(int i3=0; i3<D1D; ++i3) {
         for(int i2=0; i2<D1D; ++i2) {

           BX[i2][i3][j1] = 0.0;
           GX[i2][i3][j1] = 0.0;
           for(int i1=0; i1<D1D; ++i1) {
             BX[i2][i3][j1] += B(j1, i1) * xloc(i1,i2,i3,e);
             GX[i2][i3][j1] += G(j1, i1) * xloc(i1,i2,i3,e);
           }
         }
       }
     }

     double BBX[MAX_D1D][MAX_Q1D][MAX_Q1D];
     double GBX[MAX_D1D][MAX_Q1D][MAX_Q1D];
     double BGX[MAX_D1D][MAX_Q1D][MAX_Q1D];

     for(int j1=0; j1<Q1D; ++j1) {
       for(int i3=0; i3<Q1D; ++i3) {
         for(int i2=0; i2<D1D; ++i2) {

           BBX[i2][i3][j1] = 0.0;
           GBX[i2][i3][j1] = 0.0;
           BGX[i2][i3][j1] = 0.0;
           for(int i1=0; i1<D1D; ++i1) {
             BBX[i2][i3][j1] += B(j1, i1) * BX[i1][i2][i3];
             GBX[i2][i3][j1] += G(j1, i1) * BX[i1][i2][i3];
             BGX[i2][i3][j1] += B(j1, i1) * GX[i1][i2][i3];
           }

         }
       }
     }

     double GBBX[MAX_Q1D][MAX_Q1D][MAX_Q1D];
     double BGBX[MAX_Q1D][MAX_Q1D][MAX_Q1D];
     double BBGX[MAX_Q1D][MAX_Q1D][MAX_Q1D];

     for(int j1=0; j1<Q1D; ++j1) {
       for(int i3=0; i3<Q1D; ++i3) {
         for(int i2=0; i2<Q1D; ++i2) {

           GBBX[i2][i3][j1] = 0.0;
           BGBX[i2][i3][j1] = 0.0;
           BBGX[i2][i3][j1] = 0.0;
           for(int i1=0; i1<D1D; ++i1) {
             GBBX[i2][i3][j1] += G(j1, i1) * BBX[i1][i2][i3];
             BGBX[i2][i3][j1] += B(j1, i1) * GBX[i1][i2][i3];
             BBGX[i2][i3][j1] += B(j1, i1) * BGX[i1][i2][i3];
           }

         }
       }
     }

     double Z[MAX_Q1D][MAX_Q1D][MAX_Q1D];
     for(int k3=0; k3<Q1D; ++k3) {
       for(int k2=0; k2<Q1D; ++k2) {
         for(int k1=0; k1<Q1D; ++k1) {

         double dot(0.0);
         {
           dot += D(0, k1, k2, k3, e) * BBGX[k1][k2][k3];
           dot += D(1, k1, k2, k3, e) * BGBX[k1][k2][k3];
           dot += D(2, k1, k2, k3, e) * GBBX[k1][k2][k3];
         }
         Z[k1][k2][k3] = dot;
         }
       }
     }

     //Apply (B1d)^T 3 more times
     double BZ[MAX_Q1D][MAX_Q1D][MAX_D1D];
     for(int j1=0; j1<D1D; ++j1) {
       for(int i3=0; i3<Q1D; ++i3) {
         for(int i2=0; i2<Q1D; ++i2) {

           BZ[i2][i3][j1]=0.0;
           for(int i1=0; i1<Q1D; ++i1) {
             BZ[i2][i3][j1] += Bt(j1,i1)*Z[i1][i2][i3];
           }

         }
       }
     }

     double BBZ[MAX_Q1D][MAX_D1D][MAX_D1D];
     for(int j1=0; j1<D1D; ++j1) {
       for(int i3=0; i3<D1D; ++i3) {
         for(int i2=0; i2<Q1D; ++i2) {

           BBZ[i2][i3][j1]=0.0;
           for(int i1=0; i1<Q1D; ++i1) {
             BBZ[i2][i3][j1] += Bt(j1,i1)*BZ[i1][i2][i3];
           }

         }
       }
     }

     for(int j1=0; j1<D1D; ++j1) {
       for(int i3=0; i3<D1D; ++i3) {
         for(int i2=0; i2<D1D; ++i2) {

           double dot(0.0);
           for(int i1=0; i1<Q1D; ++i1) {
             dot += Bt(j1,i1)*BBZ[i1][i2][i3];
           }
           yloc(i2,i3,j1,e) += dot;
         }
       }
     }

   });

}


static void PAConvectionApply(const int dim,
                              const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &B,
                              const Array<double> &G,
                              const Array<double> &Bt,
                              const Array<double> &Gt,
                              const Vector &op,
                              const Vector &x,
                              Vector &y)
{

  //printf("dim %d D1D %d Q1D %d \n",dim,D1D,Q1D);
  if(dim==2)
  {
    switch ((D1D << 4 ) | Q1D)
    {
         case 0x22: PAConvectionApply2D<2,2>(NE, B, G, Bt, Gt, op, x, y); break;
         case 0x33: PAConvectionApply2D<3,3>(NE, B, G, Bt, Gt, op, x, y); break;
         case 0x44: PAConvectionApply2D<4,4>(NE, B, G, Bt, Gt, op, x, y); break;
         case 0x55: PAConvectionApply2D<5,5>(NE, B, G, Bt, Gt, op, x, y); break;
         default: mfem_error("convection kernel order not supported \n"); break;
    }
    return;
  }
  if(dim == 3)
  {
    switch ((D1D << 4 ) | Q1D)
    {
        case 0x23: PAConvectionApply3D<2,3>(NE, B, G, Bt, G, op, x, y); break;
        case 0x34: PAConvectionApply3D<3,4>(NE, B, G, Bt, G, op, x, y); break;
        case 0x45: PAConvectionApply3D<4,5>(NE, B, G, Bt, G, op, x, y); break;
        case 0x56: PAConvectionApply3D<5,6>(NE, B, G, Bt, G, op, x, y); break;
         default: mfem_error("convection kernel order not supported \n"); break;
    }
    return;
  }
  MFEM_ABORT("Unknown kernel.");

}

void ConvectionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
  PAConvectionApply(dim, dofs1D, quad1D, ne,maps->B, maps->G, maps->Bt, maps->Gt, pa_data, x, y);
}

} // namespace mfem
