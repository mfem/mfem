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
#include "libceed/mass.hpp"

using namespace std;

namespace mfem
{

// PA H(curl) Mass Assemble 2D kernel
static void PAHcurlSetup2D(const int Q1D,
                           const int NE,
                           const Array<double> &w,
                           const Vector &j,
                           Vector &_coeff,
                           Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();

   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, 3, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double c_detJ = W[q] * coeff(q, e) / ((J11*J22)-(J21*J12));
         y(q,0,e) =  c_detJ * (J12*J12 + J22*J22); // 1,1
         y(q,1,e) = -c_detJ * (J12*J11 + J22*J21); // 1,2
         y(q,2,e) =  c_detJ * (J11*J11 + J21*J21); // 2,2
      }
   });
}

// PA H(curl) Mass Assemble 3D kernel
static void PAHcurlSetup3D(const int Q1D,
                           const int NE,
                           const Array<double> &w,
                           const Vector &j,
                           Vector &_coeff,
                           Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, 6, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
         /* */               J21 * (J12 * J33 - J32 * J13) +
         /* */               J31 * (J12 * J23 - J22 * J13);
         const double c_detJ = W[q] * coeff(q, e) / detJ;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J32 * J13) - (J12 * J33);
         const double A13 = (J12 * J23) - (J22 * J13);
         const double A21 = (J31 * J23) - (J21 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J21 * J13) - (J11 * J23);
         const double A31 = (J21 * J32) - (J31 * J22);
         const double A32 = (J31 * J12) - (J11 * J32);
         const double A33 = (J11 * J22) - (J12 * J21);
         // detJ J^{-1} J^{-T} = (1/detJ) adj(J) adj(J)^T
         y(q,0,e) = c_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
         y(q,1,e) = c_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
         y(q,2,e) = c_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
         y(q,3,e) = c_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
         y(q,4,e) = c_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
         y(q,5,e) = c_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
      }
   });
}

void VectorFEMassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fel = fes.GetFE(0);

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*el, *el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());

   Vector coeff(ne * nq);
   coeff = 1.0;
   if (Q)
   {
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
	   coeff[p + (e * nq)] = Q->Eval(*tr, ir->IntPoint(p));
         }
      }
   }

   if (el->GetDerivType() == mfem::FiniteElement::CURL && dim == 3)
   {
      PAHcurlSetup3D(quad1D, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else if (el->GetDerivType() == mfem::FiniteElement::CURL && dim == 2)
   {
      PAHcurlSetup2D(quad1D, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

static void PAHcurlMassApply2D(const int D1D,
                               const int Q1D,
                               const int NE,
                               const Array<double> &_Bo,
                               const Array<double> &_Bc,
                               const Array<double> &_Bot,
                               const Array<double> &_Bct,
                               const Vector &_op,
                               const Vector &_x,
                               Vector &_y)
{
   constexpr static int VDIM = 2;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 3, NE);
   auto x = Reshape(_x.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][VDIM];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               mass[qy][qx][c] = 0.0;
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            double massX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               massX[qx] = 0.0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qy][qx][c] += massX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double O11 = op(qx,qy,0,e);
            const double O12 = op(qx,qy,1,e);
            const double O22 = op(qx,qy,2,e);
            const double massX = mass[qy][qx][0];
            const double massY = mass[qy][qx][1];
            mass[qy][qx][0] = (O11*massX)+(O12*massY);
            mass[qy][qx][1] = (O12*massX)+(O22*massY);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            double massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

static void PAHcurlMassAssembleDiagonal2D(const int D1D,
                                          const int Q1D,
                                          const int NE,
                                          const Array<double> &_Bo,
                                          const Array<double> &_Bc,
                                          const Vector &_op,
                                          Vector &_diag)
{
   constexpr static int VDIM = 2;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 3, NE);
   auto diag = Reshape(_diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         double mass[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               mass[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);

                  mass[qx] += wy * wy * ((c == 0) ? op(qx,qy,0,e) : op(qx,qy,2,e));
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  diag(dx + (dy * D1Dx) + osc, e) += mass[qx] * wx * wx;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   }); // end of element loop
}

static void PAHcurlMassAssembleDiagonal3D(const int D1D,
                                          const int Q1D,
                                          const int NE,
                                          const Array<double> &_Bo,
                                          const Array<double> &_Bc,
                                          const Vector &_op,
                                          Vector &_diag)
{
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto diag = Reshape(_diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         const int opc = (c == 0) ? 0 : ((c == 1) ? 3 : 5);

         double mass[MAX_Q1D];

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qx] = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);

                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        const double wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);

                        mass[qx] += wy * wy * wz * wz * op(qx,qy,qz,opc,e);
                     }
                  }
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                     diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += mass[qx] * wx * wx;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

void VectorFEMassIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (dim == 3)
      PAHcurlMassAssembleDiagonal3D(dofs1D, quad1D, ne, 
                                    mapsO->B, mapsC->B, pa_data, diag);
   else
      PAHcurlMassAssembleDiagonal2D(dofs1D, quad1D, ne, 
                                    mapsO->B, mapsC->B, pa_data, diag);
}

static void PAHcurlMassApply3D(const int D1D,
                               const int Q1D,
                               const int NE,
                               const Array<double> &_Bo,
                               const Array<double> &_Bc,
                               const Array<double> &_Bot,
                               const Array<double> &_Bct,
                               const Vector &_op,
                               const Vector &_x,
                               Vector &_y)
{
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double massXY[MAX_Q1D][MAX_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massXY[qy][qx] = 0.0;
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double massX[MAX_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  massX[qx] = 0.0;
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     massX[qx] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = massX[qx];
                     massXY[qy][qx] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     mass[qz][qy][qx][c] += massXY[qy][qx] * wz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);
               const double massX = mass[qz][qy][qx][0];
               const double massY = mass[qz][qy][qx][1];
               const double massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[MAX_D1D][MAX_D1D];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

void VectorFEMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      PAHcurlMassApply3D(dofs1D, quad1D, ne, mapsO->B, mapsC->B, mapsO->Bt,
                         mapsC->Bt, pa_data, x, y);
   }
   else
   {
      PAHcurlMassApply2D(dofs1D, quad1D, ne, mapsO->B, mapsC->B, mapsO->Bt,
                         mapsC->Bt, pa_data, x, y);
   }
}

// PA H(curl) curl-curl assemble 2D kernel
static void PACurlCurlSetup2D(const int Q1D,
                              const int NE,
                              const Array<double> &w,
                              const Vector &j,
                              Vector &_coeff,
                              Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double detJ = (J11*J22)-(J21*J12);
         y(q,e) = W[q] * coeff(q,e) / detJ;
      }
   });
}

// PA H(curl) curl-curl assemble 3D kernel
static void PACurlCurlSetup3D(const int Q1D,
                              const int NE,
                              const Array<double> &w,
                              const Vector &j,
                              Vector &_coeff,
                              Vector &op,
                              Vector &op2)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto coeff = Reshape(_coeff.Read(), NQ, NE);
   auto y = Reshape(op.Write(), NQ, 10, NE);
   auto y2 = Reshape(op2.Write(), NQ, 6, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
         /* */               J21 * (J12 * J33 - J32 * J13) +
         /* */               J31 * (J12 * J23 - J22 * J13);
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J32 * J13) - (J12 * J33);
         const double A13 = (J12 * J23) - (J22 * J13);
         const double A21 = (J31 * J23) - (J21 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J21 * J13) - (J11 * J23);
         const double A31 = (J21 * J32) - (J31 * J22);
         const double A32 = (J31 * J12) - (J11 * J32);
         const double A33 = (J11 * J22) - (J12 * J21);
         // J^{-1} = (1/detJ) adj(J)
         y(q,0,e) = A11 / detJ;
         y(q,1,e) = A12 / detJ;
         y(q,2,e) = A13 / detJ;
         y(q,3,e) = A21 / detJ;
         y(q,4,e) = A22 / detJ;
         y(q,5,e) = A23 / detJ;
         y(q,6,e) = A31 / detJ;
         y(q,7,e) = A32 / detJ;
         y(q,8,e) = A33 / detJ;
         y(q,9,e) = W[q] * coeff(q,e) * detJ;

         // set y2 to the 6 entries of J^T J / det^2
         const double c_detJ = W[q] * coeff(q,e) / detJ;

         y2(q,0,e) = c_detJ * (J11*J11 + J21*J21 + J31*J31); // 1,1
         y2(q,1,e) = c_detJ * (J11*J12 + J21*J22 + J31*J32); // 1,2
         y2(q,2,e) = c_detJ * (J11*J13 + J21*J23 + J31*J33); // 1,3
         y2(q,3,e) = c_detJ * (J12*J12 + J22*J22 + J32*J32); // 2,2
         y2(q,4,e) = c_detJ * (J12*J13 + J22*J23 + J32*J33); // 2,3
         y2(q,5,e) = c_detJ * (J13*J13 + J23*J23 + J33*J33); // 3,3
      }
   });
}

void CurlCurlIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fel = fes.GetFE(0);

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*el, *el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   const int ndata = (dim == 2) ? 1 : 10;
   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   Vector coeff(ne * nq);
   coeff = 1.0;
   if (Q)
   {
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
	    coeff[p + (e * nq)] = Q->Eval(*tr, ir->IntPoint(p));
         }
      }
   }

   if (el->GetDerivType() == mfem::FiniteElement::CURL && dim == 3)
   {
      pa_data_2.SetSize(6 * nq * ne, Device::GetMemoryType());

      PACurlCurlSetup3D(quad1D, ne, ir->GetWeights(), geom->J,
                        coeff, pa_data, pa_data_2);
   }
   else if (el->GetDerivType() == mfem::FiniteElement::CURL && dim == 2)
   {
      PACurlCurlSetup2D(quad1D, ne, ir->GetWeights(), geom->J,
                        coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

static void PACurlCurlApply2D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &_Bo,
                              const Array<double> &_Bot,
                              const Array<double> &_Gc,
                              const Array<double> &_Gct,
                              const Vector &_op,
                              const Vector &_x,
                              Vector &_y)
{
   constexpr static int VDIM = 2;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), 2*(D1D-1)*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double curl[MAX_Q1D][MAX_Q1D];

      // curl[qy][qx] will be computed as du_y/dx - du_x/dy

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            curl[qy][qx] = 0;
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            double gradX[MAX_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx] = 0;
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               const double t = x(dx + (dy * D1Dx) + osc, e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx] += t * ((c == 0) ? Bo(qx,dx) : Gc(qx,dx));
               }
            }

            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = (c == 0) ? -Gc(qy,dy) : Bo(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  curl[qy][qx] += gradX[qx] * wy;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop (c) over components

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            curl[qy][qx] *= op(qx,qy,e);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            double gradX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               gradX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  gradX[dx] += curl[qy][qx] * ((c == 0) ? Bot(dx,qx) : Gct(dx,qx));
               }
            }
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const double wy = (c == 0) ? -Gct(dy,qy) : Bot(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += gradX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }  // loop qy
   }); // end of element loop
}

static void PACurlCurlApply3D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &_Bo,
                              const Array<double> &_Bc,
                              const Array<double> &_Bot,
                              const Array<double> &_Bct,
                              const Array<double> &_Go,
                              const Array<double> &_Gc,
                              const Array<double> &_Got,
                              const Array<double> &_Gct,
                              const Vector &_op,
                              const Vector &_x,
                              Vector &_y)
{
   // Note that _Go and _Got are never actually used. They are used in the diagonal of the gradient, which is not used in the curl.
   // This implementation is based on the identity [\nabla\times u] F = dF^{-T} [\hat{\nabla}\times\hat{u}] dF^{-1} (p. 77 of Monk).
   // It may have been simpler to use the identity (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk).

   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto Go = Reshape(_Go.Read(), Q1D, D1D-1);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto Got = Reshape(_Got.Read(), D1D-1, Q1D);
   auto Gct = Reshape(_Gct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 10, NE);
   auto x = Reshape(_x.Read(), 3*(D1D-1)*D1D*D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   int idJ[3][3];

   idJ[0][0] = 0;
   idJ[0][1] = 1;
   idJ[0][2] = 2;
   idJ[1][0] = 3;
   idJ[1][1] = 4;
   idJ[1][2] = 5;
   idJ[2][0] = 6;
   idJ[2][1] = 7;
   idJ[2][2] = 8;

   MFEM_FORALL(e, NE,
   {
      double grad[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM][VDIM];

      // grad[qz][qy][qx][c][d] will be computed as the partial derivative of component c with respect to spatial variable d.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  for (int d = 0; d < VDIM; ++d)
                  {
                     grad[qz][qy][qx][c][d] = 0.0;
                  }
               }
            }
         }
      }

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         for (int dz = 0; dz < D1Dz; ++dz)
         {
            double gradXY[MAX_Q1D][MAX_Q1D][3];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 3; ++d)
                  {
                     gradXY[qy][qx][d] = 0.0;
                  }
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               double gradX[MAX_Q1D][2];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int d = 0; d < 2; ++d)
                  {
                     gradX[qx][d] = 0.0;
                  }
               }

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  const double t = x(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     gradX[qx][0] += t * ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                     gradX[qx][1] += t * ((c == 0) ? Go(qx,dx) : Gc(qx,dx));
                  }
               }

               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : Bc(qy,dy);
                  const double wDy = (c == 1) ? Go(qy,dy) : Gc(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = gradX[qx][0];
                     const double wDx = gradX[qx][1];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx * wDy;
                     gradXY[qy][qx][2] += wx * wy;
                  }
               }
            }

            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = (c == 2) ? Bo(qz,dz) : Bc(qz,dz);
               const double wDz = (c == 2) ? Go(qz,dz) : Gc(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     grad[qz][qy][qx][c][0] += gradXY[qy][qx][0] * wz;
                     grad[qz][qy][qx][c][1] += gradXY[qy][qx][1] * wz;
                     grad[qz][qy][qx][c][2] += gradXY[qy][qx][2] * wDz;
                  }
               }
            }
         }

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop (c) over components

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               double curlRef[3][3];
               double invJ[3][3];

               // op stores the entries of J^{-1} and det.

               invJ[0][0] = op(qx,qy,qz,0,e);
               invJ[0][1] = op(qx,qy,qz,1,e);
               invJ[0][2] = op(qx,qy,qz,2,e);
               invJ[1][0] = op(qx,qy,qz,3,e);
               invJ[1][1] = op(qx,qy,qz,4,e);
               invJ[1][2] = op(qx,qy,qz,5,e);
               invJ[2][0] = op(qx,qy,qz,6,e);
               invJ[2][1] = op(qx,qy,qz,7,e);
               invJ[2][2] = op(qx,qy,qz,8,e);

               const double det = op(qx,qy,qz,9,
                                     e);  // determinant times quadrature weight times coefficient
               MFEM_VERIFY(det > 0, "");

               for (int c = 0; c < 3; ++c)
               {
                  for (int d = 0; d < 3; ++d)
                  {
                     curlRef[c][d] = grad[qz][qy][qx][c][d] - grad[qz][qy][qx][d][c];
                  }
               }

               // Set grad[qz][qy][qx] = J^{-T} curlRef J^{-1}
               for (int i=0; i<3; ++i)
               {
                  for (int j=0; j<3; ++j)
                  {
                     grad[qz][qy][qx][i][j] = 0;
                     for (int k=0; k<3; ++k)
                     {
                        double curl_invJ_kj = 0;

                        for (int l=0; l<3; ++l)
                        {
                           curl_invJ_kj += curlRef[k][l] * invJ[l][j];
                        }

                        grad[qz][qy][qx][i][j] += invJ[k][i] * curl_invJ_kj;
                     }
                  }
               }

               // Now curl v = [g[2][1], g[0][2], g[1][0], where g = grad[qz][qy][qx].

               const double curlx = grad[qz][qy][qx][2][1];
               const double curly = grad[qz][qy][qx][0][2];
               const double curlz = grad[qz][qy][qx][1][0];

               // Set g[0][:] = J^{-1}_{(:,1)} g[2][1]
               //     g[1][:] = J^{-1}_{(:,2)} g[0][2]
               //     g[2][:] = J^{-1}_{(:,0)} g[1][0]
               // Also scale by det.
               for (int i=0; i<3; ++i)
               {
                  grad[qz][qy][qx][0][i] = invJ[i][1] * curlx * det;
                  grad[qz][qy][qx][1][i] = invJ[i][2] * curly * det;
                  grad[qz][qy][qx][2][i] = invJ[i][0] * curlz * det;
               }
            }
         }
      }

      // Note that curl does not simplify as a tensor product of derivatives, like for diffusion.
      // All 6 of the off-diagonal partial derivatives must be computed and stored, before computing curl,
      // which involves a transformation with the Jacobian at each quadrature point.

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double gradXY[MAX_D1D][MAX_D1D][2][2][6];

         osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     for (int d = 0; d < 6; ++d)
                     {
                        gradXY[dy][dx][0][n][d] = 0;
                        gradXY[dy][dx][1][n][d] = 0;
                     }
                  }
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double gradX[MAX_D1D][2][2][6];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int n = 0; n < 2; ++n)
                  {
                     for (int d = 0; d < 6; ++d)
                     {
                        gradX[dx][0][n][d] = 0;
                        gradX[dx][1][n][d] = 0;
                     }
                  }
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     const double wx = ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
                     const double wDx = ((c == 0) ? Got(dx,qx) : Gct(dx,qx));

                     // The pattern for all c is for each i != c, we store 6 quantities for J^{-1}_{(c,m)} g[n][i], J^{-1}_{(i,m)} g[n][c].
                     int d = 0;
                     for (int i = 0; i < 3; ++i)
                     {
                        if (i != c)
                        {
                           for (int n = 0; n < 3; ++n)
                           {
                              const int m = (n + 2) % 3;
                              const double invJcm = op(qx,qy,qz,idJ[c][m],e);
                              const double invJim = op(qx,qy,qz,idJ[i][m],e);

                              gradX[dx][0][d][2*n] += invJcm * grad[qz][qy][qx][n][i] *
                                                      wx; // J^{-1}_{(c,m)} g[n][i]
                              gradX[dx][0][d][(2*n)+1] += invJim * grad[qz][qy][qx][n][c] *
                                                          wx; // J^{-1}_{(i,m)} g[n][c]

                              gradX[dx][1][d][2*n] += invJcm * grad[qz][qy][qx][n][i] *
                                                      wDx; // J^{-1}_{(c,m)} g[n][i]
                              gradX[dx][1][d][(2*n)+1] += invJim * grad[qz][qy][qx][n][c] *
                                                          wDx; // J^{-1}_{(i,m)} g[n][c]
                           }

                           d++;
                        }
                     }
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  const double wDy = (c == 1) ? Got(dy,qy) : Gct(dy,qy);

                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // The pattern for all c is for each i != c, we store 6 quantities for J^{-1}_{(c,m)} g[n][i], J^{-1}_{(i,m)} g[n][c].
                     for (int d = 0; d < 2; ++d)
                     {
                        for (int n = 0; n < 6; ++n)
                        {
                           if (c == 0)  // skip wDx
                           {
                              gradXY[dy][dx][0][d][n] += gradX[dx][0][d][n] * wDy; // wx * wDy
                              gradXY[dy][dx][1][d][n] += gradX[dx][0][d][n] * wy;  // wx * wy
                           }
                           else if (c == 1)  // skip wDy
                           {
                              gradXY[dy][dx][0][d][n] += gradX[dx][1][d][n] * wy;  // wDx * wy
                              gradXY[dy][dx][1][d][n] += gradX[dx][0][d][n] * wy;  // wx * wy
                           }
                           else // c == 2, skip wDz
                           {
                              gradXY[dy][dx][0][d][n] += gradX[dx][1][d][n] * wy;  // wDx * wy
                              gradXY[dy][dx][1][d][n] += gradX[dx][0][d][n] * wDy; // wx * wDy
                           }
                        }
                     }
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
               const double wDz = (c == 2) ? Got(dz,qz) : Gct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     // [gradXY[dy][dx][0] * wz, gradXY[dy][dx][1] * wz, gradXY[dy][dx][2] * wDz] is grad(u_c), except for the c entry (not used).

                     // 21 contribution is (J^{-1}_{(:,2)})^T [curl u] g[0][:]
                     // 02 contribution is (J^{-1}_{(:,0)})^T [curl u] g[1][:]
                     // 10 contribution is (J^{-1}_{(:,1)})^T [curl u] g[2][:]

                     // The pattern for all c is for each i != c, we store 6 quantities for J^{-1}_{(c,m)} g[n][i], J^{-1}_{(i,m)} g[n][c].
                     // We do not need the derivative of component u_c with respect to x_c.

                     for (int n = 0; n < 3; ++n)
                     {
                        // Note that there are entries of gradXY that do not get used. This could be optimized further,
                        // perhaps by using the idenity (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u}.
                        const double t1 = gradXY[dy][dx][0][0][2*n];
                        const double t2 = gradXY[dy][dx][0][0][(2*n)+1];

                        //const double t3 = gradXY[dy][dx][0][1][2*n];  // not used
                        //const double t4 = gradXY[dy][dx][0][1][(2*n)+1];  // not used

                        //const double t5 = gradXY[dy][dx][1][0][2*n];  // not used
                        //const double t6 = gradXY[dy][dx][1][0][(2*n)+1];  // not used

                        const double t7 = gradXY[dy][dx][1][1][2*n];
                        const double t8 = gradXY[dy][dx][1][1][(2*n)+1];

                        if (c == 0)
                        {
                           // For 21, 02, 10, the contribution is
                           //  J^{-1}_{(0,m)} { (u_0)_{x_1} g[n][1] + (u_0)_{x_2} g[n][2] } +
                           // -J^{-1}_{(1,m)} (u_0)_{x_1} g[n][0]
                           // -J^{-1}_{(2,m)} (u_0)_{x_2} g[n][0]
                           // where m = 2, 0, 1, and n = 0, 1, 2, respectively.
                           // However, J is not available, since we already summed over quadrature points.
                           // Thus for i = 1, we store the 6 summed quantities gradXY[dy][dx][i] times J^{-1}_{(0,m)} g[n][1], J^{-1}_{(1,m)} g[n][0];
                           //      for i = 2, we store the 6 summed quantities gradXY[dy][dx][i] times J^{-1}_{(0,m)} g[n][2], J^{-1}_{(2,m)} g[n][0].

                           // t1 = wx * wDy times J^{-1}_{(0,m)} g[n][1]
                           // t2 = wx * wDy times J^{-1}_{(1,m)} g[n][0]
                           // t3 = wx * wDy times J^{-1}_{(0,m)} g[n][2]
                           // t4 = wx * wDy times J^{-1}_{(2,m)} g[n][0]
                           // t5 = wx * wy times J^{-1}_{(0,m)} g[n][1]
                           // t6 = wx * wy times J^{-1}_{(1,m)} g[n][0]
                           // t7 = wx * wy times J^{-1}_{(0,m)} g[n][2]
                           // t8 = wx * wy times J^{-1}_{(2,m)} g[n][0]

			   y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += ((t1 * wz) + (t7 * wDz) - (t2 * wz) - (t8 * wDz));
                        }
                        else if (c == 1)
                        {
                           // For 21, 02, 10, the contribution is
                           // -J^{-1}_{(0,m)} (u_1)_{x_0} g[n][1] +
                           //  J^{-1}_{(1,m)} { (u_1)_{x_0} g[n][0] + (u_1)_{x_2} g[n][2] } +
                           // -J^{-1}_{(2,m)} (u_1)_{x_2} g[n][1]
                           // where m = 2, 0, 1, and n = 0, 1, 2, respectively.
                           // However, J is not available, since we already summed over quadrature points.
                           // Thus for i = 0, we store the 6 summed quantities gradXY[dy][dx][i] times J^{-1}_{(1,m)} g[n][0], J^{-1}_{(0,m)} g[n][1];
                           //      for i = 2, we store the 6 summed quantities gradXY[dy][dx][i] times J^{-1}_{(1,m)} g[n][2], J^{-1}_{(2,m)} g[n][1].

                           // t1 = wDx * wy times J^{-1}_{(1,m)} g[n][0]
                           // t2 = wDx * wy times J^{-1}_{(0,m)} g[n][1]
                           // t3 = wDx * wy times J^{-1}_{(1,m)} g[n][2]
                           // t4 = wDx * wy times J^{-1}_{(2,m)} g[n][1]
                           // t5 = wx * wy times J^{-1}_{(1,m)} g[n][0]
                           // t6 = wx * wy times J^{-1}_{(0,m)} g[n][1]
                           // t7 = wx * wy times J^{-1}_{(1,m)} g[n][2]
                           // t8 = wx * wy times J^{-1}_{(2,m)} g[n][1]

			   y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += (-(t2 * wz) + (t1 * wz) + (t7 * wDz) - (t8 * wDz));
                        }
                        else  // c == 2
                        {
                           // For 21, 02, 10, the contribution is
                           // -J^{-1}_{(0,m)} (u_2)_{x_0} g[n][2] +
                           // -J^{-1}_{(1,m)} (u_2)_{x_1} g[n][2] +
                           //  J^{-1}_{(2,m)} { (u_2)_{x_0} g[n][0] + (u_2)_{x_1} g[n][1] } +
                           // where m = 2, 0, 1, and n = 0, 1, 2, respectively.
                           // However, J is not available, since we already summed over quadrature points.
                           // Thus for i = 0, we store the 6 summed quantities gradXY[dy][dx][i] times J^{-1}_{(2,m)} g[n][0], J^{-1}_{(0,m)} g[n][2];
                           //      for i = 1, we store the 6 summed quantities gradXY[dy][dx][i] times J^{-1}_{(2,m)} g[n][1], J^{-1}_{(1,m)} g[n][2].

                           // t1 = wDx * wy times J^{-1}_{(2,m)} g[n][0]
                           // t2 = wDx * wy times J^{-1}_{(0,m)} g[n][2]
                           // t3 = wDx * wy times J^{-1}_{(2,m)} g[n][1]
                           // t4 = wDx * wy times J^{-1}_{(1,m)} g[n][2]
                           // t5 = wx * wDy times J^{-1}_{(2,m)} g[n][0]
                           // t6 = wx * wDy times J^{-1}_{(0,m)} g[n][2]
                           // t7 = wx * wDy times J^{-1}_{(2,m)} g[n][1]
                           // t8 = wx * wDy times J^{-1}_{(1,m)} g[n][2]

			   y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += wz * (-t2 - t8 + t1 + t7);
                        }
                     }
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

void CurlCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
      PACurlCurlApply3D(dofs1D, quad1D, ne, mapsO->B, mapsC->B, mapsO->Bt,
                        mapsC->Bt,
                        mapsO->G, mapsC->G, mapsO->Gt, mapsC->Gt, pa_data, x, y);
   else
      PACurlCurlApply2D(dofs1D, quad1D, ne, mapsO->B, mapsO->Bt,
                        mapsC->G, mapsC->Gt, pa_data, x, y);
}

static void PACurlCurlAssembleDiagonal2D(const int D1D,
                                         const int Q1D,
                                         const int NE,
                                         const Array<double> &_Bo,
                                         const Array<double> &_Gc,
                                         const Vector &_op,
                                         Vector &_diag)
{
   constexpr static int VDIM = 2;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto diag = Reshape(_diag.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y components
      {
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         double t[MAX_Q1D];

         for (int dy = 0; dy < D1Dy; ++dy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               t[qx] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = (c == 1) ? Bo(qy,dy) : -Gc(qy,dy);
                  t[qx] += wy * wy * op(qx,qy,e);
               }
            }

            for (int dx = 0; dx < D1Dx; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx = ((c == 0) ? Bo(qx,dx) : Gc(qx,dx));
                  diag(dx + (dy * D1Dx) + osc, e) += t[qx] * wx * wx;
               }
            }
         }

         osc += D1Dx * D1Dy;
      }  // loop c
   }); // end of element loop
}

static void PACurlCurlAssembleDiagonal3D(const int D1D,
                                         const int Q1D,
                                         const int NE,
                                         const Array<double> &_Bo,
                                         const Array<double> &_Bc,
                                         const Array<double> &_Go,
                                         const Array<double> &_Gc,
                                         const Vector &_op,
                                         Vector &_diag)
{
   constexpr static int VDIM = 3;

   auto Bo = Reshape(_Bo.Read(), Q1D, D1D-1);
   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Go = Reshape(_Go.Read(), Q1D, D1D-1);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto diag = Reshape(_diag.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
      // (\nabla\times u) \cdot (\nabla\times u) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{u}
      // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
      // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
      // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

      // For each c, we will keep 6 arrays for derivatives multiplied by the 6 entries of the symmetric 3x3 matrix (dF^T dF).

      int osc = 0;

      for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
      {
         const int D1Dz = (c == 2) ? D1D - 1 : D1D;
         const int D1Dy = (c == 1) ? D1D - 1 : D1D;
         const int D1Dx = (c == 0) ? D1D - 1 : D1D;

         double zt[MAX_Q1D][MAX_Q1D][MAX_D1D][6][3];

         // z contraction
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int dz = 0; dz < D1Dz; ++dz)
               {
                  for (int i=0; i<6; ++i)
                  {
                     for (int d=0; d<3; ++d)
                     {
                        zt[qx][qy][dz][i][d] = 0.0;
                     }
                  }

                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double wz = ((c == 2) ? Bo(qz,dz) : Bc(qz,dz));
                     const double wDz = ((c == 2) ? Go(qz,dz) : Gc(qz,dz));

                     for (int i=0; i<6; ++i)
                     {
                        zt[qx][qy][dz][i][0] += wz * wz * op(qx,qy,qz,i,e);
                        zt[qx][qy][dz][i][1] += wDz * wz * op(qx,qy,qz,i,e);
                        zt[qx][qy][dz][i][2] += wDz * wDz * op(qx,qy,qz,i,e);
                     }
                  }
               }
            }
         }  // end of z contraction

         double yt[MAX_Q1D][MAX_D1D][MAX_D1D][6][3][3];

         // y contraction
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int dz = 0; dz < D1Dz; ++dz)
            {
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int i=0; i<6; ++i)
                  {
                     for (int d=0; d<3; ++d)
                        for (int j=0; j<3; ++j)
                        {
                           yt[qx][dy][dz][i][d][j] = 0.0;
                        }
                  }

                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const double wy = ((c == 1) ? Bo(qy,dy) : Bc(qy,dy));
                     const double wDy = ((c == 1) ? Go(qy,dy) : Gc(qy,dy));

                     for (int i=0; i<6; ++i)
                     {
                        for (int d=0; d<3; ++d)
                        {
                           yt[qx][dy][dz][i][d][0] += wy * wy * zt[qx][qy][dz][i][d];
                           yt[qx][dy][dz][i][d][1] += wDy * wy * zt[qx][qy][dz][i][d];
                           yt[qx][dy][dz][i][d][2] += wDy * wDy * zt[qx][qy][dz][i][d];
                        }
                     }
                  }
               }
            }
         }  // end of y contraction

         // x contraction
         for (int dz = 0; dz < D1Dz; ++dz)
         {
            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx = ((c == 0) ? Bo(qx,dx) : Bc(qx,dx));
                     const double wDx = ((c == 0) ? Go(qx,dx) : Gc(qx,dx));

                     // Using (\nabla\times u) F = 1/det(dF) dF \hat{\nabla}\times\hat{u} (p. 78 of Monk), we get
                     // (\nabla\times u) \cdot (\nabla\times u) = 1/det(dF)^2 \hat{\nabla}\times\hat{u}^T dF^T dF \hat{\nabla}\times\hat{u}
                     // If c = 0, \hat{\nabla}\times\hat{u} reduces to [0, (u_0)_{x_2}, -(u_0)_{x_1}]
                     // If c = 1, \hat{\nabla}\times\hat{u} reduces to [-(u_1)_{x_2}, 0, (u_1)_{x_0}]
                     // If c = 2, \hat{\nabla}\times\hat{u} reduces to [(u_2)_{x_1}, -(u_2)_{x_0}, 0]

                     /*
                       const double O11 = op(q,0,e);
                       const double O12 = op(q,1,e);
                       const double O13 = op(q,2,e);
                       const double O22 = op(q,3,e);
                       const double O23 = op(q,4,e);
                       const double O33 = op(q,5,e);
                     */

                     if (c == 0)
                     {
                        // (u_0)_{x_2} (O22 (u_0)_{x_2} - O23 (u_0)_{x_1}) - (u_0)_{x_1} (O32 (u_0)_{x_2} - O33 (u_0)_{x_1})

                        // (u_0)_{x_2} O22 (u_0)_{x_2}
		        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += yt[qx][dy][dz][3][2][0] * wx * wx;

                        // -(u_0)_{x_2} O23 (u_0)_{x_1} - (u_0)_{x_1} O32 (u_0)_{x_2}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += -2.0 * yt[qx][dy][dz][4][1][1] * wx * wx;
			  
                        // (u_0)_{x_1} O33 (u_0)_{x_1}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += yt[qx][dy][dz][5][0][2] * wx * wx;
                     }
                     else if (c == 1)
                     {
                        // (u_1)_{x_2} (O11 (u_1)_{x_2} - O13 (u_1)_{x_0}) + (u_1)_{x_0} (-O31 (u_1)_{x_2} + O33 (u_1)_{x_0})

                        // (u_1)_{x_2} O11 (u_1)_{x_2}
		        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += yt[qx][dy][dz][0][2][0] * wx * wx;

                        // -(u_1)_{x_2} O13 (u_1)_{x_0} - (u_1)_{x_0} O31 (u_1)_{x_2}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += -2.0 * yt[qx][dy][dz][2][1][0] * wDx * wx;

                        // (u_1)_{x_0} O33 (u_1)_{x_0})
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += yt[qx][dy][dz][5][0][0] * wDx * wDx;
                     }
                     else
                     {
                        // (u_2)_{x_1} (O11 (u_2)_{x_1} - O12 (u_2)_{x_0}) - (u_2)_{x_0} (O21 (u_2)_{x_1} - O22 (u_2)_{x_0})

                        // (u_2)_{x_1} O11 (u_2)_{x_1}
		        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += yt[qx][dy][dz][0][0][2] * wx * wx;

                        // -(u_2)_{x_1} O12 (u_2)_{x_0} - (u_2)_{x_0} O21 (u_2)_{x_1}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += -2.0 * yt[qx][dy][dz][1][0][1] * wDx * wx;

                        // (u_2)_{x_0} O22 (u_2)_{x_0}
                        diag(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += yt[qx][dy][dz][3][0][0] * wDx * wDx;
                     }
                  }
               }
            }
         }  // end of x contraction

         osc += D1Dx * D1Dy * D1Dz;
      }  // loop c
   }); // end of element loop
}

void CurlCurlIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (dim == 3)
      PACurlCurlAssembleDiagonal3D(dofs1D, quad1D, ne, 
                                   mapsO->B, mapsC->B, mapsO->G, mapsC->G, pa_data_2, diag);
   else
      PACurlCurlAssembleDiagonal2D(dofs1D, quad1D, ne, 
                                   mapsO->B, mapsC->G, pa_data, diag);
}

void MixedVectorGradientIntegrator::AssemblePA(const FiniteElementSpace &trial_fes,
					       const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with a vector test space and H^1 trial space.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const NodalTensorFiniteElement *trial_el =
      dynamic_cast<const NodalTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only NodalTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());

   Vector coeff(ne * nq);
   coeff = 1.0;
   if (Q)
   {
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            coeff[p + (e * nq)] = Q->Eval(*tr, ir->IntPoint(p));
         }
      }
   }

   // Use the same setup functions as VectorFEMassIntegrator.
   if (test_el->GetDerivType() == mfem::FiniteElement::CURL && dim == 3)
   {
      PAHcurlSetup3D(quad1D, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else if (test_el->GetDerivType() == mfem::FiniteElement::CURL && dim == 2)
   {
      PAHcurlSetup2D(quad1D, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

// Apply to x corresponding to DOF's in H^1 (trial), whose gradients are integrated
// against H(curl) test functions corresponding to y.
static void PAHcurlH1Apply3D(const int D1D,
                             const int Q1D,
                             const int NE,
                             const Array<double> &_Bc,
                             const Array<double> &_Gc,
                             const Array<double> &_Bot,
                             const Array<double> &_Bct,
                             const Vector &_op,
                             const Vector &_x,
                             Vector &_y)
{
   constexpr static int VDIM = 3;

   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 6, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 3*(D1D-1)*D1D*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][MAX_Q1D][VDIM];

      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < VDIM; ++c)
               {
                  mass[qz][qy][qx][c] = 0.0;
               }
            }
         }
      }

      for (int dz = 0; dz < D1D; ++dz)
      {
         double gradXY[MAX_Q1D][MAX_Q1D][3];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double gradX[MAX_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * Bc(qx,dx);
                  gradX[qx][1] += s * Gc(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy  = Bc(qy,dy);
               const double wDy = Gc(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx * wDy;
                  gradXY[qy][qx][2] += wx * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz  = Bc(qz,dz);
            const double wDz = Gc(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  mass[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  mass[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  mass[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }

      // Apply D operator.
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double O11 = op(qx,qy,qz,0,e);
               const double O12 = op(qx,qy,qz,1,e);
               const double O13 = op(qx,qy,qz,2,e);
               const double O22 = op(qx,qy,qz,3,e);
               const double O23 = op(qx,qy,qz,4,e);
               const double O33 = op(qx,qy,qz,5,e);
               const double massX = mass[qz][qy][qx][0];
               const double massY = mass[qz][qy][qx][1];
               const double massZ = mass[qz][qy][qx][2];
               mass[qz][qy][qx][0] = (O11*massX)+(O12*massY)+(O13*massZ);
               mass[qz][qy][qx][1] = (O12*massX)+(O22*massY)+(O23*massZ);
               mass[qz][qy][qx][2] = (O13*massX)+(O23*massY)+(O33*massZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D; ++qz)
      {
         double massXY[MAX_D1D][MAX_D1D];

         int osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y, z components
         {
            const int D1Dz = (c == 2) ? D1D - 1 : D1D;
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massXY[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double massX[MAX_D1D];
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massX[dx] += mass[qz][qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
                  }
               }
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     massXY[dy][dx] += massX[dx] * wy;
                  }
               }
            }

            for (int dz = 0; dz < D1Dz; ++dz)
            {
               const double wz = (c == 2) ? Bot(dz,qz) : Bct(dz,qz);
               for (int dy = 0; dy < D1Dy; ++dy)
               {
                  for (int dx = 0; dx < D1Dx; ++dx)
                  {
                     y(dx + ((dy + (dz * D1Dy)) * D1Dx) + osc, e) += massXY[dy][dx] * wz;
                  }
               }
            }

            osc += D1Dx * D1Dy * D1Dz;
         }  // loop c
      }  // loop qz
   }); // end of element loop
}

// Apply to x corresponding to DOF's in H^1 (trial), whose gradients are integrated
// against H(curl) test functions corresponding to y.
static void PAHcurlH1Apply2D(const int D1D,
                             const int Q1D,
                             const int NE,
                             const Array<double> &_Bc,
                             const Array<double> &_Gc,
                             const Array<double> &_Bot,
                             const Array<double> &_Bct,
                             const Vector &_op,
                             const Vector &_x,
                             Vector &_y)
{
   constexpr static int VDIM = 2;

   auto Bc = Reshape(_Bc.Read(), Q1D, D1D);
   auto Gc = Reshape(_Gc.Read(), Q1D, D1D);
   auto Bot = Reshape(_Bot.Read(), D1D-1, Q1D);
   auto Bct = Reshape(_Bct.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, 3, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), 2*(D1D-1)*D1D, NE);

   MFEM_FORALL(e, NE,
   {
      double mass[MAX_Q1D][MAX_Q1D][VDIM];

      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < VDIM; ++c)
            {
               mass[qy][qx][c] = 0.0;
            }
         }
      }

      for (int dy = 0; dy < D1D; ++dy)
      {
         double gradX[MAX_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * Bc(qx,dx);
               gradX[qx][1] += s * Gc(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy  = Bc(qy,dy);
            const double wDy = Gc(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wx  = gradX[qx][0];
               const double wDx = gradX[qx][1];
               mass[qy][qx][0] += wDx * wy;
               mass[qy][qx][1] += wx * wDy;
            }
         }
      }

      // Apply D operator.
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double O11 = op(qx,qy,0,e);
            const double O12 = op(qx,qy,1,e);
            const double O22 = op(qx,qy,2,e);
            const double massX = mass[qy][qx][0];
            const double massY = mass[qy][qx][1];
            mass[qy][qx][0] = (O11*massX)+(O12*massY);
            mass[qy][qx][1] = (O12*massX)+(O22*massY);
         }
      }

      for (int qy = 0; qy < Q1D; ++qy)
      {
         int osc = 0;

         for (int c = 0; c < VDIM; ++c)  // loop over x, y components
         {
            const int D1Dy = (c == 1) ? D1D - 1 : D1D;
            const int D1Dx = (c == 0) ? D1D - 1 : D1D;

            double massX[MAX_D1D];
            for (int dx = 0; dx < D1Dx; ++dx)
            {
               massX[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  massX[dx] += mass[qy][qx][c] * ((c == 0) ? Bot(dx,qx) : Bct(dx,qx));
               }
            }

            for (int dy = 0; dy < D1Dy; ++dy)
            {
               const double wy = (c == 1) ? Bot(dy,qy) : Bct(dy,qy);

               for (int dx = 0; dx < D1Dx; ++dx)
               {
                  y(dx + (dy * D1Dx) + osc, e) += massX[dx] * wy;
               }
            }

            osc += D1Dx * D1Dy;
         }  // loop c
      }
   }); // end of element loop
}

void MixedVectorGradientIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
      PAHcurlH1Apply3D(dofs1D, quad1D, ne, mapsC->B, mapsC->G,
                       mapsO->Bt, mapsC->Bt, pa_data, x, y);
   else if (dim == 2)
      PAHcurlH1Apply2D(dofs1D, quad1D, ne, mapsC->B, mapsC->G,
                       mapsO->Bt, mapsC->Bt, pa_data, x, y);
}

} // namespace mfem
