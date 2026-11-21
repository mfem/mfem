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

#include "general/forall.hpp"
#include <fstream>
#include <iomanip>

using namespace mfem;


class ShearStressNormalTractionCoefficient : public VectorCoefficient
{
public:
   // Providing kinematic viscosity coefficient and density
   ShearStressNormalTractionCoefficient(ParGridFunction *u,
                                        Coefficient *nu_coeff_, real_t density_) :
      VectorCoefficient(u->FESpace()->GetVDim()),
      dim(u->ParFESpace()->GetMesh()->Dimension()),
      nu_coeff(nu_coeff_),
      u_gf(u),
      density(density_)
   {
      nor.SetSize(dim);
      dudx.SetSize(vdim, dim);
      A.SetSize(vdim, dim);
   }

   // Providing dynamic viscosity coefficient
   ShearStressNormalTractionCoefficient(ParGridFunction *u,
                                        Coefficient *mu_coeff_, bool symmetric_tensor = false) :
      VectorCoefficient(u->FESpace()->GetVDim()),
      dim(u->ParFESpace()->GetMesh()->Dimension()),
      nu_coeff(mu_coeff_),
      u_gf(u),
      density(1.0),
      symmetric_tensor(symmetric_tensor)
   {
      nor.SetSize(dim);
      dudx.SetSize(vdim, dim);
      A.SetSize(vdim, dim);
   }

   using VectorCoefficient::Eval;

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      V.SetSize(vdim);

      T.SetIntPoint(&ip);
      u_gf->GetVectorGradient(T, dudx);
      real_t nu = nu_coeff->Eval(T, ip);
      nu *= density; // mu = nu_coeff * density  --> dynamic viscosity

      CalcOrtho(T.Jacobian(), nor);
      const real_t scale = nor.Norml2();
      nor *= (1.0 / scale);

      // If symmetric_tensor is true, then compute the symmetric part of the gradient
      // (mu (grad(u) + grad(u)^T))
      // Otherwise, compute the non-symmetric version (mu grad(u))
      for (int vd = 0; vd < vdim; vd++)
      {
         for (int d = 0; d < dim; d++)
         {
            A(vd, d) = nu * dudx(vd, d);
            A(vd, d) += symmetric_tensor ? nu * dudx(d,
                                                     vd) : 0.0; // add the transpose term if symmetric
         }
      }

      for (int vd = 0; vd < vdim; vd++)
      {
         real_t s = 0.0;
         for (int d = 0; d < dim; d++)
         {
            s += A(vd, d) * nor(d);
         }
         V(vd) = s;
      }
   }

private:
   const int dim;
   Coefficient *nu_coeff;   ///< Kinematic viscosity coefficient
   // vdim x dim
   DenseMatrix dudx, A;
   Vector nor;              ///< Normal vector
   ParGridFunction *u_gf;   ///< Velocity grid function
   real_t density;          ///< Density of the fluid
   bool symmetric_tensor =
      false; ///< Whether to compute the symmetric part of the gradient
};

class NormalComponentCoefficient : public Coefficient
{
public:
   NormalComponentCoefficient(VectorCoefficient *vcoeff_) : vcoeff(vcoeff_) {}

   using Coefficient::Eval;

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      int sdim = T.GetSpaceDim();
      MFEM_ASSERT(sdim == vcoeff->GetVDim(), "Dimension mismatch: "
                  << "T.GetSpaceDim() = " << sdim
                  << ", vcoeff->GetVDim() = " << vcoeff->GetVDim());

      T.SetIntPoint(&ip);

      // Compute normal
      nor.SetSize(sdim);
      CalcOrtho(T.Jacobian(), nor);
      const real_t scale = nor.Norml2();
      nor *= (1.0 / scale);

      // Compute vector tau \cdot n
      vcoeff->Eval(vec, T, ip);

      // Return the normal component of the provided vector  v \cdot n
      return vec * nor;
   }

private:
   VectorCoefficient *vcoeff = nullptr;
   Vector vec, nor; ///< Normal vector
};

/// Matrix coefficient defined as the outer product of the normal vector: $ a * n n^T $.
class NormalOuterProductCoefficient : public MatrixCoefficient
{
private:
   mutable Vector nor;       ///< Normal vector
   Coefficient *A = nullptr; ///< Optional scaling coefficient.   NOT OWNED

public:
   NormalOuterProductCoefficient(int dim,
                                 Coefficient *A = nullptr) : MatrixCoefficient(dim), A(A) {}

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {

      int sdim = T.GetSpaceDim();

      T.SetIntPoint(&ip);

      // Compute normal
      nor.SetSize(sdim);
      CalcOrtho(T.Jacobian(), nor);
      const real_t scale = nor.Norml2();
      nor *= (1.0 / scale);

      // Compute outer product
      M.SetSize(sdim);
      for (int i = 0; i < sdim; i++)
      {
         for (int j = 0; j < sdim; j++)
         {
            M(i, j) = nor[i] * nor[j];
         }
      }

      if (A)
      {
         M *= A->Eval(T, ip);
      }
   }
};

/// Matrix coefficient defined as the outer product of the tangent vector: $ a * t t^T $.
class TangentOuterProductCoefficient : public MatrixCoefficient
{
   mutable Vector tangent;   ///< Tangent vector
   Coefficient *A = nullptr; ///< Optional scaling coefficient.   NOT OWNED

public:
   TangentOuterProductCoefficient(int dim,
                                  Coefficient *A = nullptr) : MatrixCoefficient(dim), A(A) {}

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      int sdim = T.GetSpaceDim();
      T.SetIntPoint(&ip);
      M.SetSize(sdim);

      if (sdim == 2)
      {
         // --- RETAIN OLD 2D VERSION (Tangent Outer Product) ---
         tangent.SetSize(sdim);
         const DenseMatrix &Jac = T.Jacobian();
         tangent(0) = Jac(0, 0);
         tangent(1) = Jac(1, 0);
         tangent *= 1.0 / tangent.Norml2();

         for (int i = 0; i < sdim; i++)
         {
            for (int j = 0; j < sdim; j++)
            {
               M(i, j) = tangent(i) * tangent(j);
            }
         }
      }
      else if (sdim == 3)
      {
         // --- 3D PROJECTION MATRIX VERSION ---
         // The projection matrix is P = I - n n^T
         // This effectively computes (u \cdot t1)t1 + (u \cdot t2)t2
         // for any orthonormal basis in the tangent plane.

         Vector nor(sdim);
         CalcOrtho(T.Jacobian(), nor);
         nor /= nor.Norml2(); // Unit normal

         for (int i = 0; i < sdim; i++)
         {
            for (int j = 0; j < sdim; j++)
            {
               M(i, j) = (i == j ? 1.0 : 0.0) - nor(i) * nor(j);
            }
         }
      }
      else
      {
         mfem_error("TangentOuterProductCoefficient: sdim must be 2 or 3.");
      }

      if (A)
      {
         M *= A->Eval(T, ip);
      }
   }

};