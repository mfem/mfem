// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include "../fe/fe_base.hpp"
#include "../../fem/fespace.hpp"

namespace mfem::future
{

/// Base class for parametric spaces
class ParameterSpace
{
public:
   ParameterSpace(int vdim = 1) : vdim(vdim) {}

   /// @brief Get vector dimension at each point
   ///
   /// This is the number of components at each point in the parametric space.
   int GetVDim() const { return vdim; }

   /// Get DofToQuad information
   const DofToQuad& GetDofToQuad() const { return dtq; }

   /// Get total size of the space (T-vector size)
   ///
   /// returns the true size vsize of the space
   virtual int GetTrueVSize() const = 0;

   /// Get local vector size (L-vector size)
   ///
   /// returns the local size of the space
   virtual int GetVSize() const = 0;

   /// Get spatial dimension
   ///
   /// returns always 1.
   int Dimension() const
   {
      return 1;
   }

   /// @brief Get T-vector to L-vector transformation
   ///
   /// returns identity by default that is lazy evaluated.
   virtual const Operator* GetProlongationMatrix() const
   {
      if (!prolongation)
      {
         prolongation.reset(new IdentityOperator(GetTrueVSize()));
      }
      return prolongation.get();
   }

   /// @brief Get L-vector to E-vector transformation
   /// @note This is a mock call to replicate interface of FiniteElementSpace.
   /// It should not be used by a user.
   ///
   /// returns identity by default that is lazy evaluated.
   virtual const Operator* GetElementRestriction(ElementDofOrdering o) const
   {
      if (!elem_restr)
      {
         elem_restr.reset(new IdentityOperator(GetVSize()));
      }
      return elem_restr.get();
   }

protected:
   int vdim;
   DofToQuad dtq;
   mutable std::unique_ptr<Operator> prolongation;
   mutable std::unique_ptr<Operator> elem_restr;
};

/// @brief Uniform parameter space
class UniformParameterSpace : public ParameterSpace
{
public:
   /// @brief Constructor for a uniform parameter space
   ///
   /// @param mesh The mesh to determine dimension and number of elements.
   /// @param ir The integration rule to determine the number of quadrature points.
   /// @param vdim The vector dimension at each point.
   /// @param used_in_tensor_product If true, the number of quadrature points is
   /// calculated as the nth root of the number of points in the integration rule,
   /// where n is the mesh dimension. If false, the number of quadrature points is
   /// taken directly from the integration rule.
   UniformParameterSpace(Mesh &mesh, const IntegrationRule &ir, int vdim,
                         bool used_in_tensor_product = true) :
      ParameterSpace(vdim)
   {
      // Setup DofToQuad information
      dtq.nqpt = (int)floor(std::pow(ir.GetNPoints(), 1.0 / mesh.Dimension()) + 0.5);
      dtq.ndof = dtq.nqpt;
      dtq.mode = used_in_tensor_product ? DofToQuad::TENSOR : DofToQuad::FULL;

      // Calculate sizes
      const int num_qp = used_in_tensor_product ?
                         static_cast<int>(std::pow(dtq.nqpt, mesh.Dimension())) :
                         ir.GetNPoints();

      tsize = vdim * num_qp * mesh.GetNE();
      lsize = tsize;
   }

   int GetTrueVSize() const override
   {
      return tsize;
   }

   int GetVSize() const override
   {
      return lsize;
   }

private:
   /// T-vector size
   int tsize;

   /// L-vector size
   int lsize;
};

class ParameterFunction : public Vector
{
public:
   ParameterFunction(ParameterSpace &space) :
      Vector(space.GetTrueVSize()),
      space(space)
   {}

   /// @brief Get the ParameterSpace
   const ParameterSpace& GetParameterSpace() const
   {
      return space;
   }

   using Vector::operator=;

private:
   /// the parametric space
   ParameterSpace &space;
};

} // namespace mfem::future
