#pragma once

#include "../linalg/linalg.hpp"
#include "../fe/fe_base.hpp"

namespace mfem
{

class ParametricSpace
{

public:
   /// spatial_dim is the dimension of the spatial domain (e.g. 2 for 2D)
   /// local_size is the size of the data on a single quadrature point
   /// element_size is the size of the data on an element divided by vdim
   /// total_size is the size of the data for all elements
   ParametricSpace(int spatial_dim, int local_size, int element_size,
                   int total_size) :
      spatial_dim(spatial_dim),
      local_size(local_size),
      element_size(element_size),
      total_size(total_size),
      identity(total_size)
   {
      // dtq.ndof = (int)floor(pow(element_size, 1.0/spatial_dim) + 0.5);
      dtq.ndof = element_size;
      dtq.nqpt = dtq.ndof;
   }

   ParametricSpace(int local_size) :
      local_size(local_size),
      element_size(local_size),
      total_size(local_size),
      identity(local_size)
   {
      dtq.ndof = (int)floor(pow(element_size, 1.0/spatial_dim) + 0.5);
      dtq.nqpt = dtq.ndof;
   }

   int Dimension() const
   {
      return spatial_dim;
   }

   int GetLocalSize() const
   {
      return local_size;
   }

   int GetElementSize() const
   {
      return element_size;
   }

   int GetTotalSize() const
   {
      return total_size;
   }

   const DofToQuad &GetDofToQuad() const
   {
      return dtq;
   }

   const Operator *GetProlongation() const
   {
      return &identity;
   }

   const Operator *GetRestriction() const
   {
      return &identity;
   }

private:
   int spatial_dim;

   // Hint for the local dimension. E.g. the size on the quadrature point or vdim.
   int local_size;

   // Size of the data on an element
   int element_size;

   int total_size;

   IdentityOperator identity;

   DofToQuad dtq;
};

class ParametricFunction : public Vector
{
public:
   ParametricFunction(ParametricSpace &space) :
      Vector(space.GetTotalSize()),
      space(space)
   {}

   ParametricSpace &space;

   using Vector::operator=;

};

}
