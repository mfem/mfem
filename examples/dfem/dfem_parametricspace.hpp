#pragma once

#include <mfem.hpp>

namespace mfem
{

class ParametricSpace
{

public:
   ParametricSpace(int local_size, int element_size, int total_size) :
      local_size(local_size),
      element_size(element_size),
      total_size(total_size),
      identity(total_size)
   {
      dtq.ndof = element_size;
      dtq.nqpt = element_size;
   }

   ParametricSpace(int local_size) :
      local_size(local_size),
      element_size(local_size),
      total_size(local_size),
      identity(local_size)
   {
      dtq.ndof = element_size;
      dtq.nqpt = element_size;
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
   // Hint for the local dimension. E.g. the size on the quadrature point or vdim.
   int local_size;

   // Size of the data on an elements
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
