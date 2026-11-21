/**
 * @file anisotropy-utils.hpp
 * @brief File containing useful functions for anisotropy calculations.
 */


#pragma once

#include "mfem.hpp"

#ifndef ANISOTROPY_HPP
#define ANISOTROPY_HPP

using namespace mfem;

std::function<void(const Vector &, DenseMatrix &)> ConductivityMatrix(const Vector &d, std::function<void(const Vector &, Vector &)> EulerAngles)
{

   return [d, EulerAngles](const Vector &x, DenseMatrix &m)
   {
      // Define dimension of problem
      const int dim = x.Size();  

      // Compute Euler angles
      Vector e(3);
      EulerAngles(x, e);
      real_t e1 = e(0); // Roll
      real_t e2 = e(1); // Pitch
      real_t e3 = e(2); // Yaw

      // Compute rotated matrix
      if (dim == 3)
      {
         // Compute cosine and sine of the angles e1, e2, e3
         const real_t c1 = cos(e1);
         const real_t s1 = sin(e1);
         const real_t c2 = cos(e2);
         const real_t s2 = sin(e2);
         const real_t c3 = cos(e3);
         const real_t s3 = sin(e3);

         // Fill the rotation matrix R with the Euler angles.
         DenseMatrix R(3, 3);
         R(0, 0) = c3 * c2;
         R(1, 0) = s3 * c2;
         R(2, 0) = -s2;         
         R(0, 1) = s1 * s2 * c3 - c1 * s3;
         R(1, 1) = s1 * s2 * s3 + c1 * c3;
         R(2, 1) = s1 * c2;
         R(0, 2) = c1 * s2 * c3 + s1 * s3;
         R(1, 2) = c1 * s2 * s3 - s1 * c3;
         R(2, 2) = c1 * c2;

         // Multiply the rotation matrix R with the diffusivity vector.
         Vector l(3);
         l(0) = d[0];
         l(1) = d[1];
         l(2) = d[2];

         // Compute m = R^t diag(l) R
         R.Transpose();
         MultADBt(R, l, R, m);
      }
      else if (dim == 2)
      {  // R^t diag(l) R
         const real_t c1 = cos(e1);
         const real_t s1 = sin(e1);
         DenseMatrix Rt(2, 2);
         Rt(0, 0) = c1;
         Rt(0, 1) = s1;
         Rt(1, 0) = -s1;
         Rt(1, 1) = c1;
         Vector l(2);
         l(0) = d[0];
         l(1) = d[1];
         MultADAt(Rt, l, m);
      }
      else
      {
         m(0, 0) = d[0];
      }
   };
}


std::function<void(const Vector &, Vector &)> FiberDirection(std::function<void(const Vector &, Vector &)> EulerAngles, int component)
{
   return [EulerAngles, component](const Vector &x, Vector &e)
   {
      // Compute Euler angles
      Vector angles(3);
      EulerAngles(x, angles);
      real_t e1 = angles(0); // Roll
      real_t e2 = angles(1); // Pitch
      real_t e3 = angles(2); // Yaw

      // Compute cosine and sine of the angles e1, e2, e3
      const real_t c1 = cos(e1);
      const real_t s1 = sin(e1);
      const real_t c2 = cos(e2);
      const real_t s2 = sin(e2);
      const real_t c3 = cos(e3);
      const real_t s3 = sin(e3);

      // Fill the rotation matrix R with the Euler angles.
      DenseMatrix R(3, 3);
      R(0, 0) = c3 * c2;
      R(1, 0) = s3 * c2;
      R(2, 0) = -s2;         
      R(0, 1) = s1 * s2 * c3 - c1 * s3;
      R(1, 1) = s1 * s2 * s3 + c1 * c3;
      R(2, 1) = s1 * c2;
      R(0, 2) = c1 * s2 * c3 + s1 * s3;
      R(1, 2) = c1 * s2 * s3 - s1 * c3;
      R(2, 2) = c1 * c2;

      // Extract the desired column from the rotation matrix R
      e.SetSize(3);
      e(0) = R(0, component);
      e(1) = R(1, component);
      e(2) = R(2, component);
   };

}

#endif // ANISOTROPY_HPP
