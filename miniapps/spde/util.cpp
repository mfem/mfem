// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

#include "util.hpp"
#include <algorithm>

namespace mfem
{

void FillWithRandomNumbers(std::vector<real_t> &x, real_t a, real_t b)
{
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<> dis(a, b);
   std::for_each(x.begin(), x.end(), [&](real_t &v) { v = dis(gen); });
}

void FillWithRandomRotations(std::vector<real_t> &x)
{
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<> dis(0, 1);
   for (size_t i = 0; i < x.size(); i += 9)
   {
      // Get a random rotation matrix via uniform Euler angles.
      real_t e1 = 2 * M_PI * dis(gen);
      real_t e2 = 2 * M_PI * dis(gen);
      real_t e3 = 2 * M_PI * dis(gen);
      const real_t c1 = cos(e1);
      const real_t s1 = sin(e1);
      const real_t c2 = cos(e2);
      const real_t s2 = sin(e2);
      const real_t c3 = cos(e3);
      const real_t s3 = sin(e3);

      // Fill the rotation matrix R with the Euler angles. See for instance
      // the definition in Wikipedia.
      x[i + 0] = c1 * c3 - c2 * s1 * s3;
      x[i + 1] = -c1 * s3 - c2 * c3 * s1;
      x[i + 2] = s1 * s2;
      x[i + 3] = c3 * s1 + c1 * c2 * s3;
      x[i + 4] = c1 * c2 * c3 - s1 * s3;
      x[i + 5] = -c1 * s2;
      x[i + 6] = s2 * s3;
      x[i + 7] = c3 * s2;
      x[i + 8] = c2;
   }
}

}  // namespace mfem
