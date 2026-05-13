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

#include "umansky.hpp"

namespace umansky
{
  
using namespace std;

real_t FindYVal(ParGridFunction &u, real_t u_target, real_t x,
                real_t y0, real_t y1)
{
   ParMesh * pmesh = u.ParFESpace()->GetParMesh();

   MPI_Comm comm = pmesh->GetComm();
   int nranks = pmesh->GetNRanks();
   Vector ucVec(nranks);

   Array<int> elem;
   Array<IntegrationPoint> ips;
   DenseMatrix point_mat(2, 1);
   point_mat(0,0) = x;

   real_t tol = 1e-5;

   real_t a = y0;
   real_t b = y1;

   real_t ua = 0.0 - u_target;
   real_t ub = 1.0 - u_target;
   real_t uc = 0.5 - u_target;

   const int nmax = 20;
   int n = 0;
   while (n < nmax)
   {
      real_t c = 0.5 * (a + b);
      point_mat(1,0) = c;
      int nfound = pmesh->FindPoints(point_mat, elem, ips);

      if (nfound != 1)
      {
         MFEM_ABORT("Point (" << x << ", " << c << ") not found");
      }

      if (elem[0] >= 0)
      {
         uc = u.GetValue(elem[0], ips[0]) - u_target;
      }
      else
      {
         uc = -DBL_MAX;
      }

      MPI_Allgather(&uc, 1, MFEM_MPI_REAL_T, ucVec.GetData(), 1,
		    MFEM_MPI_REAL_T, comm);

      for (int i=0; i<nranks; i++)
      {
         if (ucVec[i] > -0.5 * DBL_MAX)
         {
            uc = ucVec[i];
            break;
         }
      }

      if (std::abs(uc) < tol || 0.5 * (b - a) < tol)
      {
         return c;
      }

      if (ua * uc < 0_r)
      {
         b = c;
         ub = uc;
      }
      else if (ub * uc < 0_r)
      {
 	 a = c;
         ua = uc;
      }
      else
      {
	MFEM_ABORT("Bisection failed");
      }

      n++;
   }
   return -1.0;
}


real_t CalcWidth(ParGridFunction &u)
{
   ParMesh * pmesh = u.ParFESpace()->GetParMesh();

   Vector min, max;
   pmesh->GetBoundingBox(min,max);

   real_t xMid = 0.5 * (max[0] + min[0]);
   real_t y0 = min[1];
   real_t y1 = max[1];

   real_t y25 = FindYVal(u, 0.25, xMid, y0, y1);
   real_t y75 = FindYVal(u, 0.75, xMid, y0, y1);

   return y75 - y25;
}

}
