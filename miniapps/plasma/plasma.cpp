// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "plasma.hpp"

using namespace std;

namespace mfem
{

namespace plasma
{

double lambda_ei(double Te, double ne, double zi)
{
   if (Te < 10.0 * zi * zi)
   {
      return 23.0 - log(1.0e-3 * zi * sqrt(ne / pow(Te, 3)));
   }
   else
   {
      return 23.0 - log(1.0e-3 * sqrt(0.1 * ne) / Te);
   }
}
/// Derivative of lambda_ei wrt Te
double dlambda_ei_dTe(double Te, double ne, double zi)
{
   if (Te < 10.0 * zi * zi)
   {
      return 1.5 / Te;
   }
   else
   {
      return 1.0 / Te;
   }
}

} // namespace plasma

} // namespace mfem
