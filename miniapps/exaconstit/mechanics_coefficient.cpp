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

// Implementation of Coefficient class

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
//#include "fem.hpp"


//#include <cmath>
//#include <limits>

namespace mfem
{

void QuadratureVectorFunctionCoefficient::Eval(Vector &V,
                                               ElementTransformation &T,
                                               const IntegrationPoint &ip)
{
   mfem_error ("QuadratureVectorFunctionCoefficient::Eval (...)\n"
               "   is not implemented for this class.");
   return;
}

void QuadratureVectorFunctionCoefficient::EvalQ(Vector &V,
                                                ElementTransformation &T,
                                                const int ip_num)
{
   int elem_no = T.ElementNo;
   QuadF->GetElementValues(elem_no, ip_num, V);
   return;
}
#endif

}
