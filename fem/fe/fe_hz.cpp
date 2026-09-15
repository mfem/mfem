// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#include "fe_hz.hpp"

namespace mfem
{

HuZhangTriangleFiniteElement::HuZhangTriangleFiniteElement()
   : CubicSymmetricTriangleFiniteElement(false) { }

} // namespace mfem
