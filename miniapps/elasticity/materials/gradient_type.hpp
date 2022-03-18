#ifndef MFEM_ELASTICITY_GRADIENT_TYPE_HPP
#define MFEM_ELASTICITY_GRADIENT_TYPE_HPP

enum class GradientType
{
   Symbolic,
   EnzymeFwd,
   EnzymeRev,
   FiniteDiff,
   DualNumbers
};

#endif