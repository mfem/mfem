#pragma once
#include "mfem.hpp"
namespace mfem
{
GridFunction *MakeGridFunction(FiniteElementSpace *fes);
LinearForm *MakeLinearForm(FiniteElementSpace *fes);
NonlinearForm *MakeNonlinearForm(FiniteElementSpace *fes);
BilinearForm *MakeBilinearForm(FiniteElementSpace *fes);
MixedBilinearForm *MakeMixedBilinearForm(FiniteElementSpace *trial_fes, FiniteElementSpace *test_fes);
}