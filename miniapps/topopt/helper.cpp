#include "mfem.hpp"
#include "helper.hpp"
namespace mfem
{
GridFunction *MakeGridFunction(FiniteElementSpace *fes)
{
    #ifdef MFEM_USE_MPI
    auto pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
    if (pfes)
    {
      return new ParGridFunction(pfes);
    }
    else
    {
      return new GridFunction(fes);
    }
    #else
    return new GridFunction(fes);
    #endif
}
LinearForm *MakeLinearForm(FiniteElementSpace *fes)
{
    #ifdef MFEM_USE_MPI
    auto pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
    if (pfes)
    {
      return new ParLinearForm(pfes);
    }
    else
    {
      return new LinearForm(fes);
    }
    #else
    return new LinearForm(fes);
    #endif
}
NonlinearForm *MakeNonlinearForm(FiniteElementSpace *fes)
{
    #ifdef MFEM_USE_MPI
    auto pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
    if (pfes)
    {
      return new ParNonlinearForm(pfes);
    }
    else
    {
      return new NonlinearForm(fes);
    }
    #else
    return new NonlinearForm(fes);
    #endif
}
BilinearForm *MakeBilinearForm(FiniteElementSpace *fes)
{
    #ifdef MFEM_USE_MPI
    auto pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
    if (pfes)
    {
      return new ParBilinearForm(pfes);
    }
    else
    {
      return new BilinearForm(fes);
    }
    #else
    return new BilinearForm(fes);
    #endif
}
MixedBilinearForm *MakeMixedBilinearForm(FiniteElementSpace *trial_fes, FiniteElementSpace *test_fes)
{
    #ifdef MFEM_USE_MPI
    auto trial_pfes = dynamic_cast<ParFiniteElementSpace*>(trial_fes);
    auto test_pfes = dynamic_cast<ParFiniteElementSpace*>(test_fes);
    if (trial_pfes)
    {
      if (test_pfes)
      {
         return new ParMixedBilinearForm(trial_pfes, test_pfes);
      }
      else
      {
         return new MixedBilinearForm(trial_fes, test_fes);
      }
    }
    else
    {
      return new MixedBilinearForm(trial_fes, test_fes);
    }
    #else
    return new MixedBilinearForm(trial_fes, test_fes);
    #endif
}
}