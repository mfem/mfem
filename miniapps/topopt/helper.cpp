#include "helper.hpp"
#include <string>
#include <vector>
#include <iostream>
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
MixedBilinearForm *MakeMixedBilinearForm(FiniteElementSpace *trial_fes,
                                         FiniteElementSpace *test_fes)
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


TableLogger::TableLogger(std::ostream &os): os(os), var_name_printed(false),
   my_rank(0)
{
#ifdef MFEM_USE_MPI
   int mpi_is_initialized = Mpi::IsInitialized();
   if (mpi_is_initialized) { my_rank = Mpi::WorldRank(); }
#endif

}

void TableLogger::Append(const std::string name, double &val)
{
   names.push_back(name);
   monitored_double_data.push_back(&val);
   data_order.push_back(dtype::DOUBLE);
}

void TableLogger::Append(const std::string name, int &val)
{
   names.push_back(name);
   monitored_int_data.push_back(&val);
   data_order.push_back(dtype::INT);
}

void TableLogger::Print()
{
   if (my_rank == 0)
   {
      if (!var_name_printed)
      {
         var_name_printed = true;
         for (auto &name : names) { os << std::setw(10) << name << "\t";}
         os << "\b\b";
         os << "\n";
      }
      int i_double(0), i_int(0);
      for (auto d: data_order)
      {
         switch (d)
         {
            case dtype::DOUBLE:
            {
               os << std::setw(10) << *monitored_double_data[i_double++] << ",\t";
               break;
            }
            case dtype::INT:
            {
               os << std::setw(10) << *monitored_int_data[i_int++] << ",\t";
               break;
            }
            default:
            {
               MFEM_ABORT("Unknown data type. See, TableLogger::dtype");
            }
         }
      }
      os << "\b\b"; // remove the last ,\t
      os << std::endl;
   }
}
}