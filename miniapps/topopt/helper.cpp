#include "helper.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
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
   if (trial_pfes && test_pfes)
   {
      return new ParMixedBilinearForm(trial_pfes, test_pfes);
   }
   if (trial_pfes || test_pfes)
   {
      mfem_warning("One of the spaces is parallel but the other is serial. Return a serial mixed-bilinear form");
   }
#endif
   return new MixedBilinearForm(trial_fes, test_fes);
}


TableLogger::TableLogger(std::ostream &os): os(os), w(10),
   var_name_printed(false),
   my_rank(0)
{
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized()) { my_rank = Mpi::WorldRank(); }
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
         for (auto &name : names) { os << std::setw(w) << name << "\t";}
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
               os << std::setw(w) << *monitored_double_data[i_double++] << ",\t";
               if (file) { *file << std::setprecision(8) << std::scientific << *monitored_double_data[i_double++]; }
               break;
            }
            case dtype::INT:
            {
               os << std::setw(w) << *monitored_int_data[i_int++] << ",\t";
               if (file) { *file << *monitored_int_data[i_double++]; }
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
      if (file) { *file << std::endl; }
   }
}
void TableLogger::SaveWhenPrint(const char *filename, std::ios::openmode mode)
{
   file.reset(new std::ofstream(filename, mode));
}

}