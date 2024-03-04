#include "helper.hpp"
#include <string>
#include <vector>
#include <iostream>
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
GridFunction *MakeGridFunction(FiniteElementSpace *fes, double *data)
{
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
   if (pfes)
   {
      return new ParGridFunction(pfes, data);
   }
   else
   {
      return new GridFunction(fes, data);
   }
#else
   return new GridFunction(fes, data);
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
LinearForm *MakeLinearForm(FiniteElementSpace *fes, double *data)
{
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
   if (pfes)
   {
      return new ParLinearForm(pfes, data);
   }
   else
   {
      return new LinearForm(fes, data);
   }
#else
   return new LinearForm(fes, data);
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
   isRoot(true)
{
#ifdef MFEM_USE_MPI
   isRoot = Mpi::IsInitialized() ? Mpi::Root() : true;
#endif

}

void TableLogger::Append(const std::string name, double &val)
{
   names.push_back(name);
   data_double.push_back(&val);
   data_order.push_back(dtype::DOUBLE);
}

void TableLogger::Append(const std::string name, int &val)
{
   names.push_back(name);
   data_int.push_back(&val);
   data_order.push_back(dtype::INT);
}

void TableLogger::Print()
{
   if (isRoot)
   {
      if (!var_name_printed)
      {
         var_name_printed = true;
         for (auto &name : names) { os << std::setw(w) << name << "\t";}
         os << "\b\b";
         os << "\n";
         if (file && file->is_open())
         {
            for (auto &name : names) { *file << std::setw(w) << name << "\t";}
            *file << std::endl;
         }
      }
      int i_double(0), i_int(0);
      for (auto d: data_order)
      {
         switch (d)
         {
            case dtype::DOUBLE:
            {
               os << std::setw(w) << *data_double[i_double] << ",\t";
               if (file && file->is_open()) { *file << std::setprecision(8) << std::scientific << *data_double[i_double] << ",\t"; }
               i_double++;
               break;
            }
            case dtype::INT:
            {
               os << std::setw(w) << *data_int[i_int] << ",\t";
               if (file && file->is_open()) { *file << *data_int[i_int] << ",\t"; }
               i_int++;
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
void TableLogger::SaveWhenPrint(std::string filename, std::ios::openmode mode)
{
   if (isRoot)
   {
      filename = filename.append(".txt");
      file.reset(new std::ofstream);
      file->open(filename, mode);
      if (!file->is_open())
      {
         std::string msg("");
         msg += "Cannot open file ";
         msg += filename;
         MFEM_ABORT(msg);
      }
   }
}

}