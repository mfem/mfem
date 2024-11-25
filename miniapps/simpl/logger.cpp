#include "logger.hpp"

TableLogger::TableLogger(std::ostream &os)
   : os(os), w(14), var_name_printed(false), isRoot(true)
{
#ifdef MFEM_USE_MPI
   isRoot = mfem::Mpi::IsInitialized() ? mfem::Mpi::Root() : true;
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

void TableLogger::Print(bool print_varname)
{
   if (isRoot)
   {
      if (!var_name_printed || print_varname)
      {
         for (auto &name : names)
         {
            os << std::setw(w) << std::setfill(' ') << name << ",\t";
         }
         os << "\b\b";
         os << std::endl;
         if (!var_name_printed && file && file->is_open())
         {
            for (int i=0; i<names.size() - 1; i++)
            {
               *file << std::setw(w) << std::setfill(' ') << names[i] << ",\t";
            }
            *file << std::setw(w) << std::setfill(' ') << names.back() << std::endl;
         }
         var_name_printed = true;
      }
      int i(0), i_double(0), i_int(0);
      for (int i=0; i<data_order.size(); i++)
      {
         auto d = data_order[i];
         switch (d)
         {
            case dtype::DOUBLE:
            {
               os << std::setw(w) << *data_double[i_double];
               if (file && file->is_open())
               {
                  *file << std::setprecision(8) << std::scientific << std::setw(w) 
                        << std::setfill(' ') << *data_double[i_double];
               }
               i_double++;
               break;
            }
            case dtype::INT:
            {
               os << std::setw(w) << *data_int[i_int];
               if (file && file->is_open())
               {
                  *file << std::setw(w) << std::setfill(' ') << *data_int[i_int];
               }
               i_int++;
               break;
            }
            default:
            {
               MFEM_ABORT("Unknown data type. See, TableLogger::dtype");
            }
         }
         if (i < data_order.size() - 1)
         {
            os << ",\t";
            *file << ",\t";
         }
      }
      os << std::endl;
      if (file)
      {
         *file << std::endl;
      }
   }
}

void TableLogger::SaveWhenPrint(std::string filename, std::ios::openmode mode)
{
   if (isRoot)
   {
      filename = filename.append(".csv");
      file.reset(new std::fstream);
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
