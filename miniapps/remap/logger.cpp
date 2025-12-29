#include "logger.hpp"

namespace mfem
{

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

void GLVis::Append(GridFunction &gf, const char window_title[],
                   const char keys[])
{
   sockets.push_back(std::make_unique<socketstream>(hostname, port, secure));
   socketstream &socket = *sockets.back();
   if (!socket.is_open())
   {
      return;
   }
   Mesh *mesh = gf.FESpace()->GetMesh();
   gfs.Append(&gf);
   qfs.Append(nullptr);
   meshes.Append(mesh);
   socket.precision(8);
#ifdef MFEM_USE_MPI
   if (dynamic_cast<ParGridFunction*>(&gf))
   {
      parallel.Append(true);
      socket << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
   }
   else
   {
      parallel.Append(false);
   }
#endif
   socket << "solution\n" << *mesh << gf;
   if (keys)
   {
      socket << "keys " << keys << std::endl;
   }
   if (window_title)
   {
      socket << "window_title '" << window_title <<"'\n";
   }
   int row = (sockets.size() - 1) / nrWinPerRow;
   int col = (sockets.size() - 1) % nrWinPerRow;
   socket << "window_geometry "
          << w*col << " " << h*row << " "
          << w << " " << h << "\n";
   socket << std::flush;
   if (parallel.Last())
   {
#ifdef MFEM_USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
   }
}

void GLVis::Append(QuadratureFunction &qf, const char window_title[],
                   const char keys[])
{
   sockets.push_back(std::make_unique<socketstream>(hostname, port, secure));
   socketstream &socket = *sockets.back();
   if (!socket.is_open())
   {
      return;
   }
   Mesh *mesh = qf.GetSpace()->GetMesh();
   qfs.Append(&qf);
   gfs.Append(nullptr);
   meshes.Append(mesh);
   socket.precision(8);
#ifdef MFEM_USE_MPI
   if (dynamic_cast<ParMesh*>(mesh))
   {
      parallel.Append(true);
      socket << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
   }
   else
   {
      parallel.Append(false);
   }
#endif
   socket << "quadrature\n" << *mesh << qf;
   if (keys)
   {
      socket << "keys " << keys << std::endl;
   }
   if (window_title)
   {
      socket << "window_title '" << window_title <<"'\n";
   }
   int row = (sockets.size() - 1) / nrWinPerRow;
   int col = (sockets.size() - 1) % nrWinPerRow;
   socket << "window_geometry "
          << w*col << " " << h*row << " "
          << w << " " << h << "\n";
   socket << std::flush;
   if (parallel.Last())
   {
#ifdef MFEM_USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
   }
}

void GLVis::Update()
{
   for (int i=0; i<sockets.size(); i++)
   {
      if (!sockets[i]->is_open() && !sockets[i]->good())
      {
         continue;
      }
#ifdef MFEM_USE_MPI
      if (parallel[i])
      {
         *sockets[i] << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
                        "\n";
      }
#endif
      if (gfs[i])
      {
         *sockets[i] << "solution\n" << *meshes[i] << *gfs[i];
      }
      else if (qfs[i])
      {
         *sockets[i] << "quadrature\n" << *meshes[i] << *qfs[i];
      }
      else
      {
         MFEM_ABORT("Unknown data type. See, GLVis::Update");
      }
      *sockets[i] << std::flush;
      if (parallel[i])
      {
#ifdef MFEM_USE_MPI
         MPI_Barrier(MPI_COMM_WORLD);
#endif
      }
   }
}

} // namespace mfem
