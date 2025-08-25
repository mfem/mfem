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

bool GLVis::Append(GridFunction *gf, QuadratureFunction *qf,
                   std::string_view window_title, std::string_view keys)
{
   MFEM_VERIFY((gf == nullptr && qf != nullptr)
               || (gf != nullptr && qf == nullptr),
               "Either GridFunction or QuadratureFunction must be provided, "
               "but not both.");
   bool is_gf = gf != nullptr;
   sockets.push_back(std::make_unique<socketstream>(hostname, port, secure));
   socketstream &socket = *sockets.back();
   if (!socket.is_open() || !socket.good())
   {
      MFEM_WARNING("GLVis: Cannot connect to " << hostname << ":" << port);
      sockets.back().reset();
      sockets.pop_back();
      return false;
   }
   socket.precision(8);
   gfs.Append(gf);
   qfs.Append(qf);

   Mesh *mesh;
   if (is_gf) { mesh = gf->FESpace()->GetMesh(); }
   else { mesh = qf->GetSpace()->GetMesh(); }
   meshes.Append(mesh);

   cfs.Append(nullptr);
   vcfs.Append(nullptr);

#ifdef MFEM_USE_MPI
   parallel.Append(false);
   myrank.Append(0);
   nrrank.Append(1);
   ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
   if (pmesh != nullptr)
   {
      parallel.Last() = true;
      nrrank.Last() = pmesh->GetNRanks();
      myrank.Last() = pmesh->GetMyRank();
      socket << "parallel " << nrrank.Last() << " " << myrank.Last() <<
                "\n";
   }
#endif
   if (is_gf)
   {
      socket << "solution\n" << *mesh << *gf;
   }
   else
   {
      socket << "quadrature\n" << *mesh << *qf << "\n";
   }


   if (!keys.empty())
   {
      socket << "keys " << keys << "\n";
      bool hasQ=false;
      if (!is_gf)
      {
         auto end_pos = std::min(keys.find(' '), keys.find('\n'));
         std::string_view actual_keys = keys.substr(0, end_pos);
         if (actual_keys.find('Q') != std::string_view::npos) { hasQ = true; }
      }
      qfkey_has_Q.Append(hasQ);
   }
   if (!window_title.empty())
   {
      socket << "window_title '" << window_title <<"'\n";
   }
   int row = (sockets.size() - 1) / nrWinPerRow;
   int col = (sockets.size() - 1) % nrWinPerRow;
   socket << " window_geometry "
          << w*col << " " << h*row << " "
          << w << " " << h << "\n";
   socket << std::flush;
#ifdef MFEM_USE_MPI
   if (parallel.Last())
   {
      MPI_Comm comm = static_cast<ParMesh*>(meshes.Last())->GetComm();
      MPI_Barrier(comm);
   }
#endif
   return true;
}

void GLVis::Append(Coefficient &cf, QuadratureSpace &qs,
                   std::string_view window_title,
                   std::string_view keys)
{
   owned_qfs.push_back(std::make_unique<QuadratureFunction>(qs));
   cf.Project(*owned_qfs.back());
   if (Append(nullptr, owned_qfs.back().get(), window_title, keys))
   {
      cfs.Last() = &cf;
   }
}

void GLVis::Append(VectorCoefficient &cf, QuadratureSpace &qs,
                   std::string_view window_title,
                   std::string_view keys)
{
   owned_qfs.push_back(std::make_unique<QuadratureFunction>(qs, cf.GetVDim()));
   cf.Project(*owned_qfs.back());
   if (Append(nullptr, owned_qfs.back().get(), window_title, keys))
   {
      vcfs.Last() = &cf;
   }
}

void GLVis::Update()
{
   for (int i=0; i<sockets.size(); i++)
   {
      if (!sockets[i]->is_open() || !sockets[i]->good())
      {
         MFEM_WARNING("GLVis: Connection to " << hostname << ":" << port
                      << " for window " << i+1 << " lost.");
         continue;
      }
#ifdef MFEM_USE_MPI
      if (parallel[i])
      {
         *sockets[i] << "parallel " << nrrank[i] << " " << myrank[i] <<
                        "\n";
      }
#endif
      if (gfs[i] != nullptr)
      {
         *sockets[i] << "solution\n" << *meshes[i] << *gfs[i];
      }
      else if (qfs[i] != nullptr)
      {
         if (cfs[i] != nullptr) { cfs[i]->Project(*qfs[i]); }
         else if (vcfs[i] != nullptr) { vcfs[i]->Project(*qfs[i]); }
         *sockets[i] << "quadrature\n" << *meshes[i] << *qfs[i];
         if (qfkey_has_Q[i]) { *sockets[i] << "keys QQQ\n"; }
      }
      *sockets[i] << std::flush;
#ifdef MFEM_USE_MPI
      if (parallel[i])
      {
         MPI_Comm comm = static_cast<ParMesh*>(meshes[i])->GetComm();
         MPI_Barrier(comm);
      }
#endif
   }
}

} // namespace mfem
