// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "url_handler.hpp"
#include "../../general/error.hpp"
#include <cstdlib>
#include <sys/stat.h>

namespace mfem
{

namespace occa
{

FileOpener::FileOpener(const std::string &prefix,
                       const std::string &env_variable)
   : pfx(prefix)
{
   const char *env_path = getenv(env_variable.c_str());
   if (!env_path) { return; }
   std::string path(env_path);
   for (std::size_t start = 0, end; start < path.size(); start = end + 1)
   {
      end = path.find(':', start);
      if (end == std::string::npos)
      {
         AddDir(path.substr(start, end));
         break;
      }
      AddDir(path.substr(start, end - start));
   }
}

bool FileOpener::AddDir(const std::string &dir)
{
   if (dir.size() == 0 || dir[0] != '/') { return false; }
   struct stat dir_stat;
   if (stat(dir.c_str(), &dir_stat)) { return false; }
   if (!S_ISDIR(dir_stat.st_mode)) { return false; }
   paths.push_back(dir + (*dir.rbegin() == '/' ? "" : "/"));
   return true;
}

bool FileOpener::handles(const std::string &filename)
{
   return filename.size() >= pfx.size() &&
          filename.compare(0, pfx.size(), pfx) == 0;
}

std::string FileOpener::expand(const std::string &filename)
{
   std::string sfx(filename.substr(pfx.size()));
   for (std::size_t i = 0; i < paths.size(); i++)
   {
      std::string file = paths[i] + sfx;
      struct stat file_stat;
      if (stat(file.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode))
      {
         return file;
      }
   }
   MFEM_ABORT("invalid url: " << filename);
   return sfx;
}

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
