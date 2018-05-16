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

#ifndef MFEM_BACKENDS_OCCA_URL_HANDLER_HPP
#define MFEM_BACKENDS_OCCA_URL_HANDLER_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include <occa.hpp>

namespace mfem
{

namespace occa
{

class FileOpener : public ::occa::io::fileOpener
{
protected:
   std::string pfx; // prefix, e.g. "mfem://"
   std::vector<std::string> paths; // paths to search for prefix replacement

public:
   FileOpener(const std::string &prefix, const std::string &env_variable);

   bool AddDir(const std::string &dir);

   virtual bool handles(const std::string &filename);
   virtual std::string expand(const std::string &filename);
};

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#endif // MFEM_BACKENDS_OCCA_URL_HANDLER_HPP
