// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_KERNEL_REPORTER_HPP
#define MFEM_KERNEL_REPORTER_HPP

#include "../config/config.hpp"

#ifdef MFEM_REPORT_KERNELS

#include "../general/globals.hpp"
#include <set>
#include <sstream>
#include <string>

#define MFEM_STR_(X) #X
#define MFEM_STR(X) MFEM_STR_(X)
#define MFEM_KERNEL_NAME(KernelName)  \
   __FILE__ ":" MFEM_STR(__LINE__) " : " #KernelName

namespace mfem
{

namespace
{

template <typename Last>
static void Stringify_(std::ostream &o, Last &&arg)
{
   o << arg;
}

template <typename T1, typename T2, typename... Rest>
static void Stringify_(std::ostream &o, T1 &&a1, T2 &&a2, Rest&&... rest)
{
   o << int(a1) << ",";
   Stringify_(o, a2, rest...);
}

template <typename... Args>
static std::string Stringify(Args&&... args)
{
   std::stringstream o;
   Stringify_(o, args...);
   return o.str();
}

} // namespace

template <typename... Params>
void ReportFallback(const std::string &kernel_name, Params&&... params)
{
   static std::set<std::string> reported_fallbacks;
   const std::string requested_kernel =
      kernel_name + "<" + Stringify(params...) + ">";
   if (reported_fallbacks.find(requested_kernel) == reported_fallbacks.end())
   {
      reported_fallbacks.insert(requested_kernel);
      mfem::err << "Fallback kernel. Requested "
                << requested_kernel << std::endl;
   }
}

} // namespace mfem

#else // #ifdef MFEM_REPORT_KERNELS

// No-op
#define MFEM_KERNEL_NAME(KernelName) ""
template <typename... T> void ReportFallback(T&&...) { }

#endif

#endif
