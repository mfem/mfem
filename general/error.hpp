// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ERROR_HPP
#define MFEM_ERROR_HPP

#include "../config/config.hpp"
#include <iomanip>
#include <sstream>
#ifdef MFEM_USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace mfem
{

/// Action to take when MFEM encounters an error.
enum ErrorAction
{
   MFEM_ERROR_ABORT = 0, /**<
      Abort execution using abort() or MPI_Abort(). This is the default error
      action when the build option MFEM_USE_EXCEPTIONS is set to NO. */
   MFEM_ERROR_THROW      /**<
      Throw an ErrorException. Requires the build option MFEM_USE_EXCEPTIONS=YES
      in which case it is also the default error action. */
};

/// Set the action MFEM takes when an error is encountered.
void set_error_action(ErrorAction action);
/// Get the action MFEM takes when an error is encountered.
ErrorAction get_error_action();

#ifdef MFEM_USE_EXCEPTIONS
/** @brief Exception class thrown when MFEM encounters an error and the current
    ErrorAction is set to MFEM_ERROR_THROW. */
class ErrorException: public std::exception
{
private:
   std::string msg;
public:
   explicit ErrorException(const std::string & in_msg) : msg(in_msg) { }
   virtual ~ErrorException() throw() { }
   virtual const char* what() const throw();
};
#endif

void mfem_backtrace(int mode = 0, int depth = -1);

/** @brief Function called when an error is encountered. Used by the macros
    MFEM_ABORT, MFEM_ASSERT, MFEM_VERIFY. */
void mfem_error(const char *msg = NULL);

/// Function called by the macro MFEM_WARNING.
void mfem_warning(const char *msg = NULL);

}

#ifndef _MFEM_FUNC_NAME
#ifndef _MSC_VER
// This is nice because it shows the class and method name
#define _MFEM_FUNC_NAME __PRETTY_FUNCTION__
// This one is C99 standard.
//#define _MFEM_FUNC_NAME __func__
#else
// for Visual Studio C++
#define _MFEM_FUNC_NAME __FUNCSIG__
#endif
#endif

#define MFEM_LOCATION \
   "\n ... in function: " << _MFEM_FUNC_NAME << \
   "\n ... in file: " << __FILE__ << ':' << __LINE__ << '\n'

// Common error message and abort macro
#define _MFEM_MESSAGE(msg, warn)                                        \
   {                                                                    \
      std::ostringstream mfemMsgStream;                                 \
      mfemMsgStream << std::setprecision(16);                           \
      mfemMsgStream << std::setiosflags(std::ios_base::scientific);     \
      mfemMsgStream << msg << MFEM_LOCATION;                            \
      if (!(warn))                                                      \
         mfem::mfem_error(mfemMsgStream.str().c_str());                 \
      else                                                              \
         mfem::mfem_warning(mfemMsgStream.str().c_str());               \
   }

// Outputs lots of useful information and aborts.
// For all of these functions, "msg" is pushed to an ostream, so you can
// write useful (if complicated) error messages instead of writing
// out to the screen first, then calling abort.  For example:
// MFEM_ABORT( "Unknown geometry type: " << type );
#define MFEM_ABORT(msg) _MFEM_MESSAGE("MFEM abort: " << msg, 0)

// Does a check, and then outputs lots of useful information if the test fails
#define MFEM_VERIFY(x, msg)                             \
   if (!(x))                                            \
   {                                                    \
      _MFEM_MESSAGE("Verification failed: ("            \
                    << #x << ") is false:\n --> " << msg, 0); \
   }

// Use this if the only place your variable is used is in ASSERTs
// For example, this code snippet:
//   int err = MPI_Reduce(ldata, maxdata, 5, MPI_INT, MPI_MAX, 0, MyComm);
//   MFEM_CONTRACT_VAR(err);
//   MFEM_ASSERT( err == 0, "MPI_Reduce gave an error with length "
//                       << ldata );
#define MFEM_CONTRACT_VAR(x) if (false && (&x)+1){}

// Now set up some optional checks, but only if the right flags are on
#ifdef MFEM_DEBUG

#define MFEM_ASSERT(x, msg)                             \
   if (!(x))                                            \
   {                                                    \
      _MFEM_MESSAGE("Assertion failed: ("               \
                    << #x << ") is false:\n --> " << msg, 0); \
   }

// A macro that exposes its argument in debug mode only.
#define MFEM_DEBUG_DO(x) x

#else

// Get rid of all this code, since we're not checking.
#define MFEM_ASSERT(x, msg)

// A macro that exposes its argument in debug mode only.
#define MFEM_DEBUG_DO(x)

#endif

// Generate a warning message - always generated, regardless of MFEM_DEBUG.
#define MFEM_WARNING(msg) _MFEM_MESSAGE("MFEM Warning: " << msg, 1)

// Macro that checks (in MFEM_DEBUG mode) that i is in the range [imin,imax).
#define MFEM_ASSERT_INDEX_IN_RANGE(i,imin,imax) \
   MFEM_ASSERT((imin) <= (i) && (i) < (imax), \
   "invalid index " #i << " = " << (i) << \
   ", valid range is [" << (imin) << ',' << (imax) << ')')


// Additional abort functions for HIP
#if defined(MFEM_USE_HIP)
template<typename T>
__host__ void abort_msg(T & msg)
{
   MFEM_ABORT(msg);
}

template<typename T>
__device__ void abort_msg(T & msg)
{
   abort();
}
#endif

// Abort inside a device kernel
#if defined(__CUDA_ARCH__)
#define MFEM_ABORT_KERNEL(msg) \
   {                           \
      printf(msg);             \
      asm("trap;");            \
   }
#elif defined(MFEM_USE_HIP)
#define MFEM_ABORT_KERNEL(msg) \
   {                           \
      abort_msg(msg);          \
   }
#else
#define MFEM_ABORT_KERNEL(msg) MFEM_ABORT(msg)
#endif

#endif
