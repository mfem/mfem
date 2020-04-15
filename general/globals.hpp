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

#ifndef MFEM_GLOBALS_HPP
#define MFEM_GLOBALS_HPP

#include "../config/config.hpp"
#include <iostream>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

/// Simple extension of std::ostream.
/** This class adds the ability to enable and disable the stream. The associated
    std::streambuf and tied std::ostream can be replaced with that of any
    std::ostream. */
class OutStream : public std::ostream
{
protected:
   // Pointer that stores the associated streambuf when output is disabled.
   std::streambuf *m_rdbuf;
   // Pointer that stores the tied ostream when output is disabled.
   std::ostream *m_tie;

   void Init();

public:
   /** @brief Construct an OutStream from the given stream @a out, by using its
       `rdbuf()`. */
   OutStream(std::ostream &out) : std::ostream(NULL) { SetStream(out); }

   /** @brief Replace the `rdbuf()` and `tie()` of the OutStream with that of
       @a out, enabling output. */
   void SetStream(std::ostream &out)
   {
      rdbuf(m_rdbuf = out.rdbuf()); tie(m_tie = out.tie()); Init();
   }

   /// Enable output.
   void Enable() { if (!IsEnabled()) { rdbuf(m_rdbuf); tie(m_tie); } }
   /// Disable output.
   void Disable()
   {
      if (IsEnabled()) { m_rdbuf = rdbuf(NULL); m_tie = tie(NULL); }
   }
   /// Check if output is enabled.
   bool IsEnabled() const { return (rdbuf() != NULL); }
};


/** @brief Global stream used by the library for standard output. Initially it
    uses the same std::streambuf as std::cout, however that can be changed.
    @sa OutStream. */
extern OutStream out;
/** @brief Global stream used by the library for standard error output.
    Initially it uses the same std::streambuf as std::cerr, however that can be
    changed.
    @sa OutStream. */
extern OutStream err;


/** @brief Construct a string of the form "<prefix><myid><suffix>" where the
    integer @a myid is padded with leading zeros to be at least @a width digits
    long. */
/** This is a convenience function, e.g. to redirect mfem::out to individual
    files for each rank, one can use:
    \code
       std::ofstream out_file(MakeParFilename("app_out.", myid).c_str());
       mfem::out.SetStream(out_file);
    \endcode
*/
std::string MakeParFilename(const std::string &prefix, const int myid,
                            const std::string suffix = "", const int width = 6);


#ifdef MFEM_USE_MPI

/** @name MFEM "global" communicator functions.

    Functions for getting and setting the MPI communicator used by the library
    as the "global" communicator.

    This "global" communicator is used for example in the function mfem_error(),
    which is invoked when an error is detected - the "global" communicator is
    used as a parameter to MPI_Abort() to terminate all "global" tasks. */
///@{

/// Get MFEM's "global" MPI communicator.
MPI_Comm GetGlobalMPI_Comm();

/// Set MFEM's "global" MPI communicator.
void SetGlobalMPI_Comm(MPI_Comm comm);

///@}

#endif

} // namespace mfem


// Request a global object to be instantiated for each thread in its TLS.
#define MFEM_THREAD_LOCAL thread_local


// MFEM_DEPRECATED macro to mark obsolete functions and methods
// see https://stackoverflow.com/questions/295120/c-mark-as-deprecated
#if defined(__GNUC__) || defined(__clang__)
#define MFEM_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define MFEM_DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: You need to implement MFEM_DEPRECATED for this compiler")
#define MFEM_DEPRECATED
#endif


#endif
