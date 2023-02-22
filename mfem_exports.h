// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_EXPORTS_HPP
#define MFEM_EXPORTS_HPP

//
// Header that defines symbol export logic.
//

#if defined(MFEM_EXPORTS) || defined(mfem_EXPORTS)
    /* define catch all def */
    #define MFEM_EXPORTS_DEFINED 1
#endif

#if defined(_WIN32)
    // this needs to be defined via CMake when
    // compiling mfem shared libs on Windows
    // (it should not be defined when mfem libs are used)
    #if defined(MFEM_WINDOWS_DLL_EXPORTS)
        #if defined(MFEM_EXPORTS_DEFINED)
            #define MFEM_API __declspec(dllexport)
        #else
            #define MFEM_API __declspec(dllimport)
        #endif
    #else
        #define MFEM_API /* empty for static builds */
    #endif

    #if defined(_MSC_VER)
        /* Turn off warning about lack of DLL interface */
        #pragma warning(disable:4251)
        /* Turn off warning non-dll class is base for dll-interface class */
        #pragma warning(disable:4275)
        /* Turn off warning about identifier truncation */
        #pragma warning(disable:4786)
    #endif
#else
    #define CONDUIT_API /* default */
#endif

#endif // end MFEM_EXPORTS_HPP