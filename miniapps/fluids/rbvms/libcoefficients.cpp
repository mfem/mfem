// This file is part of the RBVMS application. For more information and source
// code availability visit https://idoakkerman.github.io/
//
// RBVMS is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.
//------------------------------------------------------------------------------

#include "libcoefficients.hpp"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
using namespace mfem;

// Get the function from the library
void LibCoefficient::GetLibFunction(std::string libName,
                                    std::vector<std::string> funNames,
                                    bool required)
{
   // Release previously loaded library
   if (libHandle)
   {
#ifdef _WIN32
      FreeLibrary(libHandle);
#else
      dlclose(libHandle);
#endif
      libHandle = nullptr;
   }

   TDFunction = nullptr;

   // Open library
#ifdef _WIN32
   libHandle = LoadLibraryA(libName.c_str());
#else
   libHandle = dlopen (libName.c_str(), RTLD_LAZY);
#endif
   if (!libHandle)
   {
      mfem::out<<"Library "<<libName<<" not found."<<std::endl;
      if (required)
      {
         mfem_error("Can not obtain required function.\n");
      }
      mfem::out <<"Functions:";
      for (std::size_t i = 0; i < funNames.size(); i++)
      {
         mfem::out<<" "<<funNames[i];
      }
      mfem::out<<" will be set to val = "<<val<<std::endl;
      return;
   }

   // Search Function
   TDFunction = nullptr;
   for (std::size_t i = 0; i < funNames.size(); i++)
   {
#ifdef _WIN32
      TDFunction = (TDFunPtr)GetProcAddress(libHandle, funNames[i].c_str());
#else
      TDFunction = (TDFunPtr)dlsym(libHandle, funNames[i].c_str());
#endif
      if (TDFunction) { break; }
   }

   // Check if function is found
   if (!TDFunction)
   {
      mfem::out <<"Functions:";
      for (std::size_t i = 0; i < funNames.size(); i++)
      {
         mfem::out<<" "<<funNames[i];
      }
      mfem::out<<" can not be found in "<<libName<<std::endl;

      if (required)
      {
         mfem_error("Can not obtain required function.\n");
      }
      mfem::out<<"Function will be set to val = "<<val<<std::endl;
   }
}

// Evaluate
real_t LibCoefficient::Eval(ElementTransformation &T,
                            const IntegrationPoint &ip)
{
   // Homegenous if not defined
   if (!TDFunction) { return val; }

   // Evaluate library function
   T.Transform(ip, x);
   return ((*TDFunction)(x.GetData(),x.Size(),GetTime()));
}

// Destructor
LibCoefficient::~LibCoefficient()
{
   TDFunction = nullptr;
   if (libHandle)
   {
#ifdef _WIN32
      FreeLibrary(libHandle);
#else
      dlclose(libHandle);
#endif
      libHandle = nullptr;
   }
}


// Get the function from the library
void LibVectorCoefficient::GetLibFunction(std::string libName,
                                          std::vector<std::string> funNames,
                                          bool required)
{
   // Release previously loaded library
   if (libHandle)
   {
#ifdef _WIN32
      FreeLibrary(libHandle);
#else
      dlclose(libHandle);
#endif
      libHandle = nullptr;
   }

   TDFunction = nullptr;

   // Open library
#ifdef _WIN32
   libHandle = LoadLibraryA(libName.c_str());
#else
   libHandle = dlopen (libName.c_str(), RTLD_LAZY);
#endif
   if (!libHandle)
   {
      mfem::out<<"Library "<<libName<<" not found."<<std::endl;
      if (required)
      {
         mfem_error("Can not obtain required function.\n");
      }
      mfem::out <<"Functions:";
      for (std::size_t i = 0; i < funNames.size(); i++)
      {
         mfem::out<<" "<<funNames[i];
      }
      mfem::out<<" will be set to homogenous."<<std::endl;
      return;
   }

   // Search Function
   TDFunction = nullptr;
   for (std::size_t i = 0; i < funNames.size(); i++)
   {
#ifdef _WIN32
      TDFunction = (TDFunPtr)GetProcAddress(libHandle, funNames[i].c_str());
#else
      TDFunction = (TDFunPtr)dlsym(libHandle, funNames[i].c_str());
#endif
      if (TDFunction) { break; }
   }

   // Check if function is found
   if (!TDFunction)
   {
      mfem::out <<"Functions:";
      for (std::size_t i = 0; i < funNames.size(); i++)
      {
         mfem::out<<" "<<funNames[i];
      }
      mfem::out<<" can not be found in "<<libName<<std::endl;

      if (required)
      {
         mfem_error("Can not obtain required function.\n");
      }
      mfem::out<<"Function will be set to homogenous."<<std::endl;
   }
}

// Evaluate coefficient
void LibVectorCoefficient::Eval(Vector &V,
                                ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V.SetSize(vdim);
   // Homegenous if not defined
   if (!TDFunction)
   {
      V = 0.0;
      return;
   }


   Vector x;
   // Evaluate library function
   T.Transform(ip, x);
   (*TDFunction)(x.GetData(), x.Size(), GetTime(), V.GetData(), V.Size());
}

//  Destructor
LibVectorCoefficient::~LibVectorCoefficient()
{
   TDFunction = nullptr;
   if (libHandle)
   {
#ifdef _WIN32
      FreeLibrary(libHandle);
#else
      dlclose(libHandle);
#endif
      libHandle = nullptr;
   }
}
