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
using namespace std;
using namespace mfem;

// Get the function from the library
void LibCoefficient::GetLibFunction(string libName,
                                    vector<string> funNames,
                                    bool required)
{
   // Open library
#ifdef _WIN32
   libhandle = LoadLibraryA(libName.c_str());
#else
   libHandle = dlopen (libName.c_str(), RTLD_LAZY);
#endif
   if (!libHandle)
   {
      cout<<"Library "<<libName<<" not found."<<endl;
      if (required)
      {
         mfem_error("Can not obtain required function.\n");
      }
      cout <<"Functions:";
      for (std::size_t i = 0; i < funNames.size(); i++)
      {
         cout<<" "<<funNames[i];
      }
      cout<<" will be set to val = "<<val<< endl;
      return;
   }

   // Search Function
   TDFunction = nullptr;
   for (std::size_t i = 0; i < funNames.size(); i++)
   {
#ifdef _WIN32
      TDFunction = (TDFunPtr)GetProcAddress((HMODULE)libHandle, funNames[i].c_str());
#else
      TDFunction = (TDFunPtr)dlsym(libHandle, funNames[i].c_str());
#endif
      if (TDFunction) { break; }
   }

   // Check if function is found
   if (!TDFunction)
   {
      cout <<"Functions:";
      for (std::size_t i = 0; i < funNames.size(); i++)
      {
         cout<<" "<<funNames[i];
      }
      cout<<" can not be found in "<<libName<<endl;

      if (required)
      {
         mfem_error("Can not obtain required function.\n");
      }
      cout<<"Function will be set to val = "<<val<< endl;
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
   if (libHandle)
   {
#ifdef _WIN32
      FreeLibrary((HMODULE)libhandle);
#else
      dlclose(libHandle);
#endif
   }
}


// Get the function from the library
void LibVectorCoefficient::GetLibFunction(string libName,
                                          vector<string> funNames,
                                          bool required)
{
   // Open library
#ifdef _WIN32
   libhandle = LoadLibraryA(libName.c_str());
#else
   libHandle = dlopen (libName.c_str(), RTLD_LAZY);
#endif
   if (!libHandle)
   {
      cout<<"Library "<<libName<<" not found."<<endl;
      if (required)
      {
         mfem_error("Can not obtain required function.\n");
      }
      cout <<"Functions:";
      for (std::size_t i = 0; i < funNames.size(); i++)
      {
         cout<<" "<<funNames[i];
      }
      cout<<" will be set to homogenous."<<endl;
      return;
   }

   // Search Function
   TDFunction = nullptr;
   for (std::size_t i = 0; i < funNames.size(); i++)
   {
#ifdef _WIN32
      TDFunction = (TDFunPtr)GetProcAddress((HMODULE)libHandle, funNames[i].c_str());
#else
      TDFunction = (TDFunPtr)dlsym(libHandle, funNames[i].c_str());
#endif
      if (TDFunction) { break; }
   }

   // Check if function is found
   if (!TDFunction)
   {
      cout <<"Functions:";
      for (std::size_t i = 0; i < funNames.size(); i++)
      {
         cout<<" "<<funNames[i];
      }
      cout<<" can not be found in "<<libName<<endl;

      if (required)
      {
         mfem_error("Can not obtain required function.\n");
      }
      cout<<"Function will be set to homogenous."<<endl;
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

   // Evaluate library function
   T.Transform(ip, x);
   (*TDFunction)(x.GetData(), x.Size(), GetTime(), V.GetData(), V.Size());
}

//  Destructor
LibVectorCoefficient::~LibVectorCoefficient()
{
   if (libHandle)
   {
#ifdef _WIN32
      FreeLibrary(l(HMODULE)ibhandle);
#else
      dlclose(libHandle);
#endif
   }
}
