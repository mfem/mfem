// This file is part of the RBVMS application. For more information and source
// code availability visit https://idoakkerman.github.io/
//
// RBVMS is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.
//------------------------------------------------------------------------------

#include "libcoefficients.hpp"
#include <dlfcn.h>

using namespace std;
using namespace mfem;

// Get the function from the library
void LibCoefficient::GetLibFunction(string libName,
                                    vector<string> funNames,
                                    bool required)
{
   // Open library
   libHandle = dlopen (libName.c_str(), RTLD_LAZY);
   if (!libHandle)
   {
      cout<<"Library "<<libName<<" not found."<<endl;
      if (required)
      {
         mfem_error("Can not obtain required function.\n");
      }
      cout <<"Functions:";
      for (int i = 0; i < funNames.size(); i++)
      {
         cout<<" "<<funNames[i];
      }
      cout<<" will be set to val = "<<val<< endl;
      return;
   }

   // Search Function
   TDFunction = nullptr;
   for (int i = 0; i < funNames.size(); i++)
   {
      TDFunction = (TDFunPtr)dlsym(libHandle, funNames[i].c_str());
      if (TDFunction) { break; }
   }

   // Check if function is found
   if (!TDFunction)
   {
      cout <<"Functions:";
      for (int i = 0; i < funNames.size(); i++)
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
   if (libHandle) { dlclose(libHandle); }
}


// Get the function from the library
void LibVectorCoefficient::GetLibFunction(string libName,
                                          vector<string> funNames,
                                          bool required)
{
   // Open library
   libHandle = dlopen (libName.c_str(), RTLD_LAZY);
   if (!libHandle)
   {
      cout<<"Library "<<libName<<" not found."<<endl;
      if (required)
      {
         mfem_error("Can not obtain required function.\n");
      }
      cout <<"Functions:";
      for (int i = 0; i < funNames.size(); i++)
      {
         cout<<" "<<funNames[i];
      }
      cout<<" will be set to homogenous."<<endl;
      return;
   }

   // Search Function
   TDFunction = nullptr;
   for (int i = 0; i < funNames.size(); i++)
   {
      TDFunction = (TDFunPtr)dlsym(libHandle, funNames[i].c_str());
      if (TDFunction) { break; }
   }

   // Check if function is found
   if (!TDFunction)
   {
      cout <<"Functions:";
      for (int i = 0; i < funNames.size(); i++)
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
   // dlclose(libHandle);
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
   if (libHandle) { dlclose(libHandle); }
}
