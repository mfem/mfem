// This file is part of the RBVMS application. For more information and source
// code availability visit https://idoakkerman.github.io/
//
// RBVMS is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.
//------------------------------------------------------------------------------

#ifndef RBVMS_COEFF_HPP
#define RBVMS_COEFF_HPP

#include "mfem.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace mfem;

/// Coefficient class with a function defined in a seperate C-function
class LibCoefficient : public Coefficient
{
protected:
   typedef real_t (*TDFunPtr)(real_t *, int, real_t);

   // typedef double (*TDFunPtr)(double *, int, double);
   TDFunPtr TDFunction = nullptr;
#ifdef _WIN32
   HMODULE libHandle = nullptr;
#else
   void* libHandle = nullptr;
#endif
   Vector x;
   real_t val;

   /** Get a @a funNames C-function from the @a libName library
        - @a libName path+name of the library to use.
        - @a funNames list of function names to look for in the library.
          The first one found in library will be used.
        - @a required specify if the function needs to be defined.
          If true and function is not found, an error will be provided.
          If false the default value @a val is used. */
   void GetLibFunction(std::string libName,
                       std::vector<std::string> funNames,
                       bool required);
public:
   /** Get a @a funName C-function from the @a libName library
        - @a libName path+name of the library to use.
        - @a funName the function name to look for in the library.
        - @a required specify if the function needs to be defined.
          If true and function is not found, an error will be provided.
          If false the default value @a val is used.
        - @a val the value to use if the function is not found.*/
   LibCoefficient(std::string libName, std::string funName,
                  bool required = true, real_t val = 0.0): val(val)
   {
      GetLibFunction(libName, std::vector<std::string>({funName}), required);
   }

   /** Get a @a funNames C-function from the @a libName library
        - @a libName path+name of the library to use.
        - @a funNames list of function names to look for in the library.
          The first one found in library will be used.
        - @a required specify if the function needs to be defined.
          If true and function is not found, an error will be provided.
          If false the default value @a val is used.
        - @a val the value to use if the function is not found.*/
   LibCoefficient(std::string libName, std::vector<std::string> funNames,
                  bool required = true, real_t val = 0.0): val(val)
   {
      GetLibFunction(libName, funNames, required);
   }

   LibCoefficient(const LibCoefficient&) = delete;
   LibCoefficient& operator=(const LibCoefficient&) = delete;
   LibCoefficient(LibCoefficient&&) = delete;
   LibCoefficient& operator=(LibCoefficient&&) = delete;

   bool Foundfunction() { return TDFunction != nullptr; }

   /// Evaluate
   using Coefficient::Eval;

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;

   /// Destructor
   ~LibCoefficient();
};

/// VectorCoefficient class with a function defined in a seperate C-function
class LibVectorCoefficient : public VectorCoefficient
{
protected:
   typedef void (*TDFunPtr)(real_t *, int, real_t, real_t *, int);
   TDFunPtr TDFunction = nullptr;
#ifdef _WIN32
   HMODULE libHandle = nullptr;
#else
   void* libHandle = nullptr;
#endif
   Vector x;

   /** Get a @a funNames C-function from the @a libName library
        - @a libName path+name of the library to use.
        - @a funNames list of function names to look for in the library.
          The first one found in library will be used.
        - @a required specify if the function needs to be defined.
          If true and function is not found, an error will be provided.
          If false the default value @a val is used. */
   void GetLibFunction(std::string libName,
                       std::vector<std::string> funNames,
                       bool required);
public:
   /** Get a @a funName C-function from the @a libName library
        - @a libName path+name of the library to use.
        - @a funName the function name to look for in the library.
        - @a required specify if the function needs to be defined.
          If true and function is not found, an error will be provided.
          If false the default value @a val is used.
        - @a val the value to use if the function is not found.*/
   LibVectorCoefficient(int vdim, std::string libName, std::string funName,
                        bool required = true) : VectorCoefficient(vdim)
   {
      GetLibFunction(libName, std::vector<std::string>({funName}), required);
   }

   LibVectorCoefficient(const LibCoefficient&) = delete;
   LibVectorCoefficient& operator=(const LibCoefficient&) = delete;
   LibVectorCoefficient(LibCoefficient&&) = delete;
   LibVectorCoefficient& operator=(LibCoefficient&&) = delete;
   
   bool Foundfunction() { return TDFunction != nullptr; }

   /** Get a @a funNames C-function from the @a libName library
        - @a libName path+name of the library to use.
        - @a funNames list of function names to look for in the library.
          The first one found in library will be used.
        - @a required specify if the function needs to be defined.
          If true and function is not found, an error will be provided.
          If false the default value @a val is used.
        - @a val the value to use if the function is not found.*/
   LibVectorCoefficient(int vdim, std::string libName,
                        std::vector<std::string> funNames,
                        bool required = true) : VectorCoefficient(vdim)
   {
      GetLibFunction(libName, funNames, required);
   }

   /// Evaluate
   using VectorCoefficient::Eval;
   void Eval(Vector &V,
             ElementTransformation &T,
             const IntegrationPoint &ip) override;

   /// Destructor
   ~LibVectorCoefficient();
};

#endif
