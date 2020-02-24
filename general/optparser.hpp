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

#ifndef MFEM_OPTPARSER
#define MFEM_OPTPARSER

#include "../config/config.hpp"
#include "array.hpp"

namespace mfem
{

class Vector;

/** Class for parsing command-line options.

    The class is initialized with argc and argv, and new options are added with
    the AddOption method. Currently options of type bool, int, double, char*,
    mfem::Array<int>, and mfem::Vector are supported.

    See the MFEM examples for sample use.
*/
class OptionsParser
{
public:
   enum OptionType { INT, DOUBLE, STRING, ENABLE, DISABLE, ARRAY, VECTOR };

private:
   struct Option
   {
      OptionType type;
      void *var_ptr;
      const char *short_name;
      const char *long_name;
      const char *description;
      bool required;

      Option() = default;

      Option(OptionType _type, void *_var_ptr, const char *_short_name,
             const char *_long_name, const char *_description, bool req)
         : type(_type), var_ptr(_var_ptr), short_name(_short_name),
           long_name(_long_name), description(_description), required(req) { }
   };

   int argc;
   char **argv;
   Array<Option> options;
   Array<int> option_check;
   // error_type can be:
   //  0 - no error
   //  1 - print help message
   //  2 - unrecognized option at argv[error_idx]
   //  3 - missing argument for the last option argv[argc-1]
   //  4 - option with index error_idx is specified multiple times
   //  5 - invalid argument in argv[error_idx] for option in argv[error_idx-1]
   //  6 - required option with index error_idx is missing
   int error_type, error_idx;

   static void WriteValue(const Option &opt, std::ostream &out);

public:
   OptionsParser(int _argc, char *_argv[])
      : argc(_argc), argv(_argv)
   {
      error_type = error_idx = 0;
   }
   void AddOption(bool *var, const char *enable_short_name,
                  const char *enable_long_name, const char *disable_short_name,
                  const char *disable_long_name, const char *description,
                  bool required = false)
   {
      options.Append(Option(ENABLE, var, enable_short_name, enable_long_name,
                            description, required));
      options.Append(Option(DISABLE, var, disable_short_name, disable_long_name,
                            description, required));
   }
   void AddOption(int *var, const char *short_name, const char *long_name,
                  const char *description, bool required = false)
   {
      options.Append(Option(INT, var, short_name, long_name, description,
                            required));
   }
   void AddOption(double *var, const char *short_name, const char *long_name,
                  const char *description, bool required = false)
   {
      options.Append(Option(DOUBLE, var, short_name, long_name, description,
                            required));
   }
   void AddOption(const char **var, const char *short_name,
                  const char *long_name, const char *description,
                  bool required = false)
   {
      options.Append(Option(STRING, var, short_name, long_name, description,
                            required));
   }
   void AddOption(Array<int> * var, const char *short_name,
                  const char *long_name, const char *description,
                  bool required = false)
   {
      options.Append(Option(ARRAY, var, short_name, long_name, description,
                            required));
   }
   void AddOption(Vector * var, const char *short_name,
                  const char *long_name, const char *description,
                  bool required = false)
   {
      options.Append(Option(VECTOR, var, short_name, long_name, description,
                            required));
   }

   /** Parse the command-line options. Note that this function expects all the
       options provided through the command line to have a corresponding
       AddOption. In particular, this function cannot be used for partial
       parsing. */
   void Parse();
   bool Good() const { return (error_type == 0); }
   bool Help() const { return (error_type == 1); }
   void PrintOptions(std::ostream &out) const;
   void PrintError(std::ostream &out) const;
   void PrintHelp(std::ostream &out) const;
   void PrintUsage(std::ostream &out) const;
};

}

#endif
