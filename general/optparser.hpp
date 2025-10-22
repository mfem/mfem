// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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
   enum OptionType { INT, DOUBLE, STRING, STD_STRING, ENABLE, DISABLE, ARRAY, VECTOR };

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

      Option(OptionType type_, void *var_ptr_, const char *short_name_,
             const char *long_name_, const char *description_, bool req)
         : type(type_), var_ptr(var_ptr_), short_name(short_name_),
           long_name(long_name_), description(description_), required(req) { }
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

   /// Construct a command line option parser with 'argc_' and 'argv_'.
   OptionsParser(int argc_, char *argv_[])
      : argc(argc_), argv(argv_)
   {
      error_type = error_idx = 0;
   }

   /** @brief Add a boolean option and set 'var' to receive the value.
       Enable/disable tags are used to set the bool to true/false
       respectively. */
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

   /// Add an integer option and set 'var' to receive the value.
   void AddOption(int *var, const char *short_name, const char *long_name,
                  const char *description, bool required = false)
   {
      options.Append(Option(INT, var, short_name, long_name, description,
                            required));
   }

   /// Add a double option and set 'var' to receive the value.
   void AddOption(real_t *var, const char *short_name, const char *long_name,
                  const char *description, bool required = false)
   {
      options.Append(Option(DOUBLE, var, short_name, long_name, description,
                            required));
   }

   /// Add a string (char*) option and set 'var' to receive the value.
   void AddOption(const char **var, const char *short_name,
                  const char *long_name, const char *description,
                  bool required = false)
   {
      options.Append(Option(STRING, var, short_name, long_name, description,
                            required));
   }

   /// Add a string (std::string) option and set 'var' to receive the value.
   void AddOption(std::string *var, const char *short_name,
                  const char *long_name, const char *description,
                  bool required = false)
   {
      options.Append(Option(STD_STRING, var, short_name, long_name, description,
                            required));
   }

   /** Add an integer array (separated by spaces) option and set 'var' to
       receive the values. */
   void AddOption(Array<int> * var, const char *short_name,
                  const char *long_name, const char *description,
                  bool required = false)
   {
      options.Append(Option(ARRAY, var, short_name, long_name, description,
                            required));
   }

   /** Add a vector (doubles separated by spaces) option and set 'var' to
       receive the values. */
   void AddOption(Vector * var, const char *short_name,
                  const char *long_name, const char *description,
                  bool required = false)
   {
      options.Append(Option(VECTOR, var, short_name, long_name, description,
                            required));
   }

   /** @brief Parse the command-line options.
       Note that this function expects all the options provided through the
       command line to have a corresponding AddOption. In particular, this
       function cannot be used for partial parsing. */
   void Parse();

   /// Parse the command line options, and exit with an error if the options
   /// cannot be parsed successfully. The selected options are printed to the
   /// given stream (defaulting to mfem::out).
   void ParseCheck(std::ostream &out = mfem::out);

   /// Return true if the command line options were parsed successfully.
   bool Good() const { return (error_type == 0); }

   /// Return true if we are flagged to print the help message.
   bool Help() const { return (error_type == 1); }

   /// Print the options
   void PrintOptions(std::ostream &out) const;

   /// Print the error message
   void PrintError(std::ostream &out) const;

   /// Print the help message
   void PrintHelp(std::ostream &out) const;

   /// Print the usage message
   void PrintUsage(std::ostream &out) const;
};

}

#endif
