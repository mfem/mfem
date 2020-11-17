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

#include "optparser.hpp"
#include "../linalg/vector.hpp"
#include <cctype>

namespace mfem
{

using namespace std;

int isValidAsInt(char * s)
{
   if ( s == NULL || *s == '\0' )
   {
      return 0;   // Empty string
   }

   if ( *s == '+' || *s == '-' )
   {
      ++s;
   }

   if ( *s == '\0')
   {
      return 0;   // sign character only
   }

   while (*s)
   {
      if ( !isdigit(*s) )
      {
         return 0;
      }
      ++s;
   }

   return 1;
}

int isValidAsDouble(char * s)
{
   // A valid floating point number for atof using the "C" locale is formed by
   // - an optional sign character (+ or -),
   // - followed by a sequence of digits, optionally containing a decimal-point
   //   character (.),
   // - optionally followed by an exponent part (an e or E character followed by
   //   an optional sign and a sequence of digits).

   if ( s == NULL || *s == '\0' )
   {
      return 0;   // Empty string
   }

   if ( *s == '+' || *s == '-' )
   {
      ++s;
   }

   if ( *s == '\0')
   {
      return 0;   // sign character only
   }

   while (*s)
   {
      if (!isdigit(*s))
      {
         break;
      }
      ++s;
   }

   if (*s == '\0')
   {
      return 1;   // s = "123"
   }

   if (*s == '.')
   {
      ++s;
      while (*s)
      {
         if (!isdigit(*s))
         {
            break;
         }
         ++s;
      }
      if (*s == '\0')
      {
         return 1;   // this is a fixed point double s = "123." or "123.45"
      }
   }

   if (*s == 'e' || *s == 'E')
   {
      ++s;
      return isValidAsInt(s);
   }
   else
   {
      return 0;   // we have encounter a wrong character
   }
}

void parseArray(char * str, Array<int> & var)
{
   var.SetSize(0);
   std::stringstream input(str);
   int val;
   while ( input >> val)
   {
      var.Append(val);
   }
}

void parseVector(char * str, Vector & var)
{
   int nentries = 0;
   double val;
   {
      std::stringstream input(str);
      while ( input >> val)
      {
         ++nentries;
      }
   }

   var.SetSize(nentries);
   {
      nentries = 0;
      std::stringstream input(str);
      while ( input >> val)
      {
         var(nentries++) = val;
      }
   }
}

void OptionsParser::Parse()
{
   option_check.SetSize(options.Size());
   option_check = 0;
   for (int i = 1; i < argc; )
   {
      if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
      {
         // print help message
         error_type = 1;
         return;
      }

      for (int j = 0; true; j++)
      {
         if (j >= options.Size())
         {
            // unrecognized option
            error_type = 2;
            error_idx = i;
            return;
         }

         if (strcmp(argv[i], options[j].short_name) == 0 ||
             strcmp(argv[i], options[j].long_name) == 0)
         {
            OptionType type = options[j].type;

            if ( option_check[j] )
            {
               error_type = 4;
               error_idx = j;
               return;
            }
            option_check[j] = 1;

            i++;
            if (type != ENABLE && type != DISABLE && i >= argc)
            {
               // missing argument
               error_type = 3;
               error_idx = j;
               return;
            }

            int isValid = 1;
            switch (options[j].type)
            {
               case INT:
                  isValid = isValidAsInt(argv[i]);
                  *(int *)(options[j].var_ptr) = atoi(argv[i++]);
                  break;
               case DOUBLE:
                  isValid = isValidAsDouble(argv[i]);
                  *(double *)(options[j].var_ptr) = atof(argv[i++]);
                  break;
               case STRING:
                  *(const char **)(options[j].var_ptr) = argv[i++];
                  break;
               case ENABLE:
                  *(bool *)(options[j].var_ptr) = true;
                  option_check[j+1] = 1;  // Do not allow the DISABLE Option
                  break;
               case DISABLE:
                  *(bool *)(options[j].var_ptr) = false;
                  option_check[j-1] = 1;  // Do not allow the ENABLE Option
                  break;
               case ARRAY:
                  parseArray(argv[i++], *(Array<int>*)(options[j].var_ptr) );
                  break;
               case VECTOR:
                  parseVector(argv[i++], *(Vector*)(options[j].var_ptr) );
                  break;
            }

            if (!isValid)
            {
               error_type = 5;
               error_idx = i;
               return;
            }

            break;
         }
      }
   }

   // check for missing required options
   for (int i = 0; i < options.Size(); i++)
      if (options[i].required &&
          (option_check[i] == 0 ||
           (options[i].type == ENABLE && option_check[++i] == 0)))
      {
         error_type = 6; // required option missing
         error_idx = i; // for a boolean option i is the index of DISABLE
         return;
      }

   error_type = 0;
}

void OptionsParser::WriteValue(const Option &opt, std::ostream &out)
{
   switch (opt.type)
   {
      case INT:
         out << *(int *)(opt.var_ptr);
         break;

      case DOUBLE:
         out << *(double *)(opt.var_ptr);
         break;

      case STRING:
         out << *(const char **)(opt.var_ptr);
         break;

      case ARRAY:
      {
         Array<int> &list = *(Array<int>*)(opt.var_ptr);
         out << '\'';
         if (list.Size() > 0)
         {
            out << list[0];
         }
         for (int i = 1; i < list.Size(); i++)
         {
            out << ' ' << list[i];
         }
         out << '\'';
         break;
      }

      case VECTOR:
      {
         Vector &list = *(Vector*)(opt.var_ptr);
         out << '\'';
         if (list.Size() > 0)
         {
            out << list(0);
         }
         for (int i = 1; i < list.Size(); i++)
         {
            out << ' ' << list(i);
         }
         out << '\'';
         break;
      }

      default: // provide a default to suppress warning
         break;
   }
}

void OptionsParser::PrintOptions(ostream &out) const
{
   static const char *indent = "   ";

   out << "Options used:\n";
   for (int j = 0; j < options.Size(); j++)
   {
      OptionType type = options[j].type;

      out << indent;
      if (type == ENABLE)
      {
         if (*(bool *)(options[j].var_ptr) == true)
         {
            out << options[j].long_name;
         }
         else
         {
            out << options[j+1].long_name;
         }
         j++;
      }
      else
      {
         out << options[j].long_name << " ";
         WriteValue(options[j], out);
      }
      out << '\n';
   }
}

void OptionsParser::PrintError(ostream &out) const
{
   static const char *line_sep = "";

   out << line_sep;
   switch (error_type)
   {
      case 2:
         out << "Unrecognized option: " << argv[error_idx] << '\n' << line_sep;
         break;

      case 3:
         out << "Missing argument for the last option: " << argv[argc-1] << '\n'
             << line_sep;
         break;

      case 4:
         if (options[error_idx].type == ENABLE )
            out << "Option " << options[error_idx].long_name << " or "
                << options[error_idx + 1].long_name
                << " provided multiple times\n" << line_sep;
         else if (options[error_idx].type == DISABLE)
            out << "Option " << options[error_idx - 1].long_name << " or "
                << options[error_idx].long_name
                << " provided multiple times\n" << line_sep;
         else
            out << "Option " << options[error_idx].long_name
                << " provided multiple times\n" << line_sep;
         break;

      case 5:
         out << "Wrong option format: " << argv[error_idx - 1] << " "
             << argv[error_idx] << '\n' << line_sep;
         break;

      case 6:
         out << "Missing required option: " << options[error_idx].long_name
             << '\n' << line_sep;
         break;
   }
   out << endl;
}

void OptionsParser::PrintHelp(ostream &out) const
{
   static const char *indent = "   ";
   static const char *seprtr = ", ";
   static const char *descr_sep = "\n\t";
   static const char *line_sep = "";
   static const char *types[] = { " <int>", " <double>", " <string>", "", "",
                                  " '<int>...'", " '<double>...'"
                                };

   out << indent << "-h" << seprtr << "--help" << descr_sep
       << "Print this help message and exit.\n" << line_sep;
   for (int j = 0; j < options.Size(); j++)
   {
      OptionType type = options[j].type;

      out << indent << options[j].short_name << types[type]
          << seprtr << options[j].long_name << types[type]
          << seprtr;
      if (options[j].required)
      {
         out << "(required)";
      }
      else
      {
         if (type == ENABLE)
         {
            j++;
            out << options[j].short_name << types[type] << seprtr
                << options[j].long_name << types[type] << seprtr
                << "current option: ";
            if (*(bool *)(options[j].var_ptr) == true)
            {
               out << options[j-1].long_name;
            }
            else
            {
               out << options[j].long_name;
            }
         }
         else
         {
            out << "current value: ";
            WriteValue(options[j], out);
         }
      }
      out << descr_sep;

      if (options[j].description)
      {
         out << options[j].description << '\n';
      }
      out << line_sep;
   }
}

void OptionsParser::PrintUsage(ostream &out) const
{
   static const char *line_sep = "";

   PrintError(out);
   out << "Usage: " << argv[0] << " [options] ...\n" << line_sep
       << "Options:\n" << line_sep;
   PrintHelp(out);
}

}
