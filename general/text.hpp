// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TEXT
#define MFEM_TEXT

#include <istream>
#include <iomanip>
#include <sstream>
#include <string>
#include <limits>
#include <algorithm>

namespace mfem
{

// Utilities for text parsing

inline void skip_comment_lines(std::istream &is, const char comment_char)
{
   while (1)
   {
      is >> std::ws;
      if (is.peek() != comment_char)
      {
         break;
      }
      is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
   }
}

// Check for, and remove, a trailing '\r'.
inline void filter_dos(std::string &line)
{
   if (!line.empty() && *line.rbegin() == '\r')
   {
      line.resize(line.size()-1);
   }
}

// Convert an integer to a string
inline std::string to_string(int i)
{
   std::stringstream ss;
   ss << i;

   // trim leading spaces
   std::string out_str = ss.str();
   out_str = out_str.substr(out_str.find_first_not_of(" \t"));
   return out_str;
}

// Convert an integer to a 0-padded string with the given number of 'digits'
inline std::string to_padded_string(int i, int digits)
{
   std::ostringstream oss;
   oss << std::setw(digits) << std::setfill('0') << i;
   return oss.str();
}

// Convert a string to an int
inline int to_int(const std::string& str)
{
   int i;
   std::stringstream(str) >> i;
   return i;
}

}

#endif
