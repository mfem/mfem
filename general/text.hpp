// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TEXT
#define MFEM_TEXT

#include "../config/config.hpp"
#include <istream>
#include <iomanip>
#include <sstream>
#include <string>
#include <limits>
#include <algorithm>

namespace mfem
{

// Utilities for text parsing

using std::to_string;

/// Check if the stream starts with @a comment_char. If so skip it.
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

/// Check for, and remove, a trailing '\\r' from and std::string.
inline void filter_dos(std::string &line)
{
   if (!line.empty() && *line.rbegin() == '\r')
   {
      line.resize(line.size()-1);
   }
}

/** @brief Read a string formatted using std::quoted. Return nonzero on error.

    The stream @a in must begin with @a delim. After clearing @a result and
    extracting the opening @a delim, characters are extracted from @a in and
    processed as follows:
    - if the character is @a delim, return 0;
    - if the character is different from @a escape, it is appended to @a result;
    - if the character is @a escape, the next character from @a in is extracted
      and if it is one of @a delim or @a escape, it is appended to @a result;
      otherwise, both @a escape and the character after it are appended to
      @a result; note that the latter case is not possible if the input was
      formatted with std::quoted with the same @a delim and @a escape
      characters.

    If the stream @a in does not begin with @a delim, error code 1 is returned.
    If reading the stream fails, error code 2 is returned. On success, zero is
    returned and the closing @a delim character is the last character extracted
    from @a in. */
inline int parse_quoted_string(std::string &result, std::istream &in,
                               char delim = '"', char escape = '\\')
{
   using tt = std::string::traits_type;  // std::char_traits<char>
   auto equal = [](tt::int_type c1, tt::char_type c2) -> bool
   {
      return tt::eq_int_type(c1, tt::to_int_type(c2));
   };
   result.clear();
   if (!equal(in.peek(), delim)) { return 1; }
   in.get();  // extract delim
   for (auto c = in.get(); !equal(c, delim); c = in.get())
   {
      if (equal(c, escape))
      {
         c = in.get();
         if (!equal(c, escape) && !equal(c, delim)) { result += escape; }
      }
      if (!in) { return 2; }
      result += tt::to_char_type(c);
   }
   return 0;
}

/// Convert an integer to a 0-padded string with the given number of @a digits
inline std::string to_padded_string(int i, int digits)
{
   std::ostringstream oss;
   oss << std::setw(digits) << std::setfill('0') << i;
   return oss.str();
}

/// Convert a string to an int
inline int to_int(const std::string& str)
{
   int i;
   std::stringstream(str) >> i;
   return i;
}

}

#endif
