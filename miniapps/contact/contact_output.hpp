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

#ifndef MFEM_CONTACT_OUTPUT_HPP
#define MFEM_CONTACT_OUTPUT_HPP

#include <array>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

namespace mfem
{
namespace contact_output
{

constexpr int report_width = 86;

inline std::string Center(const std::string &text, int width)
{
   const int padding = width > int(text.size()) ? width - int(text.size()) : 0;
   return std::string(padding / 2, ' ') + text +
          std::string(padding - padding / 2, ' ');
}

template <typename T>
std::string Scientific(T value)
{
   std::ostringstream text;
   text << std::scientific << std::setprecision(6) << value;
   return text.str();
}

inline void PrintLine(std::ostream &out, char fill = '-')
{
   out << "  " << std::string(report_width, fill) << '\n';
}

inline void PrintBanner(std::ostream &out, const std::string &title)
{
   out << '\n';
   PrintLine(out, '=');
   out << "  " << Center(title, report_width) << '\n';
   PrintLine(out);
}

template <std::size_t N>
void PrintTableRow(std::ostream &out, const std::array<int, N> &widths,
                   const std::array<std::string, N> &cells)
{
   out << "  ";
   for (std::size_t i = 0; i < N; i++)
   {
      if (i) { out << " | "; }
      out << Center(cells[i], widths[i]);
   }
   out << '\n';
}

template <std::size_t N>
void PrintTableSeparator(std::ostream &out, const std::array<int, N> &widths)
{
   out << "  ";
   for (std::size_t i = 0; i < N; i++)
   {
      if (i) { out << "-+-"; }
      out << std::string(widths[i], '-');
   }
   out << '\n';
}

}
}

#endif
