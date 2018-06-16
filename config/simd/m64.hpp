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
#ifndef MFEM_X86_M64_HPP
#define MFEM_X86_M64_HPP

// ****************************************************************************
// * STD integer
// ****************************************************************************
struct integer {
protected:
  int vec;
public:
  // Constructors
  inline integer():vec(){}
  inline	integer(int i):vec(i){}
  // Convertors
  inline operator int() const { return vec; }
  // Arithmetics
  friend inline integer operator *(const integer &a, const integer &b) { return a*b; }
  // [] operator
  inline int& operator[](int k) { return vec; }
  inline const int& operator[](int k) const { return vec; }
};

// ****************************************************************************
// * STD real type class
// ****************************************************************************
struct real {
 protected:
  double vec;
 public:
  // Constructors
  inline real(): vec(){}
  inline real(double a): vec(a){}
   // Convertors
  inline operator double() const { return vec; }
  // Arithmetics  
  inline real& operator+=(const real &a) { return *this = vec+a; }
  // Mixed vector-scalar operations
  inline real& operator*=(const double &f) { return *this = vec*f; }
  // [] operators
  inline const double& operator[](int k) const { return vec; }
  inline double& operator[](int k) { return vec; }
};

#endif // MFEM_X86_M64_HPP
