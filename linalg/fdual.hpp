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


#ifndef FDUAL_H
#define FDUAL_H

#include <cmath>
#include <type_traits>

namespace mfem
{

namespace ad
{
// Forward AD - simple class for automatic differentiation
template<typename tbase>
class FDual
{
private:
   tbase pr;
   tbase du;

public:

   FDual():pr(0),du(0)
   {
       
   }

   
   template <class fltyp, class = typename 
   std::enable_if<std::is_arithmetic<fltyp>::value>::type>
   FDual(fltyp& f):pr(f),du(0) 
   {

   }
    
   template <class fltyp, class = typename 
   std::enable_if<std::is_arithmetic<fltyp>::value>::type>
   FDual(const fltyp& f):pr(f),du(0) 
   {

   }
   
   FDual(tbase& pr_,tbase& du_):pr(pr_),du(du_)
   {

   }

   FDual(const tbase& pr_,const tbase& du_):pr(pr_),du(du_)
   {
       
   }
   
   FDual(FDual<tbase>& nm):pr(nm.pr),du(nm.du)
   {
       
   }
   
   FDual(const FDual<tbase>& nm):pr(nm.pr),du(nm.du)
   {
       
   }
   
   tbase prim() const
   {
       return pr;
   }
   
   tbase real() const
   {
      return pr;
   }

   tbase dual() const
   {
      return du;
   }

   void set(const tbase& pr_,const tbase& du_)
   {
       pr=pr_;
       du=du_;
   }
   
   void prim(const tbase& pr_)
   {
       pr=pr_;
   }

   void real(const tbase& pr_)
   {
       pr=pr_;
   }
   
   void dual(const tbase& du_)
   {
       du=du_;
   }
   
   FDual<tbase> & operator=(tbase sc_)
   {
      pr=sc_;
      du=tbase(0);
      return *this;
   }

   FDual<tbase> & operator+=(tbase sc_)
   {
      pr=pr+sc_;
      return *this;
   }

   FDual<tbase> & operator-=(tbase sc_)
   {
      pr=pr-sc_;
      return *this;

   }

   FDual<tbase> & operator*=(tbase sc_)
   {
      pr=pr*sc_;
      du=du*sc_;
      return *this;
   }

   FDual<tbase>& operator/=(tbase sc_)
   {
      pr=pr/sc_;
      du=du/sc_;
      return *this;
   }


   FDual<tbase>& operator=(const FDual<tbase> & f)
   {
      pr = f.real();
      du = f.dual();
      return *this;
   }

   FDual<tbase>& operator+=(const FDual<tbase>& f)
   {
      pr += f.real();
      du += f.dual();
      return *this;
   }

   
   FDual<tbase>& operator-=(const FDual<tbase>& f)
   {
      pr -= f.real();
      du -= f.dual();
      return *this;
   }

   FDual<tbase>& operator*=(const FDual<tbase>& f)
   {
      du = du * f.real();
      du = du+ pr * f.dual();
      pr = pr * f.real();
      return *this;
   }

   FDual<tbase>& operator/=(const FDual<tbase>& f_)
   {
      pr = pr / f_.real();
      du = du - pr * f_.dual();
      du = du / f_.real();
      return *this;
   }
};


// non-member functions
// boolean operations
template <typename tbase>
inline
bool operator==(const FDual<tbase>& a1, const FDual<tbase>& a2)
{
   return a1.real() == a2.real();
}

template <typename tbase>
inline
bool operator==(tbase a, const FDual<tbase>& f_)
{
   return a == f_.real();
}

template <typename tbase>
inline
bool operator==(const FDual<tbase>& a, tbase b)
{
   return a.real() == b;
}


template <typename tbase>
inline
bool operator<(const FDual<tbase>& f1, const FDual<tbase>& f2)
{
   return f1.real() < f2.real();
}

template <typename tbase>
inline
bool operator<(const FDual<tbase>& f, tbase a)
{
   return f.real() < a;
}

template <typename tbase>
inline
bool operator<(tbase a, const FDual<tbase>& f)
{
   return a < f.real();
}

template <typename tbase>
inline
bool operator>(const FDual<tbase>& f1, const FDual<tbase>& f2)
{
   return f1.real() > f2.real();
}

template <typename tbase>
inline
bool operator>(const FDual<tbase>& f, tbase a)
{
   return f.real() > a;
}

template <typename tbase>
inline
bool operator>(tbase a, const FDual<tbase>& f)
{
   return (a > f.real());
}

template <typename tbase>
inline
FDual<tbase> operator-(const FDual<tbase>& f)
{
   return FDual<tbase>(-f.real(), -f.dual());
}

template <typename tbase>
inline
FDual<tbase> operator-(const FDual<tbase>& f, tbase a)
{
   return FDual<tbase>(f.real() - a, f.dual());
}

template <typename tbase>
inline
FDual<FDual<tbase>> operator-(const FDual<FDual<tbase>>& f, tbase a)
{
   return FDual<FDual<tbase>>(f.real() - a, f.dual());
}


template <typename tbase>
inline
FDual<tbase> operator+(const FDual<tbase>& f, tbase a)
{
   return FDual<tbase>(f.real() + a, f.dual());
}

template <typename tbase>
inline
FDual<FDual<tbase>> operator+(const FDual<FDual<tbase>>& f, tbase a)
{
   return FDual<FDual<tbase>>(f.real() + a, f.dual());
}


template <typename tbase>
inline
FDual<tbase> operator*(const FDual<tbase>& f, tbase a)
{
   return FDual<tbase>(f.real() * a, f.dual() * a);
}


template <typename tbase>
inline
FDual<tbase> operator/(const FDual<tbase>& f, tbase a)
{
   return FDual<tbase>(f.real() / a, f.dual() / a);
}

template <typename tbase>
inline
FDual<FDual<tbase>> operator/(const FDual<FDual<tbase>>& f, tbase a)
{
   return FDual<FDual<tbase>>(f.real() / a, f.dual() / a);
}

template <typename tbase>
inline
FDual<tbase> operator+(tbase a, const FDual<tbase>& f)
{
   return FDual<tbase>(a + f.real(), f.dual());
}

template <typename tbase>
inline
FDual<FDual<tbase>> operator+(tbase a, const FDual<FDual<tbase>>& f)
{
   return FDual<FDual<tbase>>(a + f.real(), f.dual());
}


template <typename tbase>
inline
FDual<tbase> operator-(tbase a, const FDual<tbase>& f)
{
   return FDual<tbase>(a - f.real(), -f.dual());
}

template <typename tbase>
inline
FDual<FDual<tbase>> operator-(tbase a, const FDual<FDual<tbase>>& f)
{
   return FDual<FDual<tbase>>(a - f.real(), -f.dual());
}

template <typename tbase>
inline
FDual<tbase> operator*(tbase a, const FDual<tbase>& f)
{
   return FDual<tbase>(f.real() * a, f.dual() *a);
}

template <typename tbase>
inline
FDual<FDual<tbase>> operator*(tbase a, const FDual<FDual<tbase>>& f)
{
   return FDual<FDual<tbase>>(f.real() * a, f.dual() *a);
}


template <typename tbase>
inline
FDual<tbase> operator/(tbase a, const FDual<tbase>& f)
{
   a = a / f.real();
   return FDual<tbase>(a, -a * f.dual() / f.real());
}

template <typename tbase>
inline
FDual<tbase> operator+(const FDual<tbase>& f1, const FDual<tbase>& f2)
{
   return FDual<tbase>(f1.real() + f2.real(), f1.dual() + f2.dual());
}

template <typename tbase>
inline
FDual<tbase> operator-(const FDual<tbase>& f1, const FDual<tbase>& f2)
{
   return FDual<tbase>(f1.real() - f2.real(), f1.dual() - f2.dual());
}

template <typename tbase>
inline
FDual<tbase> operator*(const FDual<tbase>& f1, const FDual<tbase>& f2)
{
   return FDual<tbase>(f1.real() * f2.real(),
                       f1.real() * f2.dual() + f1.dual() * f2.real());
}

template <typename tbase>
inline
FDual<tbase> operator/(const FDual<tbase>& f1, const FDual<tbase>& f2)
{
   tbase a=tbase(1)/f2.real();
   tbase b=f1.real()*a;
   return FDual<tbase>(b, (f1.dual() - f2.dual()*b)*a);
}

template <typename tbase>
inline
FDual<tbase> acos(const FDual<tbase>& f)
{
   return FDual<tbase>(acos(f.real()),
                       -f.dual() / sqrt(tbase(1) - f.real() * f.real()));
}

template <>
inline
FDual<double> acos(const FDual<double>& f)
{
   return FDual<double>(std::acos(f.real()),
                        -f.dual() / std::sqrt(double(1) - f.real() * f.real()));
}

template <typename tbase>
inline
FDual<tbase> asin(const FDual<tbase>& f)
{
   return FDual<tbase>(asin(f.real()),
                       f.dual() / sqrt(tbase(1) - f.real() * f.real()));
}

template <>
inline
FDual<double> asin(const FDual<double>& f)
{
   return FDual<double>(std::asin(f.real()),
                        f.dual() / std::sqrt(double(1) - f.real() * f.real()));
}

template <typename tbase>
inline
FDual<tbase> atan(const FDual<tbase>& f)
{
   return FDual<tbase>(atan(f.real()),
                       f.dual() / (tbase(1) + f.real() * f.real()));
}

template <>
inline
FDual<double> atan(const FDual<double>& f)
{
   return FDual<double>(std::atan(f.real()),
                        f.dual() / (double(1) + f.real() * f.real()));
}

template <typename tbase>
inline
FDual<tbase> cos(const FDual<tbase>& f)
{
   return FDual<tbase>(cos(f.real()), -f.dual() * sin(f.real()));
}

template <>
inline
FDual<double> cos(const FDual<double>& f)
{
   return FDual<double>(std::cos(f.real()), -f.dual() * std::sin(f.real()));
}

template <typename tbase>
inline
FDual<tbase> cosh(const FDual<tbase>& f)
{
   return FDual<tbase>(cosh(f.real()), f.dual() * sinh(f.real()));
}

template <>
inline
FDual<double> cosh(const FDual<double>& f)
{
   return FDual<double>(std::cosh(f.real()), f.dual() * std::sinh(f.real()));
}

template <typename tbase>
inline
FDual<tbase> exp(const FDual<tbase>& f)
{
   tbase x = exp(f.real());
   return FDual<tbase>(x, f.dual() * x);
}

template <>
inline
FDual<double> exp(const FDual<double>& f)
{
   double x = std::exp(f.real());
   return FDual<double>(x, f.dual() * x);
}

template <typename tbase>
inline
FDual<tbase> log(const FDual<tbase>& f)
{
   return FDual<tbase>(log(f.real()), f.dual() / f.real());
}

template <>
inline
FDual<double> log(const FDual<double>& f)
{
   return FDual<double>(std::log(f.real()), f.dual() / f.real());
}

template <typename tbase>
inline
FDual<tbase> log10(const FDual<tbase>& f)
{
   return log(f) / log(tbase(10));
}

template <>
inline
FDual<double> log10(const FDual<double>& f)
{
   return log(f) / std::log(double(10));
}

template <typename tbase>
inline
FDual<tbase> pow(const FDual<tbase>& a, const FDual<tbase>& b)
{
   return exp(log(a) * b);
}

template <typename tbase, typename tbase1>
inline
FDual<tbase> pow(const FDual<tbase>& a, const tbase1& b)
{
   return exp(log(a) * tbase(b));
}


template <typename tbase, typename tbase1>
inline
FDual<tbase> pow(const tbase1& a, const FDual<tbase>& b)
{
   return exp(log(tbase(a)) * b);
}

template <>
inline
FDual<double> pow(const double& a, const FDual<double>& b)
{
   return exp(std::log(a) * b);
}

template <typename tbase>
inline
FDual<tbase> sin(const FDual<tbase>& f)
{
   return FDual<tbase>(sin(f.real()), f.dual() * cos(f.real()));
}

template <>
inline
FDual<double> sin(const FDual<double>& f)
{
   return FDual<double>(std::sin(f.real()), f.dual() * std::cos(f.real()));
}

template <typename tbase>
inline
FDual<tbase> sinh(const FDual<tbase>& f)
{
   return FDual<tbase>(sinh(f.real()), f.dual() * cosh(f.real()));
}

template <>
inline
FDual<double> sinh(const FDual<double>& f)
{
   return FDual<double>(std::sinh(f.real()), f.dual() * std::cosh(f.real()));
}


template <typename tbase>
inline
FDual<tbase> sqrt(const FDual<tbase>& f)
{
   tbase a = sqrt(f.real());
   return FDual<tbase>(a, f.dual() / (tbase(2) * a));
}

template <>
inline
FDual<double> sqrt(const FDual<double>& f)
{
   double a = std::sqrt(f.real());
   return FDual<double>(a, f.dual() / (double(2) * a));
}


template <typename tbase>
inline
FDual<tbase> tan(const FDual<tbase>& f)
{
   tbase a = tan(f.real());
   return FDual<tbase>(a,f.dual() * (tbase(1) + a * a));
}

template <>
inline
FDual<double> tan(const FDual<double>& f)
{
   double a = std::tan(f.real());
   return FDual<double>(a,f.dual() * (double(1) + a * a));
}


template <typename tbase>
inline
FDual<tbase> tanh(const FDual<tbase>& f)
{
   tbase a = tanh(f.real());
   return FDual<tbase>(a, f.dual() * (tbase(1) - a * a));
}

template <>
inline
FDual<double> tanh(const FDual<double>& f)
{
   double a = std::tanh(f.real());
   return FDual<double>(a, f.dual() * (double(1) - a * a));
}

}

}




#endif
