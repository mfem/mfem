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

#include "fem_extras.hpp"
#include "../../general/text.hpp"

using namespace std;

namespace mfem
{

namespace common
{

H1_FESpace::H1_FESpace(Mesh *m,
                       const int p, const int space_dim, const int type,
                       int vdim, int order)
   : FiniteElementSpace(m, new H1_FECollection(p,space_dim,type),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

H1_FESpace::~H1_FESpace()
{
   delete FEC_;
}

ND_FESpace::ND_FESpace(Mesh *m, const int p, const int space_dim,
                       int vdim, int order)
   : FiniteElementSpace(m, new ND_FECollection(p,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

ND_FESpace::~ND_FESpace()
{
   delete FEC_;
}

RT_FESpace::RT_FESpace(Mesh *m, const int p, const int space_dim,
                       int vdim, int order)
   : FiniteElementSpace(m, new RT_FECollection(p-1,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

RT_FESpace::~RT_FESpace()
{
   delete FEC_;
}

L2_FESpace::L2_FESpace(Mesh *m, const int p, const int space_dim,
                       int vdim, int order)
   : FiniteElementSpace(m, new L2_FECollection(p,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

L2_FESpace::~L2_FESpace()
{
   delete FEC_;
}

DiscreteInterpolationOperator::~DiscreteInterpolationOperator()
{}

DiscreteGradOperator::DiscreteGradOperator(FiniteElementSpace *dfes,
                                           FiniteElementSpace *rfes)
   : DiscreteInterpolationOperator(dfes, rfes)
{
   this->AddDomainInterpolator(new GradientInterpolator);
}

DiscreteCurlOperator::DiscreteCurlOperator(FiniteElementSpace *dfes,
                                           FiniteElementSpace *rfes)
   : DiscreteInterpolationOperator(dfes, rfes)
{
   this->AddDomainInterpolator(new CurlInterpolator);
}

DiscreteDivOperator::DiscreteDivOperator(FiniteElementSpace *dfes,
                                         FiniteElementSpace *rfes)
   : DiscreteInterpolationOperator(dfes, rfes)
{
   this->AddDomainInterpolator(new DivergenceInterpolator);
}

CoefFactory::~CoefFactory()
{
   for (int i=0; i<coefs.Size(); i++)
   {
      delete coefs[i];
   }
}

Coefficient * CoefFactory::operator()(std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;
   return (*this)(buff, input);
}

Coefficient * CoefFactory::operator()(std::string &name, std::istream &input)
{
   if (name == "ConstantCoefficient")
   {
      double val;
      input >> val;
      int c = coefs.Append(new ConstantCoefficient(val));
      return coefs[--c];
   }
   else if (name == "PWConstCoefficient")
   {
      int nvals;
      input >> nvals;
      Vector vals(nvals);
      for (int i=0; i<nvals; i++)
      {
         input >> vals[i];
      }
      int c = coefs.Append(new PWConstCoefficient(vals));
      return coefs[--c];
   }
   else if (name == "FunctionCoefficient")
   {
      int index;
      input >> index;
      MFEM_VERIFY(index >=0 && index < ext_fn.Size(),
                  "Invalid Function index read by CoefFactory");
      int c = coefs.Append(new FunctionCoefficient(ext_fn[index]));
      return coefs[--c];
   }
   else if (name == "GridFunctionCoefficient")
   {
      int index;
      input >> index;
      MFEM_VERIFY(index >=0 && index < ext_gf.Size(),
                  "Invalid GridFunction index read by CoefFactory");
      int c = coefs.Append(new GridFunctionCoefficient(ext_gf[index]));
      return coefs[--c];
   }
   else if (name == "RestrictedCoefficient")
   {
      Coefficient * rc = (*this)(input);
      int nattr;
      input >> nattr;
      Array<int> attr(nattr);
      for (int i=0; i<nattr; i++)
      {
         input >> attr[i];
      }
      int c = coefs.Append(new RestrictedCoefficient(*rc, attr));
      return coefs[--c];
   }
   else
   {
      return NULL;
   }
}

void VisualizeMesh(socketstream &sock, const char *vishost, int visport,
                   Mesh &mesh, const char *title,
                   int x, int y, int w, int h, const char *keys, bool vec)
{
   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (!sock.is_open() || !sock)
      {
         sock.open(vishost, visport);
         sock.precision(8);
         newly_opened = true;
      }
      sock << "solution\n";

      mesh.Print(sock);

      if (newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n";
         if ( keys ) { sock << "keys " << keys << "\n"; }
         else { sock << "keys maaAc\n"; }
         if ( vec ) { sock << "vvv"; }
         sock << endl;
      }

      connection_failed = !sock && !newly_opened;
   }
   while (connection_failed);
}

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    GridFunction &gf, const char *title,
                    int x, int y, int w, int h, const char *keys, bool vec)
{
   Mesh &mesh = *gf.FESpace()->GetMesh();

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (!sock.is_open() || !sock)
      {
         sock.open(vishost, visport);
         sock.precision(8);
         newly_opened = true;
      }
      sock << "solution\n";

      mesh.Print(sock);
      gf.Save(sock);

      if (newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n";
         if ( keys ) { sock << "keys " << keys << "\n"; }
         else { sock << "keys maaAc\n"; }
         if ( vec ) { sock << "vvv"; }
         sock << endl;
      }

      connection_failed = !sock && !newly_opened;
   }
   while (connection_failed);
}

} // namespace common

} // namespace mfem
