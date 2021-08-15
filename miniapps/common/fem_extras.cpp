// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
   for (int i=0; i<sCoefs.Size(); i++)
   {
      delete sCoefs[i];
   }
   for (int i=0; i<vCoefs.Size(); i++)
   {
      delete vCoefs[i];
   }
   for (int i=0; i<mCoefs.Size(); i++)
   {
      delete mCoefs[i];
   }
}

Coefficient * CoefFactory::GetScalarCoef(std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;

   return this->GetScalarCoef(buff, input);
}

VectorCoefficient * CoefFactory::GetVectorCoef(std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;

   return this->GetVectorCoef(buff, input);
}

MatrixCoefficient * CoefFactory::GetMatrixCoef(std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;

   return this->GetMatrixCoef(buff, input);
}

Coefficient * CoefFactory::GetScalarCoef(std::string &name,
                                         std::istream &input)
{
   int coef_idx = -1;
   if (name == "ConstantCoefficient")
   {
      double val;
      input >> val;
      coef_idx = sCoefs.Append(new ConstantCoefficient(val));
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
      coef_idx = sCoefs.Append(new PWConstCoefficient(vals));
   }
   else if (name == "FunctionCoefficient")
   {
      int type, index;
      input >> type >> index;
      MFEM_VERIFY(type >=0 && type <= 1,
                  "Invalid Function type read by CoefFactory");
      if (type == 0)
      {
         MFEM_VERIFY(index >=0 && index < ext_sfn.Size(),
                     "Invalid Function index read by CoefFactory");
      }
      else
      {
         MFEM_VERIFY(index >=0 && index < ext_stfn.Size(),
                     "Invalid Time dependent Function index "
                     "read by CoefFactory");
      }
      coef_idx = sCoefs.Append((type == 0) ?
                               new FunctionCoefficient(ext_sfn[index]) :
                               new FunctionCoefficient(ext_stfn[index]));
   }
   else if (name == "GridFunctionCoefficient")
   {
      int index, comp;
      input >> index >> comp;
      MFEM_VERIFY(index >=0 && index < ext_gf.Size(),
                  "Invalid GridFunction index read by CoefFactory");
      coef_idx = sCoefs.Append(new GridFunctionCoefficient(ext_gf[index],
                                                           comp));
   }
   else if (name == "DivergenceGridFunctionCoefficient")
   {
      int index;
      input >> index;
      MFEM_VERIFY(index >=0 && index < ext_gf.Size(),
                  "Invalid GridFunction index for "
                  "DivergenceGridFunctionCoefficient read by CoefFactory");
      coef_idx = sCoefs.Append(
                    new DivergenceGridFunctionCoefficient(ext_gf[index]));
   }
   else if (name == "DeltaCofficient")
   {
      int dim;
      input >> dim;
      MFEM_VERIFY(dim >=1 && dim <= 3,
                  "Invalid dimension for DeltaCoefficient "
                  "read by CoefFactory");
      double x, y, z, s;
      input >> x;
      if (dim > 1) { input >> y; }
      if (dim > 2) { input >> z; }
      input >> s;
      if (dim == 1)
      {
         coef_idx = sCoefs.Append(new DeltaCoefficient(x, s));
      }
      else if (dim == 2)
      {
         coef_idx = sCoefs.Append(new DeltaCoefficient(x, y, s));
      }
      else
      {
         coef_idx = sCoefs.Append(new DeltaCoefficient(x, y, z, s));
      }
   }
   else if (name == "RestrictedCoefficient")
   {
      Coefficient * rc = this->GetScalarCoef(input);
      int nattr;
      input >> nattr;
      Array<int> attr(nattr);
      for (int i=0; i<nattr; i++)
      {
         input >> attr[i];
      }
      coef_idx = sCoefs.Append(new RestrictedCoefficient(*rc, attr));
   }
   else if (name == "ProductCoefficient")
   {
      Coefficient * a = this->GetScalarCoef(input);
      Coefficient * b = this->GetScalarCoef(input);

      if (a != NULL && b != NULL)
      {
         coef_idx = sCoefs.Append(new ProductCoefficient(*a, *b));
      }
   }
   else
   {
      return NULL;
   }
   return sCoefs[--coef_idx];
}

VectorCoefficient * CoefFactory::GetVectorCoef(std::string &name,
                                               std::istream &input)
{
   int coef_idx = -1;
   if (name == "VectorConstantCoefficient")
   {
      int dim;
      input >> dim;
      Vector val(dim);
      for (int i=0; i<dim; i++) { input >> val[i]; }
      coef_idx = vCoefs.Append(new VectorConstantCoefficient(val));
   }
   else if (name == "VectorFunctionCoefficient")
   {
      int dim, type, index;
      input >> dim >> type >> index;
      MFEM_VERIFY(type >=0 && type <= 1,
                  "Invalid Function type read by VecCoefFactory");
      if (type == 0)
      {
         MFEM_VERIFY(index >=0 && index < ext_vfn.Size(),
                     "Invalid Vector Function index read by CoefFactory");
      }
      else
      {
         MFEM_VERIFY(index >=0 && index < ext_vtfn.Size(),
                     "Invalid Time dependent Vector Function index "
                     "read by CoefFactory");
      }
      coef_idx = vCoefs.Append((type==0) ?
                               new VectorFunctionCoefficient(dim, ext_vfn[index]) :
                               new VectorFunctionCoefficient(dim, ext_vtfn[index]));
   }
   else if (name == "VectorArrayCoefficient")
   {
      int dim;
      input >> dim;
      MFEM_VERIFY(dim > 0,
                  "Invalid dimension for VectorArrayCoefficient "
                  "read by CoefFactory");
      VectorArrayCoefficient * vCoef = new VectorArrayCoefficient(dim);
      for (int i=0; i<dim; i++)
      {
         Coefficient *sCoef = this->GetScalarCoef(input);
         vCoef->Set(i, sCoef, false);
      }
      coef_idx = vCoefs.Append(vCoef);
   }
   else if (name == "VectorGridFunctionCoefficient")
   {
      int index;
      input >> index;
      MFEM_VERIFY(index >=0 && index < ext_gf.Size(),
                  "Invalid GridFunction index read by CoefFactory");
      coef_idx = vCoefs.Append(new VectorGridFunctionCoefficient(ext_gf[index]));
   }
   else if (name == "GradientGridFunctionCoefficient")
   {
      int index;
      input >> index;
      MFEM_VERIFY(index >=0 && index < ext_gf.Size(),
                  "Invalid GridFunction index for "
                  "GradientGridFunctionCoefficient read by CoefFactory");
      coef_idx = vCoefs.Append(new GradientGridFunctionCoefficient(ext_gf[index]));
   }
   else if (name == "CurlGridFunctionCoefficient")
   {
      int index;
      input >> index;
      MFEM_VERIFY(index >=0 && index < ext_gf.Size(),
                  "Invalid GridFunction index for "
                  "CurlGridFunctionCoefficient read by CoefFactory");
      coef_idx = vCoefs.Append(new CurlGridFunctionCoefficient(ext_gf[index]));
   }
   else if (name == "VectorDeltaCofficient")
   {
      int dim;
      input >> dim;
      MFEM_VERIFY(dim >=1 && dim <= 3,
                  "Invalid dimension for DeltaCoefficient "
                  "read by CoefFactory");
      Vector dir(dim);
      for (int i=0; i<dim; i++) { input >> dir[i]; }
      double x, y, z, s;
      input >> x;
      if (dim > 1) { input >> y; }
      if (dim > 2) { input >> z; }
      input >> s;
      if (dim == 1)
      {
         coef_idx = vCoefs.Append(new VectorDeltaCoefficient(dir, x, s));
      }
      else if (dim == 2)
      {
         coef_idx = vCoefs.Append(new VectorDeltaCoefficient(dir, x, y, s));
      }
      else
      {
         coef_idx = vCoefs.Append(new VectorDeltaCoefficient(dir, x, y, z, s));
      }
   }
   else if (name == "VectorRestrictedCoefficient")
   {
      VectorCoefficient * rc = this->GetVectorCoef(input);
      int nattr;
      input >> nattr;
      Array<int> attr(nattr);
      for (int i=0; i<nattr; i++)
      {
         input >> attr[i];
      }
      coef_idx = vCoefs.Append(new VectorRestrictedCoefficient(*rc, attr));
   }
   else
   {
      return NULL;
   }
   return vCoefs[--coef_idx];
}

MatrixCoefficient * CoefFactory::GetMatrixCoef(std::string &name,
                                               std::istream &input)
{
   int coef_idx = -1;
   if (name == "MatrixConstantCoefficient")
   {
      int h, w;
      input >> h >> w;
      DenseMatrix val(h, w);
      for (int i=0; i<h; i++)
         for (int j=0; j<w; j++)
         { input >> val(i, j); }
      coef_idx = mCoefs.Append(new MatrixConstantCoefficient(val));
   }
   else if (name == "MatrixFunctionCoefficient")
   {
      int type;
      input >> type;
      MFEM_VERIFY(type >=0 && type <= 2,
                  "Invalid Function type read by MatCoefFactory");
      if (type < 2)
      {
         int dim, index;
         input >> dim >> index;
         if (type == 0)
         {
            MFEM_VERIFY(index >=0 && index < ext_mfn.Size(),
                        "Invalid Matrix Function index read by MatCoefFactory");
         }
         else
         {
            MFEM_VERIFY(index >=0 && index < ext_mtfn.Size(),
                        "Invalid Time dependent Matrix Function index "
                        "read by MatCoefFactory");
         }
         coef_idx = mCoefs.Append((type==0) ?
                                  new MatrixFunctionCoefficient(dim, ext_mfn[index]) :
                                  new MatrixFunctionCoefficient(dim, ext_mtfn[index]));
      }
      else
      {
         int h, w;
         input >> h >> w;
         DenseMatrix val(h, w);
         for (int i=0; i<h; i++)
            for (int j=0; j<w; j++)
            { input >> val(i, j); }
         Coefficient * sCoef = this->GetScalarCoef(input);
         coef_idx = mCoefs.Append(new MatrixFunctionCoefficient(val, *sCoef));
      }
   }
   else if (name == "MatrixRestrictedCoefficient")
   {
      MatrixCoefficient * rc = this->GetMatrixCoef(input);
      int nattr;
      input >> nattr;
      Array<int> attr(nattr);
      for (int i=0; i<nattr; i++)
      {
         input >> attr[i];
      }
      coef_idx = mCoefs.Append(new MatrixRestrictedCoefficient(*rc, attr));
   }
   else
   {
      return NULL;
   }
   return mCoefs[--coef_idx];
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
