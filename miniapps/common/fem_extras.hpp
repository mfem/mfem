// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FEM_EXTRAS
#define MFEM_FEM_EXTRAS

#include "mfem.hpp"
#include <cstddef>

namespace mfem
{

namespace common
{

/** The H1_FESpace class is a FiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an H1_FECollection object.
*/
class H1_FESpace : public FiniteElementSpace
{
public:
   H1_FESpace(Mesh *m,
              const int p, const int space_dim = 3,
              const int type = BasisType::GaussLobatto,
              int vdim = 1, int order = Ordering::byNODES);
   ~H1_FESpace();
private:
   const FiniteElementCollection *FEC_;
};

/** The ND_FESpace class is a FiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an ND_FECollection object.
*/
class ND_FESpace : public FiniteElementSpace
{
public:
   ND_FESpace(Mesh *m, const int p, const int space_dim,
              int vdim = 1, int order = Ordering::byNODES);
   ~ND_FESpace();
private:
   const FiniteElementCollection *FEC_;
};

/** The RT_FESpace class is a FiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an RT_FECollection object.
*/
class RT_FESpace : public FiniteElementSpace
{
public:
   RT_FESpace(Mesh *m, const int p, const int space_dim,
              int vdim = 1, int order = Ordering::byNODES);
   ~RT_FESpace();
private:
   const FiniteElementCollection *FEC_;
};


class CoefFactory
{
protected:
   Array<Coefficient*>       sCoefs; ///< Owned
   Array<VectorCoefficient*> vCoefs; ///< Owned
   Array<MatrixCoefficient*> mCoefs; ///< Owned

   Array<GridFunction*>      ext_gf; ///< Not owned

   Array<double (*)(const Vector &)> ext_sfn;          ///< Not owned
   Array<double (*)(const Vector &, double)> ext_stfn; ///< Not owned

   Array<void (*)(const Vector &, Vector &)> ext_vfn;          ///< Not owned
   Array<void (*)(const Vector &, double, Vector &)> ext_vtfn; ///< Not owned

   Array<void (*)(const Vector &, DenseMatrix &)> ext_mfn;     ///< Not owned
   Array<void (*)(const Vector &, double, DenseMatrix &)> ext_mtfn;

public:
   CoefFactory() {}

   virtual ~CoefFactory();

   int AddExternalGridFunction(GridFunction &gf) { return ext_gf.Append(&gf); }

   int AddExternalFunction(double (*fn)(const Vector &))
   { return ext_sfn.Append(fn); }

   int AddExternalFunction(double (*fn)(const Vector &, double))
   { return ext_stfn.Append(fn); }

   int AddExternalFunction(void (*fn)(const Vector &, Vector &))
   { return ext_vfn.Append(fn); }

   int AddExternalFunction(void (*fn)(const Vector &, double, Vector &))
   { return ext_vtfn.Append(fn); }

   int AddExternalFunction(void (*fn)(const Vector &, DenseMatrix &))
   { return ext_mfn.Append(fn); }

   int AddExternalFunction(void (*fn)(const Vector &, double, DenseMatrix &))
   { return ext_mtfn.Append(fn); }

   virtual Coefficient * GetScalarCoef(std::istream &input);
   virtual Coefficient * GetScalarCoef(std::string &coef_name,
                                       std::istream &input);
   virtual VectorCoefficient * GetVectorCoef(std::istream &input);
   virtual VectorCoefficient * GetVectorCoef(std::string &coef_name,
                                             std::istream &input);
   virtual MatrixCoefficient * GetMatrixCoef(std::istream &input);
   virtual MatrixCoefficient * GetMatrixCoef(std::string &coef_name,
                                             std::istream &input);
};


/// Visualize the given mesh object, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeMesh(socketstream &sock, const char *vishost, int visport,
                   Mesh &mesh, const char *title,
                   int x = 0, int y = 0, int w = 400, int h = 400,
                   const char *keys = NULL);

/// Visualize the given grid function, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    GridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    const char *keys = NULL, bool vec = false);

} // namespace common

} // namespace mfem

#endif
