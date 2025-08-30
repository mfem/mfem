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

#ifndef MFEM_VIS_OBJECT
#define MFEM_VIS_OBJECT

#include "../common/fem_extras.hpp"
#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

using common::L2_ParFESpace;

namespace plasma
{

class VisObjectBase
{
protected:
   static int Sw, Sh;
   static int Mt, Mb, Ml, Mr;
   static int Fw, Fh;
   static int Ww, Wh;
   static int offx, offy, offs;
   static int Wx, Wy, Sx, Sy;

public:

   static void SetScreenDimensions(int width, int height)
   { Sw = width; Sh = height; }

   static void SetMenuDimensions(int top, int bottom, int left, int right)
   { Mt = top; Mb = bottom; Ml = left; Mr = right; }

   static void SetFrameDimensions(int width, int height)
   { Fw = width; Fh = height; SetOffsets(offs); }

   static void SetWindowDimensions(int width, int height)
   { Ww = width; Wh = height; SetOffsets(offs); }

   static void SetOffsets(int new_screen_offset = 10)
   { offx = Ww + Fw; offy = Wh + Fh; offs = new_screen_offset; }

   static void IncrementGLVisWindowPosition();

   virtual void RegisterVisItFields(VisItDataCollection & visit_dc) = 0;

   virtual void DisplayToGLVis() = 0;

   virtual void Update() = 0;
};

class ScalarFieldVisObject : public VisObjectBase
{
protected:
   bool cyl_;
   bool pseudo_;

   int dim_;

   std::string field_name_;

   GridFunction * v_; // Field in problem domain (L2)

public:
   ScalarFieldVisObject(const std::string & field_name,
                        ParFiniteElementSpace *sfes,
                        bool cyl, bool pseudo);

   virtual ~ScalarFieldVisObject();

   virtual void RegisterVisItFields(VisItDataCollection & visit_dc);

   virtual void PrepareVisField(const ParGridFunction &u);

   virtual void PrepareVisField(Coefficient &uCoef);

   virtual void DisplayToGLVis();

   virtual void Update();
};

class ComplexScalarFieldVisObject : public VisObjectBase
{
protected:
   bool cyl_;
   bool pseudo_;

   std::string field_name_;

   ComplexGridFunction * v_; // Complex field in problem domain (L2)

public:
   ComplexScalarFieldVisObject(const std::string & field_name,
                               std::shared_ptr<L2_ParFESpace> sfes,
                               bool cyl, bool pseudo);

   virtual ~ComplexScalarFieldVisObject();

   virtual void RegisterVisItFields(VisItDataCollection & visit_dc);

   virtual void PrepareVisField(const ParComplexGridFunction &u,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef);

   virtual void PrepareVisField(Coefficient &uReCoef,
                                Coefficient &uImCoef,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef);

   virtual void DisplayToGLVis();

   virtual void Update();
};

class ScalarFieldBdrVisObject
{
protected:
   bool cyl_;
   bool pseudo_;

   int dim_;

   std::string field_name_;

   ComplexGridFunction * v_; // Complex field in problem domain (L2)

public:
   ScalarFieldBdrVisObject(const std::string & field_name,
                           L2_ParFESpace *sfes,
                           bool cyl, bool pseudo);

   virtual ~ScalarFieldBdrVisObject();

   virtual void RegisterVisItFields(VisItDataCollection & visit_dc);

   virtual void PrepareVisField(const ParComplexGridFunction &u,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef,
                                Array<int> & attr_marker);

   virtual void PrepareVisField(Coefficient &uReCoef,
                                Coefficient &uImCoef,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef,
                                Array<int> & attr_marker);

   virtual void Update();
};

class VectorFieldVisObject : public VisObjectBase
{
protected:
   std::string field_name_;

   GridFunction * v_; // field in problem domain (L2^d)

public:
   VectorFieldVisObject(const std::string & field_name,
                        std::shared_ptr<L2_ParFESpace> vfes);

   virtual ~VectorFieldVisObject();

   virtual void RegisterVisItFields(VisItDataCollection & visit_dc);

   virtual void PrepareVisField(const ParGridFunction &u);

   virtual void PrepareVisField(VectorCoefficient &uCoef);

   virtual void DisplayToGLVis();

   virtual void Update();
};

class ComplexVectorFieldVisObject : public VisObjectBase
{
protected:
   std::string field_name_;

   ComplexGridFunction * v_;    // Complex field in problem domain (L2^d)

public:
   ComplexVectorFieldVisObject(const std::string & field_name,
                               std::shared_ptr<L2_ParFESpace> vfes);

   virtual ~ComplexVectorFieldVisObject();

   virtual void RegisterVisItFields(VisItDataCollection & visit_dc);

   virtual void PrepareVisField(const ParComplexGridFunction &u,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef);

   virtual void PrepareVisField(VectorCoefficient &uReCoef,
                                VectorCoefficient &uImCoef,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef);

   virtual void DisplayToGLVis();

   virtual void Update();
};

class ComplexVectorFieldAnimObject : public VisObjectBase
{
protected:
   std::string field_name_;
   unsigned int num_frames_;

   real_t norm_r_, norm_i_;

   ParComplexGridFunction * v_; // Complex field in problem domain (L2^d)
   ParGridFunction * v_t_;      // Time-dependent field in problem domain (L2^d)

public:
   ComplexVectorFieldAnimObject(const std::string & field_name,
                                std::shared_ptr<L2_ParFESpace> vfes,
                                unsigned int num_frames = 24);
   virtual ~ComplexVectorFieldAnimObject();
   virtual void PrepareVisField(const ParComplexGridFunction &u,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef);

   virtual void PrepareVisField(VectorCoefficient &uReCoef,
                                VectorCoefficient &uImCoef,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef);

   virtual void RegisterVisItFields(VisItDataCollection & visit_dc);

   virtual void DisplayToGLVis();

   virtual void Update();
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_VIS_OBJECT
