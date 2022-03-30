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

class ScalarFieldVisObject
{
protected:
   bool cyl_;
   bool pseudo_;

   int dim_;

   std::string field_name_;

   ComplexGridFunction * v_; // Complex field in problem domain (L2)

public:
   ScalarFieldVisObject(const std::string & field_name,
                        L2_ParFESpace *sfes,
                        bool cyl, bool pseudo);

   ~ScalarFieldVisObject();

   virtual void RegisterVisItFields(VisItDataCollection & visit_dc);

   virtual void PrepareVisField(const ParComplexGridFunction &u,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef);

   virtual void PrepareVisField(Coefficient &uReCoef,
                                Coefficient &uImCoef,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef);

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

   ~ScalarFieldBdrVisObject();

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

class VectorFieldVisObject
{
protected:
   bool cyl_;
   bool pseudo_;

   int dim_;

   std::string field_name_;

   GridFunction * v_; // field in problem domain (L2^d)
   GridFunction * v_y_; // field y component in 1D (L2)
   GridFunction * v_z_; // field z component in 1D or 2D (L2)

public:
   VectorFieldVisObject(const std::string & field_name,
                        L2_ParFESpace *vfes, L2_ParFESpace *sfes,
                        bool cyl, bool pseudo);

   ~VectorFieldVisObject();

   virtual void RegisterVisItFields(VisItDataCollection & visit_dc);

   virtual void PrepareVisField(const ParGridFunction &u);

   virtual void PrepareVisField(VectorCoefficient &uCoef);

   virtual void Update();
};

class ComplexVectorFieldVisObject
{
protected:
   bool cyl_;
   bool pseudo_;

   int dim_;

   std::string field_name_;

   ComplexGridFunction * v_; // Complex field in problem domain (L2^d)
   ComplexGridFunction * v_y_; // Complex field y component in 1D (L2)
   ComplexGridFunction * v_z_; // Complex field z component in 1D or 2D (L2)

public:
   ComplexVectorFieldVisObject(const std::string & field_name,
                               L2_ParFESpace *vfes, L2_ParFESpace *sfes,
                               bool cyl, bool pseudo);

   ~ComplexVectorFieldVisObject();

   virtual void RegisterVisItFields(VisItDataCollection & visit_dc);

   virtual void PrepareVisField(const ParComplexGridFunction &u,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef);

   virtual void PrepareVisField(VectorCoefficient &uReCoef,
                                VectorCoefficient &uImCoef,
                                VectorCoefficient * kReCoef,
                                VectorCoefficient * kImCoef);

   virtual void Update();
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_VIS_OBJECT
