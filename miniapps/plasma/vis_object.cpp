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

#include "vis_object.hpp"
#include "cold_plasma_dielectric_coefs.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{
using namespace common;

namespace plasma
{

ScalarFieldVisObject::ScalarFieldVisObject(const std::string & field_name,
                                           L2_ParFESpace *sfes,
                                           bool cyl,
                                           bool pseudo)
   : cyl_(cyl),
     pseudo_(pseudo),
     dim_(-1),
     field_name_(field_name),
     v_(NULL)
{
   MFEM_VERIFY(sfes != NULL, "ScalarFieldVisObject: "
               "FiniteElementSpace sfes must be non NULL.");

   dim_ = sfes->GetParMesh()->SpaceDimension();

   v_ = new ComplexGridFunction(sfes);
}

ScalarFieldVisObject::~ScalarFieldVisObject()
{
   delete v_;
}

void ScalarFieldVisObject::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   ostringstream oss_r;
   ostringstream oss_i;

   oss_r << "Re_" << field_name_;
   oss_i << "Im_" << field_name_;

   visit_dc.RegisterField(oss_r.str(), &v_->real());
   visit_dc.RegisterField(oss_i.str(), &v_->imag());
}

void ScalarFieldVisObject::PrepareVisField(const ParComplexGridFunction &u,
                                           VectorCoefficient * kReCoef,
                                           VectorCoefficient * kImCoef)
{
   GridFunctionCoefficient u_r(&u.real());
   GridFunctionCoefficient u_i(&u.imag());

   this->PrepareVisField(u_r, u_i, kReCoef, kImCoef);
}

void ScalarFieldVisObject::PrepareVisField(Coefficient &u_r,
                                           Coefficient &u_i,
                                           VectorCoefficient * kReCoef,
                                           VectorCoefficient * kImCoef)
{
   if (kReCoef || kImCoef)
   {
      ComplexPhaseCoefficient uk_r(kReCoef, kImCoef, &u_r, &u_i,
                                   true, false);
      ComplexPhaseCoefficient uk_i(kReCoef, kImCoef, &u_r, &u_i,
                                   false, false);

      PseudoScalarCoef   psu_r(uk_r, pseudo_ && cyl_);
      PseudoScalarCoef   psu_i(uk_i, pseudo_ && cyl_);

      v_->ProjectCoefficient(psu_r, psu_i);
   }
   else
   {
      PseudoScalarCoef   psu_r(u_r, pseudo_ && cyl_);
      PseudoScalarCoef   psu_i(u_i, pseudo_ && cyl_);

      v_->ProjectCoefficient(psu_r, psu_i);
   }
}

void ScalarFieldVisObject::Update()
{
   if (v_) { v_->Update(); }
}

ScalarFieldBdrVisObject::ScalarFieldBdrVisObject(const std::string & field_name,
                                                 L2_ParFESpace *sfes,
                                                 bool cyl,
                                                 bool pseudo)
   : cyl_(cyl),
     pseudo_(pseudo),
     dim_(-1),
     field_name_(field_name),
     v_(NULL)
{
   MFEM_VERIFY(sfes != NULL, "ScalarFieldBdrVisObject: "
               "FiniteElementSpace sfes must be non NULL.");

   dim_ = sfes->GetParMesh()->SpaceDimension();

   v_ = new ComplexGridFunction(sfes);
   (*v_) = 0.0;
}

ScalarFieldBdrVisObject::~ScalarFieldBdrVisObject()
{
   delete v_;
}

void ScalarFieldBdrVisObject::RegisterVisItFields(VisItDataCollection &
                                                  visit_dc)
{
   ostringstream oss_r;
   ostringstream oss_i;

   oss_r << "Re_" << field_name_;
   oss_i << "Im_" << field_name_;

   visit_dc.RegisterField(oss_r.str(), &v_->real());
   visit_dc.RegisterField(oss_i.str(), &v_->imag());
}

void ScalarFieldBdrVisObject::PrepareVisField(const ParComplexGridFunction &u,
                                              VectorCoefficient * kReCoef,
                                              VectorCoefficient * kImCoef,
                                              Array<int> & attr_marker)
{
   GridFunctionCoefficient u_r(&u.real());
   GridFunctionCoefficient u_i(&u.imag());

   this->PrepareVisField(u_r, u_i, kReCoef, kImCoef, attr_marker);
}

void ScalarFieldBdrVisObject::PrepareVisField(Coefficient &u_r,
                                              Coefficient &u_i,
                                              VectorCoefficient * kReCoef,
                                              VectorCoefficient * kImCoef,
                                              Array<int> & attr_marker)
{
   if (kReCoef || kImCoef)
   {
      ComplexPhaseCoefficient uk_r(kReCoef, kImCoef, &u_r, &u_i,
                                   true, false);
      ComplexPhaseCoefficient uk_i(kReCoef, kImCoef, &u_r, &u_i,
                                   false, false);

      PseudoScalarCoef   psu_r(uk_r, pseudo_ && cyl_);
      PseudoScalarCoef   psu_i(uk_i, pseudo_ && cyl_);

      v_->ProjectBdrCoefficient(psu_r, psu_i, attr_marker);
   }
   else
   {
      PseudoScalarCoef   psu_r(u_r, pseudo_ && cyl_);
      PseudoScalarCoef   psu_i(u_i, pseudo_ && cyl_);

      v_->ProjectBdrCoefficient(psu_r, psu_i, attr_marker);
   }
}

void ScalarFieldBdrVisObject::Update()
{
   if (v_) { v_->Update(); }
}

VectorFieldVisObject::VectorFieldVisObject(const std::string & field_name,
                                           L2_ParFESpace *vfes,
                                           L2_ParFESpace *sfes,
                                           bool cyl,
                                           bool pseudo)
   : cyl_(cyl),
     pseudo_(pseudo),
     dim_(-1),
     field_name_(field_name),
     v_(NULL),
     v_y_(NULL),
     v_z_(NULL)
{
   MFEM_VERIFY(vfes != NULL || sfes != NULL, "VectorFieldVisObject: "
               "Either vfes or sfes must be non NULL.");

   if (vfes)
   {
      dim_ = vfes->GetParMesh()->SpaceDimension();
   }
   else
   {
      dim_ = sfes->GetParMesh()->SpaceDimension();
   }

   switch (dim_)
   {
      case 1:
         MFEM_VERIFY(sfes != NULL, "VectorFieldVisObject: "
                     "sfes must be non NULL in 1D.");
         v_   = new GridFunction(sfes);
         v_y_ = new GridFunction(sfes);
         v_z_ = new GridFunction(sfes);
         break;
      case 2:
         MFEM_VERIFY(vfes != NULL && sfes != NULL, "VectorFieldVisObject: "
                     "vfes and sfes must be non NULL in 2D.");
         v_   = new GridFunction(vfes);
         v_z_ = new GridFunction(sfes);
         break;
      case 3:
         MFEM_VERIFY(vfes != NULL, "VectorFieldVisObject: "
                     "vfes must be non NULL in 3D.");
         v_   = new GridFunction(vfes);
         break;
   }
}

VectorFieldVisObject::~VectorFieldVisObject()
{
   delete v_;
   delete v_y_;
   delete v_z_;
}

void VectorFieldVisObject::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   switch (dim_)
   {
      case 1:
      {
         ostringstream oss_x;
         ostringstream oss_y;
         ostringstream oss_z;

         oss_x << field_name_ << "x";
         oss_y << field_name_ << "y";
         oss_z << field_name_ << "z";

         visit_dc.RegisterField(oss_x.str(), v_);
         visit_dc.RegisterField(oss_y.str(), v_y_);
         visit_dc.RegisterField(oss_z.str(), v_z_);
      }
      break;
      case 2:
      {
         ostringstream oss_xy;
         ostringstream oss_z;

         oss_xy << field_name_;
         oss_z  << field_name_;

         if (!cyl_)
         {
            oss_xy << "xy";
            oss_z << "z";

            visit_dc.RegisterField(oss_xy.str(), v_);
            visit_dc.RegisterField(oss_z.str(), v_z_);
         }
         else
         {
            oss_xy << "zr";
            oss_z << "phi";

            visit_dc.RegisterField(oss_xy.str(), v_);
            visit_dc.RegisterField(oss_z.str(), v_z_);
         }
      }
      break;
      case 3:
      {
         visit_dc.RegisterField(field_name_, v_);
      }
      break;
   }
}

void VectorFieldVisObject::PrepareVisField(const ParGridFunction &u)
{
   VectorGridFunctionCoefficient uCoef(&u);

   this->PrepareVisField(uCoef);
}

void VectorFieldVisObject::PrepareVisField(VectorCoefficient &u)
{
   switch (dim_)
   {
      case 1:
      {}
      break;
      case 2:
      {
         VectorXYCoef uxy(u, pseudo_ && cyl_);
         VectorZCoef   uz(u, !pseudo_ && cyl_);

         v_->ProjectCoefficient(uxy);
         if (v_z_) { v_z_->ProjectCoefficient(uz); }
      }
      break;
      case 3:
      {}
      break;
   }
}

void VectorFieldVisObject::Update()
{
   if (v_) { v_->Update(); }
   if (v_y_) { v_y_->Update(); }
   if (v_z_) { v_z_->Update(); }
}

ComplexVectorFieldVisObject
::ComplexVectorFieldVisObject(const std::string & field_name,
                              L2_ParFESpace *vfes,
                              L2_ParFESpace *sfes,
                              bool cyl,
                              bool pseudo)
   : cyl_(cyl),
     pseudo_(pseudo),
     dim_(-1),
     field_name_(field_name),
     v_(NULL),
     v_y_(NULL),
     v_z_(NULL)
{
   MFEM_VERIFY(vfes != NULL || sfes != NULL, "ComplexVectorFieldVisObject: "
               "Either vfes or sfes must be non NULL.");

   if (vfes)
   {
      dim_ = vfes->GetParMesh()->SpaceDimension();
   }
   else
   {
      dim_ = sfes->GetParMesh()->SpaceDimension();
   }

   switch (dim_)
   {
      case 1:
         MFEM_VERIFY(sfes != NULL, "ComplexVectorFieldVisObject: "
                     "sfes must be non NULL in 1D.");
         v_   = new ComplexGridFunction(sfes);
         v_y_ = new ComplexGridFunction(sfes);
         v_z_ = new ComplexGridFunction(sfes);
         break;
      case 2:
         MFEM_VERIFY(vfes != NULL && sfes != NULL,
                     "ComplexVectorFieldVisObject: "
                     "vfes and sfes must be non NULL in 2D.");
         v_   = new ComplexGridFunction(vfes);
         v_z_ = new ComplexGridFunction(sfes);
         break;
      case 3:
         MFEM_VERIFY(vfes != NULL, "ComplexVectorFieldVisObject: "
                     "vfes must be non NULL in 3D.");
         v_   = new ComplexGridFunction(vfes);
         break;
   }
}

ComplexVectorFieldVisObject::~ComplexVectorFieldVisObject()
{
   delete v_;
   delete v_y_;
   delete v_z_;
}

void ComplexVectorFieldVisObject
::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   switch (dim_)
   {
      case 1:
      {
         ostringstream oss_x_r;
         ostringstream oss_x_i;
         ostringstream oss_y_r;
         ostringstream oss_y_i;
         ostringstream oss_z_r;
         ostringstream oss_z_i;

         oss_x_r << "Re_" << field_name_ << "x";
         oss_x_i << "Im_" << field_name_ << "x";
         oss_y_r << "Re_" << field_name_ << "y";
         oss_y_i << "Im_" << field_name_ << "y";
         oss_z_r << "Re_" << field_name_ << "z";
         oss_z_i << "Im_" << field_name_ << "z";

         visit_dc.RegisterField(oss_x_r.str(), &v_->real());
         visit_dc.RegisterField(oss_x_i.str(), &v_->imag());
         visit_dc.RegisterField(oss_y_r.str(), &v_y_->real());
         visit_dc.RegisterField(oss_y_i.str(), &v_y_->imag());
         visit_dc.RegisterField(oss_z_r.str(), &v_z_->real());
         visit_dc.RegisterField(oss_z_i.str(), &v_z_->imag());
      }
      break;
      case 2:
      {
         ostringstream oss_xy_r;
         ostringstream oss_xy_i;
         ostringstream oss_z_r;
         ostringstream oss_z_i;

         oss_xy_r << "Re_" << field_name_;
         oss_xy_i << "Im_" << field_name_;
         oss_z_r << "Re_" << field_name_;
         oss_z_i << "Im_" << field_name_;

         if (!cyl_)
         {
            oss_xy_r << "xy";
            oss_xy_i << "xy";
            oss_z_r << "z";
            oss_z_i << "z";

            visit_dc.RegisterField(oss_xy_r.str(), &v_->real());
            visit_dc.RegisterField(oss_xy_i.str(), &v_->imag());
            visit_dc.RegisterField(oss_z_r.str(), &v_z_->real());
            visit_dc.RegisterField(oss_z_i.str(), &v_z_->imag());
         }
         else
         {
            oss_xy_r << "zr";
            oss_xy_i << "zr";
            oss_z_r << "phi";
            oss_z_i << "phi";

            visit_dc.RegisterField(oss_xy_r.str(), &v_->real());
            visit_dc.RegisterField(oss_xy_i.str(), &v_->imag());
            visit_dc.RegisterField(oss_z_r.str(), &v_z_->real());
            visit_dc.RegisterField(oss_z_i.str(), &v_z_->imag());
         }
      }
      break;
      case 3:
      {
         ostringstream oss_r;
         ostringstream oss_i;

         oss_r << "Re_" << field_name_;
         oss_i << "Im_" << field_name_;

         visit_dc.RegisterField(oss_r.str(), &v_->real());
         visit_dc.RegisterField(oss_i.str(), &v_->imag());
      }
      break;
   }
}

void ComplexVectorFieldVisObject
::PrepareVisField(const ParComplexGridFunction &u,
                  VectorCoefficient * kReCoef,
                  VectorCoefficient * kImCoef)
{
   VectorGridFunctionCoefficient u_r(&u.real());
   VectorGridFunctionCoefficient u_i(&u.imag());

   this->PrepareVisField(u_r, u_i, kReCoef, kImCoef);
}

void ComplexVectorFieldVisObject::PrepareVisField(VectorCoefficient &u_r,
                                                  VectorCoefficient &u_i,
                                                  VectorCoefficient * kReCoef,
                                                  VectorCoefficient * kImCoef)
{
   if (kReCoef || kImCoef)
   {
      ComplexPhaseVectorCoefficient uk_r(kReCoef, kImCoef, &u_r, &u_i,
                                         true, false);
      ComplexPhaseVectorCoefficient uk_i(kReCoef, kImCoef, &u_r, &u_i,
                                         false, false);

      switch (dim_)
      {
         case 1:
         {}
         break;
         case 2:
         {
            VectorXYCoef ukxy_r(uk_r, pseudo_ && cyl_);
            VectorXYCoef ukxy_i(uk_i, pseudo_ && cyl_);
            VectorZCoef   ukz_r(uk_r, !pseudo_ && cyl_);
            VectorZCoef   ukz_i(uk_i, !pseudo_ && cyl_);

            v_->ProjectCoefficient(ukxy_r, ukxy_i);
            if (v_z_) { v_z_->ProjectCoefficient(ukz_r, ukz_i); }
         }
         break;
         case 3:
         {}
         break;
      }
   }
   else
   {
      switch (dim_)
      {
         case 1:
         {}
         break;
         case 2:
         {
            VectorXYCoef uxy_r(u_r, pseudo_ && cyl_);
            VectorXYCoef uxy_i(u_i, pseudo_ && cyl_);
            VectorZCoef   uz_r(u_r, !pseudo_ && cyl_);
            VectorZCoef   uz_i(u_i, !pseudo_ && cyl_);

            v_->ProjectCoefficient(uxy_r, uxy_i);
            if (v_z_) { v_z_->ProjectCoefficient(uz_r, uz_i); }
         }
         break;
         case 3:
         {}
         break;
      }
   }
}

void ComplexVectorFieldVisObject::Update()
{
   if (v_) { v_->Update(); }
   if (v_y_) { v_y_->Update(); }
   if (v_z_) { v_z_->Update(); }
}

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI
