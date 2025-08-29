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

#include "vis_object.hpp"
#include "cold_plasma_dielectric_coefs.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{
using namespace common;

namespace plasma
{

int VisObjectBase::Sw   = 1728;    // Screen Width
int VisObjectBase::Sh   = 1117;    // Screen Height
int VisObjectBase::Mt   = 66;      // Top Menu width (top)
int VisObjectBase::Mb   = 0;       // Top Menu width (bottom)
int VisObjectBase::Ml   = 0;       // Top Menu width (left)
int VisObjectBase::Mr   = 0;       // Top Menu width (right)
int VisObjectBase::Fw   = 0;       // Frame Width
int VisObjectBase::Fh   = 28;      // Frame Height
int VisObjectBase::Ww   = 420;     // Window Width
int VisObjectBase::Wh   = 260;     // Window Height
int VisObjectBase::offx = Ww + Fw; // Horizontal Offset
int VisObjectBase::offy = Wh + Fh; // Vertical Offset
int VisObjectBase::offs = 10;      // Shift after screen fills
int VisObjectBase::Sx   = Ml;      // Screen Horizontal Position
int VisObjectBase::Sy   = Mt;      // Screen Vertical Position
int VisObjectBase::Wx   = Sx;      // Window Horizontal Position
int VisObjectBase::Wy   = Sy;      // Window Vertical Position

void VisObjectBase::IncrementGLVisWindowPosition()
{
   Wx += offx;
   if (Wx + offx > Sw)
   {
      Wx = Sx;
      Wy += offy;

      if (Wy + offy > Sh)
      {
         Sx += offs;
         Sy += offs;

         if (Sx + offs > Sw || Sy + offs > Sh)
         {
            Sx = 0; Sy = 0;
         }

         Wx = Sx;
         Wy = Sy;
      }
   }
}

ScalarFieldVisObject::ScalarFieldVisObject(const std::string & field_name,
                                           ParFiniteElementSpace *sfes,
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

   v_ = new GridFunction(sfes);
}

ScalarFieldVisObject::~ScalarFieldVisObject()
{
   delete v_;
}

void ScalarFieldVisObject::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc.RegisterField(field_name_, v_);
}

void ScalarFieldVisObject::PrepareVisField(const ParGridFunction &u)
{
   GridFunctionCoefficient uCoef(&u);

   this->PrepareVisField(uCoef);
}

void ScalarFieldVisObject::PrepareVisField(Coefficient &uCoef)
{
   PseudoScalarCoef   psu(uCoef, pseudo_ && cyl_);
   v_->ProjectCoefficient(psu);
}

void ScalarFieldVisObject::DisplayToGLVis()
{
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sock;
   sock.precision(8);

   string keys = "c";

   VisualizeField(sock, vishost, visport,
                  *v_, field_name_.c_str(), Wx, Wy, Ww, Wh,
                  keys.c_str());
   IncrementGLVisWindowPosition();
}

void ScalarFieldVisObject::Update()
{
   if (v_) { v_->Update(); }
}

ComplexScalarFieldVisObject::ComplexScalarFieldVisObject(
   const std::string & field_name,
   shared_ptr<L2_ParFESpace> sfes,
   bool cyl,
   bool pseudo)
   : field_name_(field_name),
     v_(NULL)
{
   v_ = new ComplexGridFunction(sfes.get());
}

ComplexScalarFieldVisObject::~ComplexScalarFieldVisObject()
{
   delete v_;
}

void ComplexScalarFieldVisObject::RegisterVisItFields(
   VisItDataCollection & visit_dc)
{
   ostringstream oss_r;
   ostringstream oss_i;

   oss_r << "Re_" << field_name_;
   oss_i << "Im_" << field_name_;

   visit_dc.RegisterField(oss_r.str(), &v_->real());
   visit_dc.RegisterField(oss_i.str(), &v_->imag());
}

void ComplexScalarFieldVisObject::PrepareVisField(
   const ParComplexGridFunction &u,
   VectorCoefficient * kReCoef,
   VectorCoefficient * kImCoef)
{
   GridFunctionCoefficient u_r(&u.real());
   GridFunctionCoefficient u_i(&u.imag());

   this->PrepareVisField(u_r, u_i, kReCoef, kImCoef);
}

void ComplexScalarFieldVisObject::PrepareVisField(Coefficient &u_r,
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

void ComplexScalarFieldVisObject::DisplayToGLVis()
{
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sock_r, sock_i;
   sock_r.precision(8);
   sock_i.precision(8);

   ostringstream oss_r; oss_r << "Re(" << field_name_ << ")";
   ostringstream oss_i; oss_i << "Im(" << field_name_ << ")";

   string keys = "c";

   VisualizeField(sock_r, vishost, visport,
                  v_->real(), oss_r.str().c_str(), Wx, Wy, Ww, Wh,
                  keys.c_str());
   IncrementGLVisWindowPosition();

   VisualizeField(sock_i, vishost, visport,
                  v_->imag(), oss_i.str().c_str(), Wx, Wy, Ww, Wh,
                  keys.c_str());
   IncrementGLVisWindowPosition();
}

void ComplexScalarFieldVisObject::Update()
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
                                           shared_ptr<L2_ParFESpace> vfes)
   : field_name_(field_name),
     v_(new GridFunction(vfes.get()))
{}

VectorFieldVisObject::~VectorFieldVisObject()
{
   delete v_;
}

void VectorFieldVisObject::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc.RegisterField(field_name_, v_);
}

void VectorFieldVisObject::PrepareVisField(const ParGridFunction &u)
{
   VectorGridFunctionCoefficient uCoef(&u);

   this->PrepareVisField(uCoef);
}

void VectorFieldVisObject::PrepareVisField(VectorCoefficient &u)
{
   v_->ProjectCoefficient(u);
}

void VectorFieldVisObject::DisplayToGLVis()
{
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sock;
   sock.precision(8);

   string keys = "cvvv";

   VisualizeField(sock, vishost, visport,
                  *v_, field_name_.c_str(), Wx, Wy, Ww, Wh,
                  keys.c_str());
   IncrementGLVisWindowPosition();
}

void VectorFieldVisObject::Update()
{
   if (v_) { v_->Update(); }
}

ComplexVectorFieldVisObject
::ComplexVectorFieldVisObject(const std::string & field_name,
                              shared_ptr<L2_ParFESpace> vfes)
   : field_name_(field_name),
     v_(new ComplexGridFunction(vfes.get()))
{}

ComplexVectorFieldVisObject::~ComplexVectorFieldVisObject()
{
   delete v_;
}

void ComplexVectorFieldVisObject
::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   ostringstream oss_r;
   ostringstream oss_i;

   oss_r << "Re_" << field_name_;
   oss_i << "Im_" << field_name_;

   visit_dc.RegisterField(oss_r.str(), &v_->real());
   visit_dc.RegisterField(oss_i.str(), &v_->imag());
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

      v_->ProjectCoefficient(uk_r, uk_i);
   }
   else
   {
      v_->ProjectCoefficient(u_r, u_i);
   }
}

void ComplexVectorFieldVisObject::DisplayToGLVis()
{
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sock_r, sock_i;
   sock_r.precision(8);
   sock_i.precision(8);

   ostringstream oss_r; oss_r << "Re(" << field_name_ << ")";
   ostringstream oss_i; oss_i << "Im(" << field_name_ << ")";

   string keys = "cvvv";

   VisualizeField(sock_r, vishost, visport,
                  v_->real(), oss_r.str().c_str(), Wx, Wy, Ww, Wh,
                  keys.c_str());
   IncrementGLVisWindowPosition();

   VisualizeField(sock_i, vishost, visport,
                  v_->imag(), oss_i.str().c_str(), Wx, Wy, Ww, Wh,
                  keys.c_str());
   IncrementGLVisWindowPosition();
}

void ComplexVectorFieldVisObject::Update()
{
   if (v_) { v_->Update(); }
}

ComplexVectorFieldAnimObject
::ComplexVectorFieldAnimObject(const std::string & field_name,
			       shared_ptr<L2_ParFESpace> vfes,
			       unsigned int num_frames)
   : field_name_(field_name),
     num_frames_(num_frames),
     v_(new ParComplexGridFunction(vfes.get())),
     v_t_(new ParGridFunction(vfes.get()))
{}

ComplexVectorFieldAnimObject::~ComplexVectorFieldAnimObject()
{
   delete v_;
   delete v_t_;
}

void ComplexVectorFieldAnimObject
::PrepareVisField(const ParComplexGridFunction &u,
                  VectorCoefficient * kReCoef,
                  VectorCoefficient * kImCoef)
{
   VectorGridFunctionCoefficient u_r(&u.real());
   VectorGridFunctionCoefficient u_i(&u.imag());

   this->PrepareVisField(u_r, u_i, kReCoef, kImCoef);
}

void ComplexVectorFieldAnimObject::PrepareVisField(VectorCoefficient &u_r,
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

      v_->ProjectCoefficient(uk_r, uk_i);
   }
   else
   {
      v_->ProjectCoefficient(u_r, u_i);
   }

   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient zeroCoef(zeroVec);

   norm_r_ = v_->real().ComputeMaxError(zeroCoef);
   norm_i_ = v_->imag().ComputeMaxError(zeroCoef);

   *v_t_ = v_->real();
}

void ComplexVectorFieldAnimObject
::RegisterVisItFields(VisItDataCollection & visit_dc)
{
  // Animation not yet supported in VisIt output
}

void ComplexVectorFieldAnimObject::DisplayToGLVis()
{
   ParMesh &pmesh = *v_->real().ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);

   ostringstream oss;
   oss.precision(4);
   oss << "Harmonic " << field_name_ << " (t = "
       << std::setw(6) << std::setfill('0') << std::left << 0.0 << " T)";
   
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << pmesh << *v_t_
            << "window_title '" << oss.str() << "'\n"
	    << "window_geometry "
	    << (2 * Sw - 3 * Ww) / 4 << " " << (2 * Sh - 3 * Wh) / 4 << " "
	    << 3 * Ww / 2 << " " << 3 * Wh / 2 << "\n"
            << "valuerange 0.0 " << max(norm_r_, norm_i_) << "\n"
            << "autoscale off\n"
            << "keys cvvv\n" << flush;

   int i = 1;
   while (sol_sock)
   {
      real_t t = (real_t)(i % num_frames_) / num_frames_;
      oss.str("");
      oss << "Harmonic " << field_name_ << " (t = "
	  << std::setw(6) << std::setfill('0') << std::left << t << " T)";

      add( cos( 2.0 * M_PI * t), v_->real(),
           sin( 2.0 * M_PI * t), v_->imag(), *v_t_);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << pmesh << *v_t_
               << "window_title '" << oss.str() << "'" << flush;
      i++;
   }
}

void ComplexVectorFieldAnimObject::Update()
{
   if (v_) { v_->Update(); }
   if (v_t_) { v_t_->Update(); }
}

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI
