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

#include "qfunction.hpp"
#include "quadinterpolator.hpp"
#include "quadinterpolator_face.hpp"
#include "../general/forall.hpp"
#include "../mesh/pmesh.hpp"

namespace mfem
{

QuadratureFunction &QuadratureFunction::operator=(real_t value)
{
   Vector::operator=(value);
   return *this;
}

QuadratureFunction &QuadratureFunction::operator=(const Vector &v)
{
   MFEM_ASSERT(qspace && v.Size() == this->Size(), "");
   Vector::operator=(v);
   return *this;
}


QuadratureFunction::QuadratureFunction(Mesh *mesh, std::istream &in)
   : QuadratureFunction()
{
   const char *msg = "invalid input stream";
   std::string ident;

   qspace = new QuadratureSpace(mesh, in);
   own_qspace = true;

   in >> ident; MFEM_VERIFY(ident == "VDim:", msg);
   in >> vdim;

   Load(in, vdim*qspace->GetSize());
}

void QuadratureFunction::Save(std::ostream &os) const
{
   GetSpace()->Save(os);
   os << "VDim: " << vdim << '\n'
      << '\n';
   Vector::Print(os, vdim);
   os.flush();
}

void QuadratureFunction::ProjectGridFunctionFallback(const GridFunction &gf)
{
   if (gf.VectorDim() == 1)
   {
      GridFunctionCoefficient coeff(&gf);
      coeff.Coefficient::Project(*this);
   }
   else
   {
      VectorGridFunctionCoefficient coeff(&gf);
      coeff.VectorCoefficient::Project(*this);
   }
}

void QuadratureFunction::ProjectGridFunction(const GridFunction &gf)
{
   SetVDim(gf.VectorDim());

   if (auto *qs_elem = dynamic_cast<QuadratureSpace*>(qspace))
   {
      const FiniteElementSpace &gf_fes = *gf.FESpace();
      const bool use_tensor_products = UsesTensorBasis(gf_fes);
      const ElementDofOrdering ordering = use_tensor_products ?
                                          ElementDofOrdering::LEXICOGRAPHIC :
                                          ElementDofOrdering::NATIVE;

      // Use quadrature interpolator to go from E-vector to Q-vector
      const QuadratureInterpolator *qi =
         gf_fes.GetQuadratureInterpolator(*qs_elem);

      // If quadrature interpolator doesn't support this space, then fallback
      // on slower (non-device) version, and return early.
      if (!qi)
      {
         ProjectGridFunctionFallback(gf);
         return;
      }

      // Use element restriction to go from L-vector to E-vector
      const Operator *R = gf_fes.GetElementRestriction(ordering);
      Vector e_vec(R->Height());
      R->Mult(gf, e_vec);

      qi->SetOutputLayout(QVectorLayout::byVDIM);
      qi->DisableTensorProducts(!use_tensor_products);
      qi->PhysValues(e_vec, *this);
   }
   else if (auto *qs_face = dynamic_cast<FaceQuadratureSpace*>(qspace))
   {
      const FiniteElementSpace &gf_fes = *gf.FESpace();
      const FaceType face_type = qs_face->GetFaceType();
      const bool use_tensor_products = UsesTensorBasis(gf_fes);
      const ElementDofOrdering ordering = use_tensor_products ?
                                          ElementDofOrdering::LEXICOGRAPHIC :
                                          ElementDofOrdering::NATIVE;

      // Use quadrature interpolator to go from E-vector to Q-vector
      const FaceQuadratureInterpolator *qi =
         gf_fes.GetFaceQuadratureInterpolator(qspace->GetIntRule(0), face_type);

      // If quadrature interpolator doesn't support this space, then fallback
      // on slower (non-device) version, and return early. Also, currently,
      // ElementDofOrdering::NATIVE in FaceRestriction, so fall back in that
      // case too.
      if (qi == nullptr || ordering == ElementDofOrdering::NATIVE)
      {
         ProjectGridFunctionFallback(gf);
         return;
      }

      // Use element restriction to go from L-vector to E-vector
      const Operator *R = gf_fes.GetFaceRestriction(
                             ordering, face_type, L2FaceValues::SingleValued);
      Vector e_vec(R->Height());
      R->Mult(gf, e_vec);

      qi->SetOutputLayout(QVectorLayout::byVDIM);
      qi->DisableTensorProducts(!use_tensor_products);
      qi->Values(e_vec, *this);
   }
   else
   {
      // This branch should be unreachable
      MFEM_ABORT("Unsupported case.");
   }
}

std::ostream &operator<<(std::ostream &os, const QuadratureFunction &qf)
{
   qf.Save(os);
   return os;
}

void QuadratureFunction::SaveVTU(std::ostream &os, VTKFormat format,
                                 int compression_level,
                                 const std::string &field_name) const
{
   os << R"(<VTKFile type="UnstructuredGrid" version="2.2")";
   if (compression_level != 0)
   {
      os << R"( compressor="vtkZLibDataCompressor")";
   }
   os << " byte_order=\"" << VTKByteOrder() << "\">\n";
   os << "<UnstructuredGrid>\n";

   const char *fmt_str = (format == VTKFormat::ASCII) ? "ascii" : "binary";
   const char *type_str = (format != VTKFormat::BINARY32) ? "Float64" : "Float32";
   std::vector<char> buf;

   const int np = qspace->GetSize();
   const int ne = qspace->GetNE();
   const int sdim = qspace->GetMesh()->SpaceDimension();

   // For quadrature functions, each point is a vertex cell, so number of cells
   // is equal to number of points
   os << "<Piece NumberOfPoints=\"" << np
      << "\" NumberOfCells=\"" << np << "\">\n";

   // print out the points
   os << "<Points>\n";
   os << "<DataArray type=\"" << type_str
      << "\" NumberOfComponents=\"3\" format=\"" << fmt_str << "\">\n";

   Vector pt(sdim);
   for (int i = 0; i < ne; i++)
   {
      ElementTransformation &T = *qspace->GetTransformation(i);
      const IntegrationRule &ir = GetIntRule(i);
      for (int j = 0; j < ir.Size(); j++)
      {
         T.Transform(ir[j], pt);
         WriteBinaryOrASCII(os, buf, pt[0], " ", format);
         if (sdim > 1) { WriteBinaryOrASCII(os, buf, pt[1], " ", format); }
         else { WriteBinaryOrASCII(os, buf, 0.0, " ", format); }
         if (sdim > 2) { WriteBinaryOrASCII(os, buf, pt[2], "", format); }
         else { WriteBinaryOrASCII(os, buf, 0.0, "", format); }
         if (format == VTKFormat::ASCII) { os << '\n'; }
      }
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>\n";
   os << "</Points>\n";

   // Write cells (each cell is just a vertex)
   os << "<Cells>\n";
   // Connectivity
   os << R"(<DataArray type="Int32" Name="connectivity" format=")"
      << fmt_str << "\">\n";

   for (int i=0; i<np; ++i) { WriteBinaryOrASCII(os, buf, i, "\n", format); }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>\n";
   // Offsets
   os << R"(<DataArray type="Int32" Name="offsets" format=")"
      << fmt_str << "\">\n";
   for (int i=0; i<np; ++i) { WriteBinaryOrASCII(os, buf, i, "\n", format); }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>\n";
   // Types
   os << R"(<DataArray type="UInt8" Name="types" format=")"
      << fmt_str << "\">\n";
   for (int i = 0; i < np; i++)
   {
      uint8_t vtk_cell_type = VTKGeometry::POINT;
      WriteBinaryOrASCII(os, buf, vtk_cell_type, "\n", format);
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>\n";
   os << "</Cells>\n";

   os << "<PointData>\n";
   os << "<DataArray type=\"" << type_str << "\" Name=\"" << field_name
      << "\" format=\"" << fmt_str << "\" NumberOfComponents=\"" << vdim << "\" "
      << VTKComponentLabels(vdim) << " "
      << ">\n";
   for (int i = 0; i < ne; i++)
   {
      DenseMatrix vals;
      GetValues(i, vals);
      for (int j = 0; j < vals.Size(); ++j)
      {
         for (int vd = 0; vd < vdim; ++vd)
         {
            WriteBinaryOrASCII(os, buf, vals(vd, j), " ", format);
         }
         if (format == VTKFormat::ASCII) { os << '\n'; }
      }
   }
   if (format != VTKFormat::ASCII)
   {
      WriteBase64WithSizeAndClear(os, buf, compression_level);
   }
   os << "</DataArray>\n";
   os << "</PointData>\n";

   os << "</Piece>\n";
   os << "</UnstructuredGrid>\n";
   os << "</VTKFile>" << std::endl;
}

void QuadratureFunction::SaveVTU(const std::string &filename, VTKFormat format,
                                 int compression_level,
                                 const std::string &field_name) const
{
   std::ofstream f(filename + ".vtu");
   SaveVTU(f, format, compression_level, field_name);
}

static real_t ReduceReal(const Mesh *mesh, real_t value)
{
#ifdef MFEM_USE_MPI
   if (auto *pmesh = dynamic_cast<const ParMesh*>(mesh))
   {
      MPI_Comm comm = pmesh->GetComm();
      MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);
   }
#endif
   return value;
}

real_t QuadratureFunction::Integrate() const
{
   MFEM_VERIFY(vdim == 1, "Only scalar functions are supported.")
   const real_t local_integral = InnerProduct(*this, qspace->GetWeights());
   return ReduceReal(qspace->GetMesh(), local_integral);
}

void QuadratureFunction::Integrate(Vector &integrals) const
{
   integrals.SetSize(vdim);

   const Vector &weights = qspace->GetWeights();
   QuadratureFunction component(qspace);
   const int N = component.Size();
   const int VDIM = vdim; // avoid capturing 'this' in lambda body
   const real_t *d_v = Read();

   for (int vd = 0; vd < vdim; ++vd)
   {
      // Extract the component 'vd' into component.
      real_t *d_c = component.Write();
      mfem::forall(N, [=] MFEM_HOST_DEVICE (int i)
      {
         d_c[i] = d_v[vd + i*VDIM];
      });
      integrals[vd] = ReduceReal(qspace->GetMesh(),
                                 InnerProduct(component, weights));
   }
}

}
