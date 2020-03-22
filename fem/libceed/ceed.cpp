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

#include "ceed.hpp"

#ifdef MFEM_USE_CEED
#include "../../general/device.hpp"
#include "../../fem/gridfunc.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#ifndef _WIN32
typedef struct stat struct_stat;
#else
#define stat(dir, buf) _stat(dir, buf)
#define S_ISDIR(mode) _S_IFDIR(mode)
typedef struct _stat struct_stat;
#endif

namespace mfem
{

namespace internal
{

extern Ceed ceed;

std::string ceed_path;

}

void InitCeedCoeff(Coefficient* Q, CeedData* ptr)
{
   if (ConstantCoefficient* coeff = dynamic_cast<ConstantCoefficient*>(Q))
   {
      CeedConstCoeff* ceedCoeff = new CeedConstCoeff{coeff->constant};
      ptr->coeff_type = CeedCoeff::Const;
      ptr->coeff = (void*)ceedCoeff;
   }
   else if (GridFunctionCoefficient* coeff =
               dynamic_cast<GridFunctionCoefficient*>(Q))
   {
      CeedGridCoeff* ceedCoeff = new CeedGridCoeff;
      ceedCoeff->coeff = coeff->GetGridFunction();
      ptr->coeff_type = CeedCoeff::Grid;
      ptr->coeff = (void*)ceedCoeff;
   }
   else
   {
      MFEM_ABORT("This type of Coefficient is not supported.");
   }
}

static CeedElemTopology GetCeedTopology(Geometry::Type geom)
{
   switch (geom)
   {
      case Geometry::SEGMENT:
         return CEED_LINE;
      case Geometry::TRIANGLE:
         return CEED_TRIANGLE;
      case Geometry::SQUARE:
         return CEED_QUAD;
      case Geometry::TETRAHEDRON:
         return CEED_TET;
      case Geometry::CUBE:
         return CEED_HEX;
      case Geometry::PRISM:
         return CEED_PRISM;
      default:
         MFEM_ABORT("This type of element is not supported");
         return CEED_PRISM;
   }
}

static void InitCeedNonTensorBasisAndRestriction(const FiniteElementSpace &fes,
                                                 const IntegrationRule &ir,
                                                 Ceed ceed, CeedBasis *basis,
                                                 CeedElemRestriction *restr)
{
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fe = fes.GetFE(0);
   const int order = fes.GetOrder(0);
   const int dim = mesh->Dimension();
   const int P = fe->GetDof();
   const int Q = ir.GetNPoints();
   DenseMatrix shape(P, Q);
   Vector grad(P*dim*Q);
   DenseMatrix qref(dim, Q);
   Vector qweight(Q);
   Vector shape_i(P);
   DenseMatrix grad_i(P, dim);
   const Table &el_dof = fes.GetElementToDofTable();
   Array<int> tp_el_dof(el_dof.Size_of_connections());
   const TensorBasisElement * tfe =
      dynamic_cast<const TensorBasisElement *>(fe);
   if (tfe) // Lexicographic ordering using dof_map
   {
      const Array<int>& dof_map = tfe->GetDofMap();
      for (int i = 0; i < Q; i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         qref(0,i) = ip.x;
         if (dim>1) { qref(1,i) = ip.y; }
         if (dim>2) { qref(2,i) = ip.z; }
         qweight(i) = ip.weight;
         fe->CalcShape(ip, shape_i);
         fe->CalcDShape(ip, grad_i);
         for (int j = 0; j < P; j++)
         {
            shape(j, i) = shape_i(dof_map[j]);
            for (int d = 0; d < dim; ++d)
            {
               grad(j+i*P+d*Q*P) = grad_i(dof_map[j], d);
            }
         }
      }

      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const int el_offset = fe->GetDof() * i;
         for (int j = 0; j < fe->GetDof(); j++)
         {
            tp_el_dof[j + el_offset] = el_dof.GetJ()[dof_map[j] + el_offset];
         }
      }
   }
   else  // Native ordering
   {
      for (int i = 0; i < Q; i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         qref(0,i) = ip.x;
         if (dim>1) { qref(1,i) = ip.y; }
         if (dim>2) { qref(2,i) = ip.z; }
         qweight(i) = ip.weight;
         fe->CalcShape(ip, shape_i);
         fe->CalcDShape(ip, grad_i);
         for (int j = 0; j < P; j++)
         {
            shape(j, i) = shape_i(j);
            for (int d = 0; d < dim; ++d)
            {
               grad(j+i*P+d*Q*P) = grad_i(j, d);
            }
         }
      }

      const FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         for (int i = 0; i < P; i++)
         {
            tp_el_dof[i + e*P] = el_dof.GetJ()[i + e*P];
         }
      }
   }
   CeedBasisCreateH1(ceed, GetCeedTopology(fe->GetGeomType()), fes.GetVDim(),
                     fe->GetDof(), ir.GetNPoints(), shape.GetData(),
                     grad.GetData(), qref.GetData(), qweight.GetData(), basis);
   CeedInterlaceMode imode = CEED_NONINTERLACED;
   if (fes.GetOrdering()==Ordering::byVDIM)
   {
      imode = CEED_INTERLACED;
   }
   CeedElemRestrictionCreate(ceed, imode, mesh->GetNE(), fe->GetDof(),
                             fes.GetNDofs(), fes.GetVDim(), CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitCeedTensorBasisAndRestriction(const FiniteElementSpace &fes,
                                              const IntegrationRule &ir,
                                              Ceed ceed, CeedBasis *basis,
                                              CeedElemRestriction *restr)
{
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fe = fes.GetFE(0);
   const int order = fes.GetOrder(0);
   const TensorBasisElement * tfe =
      dynamic_cast<const TensorBasisElement *>(fe);
   MFEM_VERIFY(tfe, "invalid FE");
   const Array<int>& dof_map = tfe->GetDofMap();
   const FiniteElement *fe1d =
      fes.FEColl()->FiniteElementForGeometry(Geometry::SEGMENT);
   DenseMatrix shape1d(fe1d->GetDof(), ir.GetNPoints());
   DenseMatrix grad1d(fe1d->GetDof(), ir.GetNPoints());
   Vector qref1d(ir.GetNPoints()), qweight1d(ir.GetNPoints());
   Vector shape_i(shape1d.Height());
   DenseMatrix grad_i(grad1d.Height(), 1);
   const H1_SegmentElement *h1_fe1d =
      dynamic_cast<const H1_SegmentElement *>(fe1d);
   MFEM_VERIFY(h1_fe1d, "invalid FE");
   const Array<int> &dof_map_1d = h1_fe1d->GetDofMap();
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      qref1d(i) = ip.x;
      qweight1d(i) = ip.weight;
      fe1d->CalcShape(ip, shape_i);
      fe1d->CalcDShape(ip, grad_i);
      for (int j = 0; j < shape1d.Height(); j++)
      {
         shape1d(j, i) = shape_i(dof_map_1d[j]);
         grad1d(j, i) = grad_i(dof_map_1d[j], 0);
      }
   }
   CeedBasisCreateTensorH1(ceed, mesh->Dimension(), fes.GetVDim(), order + 1,
                           ir.GetNPoints(), shape1d.GetData(),
                           grad1d.GetData(), qref1d.GetData(),
                           qweight1d.GetData(), basis);

   const Table &el_dof = fes.GetElementToDofTable();
   Array<int> tp_el_dof(el_dof.Size_of_connections());
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const int el_offset = fe->GetDof() * i;
      for (int j = 0; j < fe->GetDof(); j++)
      {
         tp_el_dof[j + el_offset] = el_dof.GetJ()[dof_map[j] + el_offset];
      }
   }
   CeedInterlaceMode imode = CEED_NONINTERLACED;
   if (fes.GetOrdering()==Ordering::byVDIM)
   {
      imode = CEED_INTERLACED;
   }
   CeedElemRestrictionCreate(ceed, imode, mesh->GetNE(), fe->GetDof(),
                             fes.GetNDofs(), fes.GetVDim(), CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

void InitCeedBasisAndRestriction(const FiniteElementSpace &fes,
                                 const IntegrationRule &irm,
                                 Ceed ceed, CeedBasis *basis,
                                 CeedElemRestriction *restr)
{
   if (UsesTensorBasis(fes))
   {
      const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, irm.GetOrder());
      InitCeedTensorBasisAndRestriction(fes, ir, ceed, basis, restr);
   }
   else
   {
      InitCeedNonTensorBasisAndRestriction(fes, irm, ceed, basis, restr);
   }
}

const std::string &GetCeedPath()
{
   if (internal::ceed_path.empty())
   {
      const char *install_dir = MFEM_INSTALL_DIR "/include/mfem/fem/libceed";
      const char *source_dir = MFEM_SOURCE_DIR "/fem/libceed";
      struct_stat m_stat;
      if (stat(install_dir, &m_stat) == 0 && S_ISDIR(m_stat.st_mode))
      {
         internal::ceed_path = install_dir;
      }
      else if (stat(source_dir, &m_stat) == 0 && S_ISDIR(m_stat.st_mode))
      {
         internal::ceed_path = source_dir;
      }
      else
      {
         MFEM_ABORT("Cannot find libCEED kernels in MFEM_INSTALL_DIR or "
                    "MFEM_SOURCE_DIR");
      }
      // Could be useful for debugging:
      // mfem::out << "Using libCEED dir: " << internal::ceed_path << std::endl;
   }
   return internal::ceed_path;
}

} // namespace mfem

#endif // MFEM_USE_CEED
