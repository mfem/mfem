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

void InitCeedTensorBasisAndRestriction(const mfem::FiniteElementSpace &fes,
                                       const mfem::IntegrationRule &ir,
                                       Ceed ceed, CeedBasis *basis,
                                       CeedElemRestriction *restr)
{
   mfem::Mesh *mesh = fes.GetMesh();
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const int order = fes.GetOrder(0);
   mfem::Array<int> dof_map;
   switch (mesh->Dimension())
   {
      case 1:
      {
         const mfem::H1_SegmentElement *h1_fe =
            dynamic_cast<const mfem::H1_SegmentElement *>(fe);
         MFEM_VERIFY(h1_fe, "invalid FE");
         h1_fe->GetDofMap().Copy(dof_map);
         break;
      }
      case 2:
      {
         const mfem::H1_QuadrilateralElement *h1_fe =
            dynamic_cast<const mfem::H1_QuadrilateralElement *>(fe);
         MFEM_VERIFY(h1_fe, "invalid FE");
         h1_fe->GetDofMap().Copy(dof_map);
         break;
      }
      case 3:
      {
         const mfem::H1_HexahedronElement *h1_fe =
            dynamic_cast<const mfem::H1_HexahedronElement *>(fe);
         MFEM_VERIFY(h1_fe, "invalid FE");
         h1_fe->GetDofMap().Copy(dof_map);
         break;
      }
   }
   const mfem::FiniteElement *fe1d =
      fes.FEColl()->FiniteElementForGeometry(mfem::Geometry::SEGMENT);
   mfem::DenseMatrix shape1d(fe1d->GetDof(), ir.GetNPoints());
   mfem::DenseMatrix grad1d(fe1d->GetDof(), ir.GetNPoints());
   mfem::Vector qref1d(ir.GetNPoints()), qweight1d(ir.GetNPoints());
   mfem::Vector shape_i(shape1d.Height());
   mfem::DenseMatrix grad_i(grad1d.Height(), 1);
   const mfem::H1_SegmentElement *h1_fe1d =
      dynamic_cast<const mfem::H1_SegmentElement *>(fe1d);
   MFEM_VERIFY(h1_fe1d, "invalid FE");
   const mfem::Array<int> &dof_map_1d = h1_fe1d->GetDofMap();
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const mfem::IntegrationPoint &ip = ir.IntPoint(i);
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

   const mfem::Table &el_dof = fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
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
