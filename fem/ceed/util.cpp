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

#include "util.hpp"

#include "../../general/device.hpp"
#include "../../fem/gridfunc.hpp"
#include "../../linalg/dtensor.hpp"

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

int DeviceKernelsVersion()
{
   return Device::DeviceKernelsVersion();
}

bool DeviceCanUseNonDeterministicKernels()
{
   return Device::IsNonDeterministicKernelsEnabled();
}

bool DeviceCanUseCeed()
{
   return Device::Allows(Backend::CEED_MASK);
}

namespace ceed
{

void RemoveBasisAndRestriction(const mfem::FiniteElementSpace *fes)
{
#ifdef MFEM_USE_CEED
   auto itb = mfem::internal::ceed_basis_map.begin();
   while (itb != mfem::internal::ceed_basis_map.end())
   {
      if (std::get<0>(itb->first)==fes)
      {
         CeedBasisDestroy(&itb->second);
         itb = mfem::internal::ceed_basis_map.erase(itb);
      }
      else
      {
         itb++;
      }
   }
   auto itr = mfem::internal::ceed_restr_map.begin();
   while (itr != mfem::internal::ceed_restr_map.end())
   {
      if (std::get<0>(itr->first)==fes)
      {
         CeedElemRestrictionDestroy(&itr->second);
         itr = mfem::internal::ceed_restr_map.erase(itr);
      }
      else
      {
         itr++;
      }
   }
#else
   MFEM_CONTRACT_VAR(fes);
#endif
}

#ifdef MFEM_USE_CEED

void InitVector(const mfem::Vector &v, CeedVector &cv)
{
   CeedVectorCreate(mfem::internal::ceed, v.Size(), &cv);
   CeedScalar *cv_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(mfem::internal::ceed, &mem);
   if ( Device::Allows(Backend::DEVICE_MASK) && mem==CEED_MEM_DEVICE )
   {
      cv_ptr = const_cast<CeedScalar*>(v.Read());
   }
   else
   {
      cv_ptr = const_cast<CeedScalar*>(v.HostRead());
      mem = CEED_MEM_HOST;
   }
   CeedVectorSetArray(cv, mem, CEED_USE_POINTER, cv_ptr);
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
         return CEED_PRISM; // Silence warning
   }
}

static void InitNonTensorBasis(const mfem::FiniteElementSpace &fes,
                               const mfem::IntegrationRule &ir,
                               Ceed ceed, CeedBasis *basis)
{
   const mfem::DofToQuad &maps = fes.GetFE(0)->
                                 GetDofToQuad(ir,mfem::DofToQuad::FULL);
   mfem::Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const int ndofs = maps.ndof;
   const int nqpts = maps.nqpt;
   mfem::DenseMatrix qX(dim,nqpts);
   mfem::Vector qW(nqpts);
   for (int i = 0; i < nqpts; i++)
   {
      const mfem::IntegrationPoint &ip = ir.IntPoint(i);
      qX(0,i) = ip.x;
      if (dim>1) { qX(1,i) = ip.y; }
      if (dim>2) { qX(2,i) = ip.z; }
      qW(i) = ip.weight;
   }
   CeedBasisCreateH1(ceed, GetCeedTopology(fes.GetFE(0)->GetGeomType()),
                     fes.GetVDim(), ndofs, nqpts,
                     maps.Bt.GetData(), maps.Gt.GetData(),
                     qX.GetData(), qW.GetData(), basis);
}

static void InitNonTensorRestriction(const mfem::FiniteElementSpace &fes,
                                     Ceed ceed, CeedElemRestriction *restr)
{
   mfem::Mesh *mesh = fes.GetMesh();
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const int P = fe->GetDof();
   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const mfem::Table &el_dof = fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   if (tfe) // Lexicographic ordering using dof_map
   {
      const mfem::Array<int>& dof_map = tfe->GetDofMap();
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const int el_offset = P * i;
         for (int j = 0; j < P; j++)
         {
            tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[dof_map[j]+el_offset];
         }
      }
   }
   else  // Native ordering
   {
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         for (int i = 0; i < P; i++)
         {
            tp_el_dof[i + e*P] = stride*el_dof.GetJ()[i + e*P];
         }
      }
   }
   CeedElemRestrictionCreate(ceed, mesh->GetNE(), P, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitTensorBasis(const mfem::FiniteElementSpace &fes,
                            const mfem::IntegrationRule &ir,
                            Ceed ceed, CeedBasis *basis)
{
   const mfem::DofToQuad &maps =
      fes.GetFE(0)->GetDofToQuad(ir, mfem::DofToQuad::TENSOR);
   mfem::Mesh *mesh = fes.GetMesh();
   const int ndofs = maps.ndof;
   const int nqpts = maps.nqpt;
   mfem::Vector qX(nqpts), qW(nqpts);
   const mfem::IntegrationRule &ir1d =
      IntRules.Get(Geometry::SEGMENT, ir.GetOrder());
   for (int i = 0; i < nqpts; i++)
   {
      const mfem::IntegrationPoint &ip = ir1d.IntPoint(i);
      qX(i) = ip.x;
      qW(i) = ip.weight;
   }
   CeedBasisCreateTensorH1(ceed, mesh->Dimension(), fes.GetVDim(), ndofs,
                           nqpts, maps.Bt.GetData(),
                           maps.Gt.GetData(), qX.GetData(),
                           qW.GetData(), basis);
}

void InitTensorRestriction(const mfem::FiniteElementSpace &fes,
                           Ceed ceed, CeedElemRestriction *restr)
{
   mfem::Mesh *mesh = fes.GetMesh();
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   MFEM_VERIFY(tfe, "invalid FE");
   const mfem::Array<int>& dof_map = tfe->GetDofMap();

   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const mfem::Table &el_dof = fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   const int dof = fe->GetDof();
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   if (dof_map.Size()>0)
   {
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const int el_offset = dof * i;
         for (int j = 0; j < dof; j++)
         {
            tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[dof_map[j]+el_offset];
         }
      }
   }
   else // dof_map.Size == 0, means dof_map[j]==j;
   {
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const int el_offset = dof * i;
         for (int j = 0; j < dof; j++)
         {
            tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[j+el_offset];
         }
      }
   }
   CeedElemRestrictionCreate(ceed, mesh->GetNE(), dof, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

void InitStridedRestriction(const mfem::FiniteElementSpace &fes,
                            CeedInt nelem, CeedInt nqpts, CeedInt qdatasize,
                            const CeedInt *strides,
                            CeedElemRestriction *restr)
{
   RestrKey restr_key(&fes, nelem, nqpts, qdatasize, restr_type::Strided);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      CeedElemRestrictionCreateStrided(mfem::internal::ceed, nelem, nqpts, qdatasize,
                                       nelem*nqpts*qdatasize,
                                       strides,
                                       restr);
      // Will be automatically destroyed when @a fes gets destroyed.
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitBasisAndRestriction(const FiniteElementSpace &fes,
                             const IntegrationRule &irm,
                             Ceed ceed, CeedBasis *basis,
                             CeedElemRestriction *restr)
{
   // Check for FES -> basis, restriction in hash tables
   const mfem::Mesh *mesh = fes.GetMesh();
   const mfem::FiniteElement *fe = fes.GetFE(0);
   const int P = fe->GetDof();
   const int Q = irm.GetNPoints();
   const int nelem = mesh->GetNE();
   const int ncomp = fes.GetVDim();
   BasisKey basis_key(&fes, &irm, ncomp, P, Q);
   auto basis_itr = mfem::internal::ceed_basis_map.find(basis_key);
   RestrKey restr_key(&fes, nelem, P, ncomp, restr_type::Standard);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retreive key values
   if (basis_itr == mfem::internal::ceed_basis_map.end())
   {
      if (UsesTensorBasis(fes))
      {
         InitTensorBasis(fes, irm, ceed, basis);
      }
      else
      {
         InitNonTensorBasis(fes, irm, ceed, basis);
      }
      mfem::internal::ceed_basis_map[basis_key] = *basis;
   }
   else
   {
      *basis = basis_itr->second;
   }
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      if (UsesTensorBasis(fes))
      {
         InitTensorRestriction(fes, ceed, restr);
      }
      else
      {
         InitNonTensorRestriction(fes, ceed, restr);
      }
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

// Assumes a tensor-product operator with one active field
int CeedOperatorGetActiveField(CeedOperator oper, CeedOperatorField *field)
{
   int ierr;
   Ceed ceed;
   ierr = CeedOperatorGetCeed(oper, &ceed); CeedChk(ierr);

   CeedQFunction qf;
   bool isComposite;
   ierr = CeedOperatorIsComposite(oper, &isComposite); CeedChk(ierr);
   CeedOperator *subops;
   if (isComposite)
   {
      ierr = CeedOperatorGetSubList(oper, &subops); CeedChk(ierr);
      ierr = CeedOperatorGetQFunction(subops[0], &qf); CeedChk(ierr);
   }
   else
   {
      ierr = CeedOperatorGetQFunction(oper, &qf); CeedChk(ierr);
   }
   CeedInt numinputfields, numoutputfields;
   ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
   CeedOperatorField *inputfields;
   if (isComposite)
   {
      ierr = CeedOperatorGetFields(subops[0], &inputfields, NULL); CeedChk(ierr);
   }
   else
   {
      ierr = CeedOperatorGetFields(oper, &inputfields, NULL); CeedChk(ierr);
   }

   CeedVector if_vector;
   bool found = false;
   int found_index = -1;
   for (int i = 0; i < numinputfields; ++i)
   {
      ierr = CeedOperatorFieldGetVector(inputfields[i], &if_vector); CeedChk(ierr);
      if (if_vector == CEED_VECTOR_ACTIVE)
      {
         if (found)
         {
            return CeedError(ceed, 1, "Multiple active vectors in CeedOperator!");
         }
         found = true;
         found_index = i;
      }
   }
   if (!found)
   {
      return CeedError(ceed, 1, "No active vector in CeedOperator!");
   }
   *field = inputfields[found_index];

   return 0;
}

int CeedOperatorGetActiveElemRestriction(CeedOperator oper,
                                         CeedElemRestriction* restr_out)
{
   int ierr;

   CeedOperatorField active_field;
   ierr = CeedOperatorGetActiveField(oper, &active_field); CeedChk(ierr);
   CeedElemRestriction er;
   ierr = CeedOperatorFieldGetElemRestriction(active_field, &er); CeedChk(ierr);
   *restr_out = er;

   return 0;
}

std::string ceed_path;

const std::string &GetCeedPath()
{
   if (ceed_path.empty())
   {
      const char *install_dir = MFEM_INSTALL_DIR "/include/mfem/fem/ceed";
      const char *source_dir = MFEM_SOURCE_DIR "/fem/ceed";
      struct_stat m_stat;
      if (stat(install_dir, &m_stat) == 0 && S_ISDIR(m_stat.st_mode))
      {
         ceed_path = install_dir;
      }
      else if (stat(source_dir, &m_stat) == 0 && S_ISDIR(m_stat.st_mode))
      {
         ceed_path = source_dir;
      }
      else
      {
         MFEM_ABORT("Cannot find libCEED kernels in MFEM_INSTALL_DIR or "
                    "MFEM_SOURCE_DIR");
      }
      // Could be useful for debugging:
      // out << "Using libCEED dir: " << ceed_path << std::endl;
   }
   return ceed_path;
}

#endif

} // namespace ceed

} // namespace mfem
