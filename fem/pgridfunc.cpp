// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include <iostream>
#include <limits>
#include "../general/forall.hpp"
using namespace std;

namespace mfem
{

ParGridFunction::ParGridFunction(ParFiniteElementSpace *pf, GridFunction *gf)
{
   fes = pfes = pf;
   SetDataAndSize(gf->GetData(), gf->Size());
}

ParGridFunction::ParGridFunction(ParFiniteElementSpace *pf, HypreParVector *tv)
   : GridFunction(pf), pfes(pf)
{
   Distribute(tv);
}

ParGridFunction::ParGridFunction(ParMesh *pmesh, const GridFunction *gf,
                                 const int *partitioning)
{
   const FiniteElementSpace *glob_fes = gf->FESpace();
   // duplicate the FiniteElementCollection from 'gf'
   fec_owned = FiniteElementCollection::New(glob_fes->FEColl()->Name());
   // create a local ParFiniteElementSpace from the global one:
   fes = pfes = new ParFiniteElementSpace(pmesh, glob_fes, partitioning,
                                          fec_owned);
   SetSize(pfes->GetVSize());

   if (partitioning)
   {
      // Assumption: the map "local element id" -> "global element id" is
      // increasing, i.e. the local numbering preserves the element order from
      // the global numbering.
      Array<int> gvdofs, lvdofs;
      Vector lnodes;
      int element_counter = 0;
      const int MyRank = pfes->GetMyRank();
      const int glob_ne = glob_fes->GetNE();
      for (int i = 0; i < glob_ne; i++)
      {
         if (partitioning[i] == MyRank)
         {
            const DofTransformation* const ltrans = pfes->GetElementVDofs(element_counter,
                                                                          lvdofs);
            const DofTransformation* const gtrans = glob_fes->GetElementVDofs(i, gvdofs);
            gf->GetSubVector(gvdofs, lnodes);
            if (gtrans)
            {
               gtrans->InvTransformPrimal(lnodes);
            }
            if (ltrans)
            {
               ltrans->TransformPrimal(lnodes);
            }
            SetSubVector(lvdofs, lnodes);
            element_counter++;
         }
      }
   }
}

ParGridFunction::ParGridFunction(ParMesh *pmesh, std::istream &input)
   : GridFunction(pmesh, input)
{
   // Convert the FiniteElementSpace, fes, to a ParFiniteElementSpace:
   pfes = new ParFiniteElementSpace(pmesh, fec_owned, fes->GetVDim(),
                                    fes->GetOrdering());
   delete fes;
   fes = pfes;
}

void ParGridFunction::Update()
{
   face_nbr_data.Destroy();
   GridFunction::Update();
}

void ParGridFunction::SetSpace(FiniteElementSpace *f)
{
   face_nbr_data.Destroy();
   GridFunction::SetSpace(f);
   pfes = dynamic_cast<ParFiniteElementSpace*>(f);
   MFEM_ASSERT(pfes != NULL, "not a ParFiniteElementSpace");
}

void ParGridFunction::SetSpace(ParFiniteElementSpace *f)
{
   face_nbr_data.Destroy();
   GridFunction::SetSpace(f);
   pfes = f;
}

void ParGridFunction::MakeRef(FiniteElementSpace *f, real_t *v)
{
   face_nbr_data.Destroy();
   GridFunction::MakeRef(f, v);
   pfes = dynamic_cast<ParFiniteElementSpace*>(f);
   MFEM_ASSERT(pfes != NULL, "not a ParFiniteElementSpace");
}

void ParGridFunction::MakeRef(ParFiniteElementSpace *f, real_t *v)
{
   face_nbr_data.Destroy();
   GridFunction::MakeRef(f, v);
   pfes = f;
}

void ParGridFunction::MakeRef(FiniteElementSpace *f, Vector &v, int v_offset)
{
   face_nbr_data.Destroy();
   GridFunction::MakeRef(f, v, v_offset);
   pfes = dynamic_cast<ParFiniteElementSpace*>(f);
   MFEM_ASSERT(pfes != NULL, "not a ParFiniteElementSpace");
}

void ParGridFunction::MakeRef(ParFiniteElementSpace *f, Vector &v, int v_offset)
{
   face_nbr_data.Destroy();
   GridFunction::MakeRef(f, v, v_offset);
   pfes = f;
}

void ParGridFunction::Distribute(const Vector *tv)
{
   const Operator *prolong = pfes->GetProlongationMatrix();
   prolong->Mult(*tv, *this);
}

void ParGridFunction::AddDistribute(real_t a, const Vector *tv)
{
   pfes->Dof_TrueDof_Matrix()->Mult(a, *tv, 1.0, *this);
}

HypreParVector *ParGridFunction::GetTrueDofs() const
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   GetTrueDofs(*tv);
   return tv;
}

void ParGridFunction::ParallelAverage(Vector &tv) const
{
   MFEM_VERIFY(pfes->Conforming(), "not implemented for NC meshes");
   pfes->GetProlongationMatrix()->MultTranspose(*this, tv);
   pfes->DivideByGroupSize(tv.HostReadWrite());
}

void ParGridFunction::ParallelAverage(HypreParVector &tv) const
{
   MFEM_VERIFY(pfes->Conforming(), "not implemented for NC meshes");
   pfes->GetProlongationMatrix()->MultTranspose(*this, tv);
   pfes->DivideByGroupSize(tv.HostReadWrite());
}

HypreParVector *ParGridFunction::ParallelAverage() const
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   ParallelAverage(*tv);
   return tv;
}

void ParGridFunction::ParallelProject(Vector &tv) const
{
   pfes->GetRestrictionMatrix()->Mult(*this, tv);
}

void ParGridFunction::ParallelProject(HypreParVector &tv) const
{
   pfes->GetRestrictionMatrix()->Mult(*this, tv);
}

HypreParVector *ParGridFunction::ParallelProject() const
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   ParallelProject(*tv);
   return tv;
}

void ParGridFunction::ParallelAssemble(Vector &tv) const
{
   pfes->GetProlongationMatrix()->MultTranspose(*this, tv);
}

void ParGridFunction::ParallelAssemble(HypreParVector &tv) const
{
   pfes->GetProlongationMatrix()->MultTranspose(*this, tv);
}

HypreParVector *ParGridFunction::ParallelAssemble() const
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   ParallelAssemble(*tv);
   return tv;
}

void ParGridFunction::ExchangeFaceNbrData()
{
   pfes->ExchangeFaceNbrData();

   if (pfes->GetFaceNbrVSize() <= 0)
   {
      return;
   }

   ParMesh *pmesh = pfes->GetParMesh();

   face_nbr_data.SetSize(pfes->GetFaceNbrVSize());
   send_data.SetSize(pfes->send_face_nbr_ldof.Size_of_connections());

   int *send_offset = pfes->send_face_nbr_ldof.GetI();
   const int *d_send_ldof = mfem::Read(pfes->send_face_nbr_ldof.GetJMemory(),
                                       send_data.Size());
   int *recv_offset = pfes->face_nbr_ldof.GetI();
   MPI_Comm MyComm = pfes->GetComm();

   int num_face_nbrs = pmesh->GetNFaceNeighbors();
   MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
   MPI_Request *send_requests = requests;
   MPI_Request *recv_requests = requests + num_face_nbrs;
   MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

   auto d_data = this->Read();
   auto d_send_data = send_data.Write();
   mfem::forall(send_data.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      const int ldof = d_send_ldof[i];
      d_send_data[i] = d_data[ldof >= 0 ? ldof : -1-ldof];
   });

   bool mpi_gpu_aware = Device::GetGPUAwareMPI();
   auto send_data_ptr = mpi_gpu_aware ? send_data.Read() : send_data.HostRead();
   auto face_nbr_data_ptr = mpi_gpu_aware ? face_nbr_data.Write() :
                            face_nbr_data.HostWrite();
   // Wait for the kernel to be done since it updates what's sent and it may be async
   if (mpi_gpu_aware) { MFEM_STREAM_SYNC; }
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(&send_data_ptr[send_offset[fn]],
                send_offset[fn+1] - send_offset[fn],
                MPITypeMap<real_t>::mpi_type, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(&face_nbr_data_ptr[recv_offset[fn]],
                recv_offset[fn+1] - recv_offset[fn],
                MPITypeMap<real_t>::mpi_type, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);
   MPI_Waitall(num_face_nbrs, recv_requests, statuses);

   delete [] statuses;
   delete [] requests;
}

real_t ParGridFunction::GetValue(int i, const IntegrationPoint &ip, int vdim)
const
{
   Array<int> dofs;
   Vector DofVal, LocVec;
   int nbr_el_no = i - pfes->GetParMesh()->GetNE();
   if (nbr_el_no >= 0)
   {
      int fes_vdim = pfes->GetVDim();
      const DofTransformation* const doftrans = pfes->GetFaceNbrElementVDofs(
                                                   nbr_el_no, dofs);
      const FiniteElement *fe = pfes->GetFaceNbrFE(nbr_el_no);
      if (fes_vdim > 1)
      {
         int s = dofs.Size()/fes_vdim;
         Array<int> dofs_(&dofs[(vdim-1)*s], s);
         face_nbr_data.GetSubVector(dofs_, LocVec);

         DofVal.SetSize(s);
      }
      else
      {
         face_nbr_data.GetSubVector(dofs, LocVec);
         DofVal.SetSize(dofs.Size());
      }
      if (doftrans)
      {
         doftrans->InvTransformPrimal(LocVec);
      }

      if (fe->GetMapType() == FiniteElement::VALUE)
      {
         fe->CalcShape(ip, DofVal);
      }
      else
      {
         ElementTransformation *Tr =
            pfes->GetFaceNbrElementTransformation(nbr_el_no);
         Tr->SetIntPoint(&ip);
         fe->CalcPhysShape(*Tr, DofVal);
      }
   }
   else
   {
      const DofTransformation* const doftrans = fes->GetElementDofs(i, dofs);
      fes->DofsToVDofs(vdim-1, dofs);
      DofVal.SetSize(dofs.Size());
      const FiniteElement *fe = fes->GetFE(i);
      if (fe->GetMapType() == FiniteElement::VALUE)
      {
         fe->CalcShape(ip, DofVal);
      }
      else
      {
         ElementTransformation *Tr = fes->GetElementTransformation(i);
         Tr->SetIntPoint(&ip);
         fe->CalcPhysShape(*Tr, DofVal);
      }
      GetSubVector(dofs, LocVec);
      if (doftrans)
      {
         doftrans->InvTransformPrimal(LocVec);
      }
   }

   return (DofVal * LocVec);
}

void ParGridFunction::GetVectorValue(int i, const IntegrationPoint &ip,
                                     Vector &val) const
{
   int nbr_el_no = i - pfes->GetParMesh()->GetNE();
   if (nbr_el_no >= 0)
   {
      Array<int> dofs;
      const DofTransformation* const doftrans = pfes->GetFaceNbrElementVDofs(
                                                   nbr_el_no,
                                                   dofs);
      Vector loc_data;
      face_nbr_data.GetSubVector(dofs, loc_data);
      if (doftrans)
      {
         doftrans->InvTransformPrimal(loc_data);
      }
      const FiniteElement *FElem = pfes->GetFaceNbrFE(nbr_el_no);
      int dof = FElem->GetDof();
      if (FElem->GetRangeType() == FiniteElement::SCALAR)
      {
         Vector shape(dof);
         if (FElem->GetMapType() == FiniteElement::VALUE)
         {
            FElem->CalcShape(ip, shape);
         }
         else
         {
            ElementTransformation *Tr =
               pfes->GetParMesh()->GetFaceNbrElementTransformation(nbr_el_no);
            Tr->SetIntPoint(&ip);
            FElem->CalcPhysShape(*Tr, shape);
         }
         int vdim = fes->GetVDim();
         val.SetSize(vdim);
         for (int k = 0; k < vdim; k++)
         {
            val(k) = shape * (&loc_data[dof * k]);
         }
      }
      else
      {
         int spaceDim = fes->GetMesh()->SpaceDimension();
         DenseMatrix vshape(dof, spaceDim);
         ElementTransformation *Tr =
            pfes->GetParMesh()->GetFaceNbrElementTransformation(nbr_el_no);
         Tr->SetIntPoint(&ip);
         FElem->CalcVShape(*Tr, vshape);
         val.SetSize(spaceDim);
         vshape.MultTranspose(loc_data, val);
      }
   }
   else
   {
      GridFunction::GetVectorValue(i, ip, val);
   }
}

real_t ParGridFunction::GetValue(ElementTransformation &T,
                                 const IntegrationPoint &ip,
                                 int comp, Vector *tr) const
{
   // We can assume faces and edges are local
   if (T.ElementType != ElementTransformation::ELEMENT)
   {
      return GridFunction::GetValue(T, ip, comp, tr);
   }

   // Check for evaluation in a local element
   int nbr_el_no = T.ElementNo - pfes->GetParMesh()->GetNE();
   if (nbr_el_no < 0)
   {
      return GridFunction::GetValue(T, ip, comp, tr);
   }

   // Evaluate using DoFs from a neighboring element
   if (tr)
   {
      T.SetIntPoint(&ip);
      T.Transform(ip, *tr);
   }

   Array<int> dofs;
   const FiniteElement * fe = pfes->GetFaceNbrFE(nbr_el_no);
   const DofTransformation* const doftrans = pfes->GetFaceNbrElementVDofs(
                                                nbr_el_no, dofs);

   pfes->DofsToVDofs(comp-1, dofs);
   Vector DofVal(dofs.Size()), LocVec;
   if (fe->GetMapType() == FiniteElement::VALUE)
   {
      fe->CalcShape(ip, DofVal);
   }
   else
   {
      fe->CalcPhysShape(T, DofVal);
   }
   face_nbr_data.GetSubVector(dofs, LocVec);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(LocVec);
   }


   return (DofVal * LocVec);
}

void ParGridFunction::GetVectorValue(ElementTransformation &T,
                                     const IntegrationPoint &ip,
                                     Vector &val, Vector *tr) const
{
   // We can assume faces and edges are local
   if (T.ElementType != ElementTransformation::ELEMENT)
   {
      return GridFunction::GetVectorValue(T, ip, val, tr);
   }

   // Check for evaluation in a local element
   int nbr_el_no = T.ElementNo - pfes->GetParMesh()->GetNE();
   if (nbr_el_no < 0)
   {
      return GridFunction::GetVectorValue(T, ip, val, tr);
   }

   // Evaluate using DoFs from a neighboring element
   if (tr)
   {
      T.SetIntPoint(&ip);
      T.Transform(ip, *tr);
   }

   Array<int> vdofs;
   DofTransformation * doftrans = pfes->GetFaceNbrElementVDofs(nbr_el_no, vdofs);
   Vector loc_data;
   face_nbr_data.GetSubVector(vdofs, loc_data);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(loc_data);
   }

   const FiniteElement *fe = pfes->GetFaceNbrFE(nbr_el_no);
   const int dof = fe->GetDof();
   if (fe->GetRangeType() == FiniteElement::SCALAR)
   {
      Vector shape(dof);
      if (fe->GetMapType() == FiniteElement::VALUE)
      {
         fe->CalcShape(ip, shape);
      }
      else
      {
         fe->CalcPhysShape(T, shape);
      }
      int vdim = pfes->GetVDim();
      val.SetSize(vdim);
      for (int k = 0; k < vdim; k++)
      {
         val(k) = shape * (&loc_data[dof * k]);
      }
   }
   else
   {
      int spaceDim = pfes->GetMesh()->SpaceDimension();
      int vdim = std::max(spaceDim, fe->GetRangeDim());
      DenseMatrix vshape(dof, vdim);
      fe->CalcVShape(T, vshape);
      val.SetSize(vdim);
      vshape.MultTranspose(loc_data, val);
   }
}

void ParGridFunction::CountElementsPerVDof(Array<int> &elem_per_vdof) const
{
   GridFunction::CountElementsPerVDof(elem_per_vdof);
   // Count the zones globally.
   GroupCommunicator &gcomm = this->ParFESpace()->GroupComm();
   gcomm.Reduce<int>(elem_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(elem_per_vdof);
}

void ParGridFunction::GetDerivative(int comp, int der_comp,
                                    ParGridFunction &der) const
{
   Array<int> overlap;
   AccumulateAndCountDerivativeValues(comp, der_comp, der, overlap);

   // Count the zones globally.
   GroupCommunicator &gcomm = der.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(overlap, GroupCommunicator::Sum);
   gcomm.Bcast(overlap);

   // Accumulate for all dofs.
   gcomm.Reduce<real_t>(der.HostReadWrite(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(der.HostReadWrite());

   for (int i = 0; i < overlap.Size(); i++)
   {
      der(i) /= overlap[i];
   }
}

void ParGridFunction::GetElementDofValues(int el, Vector &dof_vals) const
{
   int ne = fes->GetNE();
   if (el >= ne)
   {
      MFEM_ASSERT(face_nbr_data.Size() > 0,
                  "ParGridFunction::GetElementDofValues: ExchangeFaceNbrData "
                  "must be called before accessing face neighbor elements.");
      // Face neighbor element
      Array<int> dof_idx;
      pfes->GetFaceNbrElementVDofs(el - ne, dof_idx);
      face_nbr_data.GetSubVector(dof_idx, dof_vals);
   }
   else
   {
      GridFunction::GetElementDofValues(el, dof_vals);
   }
}

void ParGridFunction::ProjectCoefficient(Coefficient &coeff)
{
   DeltaCoefficient *delta_c = dynamic_cast<DeltaCoefficient *>(&coeff);

   if (delta_c == NULL)
   {
      GridFunction::ProjectCoefficient(coeff);
   }
   else
   {
      real_t loc_integral, glob_integral;

      ProjectDeltaCoefficient(*delta_c, loc_integral);

      MPI_Allreduce(&loc_integral, &glob_integral, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    pfes->GetComm());

      (*this) *= (delta_c->Scale() / glob_integral);
   }
}

void ParGridFunction::ProjectDiscCoefficient(VectorCoefficient &coeff)
{
   // local maximal element attribute for each dof
   Array<int> ldof_attr;

   // local projection
   GridFunction::ProjectDiscCoefficient(coeff, ldof_attr);

   // global maximal element attribute for each dof
   Array<int> gdof_attr;
   ldof_attr.Copy(gdof_attr);
   GroupCommunicator &gcomm = pfes->GroupComm();
   gcomm.Reduce<int>(gdof_attr, GroupCommunicator::Max);
   gcomm.Bcast(gdof_attr);

   // set local value to zero if global maximal element attribute is larger than
   // the local one, and mark (in gdof_attr) if we have the correct value
   for (int i = 0; i < pfes->GetVSize(); i++)
   {
      if (gdof_attr[i] > ldof_attr[i])
      {
         (*this)(i) = 0.0;
         gdof_attr[i] = 0;
      }
      else
      {
         gdof_attr[i] = 1;
      }
   }

   // parallel averaging plus interpolation to determine final values
   HypreParVector *tv = pfes->NewTrueDofVector();
   gcomm.Reduce<int>(gdof_attr, GroupCommunicator::Sum);
   gcomm.Bcast(gdof_attr);
   for (int i = 0; i < fes->GetVSize(); i++)
   {
      (*this)(i) /= gdof_attr[i];
   }
   this->ParallelAssemble(*tv);
   this->Distribute(tv);
   delete tv;
}


void ParGridFunction::ProjectDiscCoefficient(Coefficient &coeff, AvgType type)
{
   // Harmonic  (x1 ... xn) = [ (1/x1 + ... + 1/xn) / n ]^-1.
   // Arithmetic(x1 ... xn) = (x1 + ... + xn) / n.

   // Number of zones that contain a given dof.
   Array<int> zones_per_vdof;
   AccumulateAndCountZones(coeff, type, zones_per_vdof);

   // Count the zones globally.
   GroupCommunicator &gcomm = pfes->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(data, GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(data);

   ComputeMeans(type, zones_per_vdof);
}

void ParGridFunction::ProjectDiscCoefficient(VectorCoefficient &vcoeff,
                                             AvgType type)
{
   // Harmonic  (x1 ... xn) = [ (1/x1 + ... + 1/xn) / n ]^-1.
   // Arithmetic(x1 ... xn) = (x1 + ... + xn) / n.

   // Number of zones that contain a given dof.
   Array<int> zones_per_vdof;
   AccumulateAndCountZones(vcoeff, type, zones_per_vdof);

   // Count the zones globally.
   GroupCommunicator &gcomm = pfes->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(data, GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(data);

   ComputeMeans(type, zones_per_vdof);
}

void ParGridFunction::ProjectBdrCoefficient(
   Coefficient *coeff[], VectorCoefficient *vcoeff, const Array<int> &attr)
{
   Array<int> values_counter;
   AccumulateAndCountBdrValues(coeff, vcoeff, attr, values_counter);

   Vector values(Size());
   for (int i = 0; i < values.Size(); i++)
   {
      values(i) = values_counter[i] ? (*this)(i) : 0.0;
   }

   // Count the values globally.
   GroupCommunicator &gcomm = pfes->GroupComm();
   gcomm.Reduce<int>(values_counter.HostReadWrite(), GroupCommunicator::Sum);
   // Accumulate the values globally.
   gcomm.Reduce<real_t>(values.HostReadWrite(), GroupCommunicator::Sum);

   for (int i = 0; i < values.Size(); i++)
   {
      if (values_counter[i])
      {
         (*this)(i) = values(i)/values_counter[i];
      }
   }
   // Broadcast values to other processors to have a consistent GridFunction
   gcomm.Bcast<real_t>((*this).HostReadWrite());

#ifdef MFEM_DEBUG
   Array<int> ess_vdofs_marker;
   if (vcoeff) { pfes->GetEssentialVDofs(attr, ess_vdofs_marker); }
   else
   {
      ess_vdofs_marker.SetSize(Size());
      ess_vdofs_marker = 0;
      for (int i = 0; i < fes->GetVDim(); i++)
      {
         if (!coeff[i]) { continue; }
         Array<int> component_dof_marker;
         pfes->GetEssentialVDofs(attr, component_dof_marker,i);
         for (int j = 0; j<Size(); j++)
         {
            ess_vdofs_marker[j] = bool(ess_vdofs_marker[j]) ||
                                  bool(component_dof_marker[j]);
         }
      }
   }
   gcomm.Bcast<int>(values_counter.HostReadWrite());
   for (int i = 0; i < values_counter.Size(); i++)
   {
      MFEM_ASSERT(bool(values_counter[i]) == bool(ess_vdofs_marker[i]),
                  "internal error");
   }
#endif
}

void ParGridFunction::ProjectBdrCoefficientTangent(VectorCoefficient &vcoeff,
                                                   const Array<int> &bdr_attr)
{
   Array<int> values_counter;
   AccumulateAndCountBdrTangentValues(vcoeff, bdr_attr, values_counter);

   Vector values(Size());
   for (int i = 0; i < values.Size(); i++)
   {
      values(i) = values_counter[i] ? (*this)(i) : 0.0;
   }

   // Count the values globally.
   GroupCommunicator &gcomm = pfes->GroupComm();
   gcomm.Reduce<int>(values_counter.HostReadWrite(), GroupCommunicator::Sum);
   // Accumulate the values globally.
   gcomm.Reduce<real_t>(values.HostReadWrite(), GroupCommunicator::Sum);

   for (int i = 0; i < values.Size(); i++)
   {
      if (values_counter[i])
      {
         (*this)(i) = values(i)/values_counter[i];
      }
   }
   // Broadcast values to other processors to have a consistent GridFunction
   gcomm.Bcast<real_t>((*this).HostReadWrite());

#ifdef MFEM_DEBUG
   Array<int> ess_vdofs_marker;
   pfes->GetEssentialVDofs(bdr_attr, ess_vdofs_marker);
   gcomm.Bcast<int>(values_counter.HostReadWrite());
   for (int i = 0; i < values_counter.Size(); i++)
   {
      MFEM_ASSERT(bool(values_counter[i]) == bool(ess_vdofs_marker[i]),
                  "internal error: " << pfes->GetLocalTDofNumber(i) << ' ' << bool(
                     values_counter[i]));
   }
#endif
}

real_t ParGridFunction::ComputeDGFaceJumpError(Coefficient *exsol,
                                               Coefficient *ell_coeff,
                                               JumpScaling jump_scaling,
                                               const IntegrationRule *irs[]) const
{
   const_cast<ParGridFunction *>(this)->ExchangeFaceNbrData();

   int fdof, intorder, k;
   ElementTransformation *transf;
   Vector shape, el_dofs, err_val, ell_coeff_val;
   Array<int> vdofs;
   IntegrationPoint eip;
   real_t error = 0.0;

   ParMesh *mesh = pfes->GetParMesh();

   std::map<int,int> local_to_shared;
   for (int i = 0; i < mesh->GetNSharedFaces(); ++i)
   {
      int i_local = mesh->GetSharedFace(i);
      local_to_shared[i_local] = i;
   }

   for (int i = 0; i < mesh->GetNumFaces(); i++)
   {
      real_t shared_face_factor = 1.0;
      bool shared_face = false;
      int iel1, iel2, info1, info2;
      mesh->GetFaceElements(i, &iel1, &iel2);
      mesh->GetFaceInfos(i, &info1, &info2);

      real_t h = mesh->GetElementSize(iel1);
      intorder = fes->GetFE(iel1)->GetOrder();

      FaceElementTransformations *face_elem_transf;
      const FiniteElement *fe1, *fe2;
      if (info2 >= 0 && iel2 < 0)
      {
         int ishared = local_to_shared[i];
         face_elem_transf = mesh->GetSharedFaceTransformations(ishared);
         iel2 = face_elem_transf->Elem2No - mesh->GetNE();
         fe2 = pfes->GetFaceNbrFE(iel2);
         if ( (k = fe2->GetOrder()) > intorder )
         {
            intorder = k;
         }
         shared_face = true;
         shared_face_factor = 0.5;
         h = std::min(h, mesh->GetFaceNbrElementSize(iel2));
      }
      else
      {
         if (iel2 >= 0)
         {
            fe2 = pfes->GetFE(iel2);
            if ( (k = fe2->GetOrder()) > intorder )
            {
               intorder = k;
            }
            h = std::min(h, mesh->GetElementSize(iel2));
         }
         else
         {
            fe2 = NULL;
         }
         face_elem_transf = mesh->GetFaceElementTransformations(i);
      }
      int p = intorder;

      intorder = 2 * intorder;  // <-------------
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[face_elem_transf->GetGeometryType()];
      }
      else
      {
         ir = &(IntRules.Get(face_elem_transf->GetGeometryType(), intorder));
      }
      err_val.SetSize(ir->GetNPoints());
      ell_coeff_val.SetSize(ir->GetNPoints());
      // side 1
      transf = face_elem_transf->Elem1;
      fe1 = fes->GetFE(iel1);
      fdof = fe1->GetDof();
      fes->GetElementVDofs(iel1, vdofs);
      shape.SetSize(fdof);
      el_dofs.SetSize(fdof);
      for (k = 0; k < fdof; k++)
         if (vdofs[k] >= 0)
         {
            el_dofs(k) =   (*this)(vdofs[k]);
         }
         else
         {
            el_dofs(k) = - (*this)(-1-vdofs[k]);
         }
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         face_elem_transf->Loc1.Transform(ir->IntPoint(j), eip);
         fe1->CalcShape(eip, shape);
         transf->SetIntPoint(&eip);
         ell_coeff_val(j) = ell_coeff->Eval(*transf, eip);
         err_val(j) = exsol->Eval(*transf, eip) - (shape * el_dofs);
      }
      if (fe2 != NULL)
      {
         // side 2
         transf = face_elem_transf->Elem2;
         fdof = fe2->GetDof();
         shape.SetSize(fdof);
         el_dofs.SetSize(fdof);
         if (shared_face)
         {
            pfes->GetFaceNbrElementVDofs(iel2, vdofs);
            for (k = 0; k < fdof; k++)
               if (vdofs[k] >= 0)
               {
                  el_dofs(k) = face_nbr_data[vdofs[k]];
               }
               else
               {
                  el_dofs(k) = - face_nbr_data[-1-vdofs[k]];
               }
         }
         else
         {
            pfes->GetElementVDofs(iel2, vdofs);
            for (k = 0; k < fdof; k++)
               if (vdofs[k] >= 0)
               {
                  el_dofs(k) = (*this)(vdofs[k]);
               }
               else
               {
                  el_dofs(k) = - (*this)(-1 - vdofs[k]);
               }
         }
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            face_elem_transf->Loc2.Transform(ir->IntPoint(j), eip);
            fe2->CalcShape(eip, shape);
            transf->SetIntPoint(&eip);
            ell_coeff_val(j) += ell_coeff->Eval(*transf, eip);
            ell_coeff_val(j) *= 0.5;
            err_val(j) -= (exsol->Eval(*transf, eip) - (shape * el_dofs));
         }
      }
      real_t face_error = 0.0;
      transf = face_elem_transf;
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         transf->SetIntPoint(&ip);
         real_t nu = jump_scaling.Eval(h, p);
         face_error += shared_face_factor*(ip.weight * nu * ell_coeff_val(j) *
                                           transf->Weight() *
                                           err_val(j) * err_val(j));
      }
      // negative quadrature weights may cause the error to be negative
      error += fabs(face_error);
   }

   error = sqrt(error);
   return GlobalLpNorm(2.0, error, pfes->GetComm());
}

void ParGridFunction::Save(std::ostream &os) const
{
   real_t *data_  = const_cast<real_t*>(HostRead());
   for (int i = 0; i < size; i++)
   {
      if (pfes->GetDofSign(i) < 0) { data_[i] = -data_[i]; }
   }

   GridFunction::Save(os);

   for (int i = 0; i < size; i++)
   {
      if (pfes->GetDofSign(i) < 0) { data_[i] = -data_[i]; }
   }
}

void ParGridFunction::Save(const char *fname, int precision) const
{
   int rank = pfes->GetMyRank();
   ostringstream fname_with_suffix;
   fname_with_suffix << fname << "." << setfill('0') << setw(6) << rank;
   ofstream ofs(fname_with_suffix.str().c_str());
   ofs.precision(precision);
   Save(ofs);
}

void ParGridFunction::SaveAsOne(const char *fname, int precision) const
{
   ofstream ofs;
   int rank = pfes->GetMyRank();
   if (rank == 0)
   {
      ofs.open(fname);
      ofs.precision(precision);
   }
   SaveAsOne(ofs);
}

void ParGridFunction::SaveAsSerial(const char *fname, int precision,
                                   int save_rank) const
{
   ParMesh *pmesh = ParFESpace()->GetParMesh();
   Mesh serial_mesh = pmesh->GetSerialMesh(save_rank);
   GridFunction serialgf = GetSerialGridFunction(save_rank, serial_mesh);

   if (pmesh->GetMyRank() == save_rank)
   {
      serialgf.Save(fname, precision);
   }
   MPI_Barrier(pmesh->GetComm());
}

GridFunction ParGridFunction::GetSerialGridFunction(
   int save_rank, FiniteElementSpace &serial_fes) const
{
   ParFiniteElementSpace *pfespace = ParFESpace();
   ParMesh *pmesh = pfespace->GetParMesh();

   GridFunction serial_gf(&serial_fes);

   Array<real_t> vals;
   Array<int> dofs;
   MPI_Status status;

   const int vdim = pfespace->GetVDim();

   const int my_rank = pmesh->GetMyRank();
   const int nranks = pmesh->GetNRanks();
   MPI_Comm comm = pmesh->GetComm();

   if (my_rank == save_rank)
   {
      int elem_count = 0; // To keep track of element count in serial mesh

      Vector nodeval;
      for (int e = 0; e < pmesh->GetNE(); e++)
      {
         GetElementDofValues(e, nodeval);
         serial_fes.GetElementVDofs(elem_count++, dofs);
         serial_gf.SetSubVector(dofs, nodeval);
      }

      for (int p = 0; p < nranks; p++)
      {
         if (p == save_rank) { continue; }
         int n_send_recv;
         MPI_Recv(&n_send_recv, 1, MPI_INT, p, 448, comm, &status);
         vals.SetSize(n_send_recv);
         if (n_send_recv)
         {
            MPI_Recv(&vals[0], n_send_recv, MPITypeMap<real_t>::mpi_type, p, 449, comm,
                     &status);
         }
         for (int i = 0; i < n_send_recv; )
         {
            serial_fes.GetElementVDofs(elem_count++, dofs);
            serial_gf.SetSubVector(dofs, &vals[i]);
            i += dofs.Size();
         }
      }
   } // my_rank == save_rank
   else
   {
      int n_send_recv = 0;
      Vector nodeval;
      for (int e = 0; e < pmesh->GetNE(); e++)
      {
         const FiniteElement *fe = pfespace->GetFE(e);
         n_send_recv += vdim*fe->GetDof();
      }
      MPI_Send(&n_send_recv, 1, MPI_INT, save_rank, 448, comm);
      vals.Reserve(n_send_recv);
      vals.SetSize(0);
      for (int e = 0; e < pmesh->GetNE(); e++)
      {
         GetElementDofValues(e, nodeval);
         for (int j = 0; j < nodeval.Size(); j++)
         {
            vals.Append(nodeval(j));
         }
      }
      if (n_send_recv)
      {
         MPI_Send(&vals[0], n_send_recv, MPITypeMap<real_t>::mpi_type, save_rank, 449,
                  comm);
      }
   }

   return serial_gf;
}

GridFunction ParGridFunction::GetSerialGridFunction(int save_rank,
                                                    Mesh &serial_mesh) const
{
   auto *serial_fec = pfes->FEColl()->Clone(pfes->FEColl()->GetOrder());
   auto *serial_fes = new FiniteElementSpace(&serial_mesh,
                                             serial_fec,
                                             pfes->GetVDim(),
                                             pfes->GetOrdering());
   GridFunction serial_gf = GetSerialGridFunction(save_rank, *serial_fes);
   serial_gf.MakeOwner(serial_fec); // Also assumes ownership of serial_fes
   return serial_gf;
}

#ifdef MFEM_USE_ADIOS2
void ParGridFunction::Save(adios2stream &os,
                           const std::string& variable_name,
                           const adios2stream::data_type type) const
{
   real_t *data_  = const_cast<real_t*>(HostRead());
   for (int i = 0; i < size; i++)
   {
      if (pfes->GetDofSign(i) < 0) { data_[i] = -data_[i]; }
   }

   GridFunction::Save(os, variable_name, type);

   for (int i = 0; i < size; i++)
   {
      if (pfes->GetDofSign(i) < 0) { data_[i] = -data_[i]; }
   }
}
#endif

void ParGridFunction::SaveAsOne(std::ostream &os) const
{
   int i, p;

   MPI_Comm MyComm;
   MPI_Status status;
   int MyRank, NRanks;

   MyComm = pfes -> GetComm();

   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   real_t **values = new real_t*[NRanks];
   int *nv = new int[NRanks];
   int *nvdofs = new int[NRanks];
   int *nedofs = new int[NRanks];
   int *nfdofs = new int[NRanks];
   int *nrdofs = new int[NRanks];

   real_t * h_data = const_cast<real_t *>(this->HostRead());

   values[0] = h_data;
   nv[0]     = pfes -> GetVSize();
   nvdofs[0] = pfes -> GetNVDofs();
   nedofs[0] = pfes -> GetNEDofs();
   nfdofs[0] = pfes -> GetNFDofs();

   if (MyRank == 0)
   {
      pfes -> Save(os);
      os << '\n';

      for (p = 1; p < NRanks; p++)
      {
         MPI_Recv(&nv[p], 1, MPI_INT, p, 455, MyComm, &status);
         MPI_Recv(&nvdofs[p], 1, MPI_INT, p, 456, MyComm, &status);
         MPI_Recv(&nedofs[p], 1, MPI_INT, p, 457, MyComm, &status);
         MPI_Recv(&nfdofs[p], 1, MPI_INT, p, 458, MyComm, &status);
         values[p] = new real_t[nv[p]];
         MPI_Recv(values[p], nv[p], MPITypeMap<real_t>::mpi_type, p, 460, MyComm,
                  &status);
      }

      int vdim = pfes -> GetVDim();

      for (p = 0; p < NRanks; p++)
      {
         nrdofs[p] = nv[p]/vdim - nvdofs[p] - nedofs[p] - nfdofs[p];
      }

      if (pfes->GetOrdering() == Ordering::byNODES)
      {
         for (int d = 0; d < vdim; d++)
         {
            for (p = 0; p < NRanks; p++)
               for (i = 0; i < nvdofs[p]; i++)
               {
                  os << *values[p]++ << '\n';
               }

            for (p = 0; p < NRanks; p++)
               for (i = 0; i < nedofs[p]; i++)
               {
                  os << *values[p]++ << '\n';
               }

            for (p = 0; p < NRanks; p++)
               for (i = 0; i < nfdofs[p]; i++)
               {
                  os << *values[p]++ << '\n';
               }

            for (p = 0; p < NRanks; p++)
               for (i = 0; i < nrdofs[p]; i++)
               {
                  os << *values[p]++ << '\n';
               }
         }
      }
      else
      {
         for (p = 0; p < NRanks; p++)
            for (i = 0; i < nvdofs[p]; i++)
               for (int d = 0; d < vdim; d++)
               {
                  os << *values[p]++ << '\n';
               }

         for (p = 0; p < NRanks; p++)
            for (i = 0; i < nedofs[p]; i++)
               for (int d = 0; d < vdim; d++)
               {
                  os << *values[p]++ << '\n';
               }

         for (p = 0; p < NRanks; p++)
            for (i = 0; i < nfdofs[p]; i++)
               for (int d = 0; d < vdim; d++)
               {
                  os << *values[p]++ << '\n';
               }

         for (p = 0; p < NRanks; p++)
            for (i = 0; i < nrdofs[p]; i++)
               for (int d = 0; d < vdim; d++)
               {
                  os << *values[p]++ << '\n';
               }
      }

      for (p = 1; p < NRanks; p++)
      {
         values[p] -= nv[p];
         delete [] values[p];
      }
      os.flush();
   }
   else
   {
      MPI_Send(&nv[0], 1, MPI_INT, 0, 455, MyComm);
      MPI_Send(&nvdofs[0], 1, MPI_INT, 0, 456, MyComm);
      MPI_Send(&nedofs[0], 1, MPI_INT, 0, 457, MyComm);
      MPI_Send(&nfdofs[0], 1, MPI_INT, 0, 458, MyComm);
      MPI_Send(h_data, nv[0], MPITypeMap<real_t>::mpi_type, 0, 460, MyComm);
   }

   delete [] values;
   delete [] nv;
   delete [] nvdofs;
   delete [] nedofs;
   delete [] nfdofs;
   delete [] nrdofs;
}

real_t GlobalLpNorm(const real_t p, real_t loc_norm, MPI_Comm comm)
{
   real_t glob_norm;

   // negative quadrature weights may cause the local norm to be negative
   loc_norm = fabs(loc_norm);

   if (p < infinity())
   {
      loc_norm = pow(loc_norm, p);

      MPI_Allreduce(&loc_norm, &glob_norm, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);

      glob_norm = pow(fabs(glob_norm), 1.0/p);
   }
   else
   {
      MPI_Allreduce(&loc_norm, &glob_norm, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, comm);
   }

   return glob_norm;
}

void ParGridFunction::ComputeFlux(
   BilinearFormIntegrator &blfi,
   GridFunction &flux, bool wcoef, int subdomain)
{
   ParFiniteElementSpace *ffes =
      dynamic_cast<ParFiniteElementSpace*>(flux.FESpace());
   MFEM_VERIFY(ffes, "the flux FE space must be ParFiniteElementSpace");

   Array<int> count(flux.Size());
   SumFluxAndCount(blfi, flux, count, wcoef, subdomain);

   // Accumulate flux and counts in parallel
   ffes->GroupComm().Reduce<real_t>(flux.HostReadWrite(), GroupCommunicator::Sum);
   ffes->GroupComm().Bcast<real_t>(flux.HostReadWrite());

   ffes->GroupComm().Reduce<int>(count.HostReadWrite(), GroupCommunicator::Sum);
   ffes->GroupComm().Bcast<int>(count.HostReadWrite());

   // complete averaging
   for (int i = 0; i < count.Size(); i++)
   {
      if (count[i] != 0) { flux(i) /= count[i]; }
   }

   if (ffes->Nonconforming())
   {
      // On a partially conforming flux space, project on the conforming space.
      // Using this code may lead to worse refinements in ex6, so we do not use
      // it by default.

      // Vector conf_flux;
      // flux.ConformingProject(conf_flux);
      // flux.ConformingProlongate(conf_flux);
   }
}


real_t L2ZZErrorEstimator(BilinearFormIntegrator &flux_integrator,
                          const ParGridFunction &x,
                          ParFiniteElementSpace &smooth_flux_fes,
                          ParFiniteElementSpace &flux_fes,
                          Vector &errors,
                          int norm_p, real_t solver_tol, int solver_max_it)
{
   // Compute fluxes in discontinuous space
   GridFunction flux(&flux_fes);
   flux = 0.0;

   ParFiniteElementSpace *xfes = x.ParFESpace();
   Array<int> xdofs, fdofs;
   Vector el_x, el_f;

   for (int i = 0; i < xfes->GetNE(); i++)
   {
      const DofTransformation* const xtrans = xfes->GetElementVDofs(i, xdofs);
      x.GetSubVector(xdofs, el_x);
      if (xtrans)
      {
         xtrans->InvTransformPrimal(el_x);
      }

      ElementTransformation *Transf = xfes->GetElementTransformation(i);
      flux_integrator.ComputeElementFlux(*xfes->GetFE(i), *Transf, el_x,
                                         *flux_fes.GetFE(i), el_f, false);

      const DofTransformation* const ftrans = flux_fes.GetElementVDofs(i, fdofs);
      if (ftrans)
      {
         ftrans->TransformPrimal(el_f);
      }
      flux.SetSubVector(fdofs, el_f);
   }

   // Assemble the linear system for L2 projection into the "smooth" space
   ParBilinearForm *a = new ParBilinearForm(&smooth_flux_fes);
   ParLinearForm *b = new ParLinearForm(&smooth_flux_fes);
   VectorGridFunctionCoefficient f(&flux);

   const FiniteElement *smooth_flux_fe = smooth_flux_fes.GetTypicalFE();

   if (smooth_flux_fe->GetRangeType() == FiniteElement::SCALAR)
   {
      VectorMassIntegrator *vmass = new VectorMassIntegrator;
      vmass->SetVDim(smooth_flux_fes.GetVDim());
      a->AddDomainIntegrator(vmass);
      b->AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   }
   else
   {
      a->AddDomainIntegrator(new VectorFEMassIntegrator);
      b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   }

   b->Assemble();
   a->Assemble();
   a->Finalize();

   // The destination of the projected discontinuous flux
   ParGridFunction smooth_flux(&smooth_flux_fes);
   smooth_flux = 0.0;

   HypreParMatrix* A = a->ParallelAssemble();
   HypreParVector* B = b->ParallelAssemble();
   HypreParVector* X = smooth_flux.ParallelProject();

   delete a;
   delete b;

   // Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   // preconditioner from hypre.
   HypreBoomerAMG *amg = new HypreBoomerAMG(*A);
   amg->SetPrintLevel(0);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(solver_tol);
   pcg->SetMaxIter(solver_max_it);
   pcg->SetPrintLevel(0);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(*B, *X);

   // Extract the parallel grid function corresponding to the finite element
   // approximation X. This is the local solution on each processor.
   smooth_flux = *X;

   delete A;
   delete B;
   delete X;
   delete amg;
   delete pcg;

   // Proceed through the elements one by one, and find the Lp norm differences
   // between the flux as computed per element and the flux projected onto the
   // smooth_flux_fes space.
   real_t total_error = 0.0;
   errors.SetSize(xfes->GetNE());
   for (int i = 0; i < xfes->GetNE(); i++)
   {
      errors(i) = ComputeElementLpDistance(norm_p, i, smooth_flux, flux);
      total_error += pow(errors(i), norm_p);
   }

   real_t glob_error;
   MPI_Allreduce(&total_error, &glob_error, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 xfes->GetComm());

   return pow(glob_error, 1.0/norm_p);
}

} // namespace mfem

#endif // MFEM_USE_MPI
