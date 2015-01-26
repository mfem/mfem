// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_PFESPACE
#define MFEM_PFESPACE

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../linalg/hypre.hpp"
#include "../mesh/pmesh.hpp"
#include "../mesh/nurbs.hpp"
#include "fespace.hpp"

namespace mfem
{

/// Abstract parallel finite element space.
class ParFiniteElementSpace : public FiniteElementSpace
{
private:
   /// MPI data.
   MPI_Comm MyComm;
   int NRanks, MyRank;

   /// Parallel mesh.
   ParMesh *pmesh;

   /// GroupCommunicator on the local VDofs
   GroupCommunicator *gcomm;

   /// Number of true dofs in this processor (local true dofs).
   int ltdof_size;

   /// The group of each local dof.
   Array<int> ldof_group;

   /// For a local dof: the local true dof number in the master of its group.
   Array<int> ldof_ltdof;

   /// Offsets for the dofs in each processor in global numbering.
   Array<int> dof_offsets;

   /// Offsets for the true dofs in each processor in global numbering.
   Array<int> tdof_offsets;

   /// Offsets for the true dofs in neighbor processor in global numbering.
   Array<int> tdof_nb_offsets;

   /// The sign of the basis functions at the scalar local dofs.
   Array<int> ldof_sign;

   /// The matrix P (interpolation from true dof to dof).
   HypreParMatrix *P;

   ParNURBSExtension *pNURBSext()
   { return dynamic_cast<ParNURBSExtension *>(NURBSext); }

   GroupTopology &GetGroupTopo()
   { return (NURBSext) ? pNURBSext()->gtopo : pmesh->gtopo; }

   /** Create a parallel FE space stealing all data (except RefData) from the
       given FE space. This is used in SaveUpdate(). */
   ParFiniteElementSpace(ParFiniteElementSpace &pf);

   // ldof_type = 0 : DOFs communicator, otherwise VDOFs communicator
   void GetGroupComm(GroupCommunicator &gcomm, int ldof_type,
                     Array<int> *ldof_sign = NULL);

   /// Construct dof_offsets and tdof_offsets using global communication.
   void GenerateGlobalOffsets();

   /// Construct ldof_group and ldof_ltdof.
   void ConstructTrueDofs();
   void ConstructTrueNURBSDofs();

   void ApplyLDofSigns(Array<int> &dofs) const;

public:
   // Face-neighbor data
   int num_face_nbr_dofs;
   Table face_nbr_element_dof;
   Table face_nbr_gdof;
   // Local face-neighbor data
   Table send_face_nbr_ldof;

   ParFiniteElementSpace(ParMesh *pm, const FiniteElementCollection *f,
                         int dim = 1, int order = Ordering::byNODES);

   MPI_Comm GetComm() { return MyComm; }
   int GetNRanks() { return NRanks; }
   int GetMyRank() { return MyRank; }

   inline ParMesh *GetParMesh() { return pmesh; }

   int TrueVSize()          { return ltdof_size; }
   int *GetDofOffsets()     { return dof_offsets; }
   int *GetTrueDofOffsets() { return tdof_offsets; }
   int GlobalVSize()        { return Dof_TrueDof_Matrix()->GetGlobalNumRows(); }
   int GlobalTrueVSize()    { return Dof_TrueDof_Matrix()->GetGlobalNumCols(); }
   int GetDofSign(int i)    { return NURBSext ? 1 : ldof_sign[VDofToDof(i)]; }

   /// Returns indexes of degrees of freedom in array dofs for i'th element.
   virtual void GetElementDofs(int i, Array<int> &dofs) const;

   /// Returns indexes of degrees of freedom for i'th boundary element.
   virtual void GetBdrElementDofs(int i, Array<int> &dofs) const;

   /** Returns the indexes of the degrees of freedom for i'th face
       including the dofs for the edges and the vertices of the face. */
   virtual void GetFaceDofs(int i, Array<int> &dofs) const;

   /// The true dof-to-dof interpolation matrix
   HypreParMatrix *Dof_TrueDof_Matrix();

   /** Create and return a new HypreParVector on the true dofs, which is
       owned by (i.e. it must be destroyed by) the calling function. */
   HypreParVector *NewTrueDofVector()
   { return (new HypreParVector(MyComm,GlobalTrueVSize(),GetTrueDofOffsets()));}

   /// Scale a vector of true dofs
   void DivideByGroupSize(double *vec);

   /// Return a reference to the internal GroupCommunicator (on VDofs)
   GroupCommunicator &GroupComm() { return *gcomm; }

   /// Return a new GroupCommunicator on Dofs
   GroupCommunicator *ScalarGroupComm();

   /** Given an integer array on the local degrees of freedom, perform
       a bitwise OR between the shared dofs. */
   void Synchronize(Array<int> &ldof_marker) const;

   /// Determine the boundary degrees of freedom
   virtual void GetEssentialVDofs(const Array<int> &bdr_attr_is_ess,
                                  Array<int> &ess_dofs) const;

   /** If the given ldof is owned by the current processor, return its local
       tdof number, otherwise return -1 */
   int GetLocalTDofNumber(int ldof);
   /// Returns the global tdof number of the given local degree of freedom
   int GetGlobalTDofNumber(int ldof);
   /** Returns the global tdof number of the given local degree of freedom in
       the scalar vesion of the current finite element space. The input should
       be a scalar local dof. */
   int GetGlobalScalarTDofNumber(int sldof);
   int GetMyDofOffset();

   // Face-neighbor functions
   void ExchangeFaceNbrData();
   int GetFaceNbrVSize() const { return num_face_nbr_dofs; }
   void GetFaceNbrElementVDofs(int i, Array<int> &vdofs) const;
   const FiniteElement *GetFaceNbrFE(int i) const;
   int *GetFaceNbrGlobalDofMap() { return face_nbr_gdof.GetJ(); }

   void Lose_Dof_TrueDof_Matrix();
   void LoseDofOffsets() { dof_offsets.LoseData(); }
   void LoseTrueDofOffsets() { tdof_offsets.LoseData(); }

   virtual void Update();
   /// Return a copy of the current FE space and update
   virtual FiniteElementSpace *SaveUpdate();

   virtual ~ParFiniteElementSpace() { delete gcomm; delete P; }
};

}

#endif // MFEM_USE_MPI

#endif
