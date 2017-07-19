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
   Array<HYPRE_Int> dof_offsets;

   /// Offsets for the true dofs in each processor in global numbering.
   Array<HYPRE_Int> tdof_offsets;

   /// Offsets for the true dofs in neighbor processor in global numbering.
   Array<HYPRE_Int> tdof_nb_offsets;

   /// Previous 'dof_offsets' (before Update()), column partition of T.
   Array<HYPRE_Int> old_dof_offsets;

   /// The sign of the basis functions at the scalar local dofs.
   Array<int> ldof_sign;

   /// The matrix P (interpolation from true dof to dof).
   HypreParMatrix *P;

   /// The (block-diagonal) matrix R (restriction of dof to true dof)
   SparseMatrix *R;

   ParNURBSExtension *pNURBSext() const
   { return dynamic_cast<ParNURBSExtension *>(NURBSext); }

   GroupTopology &GetGroupTopo() const
   { return (NURBSext) ? pNURBSext()->gtopo : pmesh->gtopo; }

   void Construct();
   void Destroy();

   // ldof_type = 0 : DOFs communicator, otherwise VDOFs communicator
   void GetGroupComm(GroupCommunicator &gcomm, int ldof_type,
                     Array<int> *ldof_sign = NULL);

   /// Construct dof_offsets and tdof_offsets using global communication.
   void GenerateGlobalOffsets();

   /// Construct ldof_group and ldof_ltdof.
   void ConstructTrueDofs();
   void ConstructTrueNURBSDofs();

   void ApplyLDofSigns(Array<int> &dofs) const;

   /// Helper struct to store DOF dependencies in a parallel NC mesh.
   struct Dependency
   {
      int rank, dof; ///< master DOF, may be on another processor
      double coef;
      Dependency(int r, int d, double c) : rank(r), dof(d), coef(c) {}
   };

   /// Dependency list for a local vdof.
   struct DepList
   {
      Array<Dependency> list;
      int type; ///< 0 = independent, 1 = one-to-one (conforming), 2 = slave

      DepList() : type(0) {}

      bool IsTrueDof(int my_rank) const
      { return type == 0 || (type == 1 && list[0].rank == my_rank); }
   };

   void AddSlaveDependencies(DepList deps[], int master_rank,
                             const Array<int> &master_dofs, int master_ndofs,
                             const Array<int> &slave_dofs, DenseMatrix& I);

   void Add1To1Dependencies(DepList deps[], int owner_rank,
                            const Array<int> &owner_dofs, int owner_ndofs,
                            const Array<int> &dependent_dofs);

   void GetDofs(int type, int index, Array<int>& dofs);
   void ReorderFaceDofs(Array<int> &dofs, int orient);

   /// Build the P and R matrices.
   void Build_Dof_TrueDof_Matrix();

   // Used when the ParMesh is non-conforming, i.e. pmesh->pncmesh != NULL.
   // Constructs the matrices P and R. Determines ltdof_size. Calls
   // GenerateGlobalOffsets(). Constructs ldof_ltdof.
   void GetParallelConformingInterpolation();

   /** Calculate a GridFunction migration matrix after mesh load balancing.
       The result is a parallel permutation matrix that can be used to update
       all grid functions defined on this space. */
   HypreParMatrix* RebalanceMatrix(int old_ndofs,
                                   const Table* old_elem_dof);

   /** Calculate a GridFunction restriction matrix after mesh derefinement.
       The matrix is constructed so that the new grid function interpolates
       the original function, i.e., the original function is evaluated at the
       nodes of the coarse function. */
   HypreParMatrix* ParallelDerefinementMatrix(int old_ndofs,
                                              const Table *old_elem_dof);

public:
   // Face-neighbor data
   // Number of face-neighbor dofs
   int num_face_nbr_dofs;
   // Face-neighbor-element to face-neighbor dof
   Table face_nbr_element_dof;
   // Face-neighbor to ldof in the face-neighbor numbering
   Table face_nbr_ldof;
   // The global ldof indices of the face-neighbor dofs
   Array<HYPRE_Int> face_nbr_glob_dof_map;
   // Local face-neighbor data: face-neighbor to ldof
   Table send_face_nbr_ldof;

   ParFiniteElementSpace(ParMesh *pm, const FiniteElementCollection *f,
                         int dim = 1, int ordering = Ordering::byNODES);

   MPI_Comm GetComm() { return MyComm; }
   int GetNRanks() { return NRanks; }
   int GetMyRank() { return MyRank; }

   inline ParMesh *GetParMesh() { return pmesh; }

   int GetDofSign(int i)
   { return NURBSext || Nonconforming() ? 1 : ldof_sign[VDofToDof(i)]; }
   HYPRE_Int *GetDofOffsets()     { return dof_offsets; }
   HYPRE_Int *GetTrueDofOffsets() { return tdof_offsets; }
   HYPRE_Int GlobalVSize()
   { return Dof_TrueDof_Matrix()->GetGlobalNumRows(); }
   HYPRE_Int GlobalTrueVSize()
   { return Dof_TrueDof_Matrix()->GetGlobalNumCols(); }

   /// Return the number of local vector true dofs.
   virtual int GetTrueVSize() { return ltdof_size; }

   /// Returns indexes of degrees of freedom in array dofs for i'th element.
   virtual void GetElementDofs(int i, Array<int> &dofs) const;

   /// Returns indexes of degrees of freedom for i'th boundary element.
   virtual void GetBdrElementDofs(int i, Array<int> &dofs) const;

   /** Returns the indexes of the degrees of freedom for i'th face
       including the dofs for the edges and the vertices of the face. */
   virtual void GetFaceDofs(int i, Array<int> &dofs) const;

   void GetSharedEdgeDofs(int group, int ei, Array<int> &dofs) const;
   void GetSharedFaceDofs(int group, int fi, Array<int> &dofs) const;

   /// The true dof-to-dof interpolation matrix
   HypreParMatrix *Dof_TrueDof_Matrix()
   { if (!P) { Build_Dof_TrueDof_Matrix(); } return P; }

   /** @brief For a non-conforming mesh, construct and return the interpolation
       matrix from the partially conforming true dofs to the local dofs. The
       returned pointer must be deleted by the caller. */
   HypreParMatrix *GetPartialConformingInterpolation();

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
                                  Array<int> &ess_dofs,
                                  int component = -1) const;

   /** Get a list of essential true dofs, ess_tdof_list, corresponding to the
       boundary attributes marked in the array bdr_attr_is_ess. */
   virtual void GetEssentialTrueDofs(const Array<int> &bdr_attr_is_ess,
                                     Array<int> &ess_tdof_list,
                                     int component = -1);

   /** If the given ldof is owned by the current processor, return its local
       tdof number, otherwise return -1 */
   int GetLocalTDofNumber(int ldof);
   /// Returns the global tdof number of the given local degree of freedom
   HYPRE_Int GetGlobalTDofNumber(int ldof);
   /** Returns the global tdof number of the given local degree of freedom in
       the scalar version of the current finite element space. The input should
       be a scalar local dof. */
   HYPRE_Int GetGlobalScalarTDofNumber(int sldof);

   HYPRE_Int GetMyDofOffset() const;
   HYPRE_Int GetMyTDofOffset() const;

   virtual const Operator *GetProlongationMatrix()
   { return Dof_TrueDof_Matrix(); }
   /// Get the R matrix which restricts a local dof vector to true dof vector.
   virtual const SparseMatrix *GetRestrictionMatrix()
   { Dof_TrueDof_Matrix(); return R; }

   // Face-neighbor functions
   void ExchangeFaceNbrData();
   int GetFaceNbrVSize() const { return num_face_nbr_dofs; }
   void GetFaceNbrElementVDofs(int i, Array<int> &vdofs) const;
   void GetFaceNbrFaceVDofs(int i, Array<int> &vdofs) const;
   const FiniteElement *GetFaceNbrFE(int i) const;
   const FiniteElement *GetFaceNbrFaceFE(int i) const;
   const HYPRE_Int *GetFaceNbrGlobalDofMap() { return face_nbr_glob_dof_map; }

   void Lose_Dof_TrueDof_Matrix();
   void LoseDofOffsets() { dof_offsets.LoseData(); }
   void LoseTrueDofOffsets() { tdof_offsets.LoseData(); }

   bool Conforming() const { return pmesh->pncmesh == NULL; }
   bool Nonconforming() const { return pmesh->pncmesh != NULL; }

   /** Reflect changes in the mesh. Calculate one of the refinement/derefinement
       /rebalance matrices, unless want_transform is false. */
   virtual void Update(bool want_transform = true);

   /// Free ParGridFunction transformation matrix (if any), to save memory.
   virtual void UpdatesFinished()
   {
      FiniteElementSpace::UpdatesFinished();
      old_dof_offsets.DeleteAll();
   }

   virtual ~ParFiniteElementSpace() { Destroy(); }

   // Obsolete, kept for backward compatibility
   int TrueVSize() { return ltdof_size; }
};

}

#endif // MFEM_USE_MPI

#endif
