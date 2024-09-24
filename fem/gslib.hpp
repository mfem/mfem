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

#ifndef MFEM_GSLIB
#define MFEM_GSLIB

#include "../config/config.hpp"
#include "gridfunc.hpp"

#ifdef MFEM_USE_GSLIB

namespace gslib
{
struct comm;
struct findpts_data_2;
struct findpts_data_3;
struct crystal;
struct gs_data;
}

namespace mfem
{

/** \brief FindPointsGSLIB can robustly evaluate a GridFunction on an arbitrary
 *  collection of points.
 *
 *  There are three key functions in FindPointsGSLIB:
 *
 *  1. Setup - constructs the internal data structures of gslib. See \ref Setup.
 *
 *  2. FindPoints - for any given arbitrary set of points in physical space,
 *     gslib finds the element number, MPI rank, and the reference space
 *     coordinates inside the element that each point is located in. gslib also
 *     returns a code that indicates whether the point was found inside an
 *     element, on element border, or not found in the domain.
 *     For points returned as found on `element border`, the point is either
 *     on an element edge/face or near the domain boundary, and gslib also
 *     returns a distance to the border. Points near (but outside) the domain
 *     boundary must then be marked as not found using the distance returned
 *     by gslib. See \ref FindPoints.
 *
 *  3. Interpolate - Interpolates any grid function at the points found using 2.
 *     For functions in L2 finite element space, use \ref SetL2AvgType to
 *     specify how to interpolate values at points located at element boundaries
 *     where the function might be multi-valued. See \ref Interpolate.
 *
 *  FindPointsGSLIB also provides interface to use these functions through a
 *  single call.
 *
 *  For custom interpolation (e.g., evaluating strain rate tensor), we provide
 *  functions that use gslib to send element index and corresponding
 *  reference-space coordinates for each point to the mpi rank that the element
 *  is located on. Then, custom interpolation can be defined locally by the user
 *  before sending the values back to mpi ranks where the query originated from.
 *  See \ref DistributePointInfoToOwningMPIRanks and
 *  \ref DistributeInterpolatedValues.
 */
class FindPointsGSLIB
{
public:
   enum AvgType {NONE, ARITHMETIC, HARMONIC}; // Average type for L2 functions

protected:
   Mesh *mesh;
   Array<Mesh *> mesh_split;  // Meshes used to split simplices.
   // IntegrationRules for simplex->Quad/Hex and to project to highest polynomial
   // order in-case of p-refinement.
   Array<IntegrationRule *> ir_split;
   Array<FiniteElementSpace *>
   fes_rst_map; // FESpaces to map info Quad/Hex->Simplex
   Array<GridFunction *> gf_rst_map; // GridFunctions to map info Quad/Hex->Simplex
   FiniteElementCollection *fec_map_lin;
   struct gslib::findpts_data_2 *fdata2D; // gslib's internal data
   struct gslib::findpts_data_3 *fdata3D; // gslib's internal data
   struct gslib::crystal *cr;             // gslib's internal data
   struct gslib::comm *gsl_comm;          // gslib's internal data
   int dim, points_cnt;
   Array<unsigned int> gsl_code, gsl_proc, gsl_elem, gsl_mfem_elem;
   Vector gsl_mesh, gsl_ref, gsl_dist, gsl_mfem_ref;
   Array<unsigned int> recv_proc, recv_index; // data for custom interpolation
   bool setupflag;              // flag to indicate if gslib data has been setup
   double default_interp_value; // used for points that are not found in the mesh
   AvgType avgtype;             // average type used for L2 functions
   Array<int> split_element_map;
   Array<int> split_element_index;
   int        NE_split_total;
   // Tolerance to ignore points just outside elements at the boundary.
   double     bdr_tol;

   /// Use GSLIB for communication and interpolation
   virtual void InterpolateH1(const GridFunction &field_in, Vector &field_out);
   /// Uses GSLIB Crystal Router for communication followed by MFEM's
   /// interpolation functions
   virtual void InterpolateGeneral(const GridFunction &field_in,
                                   Vector &field_out);

   /// Since GSLIB is designed to work with quads/hexes, we split every
   /// triangle/tet/prism/pyramid element into quads/hexes.
   virtual void SetupSplitMeshes();

   /// Setup integration points that will be used to interpolate the nodal
   /// location at points expected by GSLIB.
   virtual void SetupIntegrationRuleForSplitMesh(Mesh *mesh,
                                                 IntegrationRule *irule,
                                                 int order);

   /// Get GridFunction value at the points expected by GSLIB.
   virtual void GetNodalValues(const GridFunction *gf_in, Vector &node_vals);

   /// Map {r,s,t} coordinates from [-1,1] to [0,1] for MFEM. For simplices,
   /// find the original element number (that was split into micro quads/hexes)
   /// during the setup phase.
   virtual void MapRefPosAndElemIndices();

public:
   FindPointsGSLIB();

#ifdef MFEM_USE_MPI
   FindPointsGSLIB(MPI_Comm comm_);
#endif

   virtual ~FindPointsGSLIB();

   /** Initializes the internal mesh in gslib, by sending the positions of the
       Gauss-Lobatto nodes of the input Mesh object \p m.
       Note: not tested with periodic (L2).
       Note: the input mesh \p m must have Nodes set.

       @param[in] m         Input mesh.
       @param[in] bb_t      (Optional) Relative size of bounding box around
                            each element.
       @param[in] newt_tol  (Optional) Newton tolerance for the gslib
                            search methods.
       @param[in] npt_max   (Optional) Number of points for simultaneous
                            iteration. This alters performance and
                            memory footprint.*/
   void Setup(Mesh &m, const double bb_t = 0.1,
              const double newt_tol = 1.0e-12,
              const int npt_max = 256);
   /** Searches positions given in physical space by \p point_pos.
       These positions can be ordered byNodes: (XXX...,YYY...,ZZZ) or
       byVDim: (XYZ,XYZ,....XYZ) specified by \p point_pos_ordering.
       This function populates the following member variables:
       #gsl_code        Return codes for each point: inside element (0),
                        element boundary (1), not found (2).
       #gsl_proc        MPI proc ids where the points were found.
       #gsl_elem        Element ids where the points were found.
                        Defaults to 0 for points that were not found.
       #gsl_mfem_elem   Element ids corresponding to MFEM-mesh where the points
                        were found. #gsl_mfem_elem != #gsl_elem for simplices
                        Defaults to 0 for points that were not found.
       #gsl_ref         Reference coordinates of the found point.
                        Ordered by vdim (XYZ,XYZ,XYZ...). Defaults to -1 for
                        points that were not found. Note: the gslib reference
                        frame is [-1,1].
       #gsl_mfem_ref    Reference coordinates #gsl_ref mapped to [0,1].
                        Defaults to 0 for points that were not found.
       #gsl_dist        Distance between the sought and the found point
                        in physical space. */
   void FindPoints(const Vector &point_pos,
                   int point_pos_ordering = Ordering::byNODES);
   /// Setup FindPoints and search positions
   void FindPoints(Mesh &m, const Vector &point_pos,
                   int point_pos_ordering = Ordering::byNODES,
                   const double bb_t = 0.1,
                   const double newt_tol = 1.0e-12,  const int npt_max = 256);

   /** Interpolation of field values at prescribed reference space positions.
       @param[in] field_in    Function values that will be interpolated on the
                              reference positions. Note: it is assumed that
                              \p field_in is in H1 and in the same space as the
                              mesh that was given to Setup().
       @param[out] field_out  Interpolated values. For points that are not found
                              the value is set to #default_interp_value. */
   virtual void Interpolate(const GridFunction &field_in, Vector &field_out);
   /** Search positions and interpolate. The ordering (byNODES or byVDIM) of
       the output values in \p field_out corresponds to the ordering used
       in the input GridFunction \p field_in. */
   void Interpolate(const Vector &point_pos, const GridFunction &field_in,
                    Vector &field_out,
                    int point_pos_ordering = Ordering::byNODES);
   /** Setup FindPoints, search positions and interpolate. The ordering (byNODES
       or byVDIM) of the output values in \p field_out corresponds to the
       ordering used in the input GridFunction \p field_in. */
   void Interpolate(Mesh &m, const Vector &point_pos,
                    const GridFunction &field_in, Vector &field_out,
                    int point_pos_ordering = Ordering::byNODES);

   /// Average type to be used for L2 functions in-case a point is located at
   /// an element boundary where the function might be multi-valued.
   virtual void SetL2AvgType(AvgType avgtype_) { avgtype = avgtype_; }

   /// Set the default interpolation value for points that are not found in the
   /// mesh.
   virtual void SetDefaultInterpolationValue(double interp_value_)
   {
      default_interp_value = interp_value_;
   }

   /// Set the tolerance for detecting points outside the 'curvilinear' boundary
   /// that gslib may return as found on the boundary. Points found on boundary
   /// with distance greater than @ bdr_tol are marked as not found.
   virtual void SetDistanceToleranceForPointsFoundOnBoundary(double bdr_tol_)
   {
      bdr_tol = bdr_tol_;
   }

   /** Cleans up memory allocated internally by gslib.
       Note that in parallel, this must be called before MPI_Finalize(), as it
       calls MPI_Comm_free() for internal gslib communicators. */
   virtual void FreeData();

   /// Return code for each point searched by FindPoints: inside element (0), on
   /// element boundary (1), or not found (2).
   virtual const Array<unsigned int> &GetCode() const { return gsl_code; }
   /// Return element number for each point found by FindPoints.
   virtual const Array<unsigned int> &GetElem() const { return gsl_mfem_elem; }
   /// Return MPI rank on which each point was found by FindPoints.
   virtual const Array<unsigned int> &GetProc() const { return gsl_proc; }
   /// Return reference coordinates for each point found by FindPoints.
   virtual const Vector &GetReferencePosition() const { return gsl_mfem_ref;  }
   /// Return distance between the sought and the found point in physical space,
   /// for each point found by FindPoints.
   virtual const Vector &GetDist()              const { return gsl_dist; }

   /// Return element number for each point found by FindPoints corresponding to
   /// GSLIB mesh. gsl_mfem_elem != gsl_elem for mesh with simplices.
   virtual const Array<unsigned int> &GetGSLIBElem() const { return gsl_elem; }
   /// Return reference coordinates in [-1,1] (internal range in GSLIB) for each
   /// point found by FindPoints.
   virtual const Vector &GetGSLIBReferencePosition() const { return gsl_ref; }

   /** @name Methods to support a custom interpolation procedure.
       \brief The physical-space point that the user seeks to interpolate at
       could be located inside an element on another mpi rank.
       To enable a custom interpolation procedure (e.g., strain tensor computation)
       we need a mechanism to first send element indices and reference-space
       coordinates to the mpi-ranks where each point is found. Then the custom
       interpolation can be done locally by the user before sending the
       interpolated values back to the mpi-ranks that the query originated from.
       Example usage looks something like this:

       FindPoints() -> DistributePointInfoToOwningMPIRanks() -> Computation by
       user -> DistributeInterpolatedValues().
   */
   ///@{
   /// Distribute element indices in #gsl_mfem_elem, the reference coordinates
   /// #gsl_mfem_ref, and the code #gsl_code to the corresponding mpi-rank
   /// #gsl_proc for each point. The received information is provided locally
   /// in \p recv_elem, \p recv_ref (ordered by vdim), and \p recv_code.
   /// Note: The user can send empty Array/Vectors to the method as they are
   /// appropriately sized and filled internally.
   virtual void DistributePointInfoToOwningMPIRanks(
      Array<unsigned int> &recv_elem, Vector &recv_ref,
      Array<unsigned int> &recv_code);
   /// Return interpolated values back to the mpi-ranks #recv_proc that had
   /// sent the element indices and corresponding reference-space coordinates.
   /// Specify \p vdim and \p ordering (by nodes or by vdim) based on how the
   /// \p int_vals are structured. The received values are filled in
   /// \p field_out consistent with the original ordering of the points that
   /// were used in \ref FindPoints.
   virtual void DistributeInterpolatedValues(const Vector &int_vals,
                                             const int vdim,
                                             const int ordering,
                                             Vector &field_out) const;
   ///@}
};

/** \brief OversetFindPointsGSLIB enables use of findpts for arbitrary number of
    overlapping grids.

    The parameters in this class are the same as FindPointsGSLIB with the
    difference of additional inputs required to account for more than 1 mesh. */
class OversetFindPointsGSLIB : public FindPointsGSLIB
{
protected:
   bool overset;
   unsigned int u_meshid;
   Vector distfint; // Used to store nodal vals of grid func. passed to findpts

public:
   OversetFindPointsGSLIB() : FindPointsGSLIB(),
      overset(true) { }

#ifdef MFEM_USE_MPI
   OversetFindPointsGSLIB(MPI_Comm comm_) : FindPointsGSLIB(comm_),
      overset(true) { }
#endif

   /** Initializes the internal mesh in gslib, by sending the positions of the
       Gauss-Lobatto nodes of the input Mesh object \p m.
       Note: not tested with periodic meshes (L2).
       Note: the input mesh \p m must have Nodes set.

       @param[in] m         Input mesh.
       @param[in] meshid    A unique # for each overlapping mesh. This id is
                            used to make sure that points being searched are not
                            looked for in the mesh that they belong to.
       @param[in] gfmax     (Optional) GridFunction in H1 that is used as a
                            discriminator when one point is located in multiple
                            meshes. The mesh that maximizes gfmax is chosen.
                            For example, using the distance field based on the
                            overlapping boundaries is helpful for convergence
                            during Schwarz iterations.
       @param[in] bb_t      (Optional) Relative size of bounding box around
                            each element.
       @param[in] newt_tol  (Optional) Newton tolerance for the gslib
                            search methods.
       @param[in] npt_max   (Optional) Number of points for simultaneous
                            iteration. This alters performance and
                            memory footprint.*/
   void Setup(Mesh &m, const int meshid, GridFunction *gfmax = NULL,
              const double bb_t = 0.1, const double newt_tol = 1.0e-12,
              const int npt_max = 256);

   /** Searches positions given in physical space by \p point_pos. All output
       Arrays and Vectors are expected to have the correct size.

       @param[in]  point_pos           Positions to be found.
       @param[in]  point_id            Index of the mesh that the point belongs
                                       to (corresponding to \p meshid in Setup).
       @param[in]  point_pos_ordering  Ordering of the points:
                                       byNodes: (XXX...,YYY...,ZZZ) or
                                       byVDim: (XYZ,XYZ,....XYZ) */
   void FindPoints(const Vector &point_pos,
                   Array<unsigned int> &point_id,
                   int point_pos_ordering = Ordering::byNODES);

   /** Search positions and interpolate */
   void Interpolate(const Vector &point_pos, Array<unsigned int> &point_id,
                    const GridFunction &field_in, Vector &field_out,
                    int point_pos_ordering = Ordering::byNODES);
   using FindPointsGSLIB::Interpolate;
};

/** \brief  Class for gather-scatter (gs) operations on Vectors based on
    corresponding global identifiers.

    This functionality is useful for gs-ops on DOF values across processor
    boundary, where the global identifier would be the corresponding true DOF
    index. Operations currently supported are min, max, sum, and multiplication.
    Note: identifier 0 does not participate in the gather-scatter operation and
    a given identifier can be included multiple times on a given rank.
    For example, consider a vector, v:
    - v = [0.3, 0.4, 0.25, 0.7] on rank1,
    - v = [0.6, 0.1] on rank 2,
    - v = [-0.2, 0.3, 0.7, 0.] on rank 3.

    Consider a corresponding Array<int>, a:
    - a = [1, 2, 3, 1] on rank 1,
    - a = [3, 2] on rank 2,
    - a = [1, 2, 0, 3] on rank 3.

    A gather-scatter "minimum" operation, done as follows:
    GSOPGSLIB gs = GSOPGSLIB(MPI_COMM_WORLD, a);
    gs.GS(v, GSOp::MIN);
    would return into v:
    - v = [-0.2, 0.1, 0., -0.2] on rank 1,
    - v = [0., 0.1] on rank 2,
    - v = [-0.2, 0.1, 0.7, 0.] on rank 3,
    where the values have been compared across all processors based on the
    integer identifier. */
class GSOPGSLIB
{
protected:
   struct gslib::crystal *cr;               // gslib's internal data
   struct gslib::comm *gsl_comm;            // gslib's internal data
   struct gslib::gs_data *gsl_data = NULL;
   int num_ids;

public:
   GSOPGSLIB(Array<long long> &ids);

#ifdef MFEM_USE_MPI
   GSOPGSLIB(MPI_Comm comm_, Array<long long> &ids);
#endif

   virtual ~GSOPGSLIB();

   /// Supported operation types. See class description.
   enum GSOp {ADD, MUL, MIN, MAX};

   /// Update the identifiers used for the gather-scatter operator.
   /// Same \p ids get grouped together and id == 0 does not participate.
   /// See class description.
   void UpdateIdentifiers(const Array<long long> &ids);

   /// Gather-Scatter operation on senddata. Must match length of unique
   /// identifiers used in the constructor. See class description.
   void GS(Vector &senddata, GSOp op);
};

} // namespace mfem

#endif // MFEM_USE_GSLIB

#endif // MFEM_GSLIB
