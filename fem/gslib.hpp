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

#ifndef MFEM_GSLIB
#define MFEM_GSLIB

#include "../config/config.hpp"
#include "gridfunc.hpp"

#ifdef MFEM_USE_GSLIB

struct comm;
struct findpts_data_2;
struct findpts_data_3;
struct array;
struct crystal;

namespace mfem
{

/** \brief FindPointsGSLIB can robustly evaluate a GridFunction on an arbitrary
 *  collection of points. There are three key functions in FindPointsGSLIB:
 *
 *  1. Setup - constructs the internal data structures of gslib.
 *
 *  2. FindPoints - for any given arbitrary set of points in physical space,
 *     gslib finds the element number, MPI rank, and the reference space
 *     coordinates inside the element that each point is located in. gslib also
 *     returns a code that indicates  wether the point was found inside an
 *     element, on element border, or not found in the domain.
 *
 *  3. Interpolate - Interpolates any gridfunction at the points found using 2.
 *
 *  FindPointsGSLIB provides interface to use these functions individually or using
 *  a single call.
 */
class FindPointsGSLIB
{
protected:
   Mesh *mesh, *meshsplit;
   IntegrationRule *ir_simplex; // IntegrationRule to split quads/hex -> simplex
   struct findpts_data_2 *fdata2D; // pointer to gslib's
   struct findpts_data_3 *fdata3D; // internal data
   int dim, points_cnt;
   Array<unsigned int> gsl_code, gsl_proc, gsl_elem, gsl_mfem_elem;
   Vector gsl_mesh, gsl_ref, gsl_dist, gsl_mfem_ref;
   bool setupflag;
   struct crystal *cr;
   struct comm *gsl_comm;
   double default_interp_value;

   GridFunction::AvgType avgtype;

   /// Get GridFunction from MFEM format to GSLIB format
   void GetNodeValues(const GridFunction &gf_in, Vector &node_vals);
   /// Get nodal coordinates from mesh to the format expected by GSLIB for quads
   /// and hexes
   void GetQuadHexNodalCoordinates();
   /// Convert simplices to quad/hexes and then get nodal coordinates for each
   /// split element into format expected by GSLIB
   void GetSimplexNodalCoordinates();

   /// Use GSLIB for communication and interpolation
   void InterpolateH1(const GridFunction &field_in, Vector &field_out);
   /// Uses GSLIB Crystal Router for communication followed by MFEM's
   /// interpolation functions
   void InterpolateGeneral(const GridFunction &field_in, Vector &field_out);
   /// Map {r,s,t} coordinates from [-1,1] to [0,1] for MFEM. For simplices mesh
   /// find the original element number (that was split into micro quads/hexes
   /// by GetSimplexNodalCoordinates())
   void MapRefPosAndElemIndices();


public:
   FindPointsGSLIB();

#ifdef MFEM_USE_MPI
   FindPointsGSLIB(MPI_Comm _comm);
#endif

   ~FindPointsGSLIB();

   /** Initializes the internal mesh in gslib, by sending the positions of the
       Gauss-Lobatto nodes of the input Mesh object @a m.
       Note: not tested with periodic (DG meshes).
       Note: the input mesh @a m must have Nodes set.

       @param[in] m         Input mesh.
       @param[in] bb_t      Relative size of bounding box around each element.
       @param[in] newt_tol  Newton tolerance for the gslib search methods.
       @param[in] npt_max   Number of points for simultaneous iteration. This
                            alters performance and memory footprint. */
   void Setup(Mesh &m, const double bb_t = 0.1, const double newt_tol = 1.0e-12,
              const int npt_max = 256);

   /** Searches positions given in physical space by @a point_pos. All output
       Arrays and Vectors are expected to have the correct size.
       @param[in]  point_pos       Positions to be found. Must by ordered by nodes
                                   (XXX...,YYY...,ZZZ).
       @param[out] gsl_codes       Return codes for each point: inside element (0),
                                   element boundary (1), not found (2).
       @param[out] gsl_proc        MPI proc ids where the points were found.
       @param[out] gsl_elem        Element ids where the points were found.
                                   Defaults to 0 for points that were not found.
       @param[out] gsl_mfem_elem   Element ids corresponding to MFEM-mesh
                                   where the points were found.
                                   @a gsl_mfem_elem != @a gsl_elem for simplices
                                   Defaults to 0 for points that were not found.
       @param[out] gsl_ref         Reference coordinates of the found point.
                                   Ordered by vdim (XYZ,XYZ,XYZ...).
                                   Note: the gslib reference frame is [-1,1].
                                   Defaults to -1 for points that were not found.
       @param[out] gsl_mfem_ref    Reference coordinates @a gsl_ref mapped to [0,1].
                                   Defaults to 0 for points that were not found.
       @param[out] gsl_dist        Distance between the sought and the found point
                                   in physical space. */
   void FindPoints(const Vector &point_pos);
   /// Setup FindPoints and search positions
   void FindPoints(Mesh &m, const Vector &point_pos, const double bb_t = 0.1,
                   const double newt_tol = 1.0e-12,  const int npt_max = 256);

   /** Interpolation of field values at prescribed reference space positions.
       @param[in] field_in    Function values that will be interpolated on the
                              reference positions. Note: it is assumed that
                              @a field_in is in H1 and in the same space as the
                              mesh that was given to Setup().
       @param[out] field_out  Interpolated values. For points that are not found
                              the value is set to #default_interp_value. */
   void Interpolate(const GridFunction &field_in, Vector &field_out);
   /** Search positions and interpolate */
   void Interpolate(const Vector &point_pos, const GridFunction &field_in,
                    Vector &field_out);
   /** Setup FindPoints, search positions and interpolate */
   void Interpolate(Mesh &m, const Vector &point_pos,
                    const GridFunction &field_in, Vector &field_out);

   /// Average type to be used for L2 functions in-case a point is located at
   /// an element boundary where the function might not have a unique value.
   void SetL2AvgType(GridFunction::AvgType avgtype_) { avgtype = avgtype_; }

   /// Set the default interpolation value for points that are not found in the
   /// mesh.
   void SetDefaultInterpolationValue(double interp_value_)
   {
      default_interp_value = interp_value_;
   }

   /** Cleans up memory allocated internally by gslib.
       Note that in parallel, this must be called before MPI_Finalize(), as
       it calls MPI_Comm_free() for internal gslib communicators. */
   void FreeData();

   /// Return code for each point searched by FindPoints: inside element (0), on
   /// element boundary (1), or not found (2).
   const Array<unsigned int> &GetCode() const { return gsl_code; }
   /// Return element number for each point found by FindPoints.
   const Array<unsigned int> &GetElem() const { return gsl_mfem_elem; }
   /// Return MPI rank on which each point was found by FindPoints.
   const Array<unsigned int> &GetProc() const { return gsl_proc; }
   /// Return reference coordinates for each point found by FindPoints.
   const Vector &GetReferencePosition() const { return gsl_mfem_ref;  }
   /// Return distance Distance between the sought and the found point
   /// in physical space, for each point found by FindPoints.
   const Vector &GetDist()              const { return gsl_dist; }

   /// Return element number for each point found by FindPoints corresponding to
   /// GSLIB mesh. gsl_mfem_elem != gsl_elem for mesh with simplices.
   const Array<unsigned int> &GetGSLIBElem() const { return gsl_elem; }
   /// Return reference coordinates in [-1,1] (internal range in GSLIB) for each
   /// point found by FindPoints.
   const Vector &GetGSLIBReferencePosition() const { return gsl_ref; }


};

} // namespace mfem

#endif //MFEM_USE_GSLIB

#endif //MFEM_GSLIB guard
