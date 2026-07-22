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
//
//      ------------------------------------------------------------------
//      Parallel Field Interp Miniapp: Transfer a grid function between meshes
//      ------------------------------------------------------------------
//
// Compile with: make pfield-interp
//
// Sample runs:
//   make pfield-interp -j10 && mpirun -np 4 pfield-interp -m1 triple-pt-1.mesh -s1 triple-pt-1.gf -m2 triple-pt-2.mesh -vis

#include "mfem.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>

namespace gslib
{
#include "gslib.h"
}

using namespace mfem;
using namespace std;

class FieldTransfer
{
private:
   unique_ptr<FindPointsGSLIB> finder;
   FiniteElementSpace *fes;
   ParFiniteElementSpace *pfes;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif
   int dim;
   bool find_points;

   void SetupMesh(Mesh &source_mesh)
   {
      MFEM_VERIFY(source_mesh.GetNodes() != NULL, "Mesh nodes are required.");
      MFEM_VERIFY(source_mesh.Dimension() > 1,
                  "GSLIB requires a 2D or a 3D mesh.");
#ifdef MFEM_USE_MPI
      ParMesh *source_pmesh = dynamic_cast<ParMesh *>(&source_mesh);
      if (source_pmesh != NULL)
      {
         comm = source_pmesh->GetComm();
         finder.reset(new FindPointsGSLIB(comm));
      }
      else
#endif
      {
#ifdef MFEM_USE_MPI
         comm = MPI_COMM_WORLD;
#endif
         finder.reset(new FindPointsGSLIB);
      }
      dim = source_mesh.Dimension();
      find_points = false;
      finder->Setup(source_mesh);
   }

   void Setup(FiniteElementSpace &source_fes)
   {
      fes = &source_fes;
      pfes = dynamic_cast<ParFiniteElementSpace *>(&source_fes);
#ifdef MFEM_USE_MPI
      if (pfes != NULL)
      {
         SetupMesh(*pfes->GetParMesh());
         return;
      }
#endif
      SetupMesh(*source_fes.GetMesh());
   }

   int SourceVectorDim() const
   {
      MFEM_VERIFY(fes != NULL,
                  "A FiniteElementSpace constructor is required.");
      const FiniteElement *fe = fes->GetTypicalFE();
      if (!fe || fe->GetRangeType() == FiniteElement::SCALAR)
      {
         return fes->GetVDim();
      }
      return fes->GetVDim()*std::max(fes->GetMesh()->SpaceDimension(),
                                     fe->GetRangeDim());
   }

   void AddTransposeContribution(int elem, const IntegrationPoint &ip,
                                 int comp, real_t value,
                                 Vector &source_values) const
   {
      const FiniteElement *fe = fes->GetFE(elem);
      const int dof = fe->GetDof();
      const int vdim = fes->GetVDim();

      Array<int> vdofs;
      fes->GetElementVDofs(elem, vdofs);
      Vector elem_values(vdofs.Size());
      elem_values = 0.0;

      if (fe->GetRangeType() == FiniteElement::SCALAR)
      {
         MFEM_VERIFY(comp < vdim,
                     "Target value component exceeds source FES vector dimension.");
         Vector shape(dof);
         fe->CalcShape(ip, shape);
         for (int j = 0; j < dof; j++)
         {
            elem_values(j + comp*dof) = value*shape(j);
         }
      }
      else
      {
         const int vector_dim = std::max(fes->GetMesh()->SpaceDimension(),
                                         fe->GetRangeDim());
         const int vd = comp/vector_dim;
         const int d = comp%vector_dim;
         MFEM_VERIFY(vd < vdim,
                     "Target value component exceeds source FES vector dimension.");

         ElementTransformation *tr = fes->GetElementTransformation(elem);
         tr->SetIntPoint(&ip);

         DenseMatrix vshape(dof, vector_dim);
         fe->CalcVShape(*tr, vshape);
         for (int j = 0; j < dof; j++)
         {
            elem_values(j + vd*dof) = value*vshape(j, d);
         }
      }

      source_values.AddElementVector(vdofs, elem_values);
   }

public:
   FieldTransfer(const FieldTransfer &) = delete;
   FieldTransfer &operator=(const FieldTransfer &) = delete;

   FieldTransfer(FiniteElementSpace &source_fes)
      : finder(nullptr), fes(NULL), pfes(NULL), dim(-1), find_points(false)
   {
      Setup(source_fes);
   }

   FieldTransfer(FiniteElementSpace &source_fes, const Vector &point_coords,
                 int point_ordering = Ordering::byNODES)
      : finder(nullptr), fes(NULL), pfes(NULL), dim(-1), find_points(false)
   {
      Setup(source_fes);
      FindPoints(point_coords, point_ordering);
   }

#ifdef MFEM_USE_MPI
   FieldTransfer(ParFiniteElementSpace &source_fes)
      : finder(nullptr), fes(NULL), pfes(NULL), dim(-1),
        find_points(false)
   {
      Setup(source_fes);
   }

   FieldTransfer(ParFiniteElementSpace &source_fes, const Vector &point_coords,
                 int point_ordering = Ordering::byNODES)
      : finder(nullptr), fes(NULL), pfes(NULL), dim(-1),
        find_points(false)
   {
      Setup(source_fes);
      FindPoints(point_coords, point_ordering);
   }
#endif

   ~FieldTransfer()
   {
      if (finder) { finder->FreeData(); }
   }

   void FindPoints(const Vector &point_coords,
                   int point_ordering = Ordering::byNODES)
   {
      MFEM_VERIFY(point_coords.Size() % dim == 0,
                  "Point coordinate vector size is not divisible by mesh dimension.");
      finder->FindPoints(point_coords, point_ordering);
      find_points = true;
   }

   void Mult(const GridFunction &source, Vector &interp_vals)
   {
      MFEM_VERIFY(find_points,
                  "Call FindPoints or use a point-list constructor before this Mult.");
      finder->Interpolate(source, interp_vals);
   }

   Vector Mult(const GridFunction &source)
   {
      Vector interp_vals;
      Mult(source, interp_vals);
      return interp_vals;
   }

   void Mult(const GridFunction &source, const Vector &point_coords,
             Vector &interp_vals, int point_ordering = Ordering::byNODES)
   {
      FindPoints(point_coords, point_ordering);
      Mult(source, interp_vals);
   }

   Vector Mult(const GridFunction &source, const Vector &point_coords,
               int point_ordering = Ordering::byNODES)
   {
      Vector interp_vals;
      Mult(source, point_coords, interp_vals, point_ordering);
      return interp_vals;
   }

   void Mult(const Vector &source, Vector &interp_vals,
             bool source_is_true_vector = false)
   {
      MFEM_VERIFY(fes != NULL,
                  "A FiniteElementSpace constructor is required for Vector input.");

      if (pfes != NULL)
      {
         ParGridFunction source_gf(pfes);
         if (source_is_true_vector ||
             (source.Size() == pfes->GetTrueVSize() &&
              source.Size() != pfes->GetVSize()))
         {
            MFEM_VERIFY(source.Size() == pfes->GetTrueVSize(),
                        "Input vector is not a parallel T-vector.");
            source_gf.Distribute(&source);
         }
         else
         {
            MFEM_VERIFY(source.Size() == pfes->GetVSize(),
                        "Input vector is not a parallel L-vector.");
            source_gf = source;
         }
         Mult(source_gf, interp_vals);
      }
      else
      {
         GridFunction source_gf(fes);
         if (source_is_true_vector ||
             (source.Size() == fes->GetTrueVSize() &&
              source.Size() != fes->GetVSize()))
         {
            if (source.Size() == fes->GetVSize())
            {
               source_gf = source;
            }
            else
            {
               const Operator *P = fes->GetProlongationMatrix();
               MFEM_VERIFY(P != NULL && source.Size() == fes->GetTrueVSize(),
                           "Input vector is not a T-vector for this space.");
               P->Mult(source, source_gf);
            }
         }
         else
         {
            MFEM_VERIFY(source.Size() == fes->GetVSize(),
                        "Input vector is not an L-vector.");
            source_gf = source;
         }
         Mult(source_gf, interp_vals);
      }
   }

   Vector Mult(const Vector &source, bool source_is_true_vector = false)
   {
      Vector interp_vals;
      Mult(source, interp_vals, source_is_true_vector);
      return interp_vals;
   }

   void Mult(const Vector &source, const Vector &point_coords,
             Vector &interp_vals, int point_ordering = Ordering::byNODES,
             bool source_is_true_vector = false)
   {
      FindPoints(point_coords, point_ordering);
      Mult(source, interp_vals, source_is_true_vector);
   }

   Vector Mult(const Vector &source, const Vector &point_coords,
               int point_ordering = Ordering::byNODES,
               bool source_is_true_vector = false)
   {
      Vector interp_vals;
      Mult(source, point_coords, interp_vals, point_ordering,
           source_is_true_vector);
      return interp_vals;
   }

   void MultTranspose(const Vector &target_values, Vector &source_values,
                      int value_ordering = -1)
   {
      MFEM_VERIFY(fes != NULL,
                  "A FiniteElementSpace constructor is required for MultTranspose.");
      MFEM_VERIFY(find_points,
                  "Call FindPoints or use a point-list constructor before MultTranspose.");

      const Array<unsigned int> &code = finder->GetCode();
      const Array<unsigned int> &proc = finder->GetProc();
      const Array<unsigned int> &elem = finder->GetElem();
      const Vector &ref = finder->GetReferencePosition();
      const int npoints = code.Size();
      const int source_vdim = SourceVectorDim();

      source_values.SetSize(fes->GetVSize());
      source_values = 0.0;

      if (npoints == 0)
      {
         MFEM_VERIFY(target_values.Size() == 0,
                     "Target value vector is inconsistent with zero points.");
         return;
      }

      MFEM_VERIFY(target_values.Size() == npoints*source_vdim,
                  "Target value vector size is not compatible with source FESpace.");

      if (value_ordering < 0) { value_ordering = fes->GetOrdering(); }

      int nsend = 0;
      for (int i = 0; i < npoints; i++)
      {
         if (code[i] != 2) { nsend += source_vdim; }
      }

      struct field_transpose_pt
      {
         double r[3], value;
         unsigned int elem, proc, comp;
      };

      gslib::comm gsl_comm;
      gslib::crystal cr;
#ifdef MFEM_USE_MPI
      comm_init(&gsl_comm, comm);
#else
      comm_init(&gsl_comm, 0);
#endif
      crystal_init(&cr, &gsl_comm);

      gslib::array outpt;
      array_init(field_transpose_pt, &outpt, nsend);
      outpt.n = nsend;

      field_transpose_pt *pt = (field_transpose_pt *)outpt.ptr;
      for (int i = 0; i < npoints; i++)
      {
         if (code[i] == 2) { continue; }
         for (int c = 0; c < source_vdim; c++)
         {
            const int value_idx = (value_ordering == Ordering::byNODES) ?
                                  i + c*npoints : i*source_vdim + c;
            for (int d = 0; d < dim; d++) { pt->r[d] = ref(i*dim + d); }
            pt->value = target_values(value_idx);
            pt->elem = elem[i];
            pt->proc = proc[i];
            pt->comp = c;
            ++pt;
         }
      }

      sarray_transfer(field_transpose_pt, &outpt, proc, 1, &cr);

      pt = (field_transpose_pt *)outpt.ptr;
      for (int i = 0; i < outpt.n; i++)
      {
         IntegrationPoint ip;
         if (dim == 2)
         {
            ip.Set2(pt[i].r[0], pt[i].r[1]);
         }
         else
         {
            ip.Set3(pt[i].r[0], pt[i].r[1], pt[i].r[2]);
         }
         AddTransposeContribution(pt[i].elem, ip, pt[i].comp, pt[i].value,
                                  source_values);
      }

      array_free(&outpt);
      crystal_free(&cr);
      comm_free(&gsl_comm);
   }

   Vector MultTranspose(const Vector &target_values, int value_ordering = -1)
   {
      Vector source_values;
      MultTranspose(target_values, source_values, value_ordering);
      return source_values;
   }

   void MultTranspose(const Vector &target_values, const Vector &point_coords,
                      Vector &source_values,
                      int point_ordering = Ordering::byNODES,
                      int value_ordering = -1)
   {
      FindPoints(point_coords, point_ordering);
      MultTranspose(target_values, source_values, value_ordering);
   }

   Vector MultTranspose(const Vector &target_values, const Vector &point_coords,
                        int point_ordering = Ordering::byNODES,
                        int value_ordering = -1)
   {
      Vector source_values;
      MultTranspose(target_values, point_coords, source_values, point_ordering,
                    value_ordering);
      return source_values;
   }

   void MultTranpose(const Vector &target_values, Vector &source_values,
                     int value_ordering = -1)
   {
      MultTranspose(target_values, source_values, value_ordering);
   }

   Vector MultTranpose(const Vector &target_values, int value_ordering = -1)
   {
      return MultTranspose(target_values, value_ordering);
   }

   const Array<unsigned int> &GetCode() const { return finder->GetCode(); }
   const Vector &GetDist() const { return finder->GetDist(); }
};

static bool UsesPointDofs(const FiniteElementCollection &fec)
{
   return dynamic_cast<const H1_FECollection *>(&fec) ||
          dynamic_cast<const L2_FECollection *>(&fec);
}

static void GetScalarDofCoordinates(ParFiniteElementSpace &fes, Vector &coords)
{
   ParMesh *pmesh = fes.GetParMesh();
   const int dim = pmesh->Dimension();
   const int ndofs = fes.GetNDofs();
   coords.SetSize(ndofs*dim);

   Array<int> visited(ndofs);
   visited = 0;
   coords = 0.0;

   Array<int> dofs;
   DenseMatrix pos;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      const FiniteElement *fe = fes.GetFE(e);
      const IntegrationRule &ir = fe->GetNodes();
      MFEM_VERIFY(ir.GetNPoints() == fe->GetDof(),
                  "The target finite element does not provide one node per dof.");

      ElementTransformation *tr = fes.GetElementTransformation(e);
      tr->Transform(ir, pos);

      fes.GetElementDofs(e, dofs);
      MFEM_VERIFY(dofs.Size() == ir.GetNPoints(),
                  "Inconsistent target dof and node counts.");

      for (int j = 0; j < dofs.Size(); j++)
      {
         const int dof = FiniteElementSpace::DecodeDof(dofs[j]);
         if (visited[dof]) { continue; }
         for (int d = 0; d < dim; d++)
         {
            coords(d*ndofs + dof) = pos(d, j);
         }
         visited[dof] = 1;
      }
   }
}

static void GetElementNodeCoordinates(ParFiniteElementSpace &fes,
                                      Vector &coords,
                                      int &points_per_elem)
{
   ParMesh *pmesh = fes.GetParMesh();
   const int dim = pmesh->Dimension();
   const int NE = fes.GetNE();
   points_per_elem = fes.GetTypicalFE()->GetNodes().GetNPoints();

   coords.SetSize(points_per_elem*NE*dim);
   DenseMatrix pos;
   for (int e = 0; e < NE; e++)
   {
      const FiniteElement *fe = fes.GetFE(e);
      const IntegrationRule &ir = fe->GetNodes();
      MFEM_VERIFY(ir.GetNPoints() == points_per_elem,
                  "Variable-order target spaces are not supported by this miniapp.");

      ElementTransformation *tr = fes.GetElementTransformation(e);
      tr->Transform(ir, pos);

      Vector rowx(coords.GetData() + e*points_per_elem, points_per_elem);
      Vector rowy(coords.GetData() + e*points_per_elem +
                  NE*points_per_elem, points_per_elem);
      pos.GetRow(0, rowx);
      pos.GetRow(1, rowy);
      if (dim == 3)
      {
         Vector rowz(coords.GetData() + e*points_per_elem +
                     2*NE*points_per_elem, points_per_elem);
         pos.GetRow(2, rowz);
      }
   }
}

static void ProjectElementNodeValues(ParFiniteElementSpace &fes,
                                     const Vector &interp_vals,
                                     int points_per_elem,
                                     ParGridFunction &target)
{
   const int NE = fes.GetNE();
   const int vdim = target.VectorDim();
   const int npoints = NE*points_per_elem;
   const int ordering = fes.GetOrdering();

   Array<int> vdofs;
   Vector elem_node_vals(points_per_elem*vdim);
   Vector elem_dof_vals;

   for (int e = 0; e < NE; e++)
   {
      for (int j = 0; j < points_per_elem; j++)
      {
         for (int d = 0; d < vdim; d++)
         {
            const int idx = (ordering == Ordering::byNODES) ?
                            d*npoints + e*points_per_elem + j :
                            (e*points_per_elem + j)*vdim + d;
            elem_node_vals(j*vdim + d) = interp_vals(idx);
         }
      }

      fes.GetElementVDofs(e, vdofs);
      elem_dof_vals.SetSize(vdofs.Size());
      fes.GetFE(e)->ProjectFromNodes(elem_node_vals,
                                     *fes.GetElementTransformation(e),
                                     elem_dof_vals);
      target.SetSubVector(vdofs, elem_dof_vals);
   }
}

static void CountFindPointsCodes(const FieldTransfer &transfer, MPI_Comm comm)
{
   const Array<unsigned int> &code = transfer.GetCode();
   const Vector &dist = transfer.GetDist();

   int found = 0, boundary = 0, not_found = 0;
   real_t max_dist = 0.0;
   for (int i = 0; i < code.Size(); i++)
   {
      if (code[i] == 0) { found++; }
      else if (code[i] == 1)
      {
         boundary++;
         max_dist = std::max(max_dist, dist(i));
      }
      else { not_found++; }
   }

   MPI_Allreduce(MPI_IN_PLACE, &found, 1, MPI_INT, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, &boundary, 1, MPI_INT, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, &not_found, 1, MPI_INT, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, &max_dist, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, comm);

   if (Mpi::Root())
   {
      cout << "FindPoints summary:"
           << "\n  points found inside:   " << found
           << "\n  points found on bdr:   " << boundary
           << "\n  points not found:      " << not_found
           << "\n  max boundary distance: " << max_dist << endl;
   }
}

static void VisualizeField(ParMesh &pmesh, ParGridFunction &field,
                           const char *title, int pos_x, int pos_y,
                           int visport, int num_procs, int myid)
{
   char vishost[] = "localhost";
   socketstream sout(vishost, visport);
   if (!sout)
   {
      if (myid == 0)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
      }
      return;
   }

   sout << "parallel " << num_procs << " " << myid << "\n";
   sout.precision(8);
   sout << "solution\n" << pmesh << field
        << "window_title '" << title << "'\n"
        << "window_geometry " << pos_x << " " << pos_y << " 600 600\n";
   if (pmesh.Dimension() == 2) { sout << "keys RmjAc\n"; }
   if (pmesh.Dimension() == 3) { sout << "keys mA\n"; }
   sout << flush;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   MPI_Comm comm = MPI_COMM_WORLD;
   const int num_procs = Mpi::WorldSize();
   const int myid = Mpi::WorldRank();

   const char *src_mesh_file = "triple-pt-1.mesh";
   const char *tar_mesh_file = "triple-pt-2.mesh";
   const char *src_sltn_file = "triple-pt-1.gf";
   const char *out_mesh_file = "pfield-interp.mesh";
   const char *out_sltn_file = "pinterpolated.gf";
   int ref_levels = 0;
   bool visualization = true;
   int visport = 19916;

   OptionsParser args(argc, argv);
   args.AddOption(&src_mesh_file, "-m1", "--mesh1",
                  "Source mesh file.");
   args.AddOption(&src_sltn_file, "-s1", "--solution1",
                  "Source GridFunction file compatible with the source mesh.");
   args.AddOption(&tar_mesh_file, "-m2", "--mesh2",
                  "Target mesh file.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of serial refinements of the target mesh.");
   args.AddOption(&out_mesh_file, "-om", "--output-mesh",
                  "Output parallel target mesh prefix.");
   args.AddOption(&out_sltn_file, "-os", "--output-solution",
                  "Output parallel interpolated GridFunction prefix.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Mesh src_mesh(src_mesh_file, 1, 1, false);
   Mesh tar_mesh(tar_mesh_file, 1, 1, false);

   const int dim = src_mesh.Dimension();
   MFEM_VERIFY(dim == tar_mesh.Dimension(),
               "Source and target meshes must have the same dimension.");
   MFEM_VERIFY(dim > 1, "GSLIB requires a 2D or a 3D mesh.");

   if (src_mesh.GetNodes() == NULL) { src_mesh.SetCurvature(1); }
   if (tar_mesh.GetNodes() == NULL) { tar_mesh.SetCurvature(1); }

   for (int lev = 0; lev < ref_levels; lev++)
   {
      tar_mesh.UniformRefinement();
   }

   ifstream src_sltn(src_sltn_file);
   MFEM_VERIFY(src_sltn, "Cannot open source GridFunction file: " <<
               src_sltn_file);
   unique_ptr<GridFunction> src_serial_gf(new GridFunction(&src_mesh, src_sltn));

   unique_ptr<int[]> src_partition(src_mesh.GeneratePartitioning(num_procs));
   unique_ptr<int[]> tar_partition(tar_mesh.GeneratePartitioning(num_procs));

   ParMesh src_pmesh(comm, src_mesh, src_partition.get());
   ParMesh tar_pmesh(comm, tar_mesh, tar_partition.get());
   ParGridFunction src_gf(&src_pmesh, src_serial_gf.get(),
                          src_partition.get());

   const FiniteElementSpace *src_fes = src_gf.FESpace();
   unique_ptr<FiniteElementCollection> tar_fec(
      src_fes->FEColl()->Clone(src_fes->GetMaxElementOrder()));
   ParFiniteElementSpace tar_fes(&tar_pmesh, tar_fec.get(),
                                 src_fes->GetVDim(), src_fes->GetOrdering());
   ParGridFunction tar_gf(&tar_fes);

   if (myid == 0)
   {
      cout << "Source FE collection: " << src_fes->FEColl()->Name() << endl
           << "Target FE collection: " << tar_fec->Name() << endl;
   }

   unique_ptr<FieldTransfer> transfer;
   Vector interp_vals;
   if (UsesPointDofs(*tar_fec))
   {
      Vector target_points;
      GetScalarDofCoordinates(tar_fes, target_points);
      transfer.reset(new FieldTransfer(*src_gf.ParFESpace(), target_points,
                                       Ordering::byNODES));
      transfer->Mult(static_cast<const Vector &>(src_gf), interp_vals);
      MFEM_VERIFY(interp_vals.Size() == tar_gf.Size(),
                  "Interpolated value vector does not match target GridFunction size.");
      tar_gf = interp_vals;
   }
   else
   {
      Vector target_points;
      int points_per_elem = 0;
      GetElementNodeCoordinates(tar_fes, target_points, points_per_elem);
      transfer.reset(new FieldTransfer(*src_gf.ParFESpace(), target_points,
                                       Ordering::byNODES));
      transfer->Mult(static_cast<const Vector &>(src_gf), interp_vals);
      ProjectElementNodeValues(tar_fes, interp_vals, points_per_elem, tar_gf);
   }

   CountFindPointsCodes(*transfer, comm);

   Vector source_transpose;
   transfer->MultTranspose(interp_vals, source_transpose);
   ParGridFunction source_transpose_gf(src_gf.ParFESpace());
   source_transpose_gf = source_transpose;

   real_t transpose_norm = source_transpose.Norml2();
   transpose_norm *= transpose_norm;
   MPI_Allreduce(MPI_IN_PLACE, &transpose_norm, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   transpose_norm = std::sqrt(transpose_norm);
   if (myid == 0)
   {
      cout << "MultTranspose source L-vector norm: " << transpose_norm << endl;
   }

   tar_pmesh.Save(out_mesh_file, 8);
   tar_gf.Save(out_sltn_file, 8);
   if (myid == 0)
   {
      cout << "Saved target mesh prefix: " << out_mesh_file << endl
           << "Saved interpolated GridFunction prefix: " << out_sltn_file
           << endl;
   }

   if (visualization)
   {
      VisualizeField(src_pmesh, src_gf, "Source mesh and solution",
                     0, 0, visport, num_procs, myid);
      VisualizeField(tar_pmesh, tar_gf, "Target mesh and interpolated solution",
                     600, 0, visport, num_procs, myid);
      VisualizeField(src_pmesh, source_transpose_gf,
                     "Source mesh and MultTranspose solution",
                     1200, 0, visport, num_procs, myid);
   }

   return 0;
}
