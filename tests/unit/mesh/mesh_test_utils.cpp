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

#include "mesh_test_utils.hpp"

#include <numeric>

namespace mfem
{


FiniteElementCollection *create_fec(FECType fectype, int p, int dim)
{
   switch (fectype)
   {
      case FECType::H1:
         return new H1_FECollection(p, dim);
      case FECType::ND:
         return new ND_FECollection(p, dim);
      case FECType::RT:
         return new RT_FECollection(p - 1, dim);
      case FECType::L2:
         return new L2_FECollection(p, dim, BasisType::GaussLobatto);
   }

   return nullptr;
}

int CheckPoisson(Mesh &mesh, int order, int disabled_boundary_attribute)
{
   constexpr int dim = 3;

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction sol(&fes);

   ConstantCoefficient one(1.0);
   BilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();

   LinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // Add in essential boundary conditions
   Array<int> ess_tdof_list;
   REQUIRE(mesh.bdr_attributes.Max() > 0);

   // Mark all boundaries essential
   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 1;
   if (disabled_boundary_attribute >= 0)
   {
      bdr_attr_is_ess[mesh.bdr_attributes.Find(disabled_boundary_attribute)] = 0;
   }

   fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
   REQUIRE(ess_tdof_list.Size() > 0);

   sol = 0.0;
   Vector B, X;
   OperatorPtr A;
   a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B);

   // Solve the system
   CG(*A, B, X, 2, 1000, 1e-20, 0.0);

   // Recover the solution
   a.RecoverFEMSolution(X, b, sol);

   // Check that X solves the system A X = B.
   A->AddMult(X, B, -1.0);
   auto residual_norm = B.Norml2();
   bool satisfy_system = residual_norm < 1e-10;
   CAPTURE(residual_norm);
   CHECK(satisfy_system);

   bool satisfy_bc = true;
   Vector tvec;
   sol.GetTrueDofs(tvec);
   for (auto dof : ess_tdof_list)
   {
      if (tvec[dof] != 0.0)
      {
         satisfy_bc = false;
         break;
      }
   }
   CHECK(satisfy_bc);
   return ess_tdof_list.Size();
};

template <typename FECollection, bool TDOF>
int CountEssentialDof(Mesh &mesh, int order, int attribute)
{
   constexpr int dim = 3;
   FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 0;
   bdr_attr_is_ess[mesh.bdr_attributes.Find(attribute)] = 1;

   if (TDOF)
   {
      Array<int> ess_tdof_list;
      fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
      return ess_tdof_list.Size();
   }
   else
   {
      // VDOF
      Array<int> ess_vdof_marker, vdof_list;
      fes.GetEssentialVDofs(bdr_attr_is_ess, ess_vdof_marker);
      fes.MarkerToList(ess_vdof_marker, vdof_list);
      return vdof_list.Size();
   }
};

template int CountEssentialDof<H1_FECollection, false>(Mesh &, int, int);
template int CountEssentialDof<ND_FECollection, false>(Mesh &, int, int);
template int CountEssentialDof<RT_FECollection, false>(Mesh &, int, int);
template int CountEssentialDof<H1_FECollection, true>(Mesh &, int, int);
template int CountEssentialDof<ND_FECollection, true>(Mesh &, int, int);
template int CountEssentialDof<RT_FECollection, true>(Mesh &, int, int);

Mesh TetStarMesh()
{
   const int nnode = 4 + 4;
   const int nelem = 5;

   Mesh mesh(3, nnode, nelem);

   // central tet
   mesh.AddVertex(0.0, 0.0, 0.0);
   mesh.AddVertex(1.0, 0.0, 0.0);
   mesh.AddVertex(0.0, 1.0, 0.0);
   mesh.AddVertex(0.0, 0.0, 1.0);

   mesh.AddVertex( 1.0,  1.0,  1.0); // opposite 0
   mesh.AddVertex(-1.0,  0.0,  0.0); // opposite 1
   mesh.AddVertex( 0.0, -1.0,  0.0); // opposite 2
   mesh.AddVertex( 0.0,  0.0, -1.0); // opposite 3

   mesh.AddTet(0, 1, 2, 3, 1); // central
   mesh.AddTet(4, 1, 2, 3, 2); // opposite 0
   mesh.AddTet(0, 5, 2, 3, 3); // opposite 1
   mesh.AddTet(0, 1, 6, 3, 4); // opposite 2
   mesh.AddTet(0, 1, 2, 7, 5); // opposite 3

   mesh.FinalizeTopology();
   mesh.Finalize(true, true);

   // Introduce internal boundary elements
   const int new_attribute = mesh.bdr_attributes.Max() + 1;
   Array<int> original_boundary_vertices;
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      mesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && mesh.GetAttribute(e1) != mesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = mesh.GetFace(f)->Duplicate(&mesh);
         new_elem->SetAttribute(new_attribute);
         new_elem->GetVertices(original_boundary_vertices);
         mesh.AddBdrElement(new_elem);
      }
   }
   mesh.SetAttributes();
   mesh.FinalizeTopology();
   mesh.Finalize(true, true);

   return mesh;
}

Mesh DividingPlaneMesh(bool tet_mesh, bool split, bool three_dim)
{
   auto mesh = three_dim ? Mesh("../../data/ref-cube.mesh") :
               Mesh("../../data/ref-square.mesh");
   {
      Array<Refinement> refs;
      refs.Append(Refinement(0, Refinement::X));
      mesh.GeneralRefinement(refs);
   }
   delete mesh.ncmesh;
   mesh.ncmesh = nullptr;
   mesh.FinalizeTopology();
   mesh.Finalize(true, true);

   mesh.SetAttribute(0, 1);
   mesh.SetAttribute(1, split ? 2 : 1);

   // Introduce internal boundary elements
   const int new_attribute = mesh.bdr_attributes.Max() + 1;
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      int e1, e2;
      mesh.GetFaceElements(f, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && mesh.GetAttribute(e1) != mesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = mesh.GetFace(f)->Duplicate(&mesh);
         new_elem->SetAttribute(new_attribute);
         mesh.AddBdrElement(new_elem);
      }
   }
   if (tet_mesh)
   {
      mesh = Mesh::MakeSimplicial(mesh);
   }
   mesh.FinalizeTopology();
   mesh.Finalize(true, true);
   return mesh;
}

Mesh OrientedTriFaceMesh(int orientation, bool add_extbdr)
{
   REQUIRE((orientation == 1 || orientation == 3 || orientation == 5));

   Mesh mesh(3, 5, 2);
   mesh.AddVertex(-1.0, 0.0, 0.0);
   mesh.AddVertex(0.0, 0.0, 0.0);
   mesh.AddVertex(0.0, 1.0, 0.0);
   mesh.AddVertex(0.0, 0.0, 1.0);

   // opposing vertex
   mesh.AddVertex(1.0, 0.0, 0.0);

   mesh.AddTet(0, 1, 2, 3, 1);

   switch (orientation)
   {
      case 1:
         mesh.AddTet(4,2,1,3,2); break;
      case 3:
         mesh.AddTet(4,3,2,1,2); break;
      case 5:
         mesh.AddTet(4,1,3,2,2); break;
   }

   mesh.FinalizeTopology(add_extbdr);
   mesh.SetAttributes();

   auto *bdr = new Triangle(1,2,3,
                            mesh.bdr_attributes.Size() == 0 ? 1 : mesh.bdr_attributes.Max() + 1);
   mesh.AddBdrElement(bdr);

   mesh.FinalizeTopology(false);
   mesh.Finalize();
   return mesh;
}

Mesh CylinderMesh(Geometry::Type el_type, bool quadratic, int variant)
{
   real_t c[3];

   const int nnodes = (el_type == Geometry::CUBE) ? 24 : 15;
   const int nelems = [&]()
   {
      switch (el_type)
      {
         case Geometry::CUBE:
            return 10;
         case Geometry::TETRAHEDRON:
            return 24;
         case Geometry::PRISM:
            return 8;
         default:
            MFEM_ABORT("Invalid choice of geometry");
            return -1;
      }
   }();

   Mesh mesh(3, nnodes, nelems);

   for (int i = 0; i < 3; i++)
   {
      if (el_type != Geometry::CUBE)
      {
         c[0] = 0.0;  c[1] = 0.0;  c[2] = 2.74 * i;
         mesh.AddVertex(c);
      }

      for (int j = 0; j < 4; j++)
      {
         if (el_type == Geometry::CUBE)
         {
            c[0] = 1.14 * ((j + 1) % 2) * (1 - j);
            c[1] = 1.14 * (j % 2) * (2 - j);
            c[2] = 2.74 * i;
            mesh.AddVertex(c);
         }

         c[0] = 2.74 * ((j + 1) % 2) * (1 - j);
         c[1] = 2.74 * (j % 2) * (2 - j);
         c[2] = 2.74 * i;
         mesh.AddVertex(c);
      }
   }

   for (int i = 0; i < 2; i++)
   {
      if (el_type == Geometry::CUBE)
      {
         mesh.AddHex(8*i, 8*i+2, 8*i+4, 8*i+6,
                     8*(i+1), 8*(i+1)+2, 8*(i+1)+4, 8*(i+1)+6);
      }

      for (int j = 0; j < 4; j++)
      {
         if (el_type == Geometry::PRISM)
         {
            switch (variant)
            {
               case 0:
                  mesh.AddWedge(5*i, 5*i+j+1, 5*i+(j+1)%4+1,
                                5*(i+1), 5*(i+1)+j+1, 5*(i+1)+(j+1)%4+1);
                  break;
               case 1:
                  mesh.AddWedge(5*i, 5*i+j+1, 5*i+(j+1)%4+1,
                                5*(i+1), 5*(i+1)+j+1, 5*(i+1)+(j+1)%4+1);
                  break;
               case 2:
                  mesh.AddWedge(5*i+(j+1)%4+1, 5*i, 5*i+j+1,
                                5*(i+1)+(j+1)%4+1, 5*(i+1), 5*(i+1)+j+1);
                  break;
            }
         }
         else if (el_type == Geometry::CUBE)
         {
            mesh.AddHex(8*i+2*j, 8*i+2*j+1, 8*i+(2*j+3)%8, 8*i+(2*j+2)%8,
                        8*(i+1)+2*j, 8*(i+1)+2*j+1, 8*(i+1)+(2*j+3)%8,
                        8*(i+1)+(2*j+2)%8);
         }
         else if (el_type == Geometry::TETRAHEDRON)
         {
            mesh.AddTet(5*i, 5*i+j+1, 5*i+(j+1)%4+1, 5*(i+1));
            mesh.AddTet(5*i+j+1, 5*i+(j+1)%4+1, 5*(i+1), 5*(i+1)+j+1);
            mesh.AddTet(5*i+(j+1)%4+1, 5*(i+1), 5*(i+1)+j+1, 5*(i+1)+(j+1)%4+1);
         }
      }
   }

   mesh.FinalizeTopology();

   if (quadratic)
   {
      mesh.SetCurvature(2);

      if (el_type == Geometry::CUBE)
      {
         auto quad_cyl_hex = [](const Vector& x, Vector& d)
         {
            d.SetSize(3);
            d = x;
            const real_t Rmax = 2.74;
            const real_t Rmin = 1.14;
            real_t ax = std::abs(x[0]);
            if (ax <= 1e-6) { return; }
            real_t ay = std::abs(x[1]);
            if (ay <= 1e-6) { return; }
            real_t r = ax + ay;
            if (r <= Rmin + 1e-6) { return; }

            real_t sx = std::copysign(1.0, x[0]);
            real_t sy = std::copysign(1.0, x[1]);

            real_t R = (Rmax - Rmin) * Rmax / (r - Rmin);
            real_t r2 = r * r;
            real_t R2 = R * R;

            real_t acosarg = 0.5 * (r + std::sqrt(2.0 * R2 - r2)) / R;
            real_t tR = std::acos(std::min(acosarg, (real_t) 1.0));
            real_t tQ = (1.0 + sx * sy * (ay - ax) / r);
            real_t tP = 0.25 * M_PI * (3.0 - (2.0 + sx) * sy);

            real_t t = tR + (0.25 * M_PI - tR) * tQ + tP;

            real_t s0 = std::sqrt(2.0 * R2 - r2);
            real_t s1 = 0.25 * std::pow(r + s0, 2);
            real_t s = std::sqrt(R2 - s1);

            d[0] = R * std::cos(t) - sx * s;
            d[1] = R * std::sin(t) - sy * s;

            return;
         };
         mesh.Transform(quad_cyl_hex);
      }
      else
      {
         auto quad_cyl = [](const Vector& x, Vector& d)
         {
            d.SetSize(3);
            d = x;
            real_t ax = std::abs(x[0]);
            real_t ay = std::abs(x[1]);
            real_t r = ax + ay;
            if (r < 1e-6) { return; }

            real_t sx = std::copysign(1.0, x[0]);
            real_t sy = std::copysign(1.0, x[1]);

            real_t t = ((2.0 - (1.0 + sx) * sy) * ax +
                        (2.0 - sy) * ay) * 0.5 * M_PI / r;
            d[0] = r * std::cos(t);
            d[1] = r * std::sin(t);

            return;
         };
         mesh.Transform(quad_cyl);
      }
   }
   mesh.Finalize(true);
   return mesh;
}

void RefineSingleAttachedElement(Mesh &mesh, int vattr, int battr,
                                 bool backwards)
{
   Array<Refinement> refs(1);
   std::vector<int> ind(mesh.GetNBE());
   if (backwards)
   {
      std::iota(ind.rbegin(), ind.rend(), 0);
   }
   else
   {
      std::iota(ind.begin(), ind.end(), 0);
   }
   for (int e : ind)
   {
      if (mesh.GetBdrAttribute(e) == battr)
      {
         int f, o, el1, el2;
         mesh.GetBdrElementFace(e, &f, &o);
         mesh.GetFaceElements(f, &el1, &el2);
         if (mesh.GetAttribute(el1) == vattr)
         { mesh.GeneralRefinement(Array<int> {el1}); return; }
         if (mesh.GetAttribute(el2) == vattr)
         { mesh.GeneralRefinement(Array<int> {el2}); return; }
      }
   }
}

void RefineSingleUnattachedElement(Mesh &mesh, int vattr, int battr,
                                   bool backwards)
{
   std::set<int> attached_elements;
   for (int e = 0; e < mesh.GetNBE(); e++)
   {
      if (mesh.GetBdrAttribute(e) == battr)
      {
         int f, o, el1, el2;
         mesh.GetBdrElementFace(e, &f, &o);
         mesh.GetFaceElements(f, &el1, &el2);
         if (mesh.GetAttribute(el1) == vattr) { attached_elements.insert(el1); }
         if (el2 >= 0 && mesh.GetAttribute(el2) == vattr) { attached_elements.insert(el2); }
      }
   }
   if (backwards)
   {
      for (int i = mesh.GetNE() - 1; i >= 0; i--)
         if (mesh.GetAttribute(i) == vattr && attached_elements.count(i) == 0)
         {
            mesh.GeneralRefinement(Array<int> {i});
            return;
         }
   }
   else
   {
      for (int i = 0; i < mesh.GetNE(); i++)
         if (mesh.GetAttribute(i) == vattr && attached_elements.count(i) == 0)
         {
            mesh.GeneralRefinement(Array<int> {i});
            return;
         }
   }
}

#ifdef MFEM_USE_MPI

void TestVectorValueInVolume(Mesh &smesh, int nc_level, int skip, bool use_ND)
{
   auto vector_exact_soln = [](const Vector& x, Vector& v)
   {
      Vector d(3);
      d[0] = -0.5; d[1] = -1; d[2] = -2; // arbitrary
      v = (d -= x);
   };

   smesh.Finalize();
   smesh.EnsureNCMesh(true);

   auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);

   // Apply refinement on face neighbors to achieve a given nc level mismatch.
   for (int i = 0; i < nc_level; ++i)
   {
      // To refine the face neighbors, need to know where they are.
      pmesh.ExchangeFaceNbrData();
      Array<int> elem_to_refine;
      // Refine only on odd ranks.
      if ((Mpi::WorldRank() + 1) % 2 == 0)
      {
         // Refine a subset of all shared faces. Using a subset helps to mix in
         // conformal faces with nonconforming faces.
         for (int n = 0; n < pmesh.GetNSharedFaces(); ++n)
         {
            if (n % skip != 0) { continue; }
            const int local_face = pmesh.GetSharedFace(n);
            const auto &face_info = pmesh.GetFaceInformation(local_face);
            REQUIRE(face_info.IsShared());
            REQUIRE(face_info.element[1].location == Mesh::ElementLocation::FaceNbr);
            elem_to_refine.Append(face_info.element[0].index);
         }
      }
      pmesh.GeneralRefinement(elem_to_refine);
   }

   // Do not rebalance again! The test is also checking for nc refinements along
   // the processor boundary.

   // Create a grid function of the mesh coordinates
   pmesh.EnsureNodes();
   pmesh.ExchangeFaceNbrData();
   GridFunction * const coords = pmesh.GetNodes();

   // Project the linear function onto the mesh. Quadratic ND tetrahedral
   // elements are the first to require face orientations.
   const int order = 2, dim = 3;
   std::unique_ptr<FiniteElementCollection> fec;
   if (use_ND)
   {
      fec = std::unique_ptr<ND_FECollection>(new ND_FECollection(order, dim));
   }
   else
   {
      fec = std::unique_ptr<RT_FECollection>(new RT_FECollection(order, dim));
   }
   ParFiniteElementSpace pnd_fes(&pmesh, fec.get());

   ParGridFunction psol(&pnd_fes);

   VectorFunctionCoefficient func(3, vector_exact_soln);
   psol.ProjectCoefficient(func);
   psol.ExchangeFaceNbrData();

   mfem::Vector value(3), exact(3), position(3);
   const IntegrationRule &ir = mfem::IntRules.Get(Geometry::Type::TETRAHEDRON,
                                                  order + 1);

   // Check that non-ghost elements match up on the serial and parallel spaces.
   bool valid = true;
   for (int n = 0; n < pmesh.GetNE(); ++n)
   {
      constexpr real_t tol = 1e-12;
      for (const auto &ip : ir)
      {
         coords->GetVectorValue(n, ip, position);
         psol.GetVectorValue(n, ip, value);

         vector_exact_soln(position, exact);
         valid &= ((value -= exact).Normlinf() < tol);
      }
   }
   CHECK(valid);

   // Loop over face neighbor elements and check the vector values match in the
   // face neighbor elements.
   valid = true;
   for (int n = 0; n < pmesh.GetNSharedFaces(); ++n)
   {
      const int local_face = pmesh.GetSharedFace(n);
      const auto &face_info = pmesh.GetFaceInformation(local_face);
      REQUIRE(face_info.IsShared());
      REQUIRE(face_info.element[1].location == Mesh::ElementLocation::FaceNbr);

      auto &T = *pmesh.GetFaceNbrElementTransformation(face_info.element[1].index);

      constexpr real_t tol = 1e-12;
      for (const auto &ip : ir)
      {
         T.SetIntPoint(&ip);
         coords->GetVectorValue(T, ip, position);
         psol.GetVectorValue(T, ip, value);

         vector_exact_soln(position, exact);
         valid &= ((value -= exact).Normlinf() < tol);
      }
   }
   CHECK(valid);
}

void CheckPoisson(ParMesh &pmesh, int order,
                  int disabled_boundary_attribute)
{
   constexpr int dim = 3;

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec);

   ParGridFunction sol(&pfes);

   ConstantCoefficient one(1.0);
   ParBilinearForm a(&pfes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();
   ParLinearForm b(&pfes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // Add in essential boundary conditions
   Array<int> ess_tdof_list;
   REQUIRE(pmesh.bdr_attributes.Max() > 0);

   Array<int> bdr_attr_is_ess(pmesh.bdr_attributes.Max());
   bdr_attr_is_ess = 1;
   if (disabled_boundary_attribute >= 0)
   {
      CAPTURE(disabled_boundary_attribute);
      bdr_attr_is_ess[pmesh.bdr_attributes.Find(disabled_boundary_attribute)] = 0;
   }

   pfes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
   int num_ess_dof = ess_tdof_list.Size();
   MPI_Allreduce(MPI_IN_PLACE, &num_ess_dof, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   REQUIRE(num_ess_dof > 0);

   sol = 0.0;
   Vector B, X;
   OperatorPtr A;
   const bool copy_interior = true; // interior(sol) --> interior(X)
   a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B, copy_interior);

   // Solve the system
   CGSolver cg(MPI_COMM_WORLD);
   HypreBoomerAMG preconditioner;
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   preconditioner.SetPrintLevel(-1);
   cg.SetPrintLevel(-1);
   cg.SetPreconditioner(preconditioner);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   // Recover the solution
   a.RecoverFEMSolution(X, b, sol);

   // Check that X solves the system A X = B.
   A->AddMult(X, B, -1.0);
   auto residual_norm = B.Norml2();
   bool satisfy_system = residual_norm < 1e-10;
   CAPTURE(residual_norm);
   CHECK(satisfy_system);

   Vector tvec;
   sol.GetTrueDofs(tvec);
   bool satisfy_bc = true;
   for (auto dof : ess_tdof_list)
   {
      if (tvec[dof] != 0.0)
      {
         satisfy_bc = false;
         break;
      }
   }
   CHECK(satisfy_bc);
};

std::unique_ptr<ParMesh> CheckParMeshNBE(Mesh &smesh,
                                         const std::unique_ptr<int[]> &partition)
{
   auto pmesh = std::unique_ptr<ParMesh>(new ParMesh(MPI_COMM_WORLD, smesh,
                                                     partition.get()));

   int nbe = pmesh->GetNBE();
   MPI_Allreduce(MPI_IN_PLACE, &nbe, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   CHECK(nbe == smesh.GetNBE());
   return pmesh;
};

bool CheckFaceInternal(ParMesh& pmesh, int f,
                       const std::map<int, int> &local_to_shared)
{
   int e1, e2;
   pmesh.GetFaceElements(f, &e1, &e2);
   int inf1, inf2, ncface;
   pmesh.GetFaceInfos(f, &inf1, &inf2, &ncface);

   if (e2 < 0 && inf2 >=0)
   {
      // Shared face on processor boundary -> Need to discover the neighbor
      // attributes
      auto FET = pmesh.GetSharedFaceTransformations(local_to_shared.at(f));

      if (FET->Elem1->Attribute != FET->Elem2->Attribute && f < pmesh.GetNumFaces())
      {
         // shared face on domain attribute boundary, which this rank owns
         return true;
      }
   }

   if (e2 >= 0 && pmesh.GetAttribute(e1) != pmesh.GetAttribute(e2))
   {
      // local face on domain attribute boundary
      return true;
   }
   return false;
};

std::array<real_t, 2> CheckL2Projection(ParMesh& pmesh, Mesh& smesh, int order,
                                        std::function<real_t(Vector const&)> exact_soln)
{
   REQUIRE(pmesh.GetGlobalNE() == smesh.GetNE());
   REQUIRE(pmesh.Dimension() == smesh.Dimension());
   REQUIRE(pmesh.SpaceDimension() == smesh.SpaceDimension());

   // Make an H1 space, then a mass matrix operator and invert it. If all
   // non-conformal constraints have been conveyed correctly, the resulting DOF
   // should match exactly on the serial and the parallel solution.

   H1_FECollection fec(order, smesh.Dimension());
   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs_coef(exact_soln);

   constexpr real_t linear_tol = 1e-16;

   // serial solve
   auto serror = [&]
   {
      FiniteElementSpace fes(&smesh, &fec);
      // solution vectors
      GridFunction x(&fes);
      x = 0.0;

      real_t snorm = x.ComputeL2Error(rhs_coef);

      LinearForm b(&fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
      b.Assemble();

      BilinearForm a(&fes);
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.Assemble();

      SparseMatrix A;
      Vector B, X;

      Array<int> empty_tdof_list;
      a.FormLinearSystem(empty_tdof_list, x, b, A, X, B);

#ifndef MFEM_USE_SUITESPARSE
      // 9. Define a simple symmetric Gauss-Seidel preconditioner and use it to
      //    solve the system AX=B with PCG.
      GSSmoother M(A);
      PCG(A, M, B, X, -1, 500, linear_tol, 0.0);
#else
      // 9. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the
      //    system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(A);
      umf_solver.Mult(B, X);
#endif

      a.RecoverFEMSolution(X, b, x);
      return x.ComputeL2Error(rhs_coef) / snorm;
   }();

   auto perror = [&]
   {
      // parallel solve
      ParFiniteElementSpace fes(&pmesh, &fec);
      ParLinearForm b(&fes);

      ParGridFunction x(&fes);
      x = 0.0;

      real_t pnorm = x.ComputeL2Error(rhs_coef);
      b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
      b.Assemble();

      ParBilinearForm a(&fes);
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.Assemble();

      HypreParMatrix A;
      Vector B, X;
      Array<int> empty_tdof_list;
      a.FormLinearSystem(empty_tdof_list, x, b, A, X, B);

      HypreBoomerAMG amg(A);
      HyprePCG pcg(A);
      amg.SetPrintLevel(-1);
      pcg.SetTol(linear_tol);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(-1);
      pcg.SetPreconditioner(amg);
      pcg.Mult(B, X);
      a.RecoverFEMSolution(X, b, x);
      return x.ComputeL2Error(rhs_coef) / pnorm;
   }();

   return {serror, perror};
}

template <typename FECollection, bool TDOF>
int CountEssentialDof(ParMesh &mesh, int order, int attribute)
{
   constexpr int dim = 3;
   FECollection fec(order, dim);
   ParFiniteElementSpace pfes(&mesh, &fec);

   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 0;
   bdr_attr_is_ess[mesh.bdr_attributes.Find(attribute)] = 1;

   Array<int> ess_tdof_list;
   pfes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
   if (TDOF)
   {
      pfes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
      return ess_tdof_list.Size();
   }
   else
   {
      // VDOF
      Array<int> ess_vdof_marker, vdof_list;
      pfes.GetEssentialVDofs(bdr_attr_is_ess, ess_vdof_marker);
      pfes.MarkerToList(ess_vdof_marker, vdof_list);
      return vdof_list.Size();
   }
};

template int CountEssentialDof<H1_FECollection, false>(ParMesh &, int, int);
template int CountEssentialDof<ND_FECollection, false>(ParMesh &, int, int);
template int CountEssentialDof<RT_FECollection, false>(ParMesh &, int, int);
template int CountEssentialDof<H1_FECollection, true>(ParMesh &, int, int);
template int CountEssentialDof<ND_FECollection, true>(ParMesh &, int, int);
template int CountEssentialDof<RT_FECollection, true>(ParMesh &, int, int);

template <typename FECollection, bool TDOF>
int ParCountEssentialDof(ParMesh &mesh, int order, int attribute)
{
   auto num_essential_dof = CountEssentialDof<FECollection, TDOF>(mesh, order,
                                                                  attribute);
   MPI_Allreduce(MPI_IN_PLACE, &num_essential_dof, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   return num_essential_dof;
};

template int ParCountEssentialDof<H1_FECollection, false>(ParMesh &, int, int);
template int ParCountEssentialDof<ND_FECollection, false>(ParMesh &, int, int);
template int ParCountEssentialDof<RT_FECollection, false>(ParMesh &, int, int);
template int ParCountEssentialDof<H1_FECollection, true>(ParMesh &, int, int);
template int ParCountEssentialDof<ND_FECollection, true>(ParMesh &, int, int);
template int ParCountEssentialDof<RT_FECollection, true>(ParMesh &, int, int);

bool CheckRPIdentity(const ParFiniteElementSpace& pfespace)
{
   const SparseMatrix *R = pfespace.GetRestrictionMatrix();
   HypreParMatrix *P = pfespace.Dof_TrueDof_Matrix();

   REQUIRE(R != nullptr);
   REQUIRE(P != nullptr);

   HypreParMatrix *hR = new HypreParMatrix(
      pfespace.GetComm(), pfespace.GlobalTrueVSize(),
      pfespace.GlobalVSize(), pfespace.GetTrueDofOffsets(),
      pfespace.GetDofOffsets(),
      const_cast<SparseMatrix*>(R)); // Non owning so cast is ok

   REQUIRE(hR->Height() == P->Width());
   REQUIRE(hR->Width() == P->Height());

   REQUIRE(hR != nullptr);
   HypreParMatrix *I = ParMult(hR, P);

   // Square matrix so the "diag" is the only bit we need.
   SparseMatrix diag;
   I->GetDiag(diag);
   bool valid = true;
   for (int i = 0; i < diag.Height(); i++)
      for (int j = 0; j < diag.Width(); j++)
      {
         // cast to const to force a zero return rather than an abort.
         valid &= const_cast<const SparseMatrix&>(diag)(i, j)  == (i == j ? 1.0 : 0.0);
      }

   delete hR;
   delete I;
   return valid;
}

#endif

} // namespace mfem
