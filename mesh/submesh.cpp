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

#include <unordered_map>
#include "submesh.hpp"
#include "../fem/gridfunc.hpp"

using namespace mfem;

struct UniqueIndexGenerator
{
  int counter = 0;
  std::unordered_map<int,int> idx;

  int Get(int i, bool &new_index)
  {
    auto f = idx.find(i);
    if (f == idx.end())
    {
      idx[i] = counter;
      new_index = true;
      return counter++;
    }
    else
    {
      new_index = false;
      return (*f).second;
    }
  }
};

bool is_submesh(const Mesh *m)
{
  return dynamic_cast<const SubMesh*>(m) != nullptr;
}

bool element_in_subdomain(Element &el, Array<int>& attributes)
{
  for (int a = 0; a < attributes.Size(); a++)
  {
    if (el.GetAttribute() == attributes[a])
    {
      return true;
    }
  }
  return false;
}

SubMesh::SubMesh(Mesh &parent, From from,
    Array<int> attributes) : parent_(parent), from_(from), attributes_(attributes)
{
  if (from == From::Domain)
  {
    InitMesh(parent.Dimension(), parent.SpaceDimension(), 0, 0, 0);

    UniqueIndexGenerator vertex_ids;
    for (int i = 0; i < parent.GetNE(); i++)
    {
      Element *pel = parent.GetElement(i);

      if (element_in_subdomain(*pel, attributes))
      {
	Array<int> v;
	pel->GetVertices(v);
	Array<int> submesh_v(v.Size());

	for (int iv = 0; iv < v.Size(); iv++)
	{
	  bool new_vertex;
	  int mesh_vertex_id = v[iv];
	  int submesh_vertex_id = vertex_ids.Get(mesh_vertex_id, new_vertex);
	  if (new_vertex)
	  {
	    AddVertex(parent.GetVertex(mesh_vertex_id));
	    parent_vertex_ids_.Append(iv);
	  }
	  submesh_v[iv] = submesh_vertex_id;
	}

	Element *el = NewElement(parent.GetElementType(i));
	el->SetVertices(submesh_v);
	AddElement(el);
	parent_element_ids_.Append(i);
      }
    }
  }
  else if (from == From::Boundary)
  {
    InitMesh(parent.Dimension() - 1, parent.SpaceDimension(), 0, 0, 0);

    UniqueIndexGenerator vertex_ids;
    for (int i = 0; i < parent.GetNBE(); i++)
    {
      Element *pel = parent.GetBdrElement(i);

      if (element_in_subdomain(*pel, attributes))
      {   
	Array<int> v;
	pel->GetVertices(v);
	Array<int> submesh_v(v.Size());

	for (int iv = 0; iv < v.Size(); iv++)
	{
	  bool new_vertex;
	  int mesh_vertex_id = v[iv];
	  int submesh_vertex_id = vertex_ids.Get(mesh_vertex_id, new_vertex);
	  if (new_vertex)
	  {
	    AddVertex(parent.GetVertex(mesh_vertex_id));
	    parent_vertex_ids_.Append(iv);
	  }
	  submesh_v[iv] = submesh_vertex_id;
	}

	Element *el = NewElement(parent.GetBdrElementType(i));
	el->SetVertices(submesh_v);
	AddElement(el);
	parent_element_ids_.Append(i);
      }
    }
  }

  // Finalize topology and generate boundary elements.
  FinalizeTopology(true);

  // If the parent Mesh has nodes and therefore is defined on a higher order
  // geometry, we define this SubMesh as a curved Mesh and transfer the
  // GridFunction from the parent Mesh to the SubMesh.
  GridFunction *parent_nodes = parent.GetNodes();
  if (parent_nodes)
  {
    const FiniteElementSpace *parent_fes = parent_nodes->FESpace();
  
    SetCurvature(
	parent_fes->FEColl()->GetOrder(),
	parent_fes->IsDGSpace(),
	spaceDim,
	parent_fes->GetOrdering());

    Transfer(*parent.GetNodes(), *GetNodes());
  }

  Finalize();
}

SubMesh::~SubMesh() {}

SubMesh SubMesh::CreateFromDomain(Mesh &parent,
    Array<int> domain_attributes)
{
  return SubMesh(parent, From::Domain, domain_attributes);
}


SubMesh SubMesh::CreateFromBoundary(Mesh &parent,
    Array<int> boundary_attributes)
{
  return SubMesh(parent, From::Boundary, boundary_attributes);
}

void SubMesh::Transfer(GridFunction &src, GridFunction &dst)
{
  Array<int> src_vdofs;
  Array<int> dst_vdofs;
  Vector vec;

  // Determine which GridFunction is defined on the SubMesh
  if (is_submesh(src.FESpace()->GetMesh()))
  {
    // SubMesh to Mesh transfer
    SubMesh *src_mesh = static_cast<SubMesh *>(src.FESpace()->GetMesh());
    Mesh *dst_mesh = dst.FESpace()->GetMesh();

    MFEM_ASSERT(src_mesh->GetParent() == dst.FESpace()->GetMesh(),
	"The Meshes of the specified GridFunction are not related in a SubMesh -> Mesh relationship.");

    auto &parent_element_ids = src_mesh->GetParentElementIDMap();

    IntegrationPointTransformation Tr;
    DenseMatrix vals, vals_transpose;
    for (int i = 0; i < src_mesh->GetNE(); i++)
    {
      src.FESpace()->GetElementVDofs(i, src_vdofs);
      if (src.FESpace()->IsDGSpace() && src_mesh->GetFrom() == From::Boundary)
      {
	MFEM_ABORT("Transferring from a surface SubMesh to a volume Mesh using L2 spaces is not implemented.");
      }
      else
      {
	if (src_mesh->GetFrom() == SubMesh::From::Domain) {
	  dst.FESpace()->GetElementVDofs(parent_element_ids[i], dst_vdofs);
	}
	else if (src_mesh->GetFrom() == SubMesh::From::Boundary) {
	  dst.FESpace()->GetBdrElementVDofs(parent_element_ids[i], dst_vdofs);
	}
	src.GetSubVector(src_vdofs, vec);
	dst.SetSubVector(dst_vdofs, vec);
      }
    }
  }
  else if (is_submesh(dst.FESpace()->GetMesh()))
  {
    // Mesh to SubMesh transfer
    Mesh *src_mesh = src.FESpace()->GetMesh();
    SubMesh *dst_mesh = static_cast<SubMesh *>(dst.FESpace()->GetMesh());

    MFEM_ASSERT(dst_mesh->GetParent() == src_mesh,
	"The Meshes of the specified GridFunction are not related in a Mesh -> SubMesh relationship.");

    auto &parent_element_ids = dst_mesh->GetParentElementIDMap();

    IntegrationPointTransformation Tr;
    DenseMatrix vals, vals_transpose;
    for (int i = 0; i < dst_mesh->GetNE(); i++)
    {
      dst.FESpace()->GetElementVDofs(i, dst_vdofs);
      if (src.FESpace()->IsDGSpace() && dst_mesh->GetFrom() == From::Boundary)
      {
	const FiniteElement *el = dst.FESpace()->GetFE(i);
	MFEM_VERIFY(dynamic_cast<const NodalFiniteElement*>(el),
	    "Destination FESpace must use nodal Finite Elements.");

	int face_info, parent_volel_id;
	src_mesh->GetBdrElementAdjacentElement(parent_element_ids[i], parent_volel_id, face_info);
	src_mesh->GetLocalFaceTransformation(
	    src_mesh->GetBdrElementType(parent_element_ids[i]),
	    src_mesh->GetElementType(parent_volel_id),
	    Tr.Transf,
	    face_info);

	IntegrationRule src_el_ir(el->GetDof());
	Tr.Transf.ElementNo = parent_volel_id;
	Tr.Transf.ElementType = ElementTransformation::ELEMENT;
	Tr.Transform(el->GetNodes(), src_el_ir);
  
	src.GetVectorValues(Tr.Transf, src_el_ir, vals);
	// vals_transpose = vals^T
	vals_transpose.Transpose(vals);
	dst.SetSubVector(dst_vdofs, vals_transpose.GetData());
      }
      else
      {
	if (dst_mesh->GetFrom() == SubMesh::From::Domain) {
	  src.FESpace()->GetElementVDofs(parent_element_ids[i], src_vdofs);
	}
	else if (dst_mesh->GetFrom() == SubMesh::From::Boundary) {
	  src.FESpace()->GetBdrElementVDofs(parent_element_ids[i], src_vdofs);
	}
	src.GetSubVector(src_vdofs, vec);
	dst.SetSubVector(dst_vdofs, vec);
      }
    }
  }
  else
  {
    MFEM_ABORT("Trying to do a transfer between GridFunctions but none of them is defined on a SubMesh");
  }
}
