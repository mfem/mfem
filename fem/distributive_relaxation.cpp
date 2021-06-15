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

// Implementation of data types for distributive relaxation smoother

#include "../linalg/vector.hpp"
#include "distributive_relaxation.hpp"
#include "fem.hpp"
#include <iostream>
#include <unordered_map>
#include <cmath>

#define DRSMOOTHER_3D_EDGES

namespace mfem
{

  DRSmoother::DRSmoother(DisjointSets *clustering, const SparseMatrix *a, bool use_composite, double sc, bool use_l1, const Operator *op)
  {
    l1 = use_l1;
    scale = sc;
    composite = use_composite;

    G = NULL;
    A = a;

    if (op != NULL) { SetOperator((Operator&) *op);}
    else { SetOperator((Operator&) *A);}

    FormG(clustering);
    tmp.SetSize(G->Width());
  }

  void DRSmoother::SetOperator(const Operator &op)
  {
    oper = &op;
    width = this->oper->Width();
    height = this->oper->Height();
  }

  DRSmoother::~DRSmoother()
  {
    if (G != NULL) delete G;
  }

  /// Matrix vector multiplication with distributive relaxation smoother.
  void DRSmoother::Mult(const Vector &b, Vector &x) const
  {
    const Vector *rhs = &b;
    if (iterative_mode) {
      MFEM_ASSERT(oper != NULL, "Must set operation before using DRSmoother");
      tmp2 = x;
      oper->Mult(x, tmp);
      add(b, -1.0, tmp, tmp);
      rhs = &tmp;
    }
    DRSmootherJacobi(*rhs, x);
    if (iterative_mode) add(tmp2, x, x);

    if (composite) {
      tmp2 = x;
      A->Jacobi2(b,tmp2,x);
    }
  }

  void DRSmoother::DRSmootherJacobi(const Vector &b, Vector &x) const
  {
    G->MultTranspose(b, tmp);

    const int end = diagonal_scaling.Size();

    for (int i = 0; i < end; ++i) tmp[i] /= diagonal_scaling[i];
    G->Mult(tmp, x);
    for (int i = 0; i < x.Size(); ++i) x[i] *= scale;
  }

  LORInfo::LORInfo(const Mesh &lor_mesh, Mesh &ho_mesh, int p)
  {
    // This function is based on the constructor Mesh(Mesh*, int, int)
    // for constructing a low-order refined mesh from a high-order mesh.
    // This method assumes that it is passed the high-order mesh used
    // to construct the LOR mesh

    const int ref_type = BasisType::GaussLobatto;

    dim = ho_mesh.Dimension();
    order = p;

    MFEM_VERIFY(order >= 1, "the order must be >= 1");
    MFEM_VERIFY(dim == 1 || dim == 2 || dim == 3,"only implemented for Segment, Quadrilateral and Hexahedron elements in 1D/2D/3D");
    MFEM_VERIFY(ho_mesh.GetNumGeometries(dim) <= 1,"meshes with mixed elements are not supported");

    // Construct a scalar H1 FE space and use its dofs as
    // the indices of the new, refined vertices.
    H1_FECollection rfec(order, dim, ref_type);
    FiniteElementSpace rfes(&ho_mesh, &rfec);

    // Add refined elements and set vertex coordinates
    Array<int> rdofs;
    H1_FECollection vertex_fec(1, dim);

    dofs = new Array<int>();

    for (int el = 0; el < ho_mesh.GetNE(); el++) {
      Geometry::Type geom = ho_mesh.GetElementBaseGeometry(el);
      RefinedGeometry &RG = *GlobGeometryRefiner.Refine(geom, order);

      rfes.GetElementDofs(el, rdofs);
      MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(),"Element degrees of freedom must have same size as refined geometry");

      const int lower = dofs->Size();
      dofs->SetSize(dofs->Size()+rdofs.Size());

      if (dim == 1) {
	(*dofs)[lower] = rdofs[0];
	(*dofs)[lower+order] = rdofs[1];
	for (int j = 1; j < order; ++j) (*dofs)[lower+j] = rdofs[j+1];
      } else if (dim == 2) {
	MFEM_ASSERT(rdofs.Size() == (order+1)*(order+1),"Wrong number of DOFs!");

	(*dofs)[lower] = rdofs[0];
	(*dofs)[lower+order] = rdofs[1];
	(*dofs)[dofs->Size()-1] = rdofs[2];
	(*dofs)[dofs->Size()-1-order] = rdofs[3];

	int idx = 4;
	for (int j = 1; j < order; ++j) {(*dofs)[lower+j] = rdofs[idx]; ++idx;}
	for (int j = 1; j < order; ++j) {(*dofs)[lower+order+j*(order+1)] = rdofs[idx]; ++idx;}
	for (int j = 1; j < order; ++j) {(*dofs)[dofs->Size()-1-j] = rdofs[idx]; ++idx;}
	for (int j = 1; j < order; ++j) {(*dofs)[dofs->Size()-1-order-j*(order+1)] = rdofs[idx];++idx;}
	for (int j = 1; j < order; ++j) {
	  for (int k = 1; k < order; ++k) {
	    (*dofs)[lower+k+j*(order+1)] = rdofs[idx]; ++idx;
	  }
	}
      } else if (dim == 3) {
	const int lx = 1;
	const int ly = order+1;
	const int lz = ly * (order+1);
	const int s = order+1;

	MFEM_ASSERT(rdofs.Size() == s*s*s,"Wrong number of DOFs!");

	// Vertices
	(*dofs)[lower] = rdofs[0];
	(*dofs)[lower+(s-1)*lx] = rdofs[1];
	(*dofs)[lower+(s-1)*lx+(s-1)*ly] = rdofs[2];
	(*dofs)[lower+(s-1)*ly] = rdofs[3];
	(*dofs)[lower+(s-1)*lz] = rdofs[4];
	(*dofs)[lower+(s-1)*lx+(s-1)*lz] = rdofs[5];
	(*dofs)[lower+(s-1)*lx+(s-1)*ly+(s-1)*lz] = rdofs[6];
	(*dofs)[lower+(s-1)*ly+(s-1)*lz] = rdofs[7];

	// Edges
	int idx = 8;
	for (int j = 0; j < 12; ++j) {
	  for (int k = 1; k < s-1; ++k) {
	    int iz,iy,ix;

	    if (j < 4) {
	      iz = 0;
	    } else if (j < 8) {
	      iz = s-1;
	    } else {
	      iz = k;
	    }

	    if (j == 0 || j == 4 || j == 8 || j == 9) {
	      iy = 0;
	    } else if (j == 2 || j == 6 || j == 10 || j == 11) {
	      iy = s-1;
	    } else {
	      iy = k;
	    }

	    if (j % 2 == 0 && j < 8) {
	      ix = k;
	    } else if (j == 1 || j == 5 || j == 9 || j == 10) {
	      ix = s-1;
	    } else {
	      ix = 0;
	    }

	    (*dofs)[lower+ix*lx+iy*ly+iz*lz] = rdofs[idx]; ++idx;
	  }
	}

	// Faces
	for (int i = 0; i < 6; ++i) {
	  for (int j = 1; j < s-1; ++j) {
	    for (int k = 1; k < s-1; ++k) {
	      int iz,iy,ix;

	      if (i == 0) {
		iz = 0;
	      } else if (i == 5) {
		iz = s-1;
	      } else {
		iz = j;
	      }

	      if (i == 0) {
		iy = s-1 - j;
	      } else if (i == 5) {
		iy = j;
	      } else if (i == 1) {
		iy = 0;
	      } else if (i == 3) {
		iy = s-1;
	      } else if (i == 2) {
		iy = k;
	      } else if (i == 4) {
		iy = s-1 - k;
	      } else {
		MFEM_ABORT("i must lie between 0 and 5");
	      }

	      if (i == 0 || i == 1 || i == 5) {
		ix = k;
	      } else if (i == 3) {
		ix = s-1 - k;
	      } else if (i == 2) {
		ix = s-1;
	      } else {
		ix = 0;
	      }

	      (*dofs)[lower+ix*lx+iy*ly+iz*lz] = rdofs[idx]; ++idx;
	    }
	  }
	}

	// Interior
	for (int iz = 1; iz < s-1; ++iz) {
	  for (int iy = 1; iy < s-1; ++iy) {
	    for (int ix = 1; ix < s-1; ++ix) {
	      (*dofs)[lower+ix*lx+iy*ly+iz*lz] = rdofs[idx]; ++idx;
	    }
	  }
	}
	MFEM_ASSERT(lower + idx == dofs->Size(), "Wrong number of elements added to array");
      } else {
	MFEM_ABORT("Only implemented for one, two, or three dimensions");
      }
    }
    num_dofs = lor_mesh.GetNV();
  }

  void DRSmoother::FormG(const DisjointSets *clustering)
  {
    const Array<int> &bounds = clustering->GetBounds();
    const Array<int> &elems  = clustering->GetElems();
    SparseMatrix *g = NULL;
    Array<double> *coeffs = NULL;
    bool form_G = l1;

    if (form_G) {
      g = new SparseMatrix(elems.Size());
    } else {
      coeffs = new Array<double>();
    }

    MFEM_ASSERT(A != NULL, "'A' matrix be defined");
    std::vector<DenseMatrix> *diag_blocks = DiagonalBlocks(A, clustering);

    for (int group = 0; group < bounds.Size()-1; ++group) {
      const int size = bounds[group+1] - bounds[group];

      if (size == 1) {
	const int i = elems[bounds[group]];
	if (form_G) {
	  g->Add(i, i, 1.0);
	}
      } else {
	// Get eigenvalues and eigenvectors of subblock
	DenseMatrix &submat = (*diag_blocks)[group];
	DenseMatrix eigenvectors(size, size);
	Vector eigenvalues(size);
	submat.Eigenvalues(eigenvalues, eigenvectors);

	Array<int> indices(size);
	elems.GetSubArray(bounds[group], size, indices);

	// Get the smallest eigenvector
	int min_idx = 0;
	double min_eigenvalue = eigenvalues.Min();
	for (; min_idx < eigenvalues.Size(); ++min_idx) {
	  if (eigenvalues[min_idx] == min_eigenvalue) break;
	}

	const int i = indices[min_idx];
	// Compute normalization for eigenvector
	double two_norm = 0.0;
	for (int l = 0; l < size; ++l) {
	  const double val = eigenvectors(l, min_idx);
	  two_norm += val*val;
	}
	two_norm = sqrt(two_norm);

	const double diag_val = eigenvectors(0, min_idx) / two_norm;
	if (form_G) {
	  g->Add(i, i, diag_val);
	} else {
	  coeffs->Append(diag_val);
	}

	for (int l = 1; l < size; ++l) {
	  const double val = eigenvectors(l, min_idx) / two_norm;
	  if (form_G) {
	    const int j = indices[l];
	    g->Add(j, i, val);
	    g->Add(j, j, 1.0);
	  } else {
	    coeffs->Append(val);
	  }
	}
      }
    }

    if (form_G) {
      g->Finalize();
      G = new DRSmootherG(g, clustering);
    } else {
      G = new DRSmootherG(clustering, coeffs);
    }

    SparseMatrix *GtAG = NULL;
    diagonal_scaling.SetSize(0);
    G->GtAG(GtAG, diagonal_scaling, *A, diag_blocks);

    if (l1 || diagonal_scaling.Size() == 01) {
      MFEM_VERIFY(GtAG != NULL, "Product GtAG not computed!");
      const int size = A->Width();

      diagonal_scaling.SetSize(size);
      diagonal_scaling = 0.0;

      const int *I = GtAG->GetI();
      const int *J = GtAG->GetJ();
      const double *data = GtAG->GetData();

      for (int i = 0; i < size; ++i) {
	const int end = I[i+1];
	for (int j = I[i]; j < end; ++j) {
	  if (J[j] == i) {
	    diagonal_scaling[i] += data[j];
	  } else if (l1) {
	    diagonal_scaling[i] += 0.5 * fabs(data[j]);
	  }
	}
	MFEM_ASSERT(diagonal_scaling[i] > 0.0,"Diagonal scaling should have positive entries");
      }
    }
    if (diag_blocks) delete diag_blocks;
    if (GtAG) delete GtAG;
  }

  std::vector<DenseMatrix> *DRSmoother::DiagonalBlocks(const SparseMatrix *A, const DisjointSets *clustering)
  {
    std::vector<DenseMatrix> *blocks = new std::vector<DenseMatrix>();

    const Array<int> &elems = clustering->GetElems();
    const Array<int> &bounds = clustering->GetBounds();

    MFEM_VERIFY(bounds.Size() > 0, "Bounds array must have been initialized");

    for (int group = 0; group < bounds.Size()-1; ++group) {
      int size = bounds[group+1] - bounds[group];
      blocks->emplace_back(size, size);
    }

    const int *I = A->GetI();
    const int *J = A->GetJ();
    const double *data = A->GetData();

    for (int i = 0; i < A->Width(); ++i) {
      // Find index of i in elems
      int idx_i;
      int group = clustering->Group(i);
      int size = bounds[group+1] - bounds[group];
      for (idx_i = 0; idx_i < size; ++idx_i) {
	if (elems[bounds[group] + idx_i] == i) break;
      }
      MFEM_VERIFY(idx_i < size, "Index " << i << " not found in elems");

      for (int k = I[i]; k < I[i+1]; ++k) {
	int j = J[k];
	if (clustering->Find(i) == clustering->Find(j)) {
	  // Find index of j in elems
	  int idx_j;
	  for (idx_j = 0; idx_j < size; ++idx_j) {
	    if (elems[bounds[group] + idx_j] == j) break;
	  }
	  MFEM_VERIFY(idx_j < size, "j not found in elems");

	  blocks->at(group)(idx_i,idx_j) = data[k];
	}
      }
    }
    return blocks;
  }

  void PrintClusteringStats(std::ostream &out, const DisjointSets *clustering)
  {
    Array<int> unique_group_sizes;
    Array<int> group_size_counts;
    const Array<int> &bounds = clustering->GetBounds();

    for (int i = 0; i < bounds.Size()-1; ++i) {
      const int size = bounds[i+1] - bounds[i];
      if (size == 1) continue;

      int j = 0;
      for (; j < unique_group_sizes.Size(); ++j) {
	if (size == unique_group_sizes[j]) {
	  group_size_counts[j]++;
	  break;
	}
      }

      if (j == unique_group_sizes.Size()) {
	unique_group_sizes.Append(size);
	group_size_counts.Append(1);
      }
    }

    if (unique_group_sizes.Size() == 0) return;

    out << "Vertex groupings: ";
    out << group_size_counts[0] << " of size " << unique_group_sizes[0];
    for (int i = 1; i < unique_group_sizes.Size(); ++i) {
      out << ", " << group_size_counts[i] << " of size " << unique_group_sizes[i];
    }
    out << std::endl;
  }

  const DRSmootherG *DRSmoother::GetG() const { return G;}

  DisjointSets *LORInfo::Cluster() const
  {
    const Array<int> &dof_arr = *dofs;

    int dofs_per_elem = 1;
    for (int d = 0; d < dim; ++d) dofs_per_elem *= order + 1;

    MFEM_VERIFY(dof_arr.Size() % dofs_per_elem == 0,"DOF array for order " << order << " and dimension " << dim << " should be a multiple of " << dofs_per_elem);
    MFEM_VERIFY(order > 1, "Order must be greater than 1");

    DisjointSets *clustering = new DisjointSets(num_dofs);

    if (dim == 1) {
      for (int base = 0; base < dof_arr.Size(); base += dofs_per_elem) {
	clustering->Union(dof_arr[base], dof_arr[base+1]);
	clustering->Union(dof_arr[base], dof_arr[base+2]);
	clustering->Union(dof_arr[base], dof_arr[base+3]);
	clustering->Union(dof_arr[base+dofs_per_elem-1], dof_arr[base+dofs_per_elem-2]);
	clustering->Union(dof_arr[base+dofs_per_elem-1], dof_arr[base+dofs_per_elem-3]);
	clustering->Union(dof_arr[base+dofs_per_elem-1], dof_arr[base+dofs_per_elem-4]);
      }
    } else if (dim == 2) {
      for (int base = 0; base < dof_arr.Size(); base += dofs_per_elem) {
	for (int d = 0; d < 2; ++d) {
	  int li, lj;
	  li = order+1;
	  lj = 1;

	  if (d == 1) std::swap(li, lj);

	  for (int i = 0; i <= order-1; i += order-1) {
	    for (int j = 2; j <= order-2; ++j) {
	      const int v1 = dof_arr[base + i * li + j * lj];
	      const int v2 = dof_arr[base + (i+1) * li + j * lj];

	      clustering->Union(v1, v2);
	    }
	  }
	}
      }
    } else if (dim == 3) {
      const int order_plus_1_sq = (order+1)*(order+1);

      for (int base = 0; base < dof_arr.Size(); base += dofs_per_elem) {
	for (int d = 0; d < 6; ++d) {
	  int li, lj, lk;

	  switch (d) {
	  case 0:
	    li = 1; lj = order+1; lk = order_plus_1_sq; break;
	  case 1:
	    lj = 1; li = order+1; lk = order_plus_1_sq; break;
	  case 2:
	    lk = 1; lj = order+1; li = order_plus_1_sq; break;
	  case 3:
	    li = 1; lk = order+1; lj = order_plus_1_sq; break;
	  case 4:
	    lj = 1; lk = order+1; li = order_plus_1_sq; break;
	  case 5:
	    lk = 1; li = order+1; lj = order_plus_1_sq; break;
	  }

	  for (int i = 0; i <= order-1; i += order-1) {
	    for (int j = 2; j <= order-2; ++j) {
	      for (int k = 2; k <= order-2; ++k) {
		const int v1 = dof_arr[base + i * li + j * lj + k*lk];
		const int v2 = dof_arr[base + (i+1) * li + j * lj + k*lk];

		clustering->Union(v1, v2);
	      }
	    }
	  }
	}

#ifdef DRSMOOTHER_3D_EDGES
	for (int d = 0; d < 3; ++d) {
	  int li = 1;
	  int lj = order+1;
	  int lk = order_plus_1_sq;

	  if (d == 1) {
	    std::swap(li, lk);
	  } else if (d == 2) {
	    std::swap(lj, lk);
	  }

	  for (int k = 2; k <= order-2; ++k) {
	    for (int i = 0; i <= order-1; i += order-1) {
	      for (int j = 0; j <= order-1; j += order-1) {
		const int v1 = dof_arr[base + i*li + j*lj + k*lk];
		const int v2 = dof_arr[base + (i+1)*li + j*lj + k*lk];
		const int v3 = dof_arr[base + i*li + (j+1)*lj + k*lk];
		const int v4 = dof_arr[base + (i+1)*li + (j+1)*lj + k*lk];

		clustering->Union(v1, v2);
		clustering->Union(v1, v3);
		clustering->Union(v1, v4);
	      }
	    }
	  }
	}
#endif
      }
    } else {
      MFEM_ABORT("Only supported for dimensions 1, 2, and 3");
    }

    clustering->Finalize();

    const Array<int> &bounds = clustering->GetBounds();
    MFEM_VERIFY(bounds.Size() > 0, "Finalized clustering should have established bounds array");

    return clustering;
  }

  void PrintClusteringForVis(std::ostream &out, const DisjointSets *clustering, const Mesh *mesh)
  {
    int dim = mesh->SpaceDimension();

    MFEM_VERIFY(clustering != NULL, "Clustering must be non-null");

    const Array<int> &elems  = clustering->GetElems();
    const Array<int> &bounds = clustering->GetBounds();

    for (int group = 0; group < bounds.Size()-1; ++group) {
      for (int i = bounds[group]; i < bounds[group+1]; ++i) {
	const double *v = mesh->GetVertex(elems[i]);
	out << v[0];
	for (int j = 1; j < dim; ++j) out << " " << v[j];

	if (i == bounds[group+1]-1) out << std::endl;
	else out << " ";
      }
    }
  }

  void DRSmoother::DiagonalDominance(const SparseMatrix *A, double &dd1, double &dd2)
  {
    const int *I = A->GetI();
    const int *J = A->GetJ();
    const double *data = A->GetData();

    dd1 = 0.0;
    dd2 = 0.0;

    for (int i = 0; i < A->Height(); ++i) {
      double off_diag = 0.0;
      double diag = 0.0;
      for (int k = I[i]; k < I[i+1]; ++k) {
	const int j = J[k];
	const double a = data[k];
	if (i == j) {
	  diag = a;
	} else {
	  off_diag += fabs(a);
	}
      }
      const double dd = off_diag / diag;
      dd1 = std::max(dd1, dd);
      dd2 += dd;
    }
    dd2 /= A->Height();
  }

  // y += scale * G' * x
  void DRSmootherG::AddMultTranspose(const Vector &x, Vector &y, double scale) const
  {
    if (!matrix_free) {
      G->AddMultTranspose(x, y, scale);
      return;
    }

    const Array<int> &bounds = clustering->GetBounds();
    const Array<int> &elems  = clustering->GetElems();

    int idx = 0;
    for (int group = 0; group < bounds.Size()-1; group++) {
      const int i = elems[bounds[group]];
      const int size = bounds[group+1] - bounds[group];
      if (size == 1) {
	y[i] += scale * x[i];
      } else {
	y[i] += scale * (*coeffs)[idx] * x[i];
	++idx;
	const int limit=bounds[group+1];
	for (int k = bounds[group]+1; k < limit; ++k) {
	  const int j = elems[k];
	  const double scaledX(scale*x[j]);
	  y[j] += scaledX;
	  y[i] += (*coeffs)[idx] * scaledX;
	  ++idx;
	}
      }
    }
  }

  // y += scale * G * x
  void DRSmootherG::AddMult(const Vector &x, Vector &y, double scale) const
  {
    if (!matrix_free) {
      G->AddMult(x, y, scale);
      return;
    }

    const Array<int> &bounds = clustering->GetBounds();
    const Array<int> &elems  = clustering->GetElems();

    int idx = 0;
    for (int group = 0; group < bounds.Size()-1; group++) {
      const int size = bounds[group+1] - bounds[group];
      const int i = elems[bounds[group]];
      if (size == 1) {
	y[i] += scale * x[i];
      } else {
	const double scaledXI(scale*x[i]);
	y[i] += ((*coeffs)[idx] * scaledXI);
	++idx;
	const int limit(bounds[group+1]);
	for (int k = bounds[group]+1; k<limit; ++k) {
	  const int j = elems[k];
	  y[j] += scale * x[j];
	  y[j] += scaledXI * (*coeffs)[idx];
	  ++idx;
	}
      }
    }
  }

  void DRSmootherG::GtAG(SparseMatrix *&GtAG_mat, Vector &GtAG_diagonal, const SparseMatrix &A, const std::vector<DenseMatrix> *diag_blocks) const
  {
    if (!matrix_free) {
      const SparseMatrix *Gt = Transpose(*G);
      //GtAG_mat = RAP(A, *G, 'P');
      GtAG_mat = RAP(*Gt, A, *G);
      GtAG_mat->GetDiag(GtAG_diagonal);
      return;
    }

    const Array<int> &bounds = clustering->GetBounds();
    const Array<int> &elems  = clustering->GetElems();

    GtAG_diagonal.SetSize(elems.Size());
    GtAG_diagonal = 0.0;

    int idx = 0;
    for (int group = 0; group < bounds.Size()-1; ++group) {
      int size = bounds[group+1] - bounds[group];
      const DenseMatrix &block = (*diag_blocks)[group];

      if (size == 1) {
	GtAG_diagonal[elems[bounds[group]]] = block(0,0);
      } else {
	const double *G_data = coeffs->GetData() + idx;
	idx += size;
	GtAG_diagonal[elems[bounds[group]]] = block.InnerProduct(G_data, G_data);
	for (int i = 1; i < size; ++i) {
	  GtAG_diagonal[elems[bounds[group]+i]] = block(i,i);
	}
      }
    }
  }

  DRSmootherG::~DRSmootherG()
  {
    if (clustering != NULL) delete clustering;
    if (coeffs != NULL) delete coeffs;
    if (G != NULL) delete G;
  }

}



