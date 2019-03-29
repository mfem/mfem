#ifndef DDOPER_HPP
#define DDOPER_HPP

#include "mfem.hpp"

using namespace mfem;
using namespace std;

hypre_CSRMatrix* GetHypreParMatrixData(const HypreParMatrix & hypParMat)
{
  // First cast the parameter to a hypre_ParCSRMatrix
  hypre_ParCSRMatrix * parcsr_op =
    (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(hypParMat);

  MFEM_ASSERT(parcsr_op != NULL,"STRUMPACK: const_cast failed in SetOperator");

  // Create the CSRMatrixMPI A_ by borrowing the internal data from a hypre_CSRMatrix.
  return hypre_MergeDiagAndOffd(parcsr_op);
}

Operator* CreateStrumpackMatrixFromHypreBlocks(MPI_Comm comm, const Array<int> & offsets, const Array2D<HypreParMatrix*> & blocks,
					       const Array2D<std::vector<int>*> & leftInjection,
					       const Array2D<std::vector<int>*> & rightInjection,
					       const Array2D<double> & coefficient)
					       
{
  const int numBlocks = offsets.Size() - 1;
  
  const int num_loc_rows = offsets[numBlocks];

  int nprocs, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  int *all_num_loc_rows = new int[nprocs];

  MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows, 1, MPI_INT, comm);

  int first_loc_row = 0;
  int glob_nrows = 0;
  for (int i=0; i<nprocs; ++i)
    {
      glob_nrows += all_num_loc_rows[i];
      if (i < rank)
	first_loc_row += all_num_loc_rows[i];
    }

  delete all_num_loc_rows;
    
  const int glob_ncols = glob_nrows;

  int *opI = new int[num_loc_rows+1];
  int *cnt = new int[num_loc_rows];

  for (int i=0; i<num_loc_rows; ++i)
    {
      opI[i] = 0;
      cnt[i] = 0;
    }

  opI[num_loc_rows] = 0;
  
  Array2D<hypre_CSRMatrix*> csr_blocks(numBlocks, numBlocks);
  
  // Loop over all blocks, to determine nnz for each row.
  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  if (blocks(i, j) == NULL)
	    {
	      csr_blocks(i, j) = NULL;
	    }
	  else
	    {
	      csr_blocks(i, j) = GetHypreParMatrixData(*(blocks(i, j)));

	      const int nrows = csr_blocks(i, j)->num_rows;

	      if (leftInjection(i, j) == NULL)
		{
		  if (nrows != offsets[i+1] - offsets[i])
		    cout << "ERROR" << endl;
		      
		  MFEM_VERIFY(nrows == offsets[i+1] - offsets[i], "");
	      
		  for (int k=0; k<nrows; ++k)
		    {
		      if (rank == 0)
			{
			  const int osk = csr_blocks(i, j)->i[k];
			  const int nnz_k = csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
			  for (int l=0; l<nnz_k; ++l)
			    {
			      if (csr_blocks(i, j)->j[osk + l] >= nrows)
				{
				  cout << "(" << i << ", " << j << ") row " << k;
				  cout << ", col " << csr_blocks(i, j)->j[osk + l] << endl;
				}
			    }
			}
			  
		      const int rowg = offsets[i] + k;
		      opI[rowg + 1] += csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
		    }
		}
	      else
		{
		  // The dimension of the range of leftInjection(i, j) is not available here.
		      
		  for (int k=0; k<nrows; ++k)
		    {
		      const int rowg = offsets[i] + k;
		      //(*(leftInjection(i, j)))[k]
		      opI[rowg + 1] += csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
		    }
		}
	    }
	}
    }

  // Now opI[i] is nnz for row i-1. Do a partial sum to get offsets.
  for (int i=0; i<num_loc_rows; ++i)
    opI[i+1] += opI[i];

  const int nnz = opI[num_loc_rows];

  int *opJ = new int[nnz];
  double *data = new double[nnz];

  // Loop over all blocks, to set matrix data.
  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  if (csr_blocks(i, j) != NULL)
	    {
	      const int nrows = csr_blocks(i, j)->num_rows;
	      const double coef = coefficient(i, j);
	      
	      MFEM_VERIFY(nrows == offsets[i+1] - offsets[i], "");
	      
	      for (int k=0; k<nrows; ++k)
		{
		  const int rowg = offsets[i] + k;
		  const int nnz_k = csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
		  const int osk = csr_blocks(i, j)->i[k];
		  
		  for (int l=0; l<nnz_k; ++l)
		    {
		      const int colg = offsets[j] + csr_blocks(i, j)->j[osk + l];

		      opJ[opI[rowg] + cnt[rowg]] = colg;
		      data[opI[rowg] + cnt[rowg]] = coef * csr_blocks(i, j)->data[osk + l];
		      cnt[rowg]++;
		    }
		}
	    }
	}
    }

  bool cntCheck = true;
  for (int i=0; i<num_loc_rows; ++i)
    {
      if (cnt[i] != opI[i+1] - opI[i])
	cntCheck = false;
    }

  MFEM_VERIFY(cntCheck, "");

  delete cnt;
  
  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  if (csr_blocks(i, j) != NULL)
	    {
	      hypre_CSRMatrixDestroy(csr_blocks(i, j));
	    }
	}
    }
  
  Operator *op = new STRUMPACKRowLocMatrix(comm, num_loc_rows, first_loc_row, glob_nrows, glob_ncols, opI, opJ, data);
  
  delete opI;
  delete opJ;
  delete data;
  
  return op;
}

// Should this be in ParFiniteElementSpace?
void FindBoundaryTrueDOFs(ParFiniteElementSpace *pfespace, set<int>& tdofsBdry)
{
  const ParMesh *pmesh = pfespace->GetParMesh();

  for (int be = 0; be < pmesh->GetNBE(); ++be)
    {
      // const int face = pmesh->GetBdrElementEdgeIndex(be);  // face index of boundary element i
      Array<int> dofs;
      pfespace->GetBdrElementDofs(be, dofs);
      for (int i=0; i<dofs.Size(); ++i)
	{
	  const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
	  const int ldof = pfespace->GetLocalTDofNumber(dof_i);  // If the DOF is owned by the current processor, return its local tdof number, otherwise -1.
	  if (ldof >= 0)
	    tdofsBdry.insert(ldof);
	}
    }
}

// This function is applicable only to convex faces, as it simply compares the vertices as sets.
bool FacesCoincideGeometrically(ParMesh *volumeMesh, const int face, ParMesh *surfaceMesh, const int elem)
{
  Array<int> faceVert;
  volumeMesh->GetFaceVertices(face, faceVert);

  Array<int> elemVert;
  surfaceMesh->GetElementVertices(elem, elemVert);

  if (faceVert.Size() != elemVert.Size())
    return false;

  for (int i=0; i<faceVert.Size(); ++i)
    {
      double *vi = volumeMesh->GetVertex(faceVert[i]);
      bool vertexFound = false;
      
      for (int j=0; j<faceVert.Size(); ++j)
	{
	  double *vj = surfaceMesh->GetVertex(elemVert[j]);

	  bool verticesEqual = true;
	  for (int k=0; k<3; ++k)
	    {
	      if (fabs(vi[k] - vj[k]) > 1.0e-12)
		verticesEqual = false;
	    }

	  if (verticesEqual)
	    vertexFound = true;
	}

      if (!vertexFound)
	return false;
    }

  return true;
}


// For InterfaceToSurfaceInjection, we need a map from DOF's on the interfaces to the corresponding DOF's on the surfaces of the subdomains.
// These maps could be created efficiently by maintaining maps between subdomain meshes and the original mesh, as well as maps between the
// interface meshes and the original mesh. The ParMesh constructor appears to keep the same ordering of elements, but it reorders the vertices.
// For interface meshes, the elements are faces, which are stored in order by the std::set<int> SubdomainInterface::faces. Therefore, creating
// these maps efficiently seems to require element maps between the original mesh and the subdomain and interface meshes. The
// InterfaceToSurfaceInjection will work by mapping interface faces to the original mesh neighboring elements, followed by mapping those
// elements to subdomain elements, determining which face of each subdomain element is on the interface geometrically, and then determining
// the DOF correspondence on each face geometrically by using GetVertexDofs, GetEdgeDofs, and GetFaceDofs (since the ordering may be different
// on the subdomain faces and interface elements). 

// For subdomain operators A^{**}, the only suboperators that use injection operators are A^{S\rho} and A^{FS}. If A^{SF} were nonzero, it
// would also use injection. The first block is for u on the entire subdomain including the interior and the surface, so injection to the
// S-rows is really injection into the true DOF's of the entire ND subdomain space. The transpose of injection is used for A^{FS}, again from
// the entire ND subdomain space to the interface. 

// For interface operators C^{**}, the S-rows are just the true DOF's of the subdomain ND space on the entire subdomain boundary. Thus we can
// use the same injection operator as for the subdomain operators. However, we must map from those ordered true DOF's to their indices within
// the set, using an std::map<int, int>. 

// The true DOF issue is complicated, because interface operators are defined on interface spaces, which may have DOF's that are not true
// DOF's in the interface space but correspond to true DOF's on the surfaces of the subdomain spaces. In the extreme case, an interface space
// may have zero true DOF's on a process, although the same process may have many true DOF's in the subdomain space on that interface. As a
// result, the subdomain would not receive the contributions from the interface operator, if it acted only on true DOF's. Instead, we must
// inject from full DOF's in the interface spaces to true DOF's in the subdomain spaces. This is also valid for the transpose of injection.
// The use of full DOF's in the interface spaces is done in InjectionOperator. Whether a DOF is true is determined by
// fespace.GetLocalTDofNumber().

// Therefore, dofmap is defined by SetInterfaceToSurfaceDOFMap() to be of full ifespace DOF size, mapping from the full ifespace DOF's to
// true subdomain DOF's in fespace.
void SetInterfaceToSurfaceDOFMap(ParFiniteElementSpace *ifespace, ParFiniteElementSpace *fespace, ParMesh *pmesh, const int sdAttribute,
				 const std::set<int>& pmeshFacesInInterface, const FiniteElementCollection *fec, std::vector<int>& dofmap)
{
  const int ifSize = ifespace->GetVSize();  // Full DOF size

  dofmap.assign(ifSize, -1);

  const double vertexTol = 1.0e-12;
  
  ParMesh *ifMesh = ifespace->GetParMesh();  // Interface mesh
  ParMesh *sdMesh = fespace->GetParMesh();  // Subdomain mesh

  // Create map from face indices in pmeshFacesInInterface to pmesh elements containing those faces.
  std::map<int, int> pmeshFaceToElem;
  std::set<int> pmeshElemsByInterface;

  for (int elId=0; elId<pmesh->GetNE(); ++elId)
    {
      if (pmesh->GetAttribute(elId) == sdAttribute)
	{
	  Array<int> elFaces, ori;
	  pmesh->GetElementFaces(elId, elFaces, ori);
	  for (int j=0; j<elFaces.Size(); ++j)
	    {
	      std::set<int>::const_iterator it = pmeshFacesInInterface.find(elFaces[j]);
	      if (it != pmeshFacesInInterface.end())
		{
		  std::map<int, int>::iterator itf = pmeshFaceToElem.find(elFaces[j]);
		  MFEM_VERIFY(itf == pmeshFaceToElem.end(), "");
		  
		  pmeshFaceToElem[elFaces[j]] = elId;

		  pmeshElemsByInterface.insert(elId);
		}
	    }
	}
    }

  // Set a map pmeshElemToSDmesh from pmesh element indices to the corresponding sdMesh element indices, only for elements neighboring the interface.
  std::map<int, int> pmeshElemToSDmesh;
  for (int elId=0; elId<sdMesh->GetNE(); ++elId)
    {
      // The sdMesh element attribute is set as the local index of the corresponding pmesh element, which is unique since SD elements do not overlap.
      const int pmeshElemId = sdMesh->GetAttribute(elId) - 1;  // 1 was added to ensure a positive attribute.
      std::set<int>::const_iterator it = pmeshElemsByInterface.find(pmeshElemId);
      if (it != pmeshElemsByInterface.end())  // if pmeshElemId neighbors the interface
	{
	  pmeshElemToSDmesh[pmeshElemId] = elId;
	}
    }
  
  // Loop over interface faces.
  int i = 0;
  for (std::set<int>::const_iterator it = pmeshFacesInInterface.begin(); it != pmeshFacesInInterface.end(); ++it, ++i)
    {
      const int pmeshFace = *it;

      // Face pmeshFace of pmesh coincides with face i of ifMesh on this process (the same face may also exist on a different process in the same ifMesh,
      // as there can be redundant overlapping faces in parallel, for communication).

      // Find the neighboring pmesh element.
      std::map<int, int>::iterator ite = pmeshFaceToElem.find(pmeshFace);

      //MFEM_VERIFY(ite != pmeshFaceToElem.end(), "");

      if (ite == pmeshFaceToElem.end())  // This process does not have an element in this subdomain neighboring the face.
	continue;
      
      MFEM_VERIFY(ite->first == pmeshFace, "");

      const int pmeshElem = ite->second;

      // Find the neighboring sdMesh element, which coincides with pmeshElem in pmesh.
      std::map<int, int>::const_iterator itse = pmeshElemToSDmesh.find(pmeshElem);
      MFEM_VERIFY(itse != pmeshElemToSDmesh.end(), "");
      MFEM_VERIFY(itse->first == pmeshElem, "");

      const int sdMeshElem = itse->second;

      // Find the face of element sdMeshElem in sdMesh that coincides geometrically with the current interface face.
      Array<int> elFaces, ori;

      sdMesh->GetElementFaces(sdMeshElem, elFaces, ori);
      int sdMeshFace = -1;
      for (int j=0; j<elFaces.Size(); ++j)
	{
	  if (FacesCoincideGeometrically(sdMesh, elFaces[j], ifMesh, i))
	    sdMeshFace = elFaces[j];
	}

      MFEM_VERIFY(sdMeshFace >= 0, "");

      // Map vertex DOF's on ifMesh face i to vertex DOF's on sdMesh face sdMeshFace.
      // TODO: is this necessary, since FiniteElementSpace::GetEdgeDofs claims to return vertex DOF's as well?
      const int nv = fec->DofForGeometry(Geometry::POINT);
      if (nv > 0)
	{
	  Array<int> ifVert, sdVert;
	  ifMesh->GetFaceVertices(i, ifVert);
	  sdMesh->GetFaceVertices(sdMeshFace, sdVert);

	  MFEM_VERIFY(ifVert.Size() == sdVert.Size(), "");
	  
	  for (int j=0; j<ifVert.Size(); ++j)
	    {
	      double *ifv = ifMesh->GetVertex(ifVert[j]);

	      bool vertexFound = false;
	      
	      for (int k=0; k<sdVert.Size(); ++k)
		{
		  double *sdv = sdMesh->GetVertex(sdVert[k]);

		  bool verticesEqual = true;
		  for (int l=0; l<3; ++l)
		    {
		      if (fabs(ifv[l] - sdv[l]) > vertexTol)
			verticesEqual = false;
		    }

		  if (verticesEqual)
		    {
		      vertexFound = true;
		      Array<int> ifdofs, sddofs;
		      ifespace->GetVertexDofs(ifVert[j], ifdofs);
		      fespace->GetVertexDofs(sdVert[k], sddofs);

		      MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
		      for (int d=0; d<ifdofs.Size(); ++d)
			{
			  const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);
			  
			  if (sdtdof >= 0)  // if this is a true DOF of fespace.
			    {
			      MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
			      dofmap[ifdofs[d]] = sdtdof;
			    }
			}
		    }
		}
	      
	      MFEM_VERIFY(vertexFound, "");
	    }
	}
      
      // Map edge DOF's on ifMesh face i to edge DOF's on sdMesh face sdMeshFace.
      const int ne = fec->DofForGeometry(Geometry::SEGMENT);
      if (ne > 0)
	{
	  // TODO: could there be multiple DOF's on an edge with different orderings (depending on orientation) in ifespace and fespace?
	  // TODO: Check orientation for ND_HexahedronElement? Does ND_TetrahedronElement have orientation?

	  Array<int> ifEdge, sdEdge, ifOri, sdOri;
	  ifMesh->GetElementEdges(i, ifEdge, ifOri);
	  sdMesh->GetFaceEdges(sdMeshFace, sdEdge, sdOri);

	  MFEM_VERIFY(ifEdge.Size() == sdEdge.Size(), "");
	  
	  for (int j=0; j<ifEdge.Size(); ++j)
	    {
	      Array<int> ifVert;
	      ifMesh->GetEdgeVertices(ifEdge[j], ifVert);

	      MFEM_VERIFY(ifVert.Size() == 2, "");

	      int sd_k = -1;
	      
	      for (int k=0; k<sdEdge.Size(); ++k)
		{
		  Array<int> sdVert;
		  sdMesh->GetEdgeVertices(sdEdge[k], sdVert);

		  MFEM_VERIFY(sdVert.Size() == 2, "");

		  bool edgesMatch = true;
		  for (int v=0; v<2; ++v)
		    {
		      double *ifv = ifMesh->GetVertex(ifVert[v]);
		      double *sdv = sdMesh->GetVertex(sdVert[v]);

		      bool verticesEqual = true;
		      for (int l=0; l<3; ++l)
			{
			  if (fabs(ifv[l] - sdv[l]) > vertexTol)
			    verticesEqual = false;
			}

		      if (!verticesEqual)
			edgesMatch = false;
		    }

		  if (edgesMatch)
		    {
		      MFEM_VERIFY(sd_k == -1, "");
		      sd_k = k;
		    }
		}

	      MFEM_VERIFY(sd_k >= 0, "");

	      Array<int> ifdofs, sddofs;
	      ifespace->GetEdgeDofs(ifEdge[j], ifdofs);
	      fespace->GetEdgeDofs(sdEdge[sd_k], sddofs);

	      MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
	      for (int d=0; d<ifdofs.Size(); ++d)
		{
		  const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);
			  
		  if (sdtdof >= 0)  // if this is a true DOF of fespace.
		    {
		      MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
		      dofmap[ifdofs[d]] = sdtdof;
		    }
		}
	    }
	}

      // Map face DOF's on ifMesh face i to face DOF's on sdMesh face sdMeshFace.
      const int nf = fec->DofForGeometry(sdMesh->GetFaceGeometryType(0));
      if (nf > 0)
	{
	  Array<int> ifdofs, sddofs;
	  ifespace->GetFaceDofs(i, ifdofs);
	  fespace->GetFaceDofs(sdMeshFace, sddofs);

	  MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
	  for (int d=0; d<ifdofs.Size(); ++d)
	    {
	      const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);
			  
	      if (sdtdof >= 0)  // if this is a true DOF of fespace.
		{
		  MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
		  dofmap[ifdofs[d]] = sdtdof;
		}
	    }
	}
    }

  // Note that some entries of dofmap may be undefined, if the corresponding subdomain DOF's in fespace are not true DOF's. 
  /*
  bool mapDefined = true;
  for (i=0; i<ifSize; ++i)
    {
      if (dofmap[i] < 0)
	mapDefined = false;
    }
  
  MFEM_VERIFY(mapDefined, "");
  */
}

// TODO: combine SetInjectionOperator and InjectionOperator as one class?
class SetInjectionOperator : public Operator
{
private:
  std::set<int> *id;
  
public:
  SetInjectionOperator(const int height, std::set<int> *a) : Operator(height, a->size()), id(a)
  {
    MFEM_VERIFY(height >= width, "SetInjectionOperator constructor");
  }

  ~SetInjectionOperator()
  {
  }
  
  virtual void Mult(const Vector & x, Vector & y) const
  {
    y = 0.0;

    int i = 0;
    for (std::set<int>::const_iterator it = id->begin(); it != id->end(); ++it, ++i)
      y[*it] = x[i];
  }

  virtual void MultTranspose(const Vector &x, Vector &y) const
  {
    int i = 0;
    for (std::set<int>::const_iterator it = id->begin(); it != id->end(); ++it, ++i)
      y[i] = x[*it];
  }
};

class InjectionOperator : public Operator
{
private:
  int *id;  // Size should be fullWidth.
  mutable ParGridFunction gf;
  int fullWidth;
  
public:
  InjectionOperator(const int height, ParFiniteElementSpace *interfaceSpace, int *a) : Operator(height, interfaceSpace->GetTrueVSize()),
										       fullWidth(interfaceSpace->GetVSize()), id(a), gf(interfaceSpace)
  {
    MFEM_VERIFY(height >= width, "InjectionOperator constructor");
  }
  
  ~InjectionOperator()
  {
  }
  
  virtual void Mult(const Vector & x, Vector & y) const
  {
    gf.SetFromTrueDofs(x);
    
    y = 0.0;
    for (int i=0; i<fullWidth; ++i)
      {
	if (id[i] >= 0)
	  y[id[i]] = gf[i];
      }
  }

  virtual void MultTranspose(const Vector &x, Vector &y) const
  {
    for (int i=0; i<fullWidth; ++i)
      {
      	if (id[i] >= 0)
	  gf[i] = x[id[i]];
      }
    
    gf.GetTrueDofs(y);
  }
};

void SetSubdomainDofsFromDomainDofs(ParFiniteElementSpace *fespaceSD, ParFiniteElementSpace *fespaceDomain, const Vector & s, Vector & ssd)
{
  MFEM_VERIFY(ssd.Size() == fespaceSD->GetTrueVSize(), "");
  MFEM_VERIFY(s.Size() == fespaceDomain->GetTrueVSize(), "");

  ParMesh *sdMesh = fespaceSD->GetParMesh();  // Subdomain mesh

  ParGridFunction s_gf(fespaceDomain);
  s_gf.SetFromTrueDofs(s);

  MFEM_VERIFY(s_gf.Size() == fespaceDomain->GetVSize(), "");

  for (int elId=0; elId<sdMesh->GetNE(); ++elId)
    {
      // The sdMesh element attribute is set as the local index of the corresponding pmesh element, which is unique since SD elements do not overlap.
      const int domainElemId = sdMesh->GetAttribute(elId) - 1;  // 1 was added to ensure a positive attribute.
      
      Array<int> sddofs;
      Array<int> dofs;
      fespaceDomain->GetElementDofs(domainElemId, dofs);
      fespaceSD->GetElementDofs(elId, sddofs);

      MFEM_VERIFY(dofs.Size() == sddofs.Size(), "");
      
      for (int i=0; i<dofs.Size(); ++i)
	{
	  const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
	  const int sddof_i = sddofs[i] >= 0 ? sddofs[i] : -1 - sddofs[i];
	  const int lsddof = fespaceSD->GetLocalTDofNumber(sddof_i);  // If the DOF is owned by the current processor, return its local tdof number, otherwise -1.

	  if (lsddof >= 0)
	    {
	      ssd[lsddof] = s_gf[dof_i];
	    }
	}
    }
}

void SetDomainDofsFromSubdomainDofs(ParFiniteElementSpace *fespaceSD, ParFiniteElementSpace *fespaceDomain, const Vector & ssd, Vector & s)
{
  MFEM_VERIFY(ssd.Size() == fespaceSD->GetTrueVSize(), "");
  MFEM_VERIFY(s.Size() == fespaceDomain->GetTrueVSize(), "");

  ParMesh *sdMesh = fespaceSD->GetParMesh();  // Subdomain mesh

  ParGridFunction ssd_gf(fespaceSD);
  ssd_gf.SetFromTrueDofs(ssd);

  MFEM_VERIFY(ssd_gf.Size() == fespaceSD->GetVSize(), "");

  for (int elId=0; elId<sdMesh->GetNE(); ++elId)
    {
      // The sdMesh element attribute is set as the local index of the corresponding pmesh element, which is unique since SD elements do not overlap.
      const int domainElemId = sdMesh->GetAttribute(elId) - 1;  // 1 was added to ensure a positive attribute.
      
      Array<int> sddofs;
      Array<int> dofs;
      fespaceDomain->GetElementDofs(domainElemId, dofs);
      fespaceSD->GetElementDofs(elId, sddofs);

      MFEM_VERIFY(dofs.Size() == sddofs.Size(), "");
      
      for (int i=0; i<dofs.Size(); ++i)
	{
	  const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
	  const int sddof_i = sddofs[i] >= 0 ? sddofs[i] : -1 - sddofs[i];
	  const int ldof = fespaceDomain->GetLocalTDofNumber(dof_i);  // If the DOF is owned by the current processor, return its local tdof number, otherwise -1.

	  if (ldof >= 0)
	    {
	      s[ldof] = ssd_gf[sddof_i];
	    }
	}
    }
}

class DDMInterfaceOperator : public Operator
{
public:
  DDMInterfaceOperator(const int numSubdomains_, const int numInterfaces_, ParMesh *pmesh_, ParMesh **pmeshSD_, ParMesh **pmeshIF_,
		       const int orderND, const int spaceDim, std::vector<SubdomainInterface> *localInterfaces_,
		       std::vector<int> *interfaceLocalIndex_) :
    numSubdomains(numSubdomains_), numInterfaces(numInterfaces_), pmeshSD(pmeshSD_), pmeshIF(pmeshIF_), fec(orderND, spaceDim),
    fecbdry(orderND, spaceDim-1), fecbdryH1(orderND, spaceDim-1), localInterfaces(localInterfaces_), interfaceLocalIndex(interfaceLocalIndex_),
    subdomainLocalInterfaces(numSubdomains_), pmeshGlobal(pmesh_),
    k2(250.0),
    alpha(1.0), beta(1.0), gamma(1.0)  // TODO: set these to the right values
  {
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MFEM_VERIFY(numSubdomains > 0, "");
    MFEM_VERIFY(interfaceLocalIndex->size() == numInterfaces, "");

    fespace = new ParFiniteElementSpace*[numSubdomains];
    Asd = new Operator*[numSubdomains];
    ASPsd = new Operator*[numSubdomains];
    invAsd = new Operator*[numSubdomains];
    injSD = new BlockOperator*[numSubdomains];
    precAsd = new Solver*[numSubdomains];
    sdND = new HypreParMatrix*[numSubdomains];
    sdNDinv = new Operator*[numSubdomains];
    bf_sdND = new ParBilinearForm*[numSubdomains];
    ySD = new Vector*[numSubdomains];

    ifespace = numInterfaces > 0 ? new ParFiniteElementSpace*[numInterfaces] : NULL;
    iH1fespace = numInterfaces > 0 ? new ParFiniteElementSpace*[numInterfaces] : NULL;

    ifNDmass = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
    ifH1mass = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
    ifNDmassInv = numInterfaces > 0 ? new Operator*[numInterfaces] : NULL;
    ifH1massInv = numInterfaces > 0 ? new Operator*[numInterfaces] : NULL;
    ifNDcurlcurl = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
    ifNDH1grad = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
    ifNDH1gradT = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
    ifND_FS = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
    
    numLocalInterfaces = localInterfaces->size();
    globalInterfaceIndex.resize(numLocalInterfaces);
    globalInterfaceIndex.assign(numLocalInterfaces, -1);
    
    for (int i=0; i<numInterfaces; ++i)
      {
	if (pmeshIF[i] == NULL)
	  {
	    ifespace[i] = NULL;
	    iH1fespace[i] = NULL;

	    ifNDmass[i] = NULL;
	    ifNDcurlcurl[i] = NULL;
	    ifNDH1grad[i] = NULL;
	    ifNDH1gradT[i] = NULL;
	    ifH1mass[i] = NULL;
	    ifND_FS[i] = NULL;
	  }
	else
	  {
	    ifespace[i] = new ParFiniteElementSpace(pmeshIF[i], &fecbdry);  // Nedelec space for f_{m,j} when interface i is the j-th interface of subdomain m. 
	    iH1fespace[i] = new ParFiniteElementSpace(pmeshIF[i], &fecbdryH1);  // H^1 space \rho_{m,j} when interface i is the j-th interface of subdomain m.

	    CreateInterfaceMatrices(i);
	  }

	const int ifli = (*interfaceLocalIndex)[i];

	MFEM_VERIFY((ifli >= 0) == (pmeshIF[i] != NULL), "");
	
	if (ifli >= 0)
	  {
	    subdomainLocalInterfaces[(*localInterfaces)[ifli].FirstSubdomain()].push_back(i);
	    subdomainLocalInterfaces[(*localInterfaces)[ifli].SecondSubdomain()].push_back(i);

	    MFEM_VERIFY(globalInterfaceIndex[ifli] == i || globalInterfaceIndex[ifli] == -1, "");

	    globalInterfaceIndex[ifli] = i;
	  }
      }
    
    // For each subdomain parallel finite element space, determine all the true DOF's on the entire boundary. Also for each interface parallel finite element space, determine the number of true DOF's. Note that a true DOF on the boundary of a subdomain may coincide with an interface DOF that is not necessarily a true DOF on the corresponding interface mesh. The size of DDMInterfaceOperator will be the sum of the numbers of true DOF's on the subdomain mesh boundaries and interfaces.
    
    block_trueOffsets.SetSize(numSubdomains + 1); // number of blocks + 1
    block_trueOffsets = 0;

    int size = 0;

    InterfaceToSurfaceInjection.resize(numSubdomains);
    InterfaceToSurfaceInjectionData.resize(numSubdomains);
    
    for (int m=0; m<numSubdomains; ++m)
      {
	InterfaceToSurfaceInjection[m].resize(subdomainLocalInterfaces[m].size());
	InterfaceToSurfaceInjectionData[m].resize(subdomainLocalInterfaces[m].size());

	if (pmeshSD[m] == NULL)
	  {
	    fespace[m] = NULL;
	  }
	else
	  {
	    fespace[m] = new ParFiniteElementSpace(pmeshSD[m], &fec);  // Nedelec space for u_m

	    /*
	    if (m == 0)
	      cout << rank << ": sd 0 ND space true size " << fespace[m]->GetTrueVSize() << ", full size " << fespace[m]->GetVSize() << endl;
	    */
	  }
	
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];
	    
	    size += ifespace[interfaceIndex]->GetTrueVSize();
	    size += iH1fespace[interfaceIndex]->GetTrueVSize();

	    block_trueOffsets[m+1] += ifespace[interfaceIndex]->GetTrueVSize();
	    block_trueOffsets[m+1] += iH1fespace[interfaceIndex]->GetTrueVSize();

	    if (fespace[m] != NULL)
	      {
		const int ifli = (*interfaceLocalIndex)[interfaceIndex];
		MFEM_VERIFY(ifli >= 0, "");
		
		SetInterfaceToSurfaceDOFMap(ifespace[interfaceIndex], fespace[m], pmeshGlobal, m+1, (*localInterfaces)[ifli].faces, &fecbdry,
					    InterfaceToSurfaceInjectionData[m][i]);
	    
		InterfaceToSurfaceInjection[m][i] = new InjectionOperator(fespace[m]->GetTrueVSize(), ifespace[interfaceIndex],
									  &(InterfaceToSurfaceInjectionData[m][i][0]));
	      }
	    else
	      {
		InterfaceToSurfaceInjection[m][i] = NULL;
	      }
	  }
      }

    tdofsBdry.resize(numSubdomains);
    trueOffsetsSD.resize(numSubdomains);

    tdofsBdryInjection = new SetInjectionOperator*[numSubdomains];
    tdofsBdryInjectionTranspose = new Operator*[numSubdomains];
    
    for (int m=0; m<numSubdomains; ++m)
      {
	if (pmeshSD[m] == NULL)
	  {
	    sdND[m] = NULL;
	    
	    /*
	    Asd[m] = NULL;
	    invAsd[m] = NULL;
	    precAsd[m] = NULL;
	    */
	    tdofsBdryInjection[m] = NULL;
	    tdofsBdryInjectionTranspose[m] = NULL;
	  }
	else
	  {
	    FindBoundaryTrueDOFs(fespace[m], tdofsBdry[m]);  // Determine all true DOF's of fespace[m] on the boundary of pmeshSD[m], representing u_m^s.
	    size += tdofsBdry[m].size();
	    block_trueOffsets[m+1] += tdofsBdry[m].size();

	    tdofsBdryInjection[m] = new SetInjectionOperator(fespace[m]->GetTrueVSize(), &(tdofsBdry[m]));
	    tdofsBdryInjectionTranspose[m] = new TransposeOperator(tdofsBdryInjection[m]);
	    
	    CreateSubdomainMatrices(m);
	  }
	
	SetOffsetsSD(m);
	
	Asd[m] = CreateSubdomainOperator(m);
	//ASPsd[m] = CreateSubdomainOperatorStrumpack(m);

	precAsd[m] = CreateSubdomainPreconditionerStrumpack(m);

	GMRESSolver *gmres = new GMRESSolver(fespace[m]->GetComm());  // TODO: this communicator is not necessarily the same as the pmeshIF communicators. Does GMRES actually use the communicator?

	gmres->SetOperator(*(Asd[m]));
	gmres->SetRelTol(1e-12);
	gmres->SetMaxIter(1000);
	gmres->SetPrintLevel(0);

	gmres->SetPreconditioner(*(precAsd[m]));

	invAsd[m] = gmres;
      }

    height = size;
    width = size;

    block_trueOffsets.PartialSum();
    MFEM_VERIFY(block_trueOffsets.Last() == size, "");

    globalInterfaceOp = new BlockOperator(block_trueOffsets);

    rowTrueOffsetsIF.resize(numLocalInterfaces);
    colTrueOffsetsIF.resize(numLocalInterfaces);
  
    rowTrueOffsetsIFR.resize(numLocalInterfaces);
    colTrueOffsetsIFR.resize(numLocalInterfaces);

    rowTrueOffsetsIFL.resize(numLocalInterfaces);
    colTrueOffsetsIFL.resize(numLocalInterfaces);
    
    rowTrueOffsetsIFBR.resize(numLocalInterfaces);
    colTrueOffsetsIFBR.resize(numLocalInterfaces);

    rowTrueOffsetsIFBL.resize(numLocalInterfaces);
    colTrueOffsetsIFBL.resize(numLocalInterfaces);

    for (int ili=0; ili<numLocalInterfaces; ++ili)
      {
	const int sd0 = (*localInterfaces)[ili].FirstSubdomain();
	const int sd1 = (*localInterfaces)[ili].SecondSubdomain();

	MFEM_VERIFY(sd0 < sd1, "");

	// Create operators for interface between subdomains sd0 and sd1, namely C_{sd0,sd1} R_{sd1}^T and the other.
	globalInterfaceOp->SetBlock(sd0, sd1, CreateInterfaceOperator(ili, 0));
	globalInterfaceOp->SetBlock(sd1, sd0, CreateInterfaceOperator(ili, 1));
      }

    // Create block diagonal operator with entries R_{sd0} A_{sd0}^{-1} R_{sd0}^T
    BlockOperator *globalSubdomainOp = new BlockOperator(block_trueOffsets);

    rowTrueOffsetsSD.resize(numSubdomains);
    colTrueOffsetsSD.resize(numSubdomains);
    
    rowSROffsetsSD.SetSize(numSubdomains + 1); // number of blocks + 1
    colSROffsetsSD.SetSize(numSubdomains + 1); // number of blocks + 1
    rowSROffsetsSD = 0;
    colSROffsetsSD = 0;

    for (int m=0; m<numSubdomains; ++m)
      {
	if (Asd[m] != NULL)
	  {
	    // Create block injection operator R_{sd0}^T from (u^s, f_i, \rho_i) space to (u, f_i, \rho_i) space.

	    rowTrueOffsetsSD[m].SetSize(2 + 1);  // Number of blocks + 1
	    colTrueOffsetsSD[m].SetSize(2 + 1);  // Number of blocks + 1

	    rowTrueOffsetsSD[m] = 0;
	    rowTrueOffsetsSD[m][1] = fespace[m]->GetTrueVSize();

	    int ifsize = 0;
	    for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	      {
		const int interfaceIndex = subdomainLocalInterfaces[m][i];
	
		MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
		MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");

		ifsize += ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
	      }

	    rowTrueOffsetsSD[m][2] = ifsize;
	    
	    colTrueOffsetsSD[m] = rowTrueOffsetsSD[m];
	    colTrueOffsetsSD[m][1] = tdofsBdry[m].size();

	    rowTrueOffsetsSD[m].PartialSum();
	    colTrueOffsetsSD[m].PartialSum();
    
	    injSD[m] = new BlockOperator(rowTrueOffsetsSD[m], colTrueOffsetsSD[m]);
	    
	    injSD[m]->SetBlock(0, 0, tdofsBdryInjection[m]);
	    injSD[m]->SetBlock(1, 1, new IdentityOperator(ifsize));

	    globalSubdomainOp->SetBlock(m, m, new TripleProductOperator(new TransposeOperator(injSD[m]), invAsd[m], injSD[m], false, false, false));

	    rowSROffsetsSD[m+1] = fespace[m]->GetTrueVSize();
	    colSROffsetsSD[m+1] = fespace[m]->GetTrueVSize();
	  }
      }

    // Create operators R_{sd0} A_{sd0}^{-1} C_{sd0,sd1} R_{sd1}^T by multiplying globalInterfaceOp on the left by globalSubdomainOp. Then add identity.
    globalOp = new SumOperator(new ProductOperator(globalSubdomainOp, globalInterfaceOp, false, false), new IdentityOperator(size), false, false, 1.0, 1.0);

    // Create source reduction operator.
    { // TODO: is this used?
      rowSROffsetsSD.PartialSum();
      colSROffsetsSD.PartialSum();
      NDinv = new BlockOperator(rowSROffsetsSD, colSROffsetsSD);

      for (int m=0; m<numSubdomains; ++m)
	{
	  if (pmeshSD[m] != NULL)
	    {
	      NDinv->SetBlock(m, m, sdNDinv[m]);
	    }
	}
    }

    /*
    { // Test
      Vector x0(Asd[0]->Height());
      Vector y0(Asd[0]->Height());
      for (int i=0; i<10; ++i)
	x0[i] = 1.0;
      
      invAsd[0]->Mult(x0, y0);
    }
    */
  }

  virtual void Mult(const Vector & x, Vector & y) const
  {
    // x and y are vectors of true DOF's on the subdomain interfaces and exterior boundary. 
    // Degrees of freedom in x and y are ordered as follows: x = [x_0, x_1, ..., x_{N-1}];
    // N = numSubdomains, and on subdomain m, x_m = [u_m^s, f_m, \rho_m];
    // u_m^s is the vector of true DOF's of u on the entire surface of subdomain m, for a field u in a Nedelec space on subdomain m;
    // f_m = [f_{m,0}, ..., f_{m,p-1}] is an auxiliary vector of true DOF's in Nedelec spaces on all p interfaces on subdomain m;
    // \rho_m = [\rho_{m,0}, ..., \rho_{m,p-1}] is an auxiliary vector of true DOF's in H^1 (actually H^{1/2}) FE spaces on all
    // p interfaces on subdomain m.
    
    // The surface of subdomain m equals the union of subdomain interfaces and a subset of the exterior boundary.
    // There are redundant DOF's for f and \rho at subdomain corner edges (intersections of interfaces), i.e. discontinuity on corners.
    // The surface bilinear forms and their matrices are defined on subdomain interfaces, not the entire subdomain boundary.
    // The surface DOF's for a subdomain are indexed according to the entire subdomain mesh boundary, and we must use maps between
    // those surface DOF's and DOF's on the individual interfaces.

    globalOp->Mult(x, y);
  }  

  void GetReducedSource(ParFiniteElementSpace *fespaceGlobal, Vector & sourceGlobal, Vector & sourceReduced) const
  {
    Vector sourceSD, wSD, vSD;

    MFEM_VERIFY(sourceReduced.Size() == block_trueOffsets[numSubdomains], "");
    MFEM_VERIFY(sourceReduced.Size() == this->Height(), "");
    
    for (int m=0; m<numSubdomains; ++m)
      {
	if (pmeshSD[m] != NULL)
	  {
	    // Map from the global u to [u_m f_m \rho_m], with blocks corresponding to subdomains, and f_m = 0, \rho_m = 0.
	    
	    sourceSD.SetSize(fespace[m]->GetTrueVSize());
	    SetSubdomainDofsFromDomainDofs(fespace[m], fespaceGlobal, sourceGlobal, sourceSD);

	    ySD[m] = new Vector(Asd[m]->Height());  // Size of [u_m f_m \rho_m]
	    wSD.SetSize(Asd[m]->Height());  // Size of [u_m f_m \rho_m]
	    
	    (*(ySD[m])) = 0.0;
	    for (int i=0; i<sourceSD.Size(); ++i)
	      (*(ySD[m]))[i] = sourceSD[i];  // Set the u_m block of ySD

	    cout << "Applying invAsd[" << m << "]" << endl;
	    
	    invAsd[m]->Mult(*(ySD[m]), wSD);

	    cout << "Done applying invAsd[" << m << "]" << endl;

	    // Extract only the [u_m^s, f_m, \rho_m] entries.
	    vSD.SetSize(block_trueOffsets[m+1] - block_trueOffsets[m]);
	    injSD[m]->MultTranspose(wSD, vSD);

	    for (int i=0; i<vSD.Size(); ++i)
	      sourceReduced[block_trueOffsets[m] + i] = vSD[i];
	  }
	else
	  {
	    ySD[m] = NULL;
	  }
      }
  }

  void RecoverDomainSolution(ParFiniteElementSpace *fespaceGlobal, const Vector & solReduced, Vector & solDomain)
  {
    MFEM_VERIFY(solReduced.Size() == this->Height(), "");
    Vector w(this->Height());
    Vector v, u, uSD;
    
    globalInterfaceOp->Mult(solReduced, w);
    
    for (int m=0; m<numSubdomains; ++m)
      {
	if (ySD[m] != NULL)
	  {
	    MFEM_VERIFY(ySD[m]->Size() == block_trueOffsets[m+1] - block_trueOffsets[m], "");

	    v.SetSize(ySD[m]->Size());
	    u.SetSize(ySD[m]->Size());
	    
	    for (int i=0; i<block_trueOffsets[m+1] - block_trueOffsets[m]; ++i)
	      v[i] = (*(ySD[m]))[i] - w[block_trueOffsets[m] + i];

	    invAsd[m]->Mult(v, u);

	    uSD.SetSize(fespace[m]->GetTrueVSize());

	    for (int i=0; i<uSD.Size(); ++i)
	      uSD[i] = u[i];
	    
	    SetDomainDofsFromSubdomainDofs(fespace[m], fespaceGlobal, uSD, solDomain);
	  }
      }
  }
  
private:

  double k2;
  
  const int numSubdomains;
  int numInterfaces, numLocalInterfaces;

  ParMesh *pmeshGlobal;
  ParMesh **pmeshSD;  // Subdomain meshes
  ParMesh **pmeshIF;  // Interface meshes
  ND_FECollection fec, fecbdry;
  H1_FECollection fecbdryH1;
  
  ParFiniteElementSpace **fespace, **ifespace, **iH1fespace;
  HypreParMatrix **ifNDmass, **ifNDcurlcurl, **ifNDH1grad, **ifNDH1gradT, **ifH1mass, **ifND_FS;
  Operator **ifNDmassInv, **ifH1massInv;
  HypreParMatrix **sdND;
  ParBilinearForm **bf_sdND;
  Operator **sdNDinv;
  Operator **Asd;
  Operator **ASPsd;
  Operator **invAsd;
  Solver **precAsd;

  // TODO: it may be possible to eliminate ifND_FS. It is assembled as a convenience, to avoid summing entries input to ASPsd.
  
  Vector **ySD;
  
  BlockOperator **injSD;
  
  std::vector<SubdomainInterface> *localInterfaces;
  std::vector<int> *interfaceLocalIndex;
  std::vector<int> globalInterfaceIndex;
  std::vector<std::vector<int> > subdomainLocalInterfaces;

  BlockOperator *globalInterfaceOp;
  
  Operator *globalOp;  // Operator for all global subdomains (blocks corresponding to non-local subdomains will be NULL).
  Array<int> block_trueOffsets;  // Offsets used in globalOp

  BlockOperator *NDinv;
  
  vector<set<int> > tdofsBdry;
  SetInjectionOperator **tdofsBdryInjection;
  Operator **tdofsBdryInjectionTranspose;
  
  double alpha, beta, gamma;
  
  std::vector<std::vector<InjectionOperator*> > InterfaceToSurfaceInjection;
  std::vector<std::vector<std::vector<int> > > InterfaceToSurfaceInjectionData;

  std::vector<Array<int> > rowTrueOffsetsSD, colTrueOffsetsSD;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIF, colTrueOffsetsIF;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFL, colTrueOffsetsIFL;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFR, colTrueOffsetsIFR;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFBL, colTrueOffsetsIFBL;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFBR, colTrueOffsetsIFBR;
  std::vector<Array<int> > trueOffsetsSD;
  Array<int> rowSROffsetsSD, colSROffsetsSD;

  // TODO: if the number of subdomains gets large, it may be better to define a local block operator only for local subdomains.

  void CreateInterfaceMatrices(const int interfaceIndex)
  {
    int num_procs, myid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    ConstantCoefficient one(1.0);
    Array<int> ess_tdof_list;  // empty

    // Nedelec interface operators

    ParBilinearForm *NDmass = new ParBilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
    NDmass->AddDomainIntegrator(new VectorFEMassIntegrator(one));
    NDmass->Assemble();

    ParBilinearForm *NDcurlcurl = new ParBilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
    NDcurlcurl->AddDomainIntegrator(new CurlCurlIntegrator(one));
    NDcurlcurl->Assemble();
    
    ParBilinearForm *ND_FS = new ParBilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
    ND_FS->AddDomainIntegrator(new VectorFEMassIntegrator(one));
    ND_FS->AddDomainIntegrator(new CurlCurlIntegrator(one));
    ND_FS->Assemble();

    ifNDmass[interfaceIndex] = new HypreParMatrix();
    ifNDcurlcurl[interfaceIndex] = new HypreParMatrix();
    ifND_FS[interfaceIndex] = new HypreParMatrix();

    NDmass->FormSystemMatrix(ess_tdof_list, *(ifNDmass[interfaceIndex]));
    NDcurlcurl->FormSystemMatrix(ess_tdof_list, *(ifNDcurlcurl[interfaceIndex]));
    ND_FS->FormSystemMatrix(ess_tdof_list, *(ifND_FS[interfaceIndex]));

    cout << myid << ": interface " << interfaceIndex << ", ND true size " << ifespace[interfaceIndex]->GetTrueVSize() << ", mass height "
	 << ifNDmass[interfaceIndex]->Height() << ", width " << ifNDmass[interfaceIndex]->Width() << ", ND V size "
	 << ifespace[interfaceIndex]->GetVSize() << endl;

    Operator *ifNDmassSP = new STRUMPACKRowLocMatrix(*(ifNDmass[interfaceIndex]));
    ifNDmassInv[interfaceIndex] = CreateStrumpackSolver(ifNDmassSP, ifespace[interfaceIndex]->GetComm());

    // H^1 interface operators

    ParBilinearForm *H1mass = new ParBilinearForm(iH1fespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
    H1mass->AddDomainIntegrator(new MassIntegrator(one));
    H1mass->Assemble();

    ifH1mass[interfaceIndex] = new HypreParMatrix();
    
    H1mass->FormSystemMatrix(ess_tdof_list, *(ifH1mass[interfaceIndex]));

    {
      /*
      Operator *ifH1massSP = new STRUMPACKRowLocMatrix(*(ifH1mass[interfaceIndex]));
      ifH1massInv[interfaceIndex] = CreateStrumpackSolver(ifH1massSP, iH1fespace[interfaceIndex]->GetComm());
      */
      
      ParBilinearForm *H1stiff = new ParBilinearForm(iH1fespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
      H1stiff->AddDomainIntegrator(new MassIntegrator(one));
      H1stiff->AddDomainIntegrator(new DiffusionIntegrator(one));
      H1stiff->Assemble();

      HypreParMatrix *ifH1stiff = new HypreParMatrix();
    
      H1stiff->FormSystemMatrix(ess_tdof_list, *ifH1stiff);
      Operator *ifH1massSP = new STRUMPACKRowLocMatrix(*ifH1stiff);
      ifH1massInv[interfaceIndex] = CreateStrumpackSolver(ifH1massSP, iH1fespace[interfaceIndex]->GetComm());
    }
    
    // Mixed interface operator
    ParMixedBilinearForm *NDH1grad = new ParMixedBilinearForm(iH1fespace[interfaceIndex], ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
    NDH1grad->AddDomainIntegrator(new MixedVectorGradientIntegrator(one));
    NDH1grad->Assemble();
    NDH1grad->Finalize();

    //ifNDH1grad[interfaceIndex] = new HypreParMatrix();
    //NDH1grad.FormSystemMatrix(ess_tdof_list, *(ifNDH1grad[interfaceIndex]));
    ifNDH1grad[interfaceIndex] = NDH1grad->ParallelAssemble();
    ifNDH1gradT[interfaceIndex] = ifNDH1grad[interfaceIndex]->Transpose();
    
    cout << myid << ": interface " << interfaceIndex << ", ND true size " << ifespace[interfaceIndex]->GetTrueVSize()
	 << ", H1 true size " << iH1fespace[interfaceIndex]->GetTrueVSize()
	 << ", NDH1 height " << ifNDH1grad[interfaceIndex]->Height() << ", width " << ifNDH1grad[interfaceIndex]->Width() << endl;
  }
  
  // Create operator C_{sd0,sd1} in the block space corresponding to [u_m^s, f_i, \rho_i]. Note that the u_m^I blocks are omitted (just zeros).
  Operator* CreateCij(const int localInterfaceIndex, const int orientation)
  {
    const int sd0 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].FirstSubdomain() : (*localInterfaces)[localInterfaceIndex].SecondSubdomain();
    const int sd1 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].SecondSubdomain() : (*localInterfaces)[localInterfaceIndex].FirstSubdomain();

    const int interfaceIndex = globalInterfaceIndex[localInterfaceIndex];

    /* REMOVE THIS
    // Nedelec interface operators
    ParBilinearForm *a = new ParBilinearForm(ifespace[interfaceIndex]);

    ConstantCoefficient one(1.0);

    a->AddDomainIntegrator(new CurlCurlIntegrator(one));
    a->AddDomainIntegrator(new VectorFEMassIntegrator(one));
    a->Assemble();
    
    HypreParMatrix A;
    Array<int> ess_tdof_list;  // empty
    a->FormSystemMatrix(ess_tdof_list, A);
    */
    
    if (orientation == 0)
      {
	rowTrueOffsetsIF[localInterfaceIndex].resize(2);
	colTrueOffsetsIF[localInterfaceIndex].resize(2);
      }
    
    rowTrueOffsetsIF[localInterfaceIndex][orientation].SetSize(3);  // Number of blocks + 1
    colTrueOffsetsIF[localInterfaceIndex][orientation].SetSize(4);  // Number of blocks + 1
    
    rowTrueOffsetsIF[localInterfaceIndex][orientation][0] = 0;
    colTrueOffsetsIF[localInterfaceIndex][orientation][0] = 0;

    /*
    // This is larger than it needs to be for this interface, because the solution space has DOF's on the entire subdomain boundaries.
    rowTrueOffsetsIF[localInterfaceIndex][1] = tdofsBdry[sd0].size();
    colTrueOffsetsIF[localInterfaceIndex][1] = tdofsBdry[sd1].size();
    */
    
    rowTrueOffsetsIF[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsetsIF[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();

    rowTrueOffsetsIF[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsetsIF[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize();

    //rowTrueOffsetsIF[localInterfaceIndex][3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsetsIF[localInterfaceIndex][orientation][3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    
    rowTrueOffsetsIF[localInterfaceIndex][orientation].PartialSum();
    colTrueOffsetsIF[localInterfaceIndex][orientation].PartialSum();
    
    BlockOperator *op = new BlockOperator(rowTrueOffsetsIF[localInterfaceIndex][orientation], colTrueOffsetsIF[localInterfaceIndex][orientation]);

    // In PengLee2012 notation, (sd0,sd1) = (m,n).
    
    // In PengLee2012 C_{mn}^{SS} corresponds to
    // -alpha <\pi_{mn}(v_m), [[u]]_{mn}>_{S_{mn}} +
    // -beta <curl_\tau \pi_{mn}(v_m), curl_\tau [[u]]_{mn}>_{S_{mn}}
    // Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the C_{mn}^{SS} block is the part
    // alpha <\pi_{mn}(v_m), \pi_{nm}(u_n)>_{S_{mn}} +
    // beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}}
    // This is an interface mass plus curl-curl stiffness matrix.

    op->SetBlock(0, 0, new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, alpha, beta));

    // In PengLee2012 C_{mn}^{SF} corresponds to
    // -<\pi_{mn}(v_m), -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
    // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the C_{mn}^{SF} block is the part
    // -<\pi_{mn}(v_m), \mu_{rn}^{-1} f_{nm}>_{S_{mn}}
    // This is an interface mass matrix.
    
    op->SetBlock(0, 1, ifNDmass[interfaceIndex], -1.0);

    // In PengLee2012 C_{mn}^{S\rho} corresponds to
    // -\gamma <\pi_{mn}(v_m), \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
    // Since <<\rho>>_{mn} = \rho_m + \rho_n, the C_{mn}^{S\rho} block is the part
    // -\gamma <\pi_{mn}(v_m), \nabla_\tau \rho_n >_{S_{mn}}
    // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
    
    op->SetBlock(0, 2, ifNDH1grad[interfaceIndex], -gamma);

    // In PengLee2012 C_{mn}^{FS} corresponds to
    // <w_m, [[u]]_{mn}>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau [[u]]_{mn}>_{S_{mn}}
    // Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the C_{mn}^{FS} block is the part
    // -<w_m, \pi_{nm}(u_n)>_{S_{mn}} - beta/alpha <curl_\tau w_m, curl_\tau \pi_{nm}(u_n)>_{S_{mn}}
    // This is an interface mass plus curl-curl stiffness matrix.

    op->SetBlock(1, 0, new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, -1.0, -beta / alpha));

    // In PengLee2012 C_{mn}^{FF} corresponds to
    // alpha^{-1} <w_m, -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
    // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the C_{mn}^{FF} block is the part
    // alpha^{-1} <w_m, \mu_{rn}^{-1} f_{nm}>_{S_{mn}}
    // This is an interface mass matrix.
    
    op->SetBlock(1, 1, ifNDmass[interfaceIndex], 1.0 / alpha);

    // In PengLee2012 C_{mn}^{F\rho} corresponds to
    // gamma / alpha <w_m, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
    // Since <<\rho>>_{mn} = \rho_m + \rho_n, the C_{mn}^{F\rho} block is the part
    // gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
    // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.

    op->SetBlock(1, 2, ifNDH1grad[interfaceIndex], gamma / alpha);

    // Row 2 is just zeros.
    
    return op;
  }

  // Create operator C_{sd0,sd1} R_{sd1}^T. The operator returned here is of size n_{sd0} by n_{sd1}, where n_{sd} is the sum of
  // tdofsBdry[sd].size() and ifespace[interfaceIndex]->GetTrueVSize() and iH1fespace[interfaceIndex]->GetTrueVSize() for all interfaces of subdomain sd.
  Operator* CreateInterfaceOperator(const int localInterfaceIndex, const int orientation)
  {
    const int sd0 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].FirstSubdomain() : (*localInterfaces)[localInterfaceIndex].SecondSubdomain();
    const int sd1 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].SecondSubdomain() : (*localInterfaces)[localInterfaceIndex].FirstSubdomain();

    const int interfaceIndex = globalInterfaceIndex[localInterfaceIndex];
    
    MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
    MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");

    // Find interface indices with respect to subdomains sd0 and sd1.
    int sd0if = -1;
    int sd1if = -1;

    int sd0os = 0;
    int sd1os = 0;
    
    int sd0osComp = 0;
    int sd1osComp = 0;
    
    for (int i=0; i<subdomainLocalInterfaces[sd0].size(); ++i)
      {
	if (subdomainLocalInterfaces[sd0][i] == interfaceIndex)
	  {
	    MFEM_VERIFY(sd0if == -1, "");
	    sd0if = i;
	  }

	if (sd0if == -1)
	  sd0os += ifespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize();
	else
	  sd0osComp += ifespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize();
      }

    for (int i=0; i<subdomainLocalInterfaces[sd1].size(); ++i)
      {
	if (subdomainLocalInterfaces[sd1][i] == interfaceIndex)
	  {
	    MFEM_VERIFY(sd1if == -1, "");
	    sd1if = i;
	  }

	if (sd1if == -1)
	  sd1os += ifespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize();
	else
	  sd1osComp += ifespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize();
      }
    
    MFEM_VERIFY(sd0if >= 0, "");
    MFEM_VERIFY(sd1if >= 0, "");

    sd0osComp -= ifespace[interfaceIndex]->GetTrueVSize();
    sd1osComp -= ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();

    Operator *Cij = CreateCij(localInterfaceIndex, orientation);

    // Cij is in the local interface space only, mapping from (u^s, f_i, \rho_i) space to (u^s, f_i) space.

    // Compose Cij on the left and right with injection operators between the subdomain surfaces and the interface.

    // Create right injection operator for sd1.

    if (orientation == 0)
      {
	rowTrueOffsetsIFR[localInterfaceIndex].resize(2);
	colTrueOffsetsIFR[localInterfaceIndex].resize(2);
	rowTrueOffsetsIFL[localInterfaceIndex].resize(2);
	colTrueOffsetsIFL[localInterfaceIndex].resize(2);

	rowTrueOffsetsIFBR[localInterfaceIndex].resize(2);
	colTrueOffsetsIFBR[localInterfaceIndex].resize(2);
	rowTrueOffsetsIFBL[localInterfaceIndex].resize(2);
	colTrueOffsetsIFBL[localInterfaceIndex].resize(2);
      }
    
    const int numBlocks = 2;  // 1 for the subdomain surface, 1 for the interface (f_{mn} and \rho_{mn}).
    rowTrueOffsetsIFR[localInterfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1
    colTrueOffsetsIFR[localInterfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1

    rowTrueOffsetsIFR[localInterfaceIndex][orientation] = 0;
    rowTrueOffsetsIFR[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
    rowTrueOffsetsIFR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
    
    rowTrueOffsetsIFR[localInterfaceIndex][orientation].PartialSum();
    
    colTrueOffsetsIFR[localInterfaceIndex][orientation] = 0;
    colTrueOffsetsIFR[localInterfaceIndex][orientation][1] = tdofsBdry[sd1].size();
    colTrueOffsetsIFR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
    
    colTrueOffsetsIFR[localInterfaceIndex][orientation].PartialSum();
    
    BlockOperator *rightInjection = new BlockOperator(rowTrueOffsetsIFR[localInterfaceIndex][orientation], colTrueOffsetsIFR[localInterfaceIndex][orientation]);

    rightInjection->SetBlock(0, 0, new ProductOperator(new TransposeOperator(InterfaceToSurfaceInjection[sd1][sd1if]),
						       tdofsBdryInjection[sd1], false, false));
    rightInjection->SetBlock(1, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize()));

    // Create left injection operator for sd0.

    rowTrueOffsetsIFL[localInterfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1
    colTrueOffsetsIFL[localInterfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1

    rowTrueOffsetsIFL[localInterfaceIndex][orientation] = 0;
    rowTrueOffsetsIFL[localInterfaceIndex][orientation][1] = tdofsBdry[sd0].size();
    rowTrueOffsetsIFL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize(); // + iH1fespace[interfaceIndex]->GetTrueVSize();
    //rowTrueOffsetsIFL[localInterfaceIndex][orientation][3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    
    rowTrueOffsetsIFL[localInterfaceIndex][orientation].PartialSum();
    
    colTrueOffsetsIFL[localInterfaceIndex][orientation] = 0;
    colTrueOffsetsIFL[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsetsIFL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize(); // + iH1fespace[interfaceIndex]->GetTrueVSize();
    
    colTrueOffsetsIFL[localInterfaceIndex][orientation].PartialSum();
    
    BlockOperator *leftInjection = new BlockOperator(rowTrueOffsetsIFL[localInterfaceIndex][orientation], colTrueOffsetsIFL[localInterfaceIndex][orientation]);

    leftInjection->SetBlock(0, 0, new ProductOperator(tdofsBdryInjectionTranspose[sd0], InterfaceToSurfaceInjection[sd0][sd0if], false, false));
    leftInjection->SetBlock(1, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));

    TripleProductOperator *CijS = new TripleProductOperator(leftInjection, Cij, rightInjection, false, false, false);

    // CijS maps from (u^s, f_i, \rho_i) space to (u^s, f_i) space.

    // Create block injection operator from (u^s, f_i) to (u^s, f_i, \rho_i) on sd0, where the range is over all sd0 interfaces.
    
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation].SetSize(4 + 1);  // Number of blocks + 1
    colTrueOffsetsIFBL[localInterfaceIndex][orientation].SetSize(2 + 1);  // Number of blocks + 1

    rowTrueOffsetsIFBL[localInterfaceIndex][orientation] = 0;
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation][1] = tdofsBdry[sd0].size();
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation][2] = sd0os;
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation][3] = ifespace[interfaceIndex]->GetTrueVSize();
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation][4] = sd0osComp;
    
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation].PartialSum();
    
    colTrueOffsetsIFBL[localInterfaceIndex][orientation] = 0;
    colTrueOffsetsIFBL[localInterfaceIndex][orientation][1] = tdofsBdry[sd0].size();
    colTrueOffsetsIFBL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize();
    
    colTrueOffsetsIFBL[localInterfaceIndex][orientation].PartialSum();
    
    BlockOperator *blockInjectionLeft = new BlockOperator(rowTrueOffsetsIFBL[localInterfaceIndex][orientation], colTrueOffsetsIFBL[localInterfaceIndex][orientation]);

    blockInjectionLeft->SetBlock(0, 0, new IdentityOperator(tdofsBdry[sd0].size()));
    blockInjectionLeft->SetBlock(2, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));

    // Create block injection operator from (u^s, f_i, \rho_i) to (u^s, f_i, \rho_i) on sd1, where the domain is over all sd1 interfaces
    // and the range is only this one interface.

    rowTrueOffsetsIFBR[localInterfaceIndex][orientation].SetSize(2 + 1);  // Number of blocks + 1
    colTrueOffsetsIFBR[localInterfaceIndex][orientation].SetSize(4 + 1);  // Number of blocks + 1

    rowTrueOffsetsIFBR[localInterfaceIndex][orientation] = 0;
    rowTrueOffsetsIFBR[localInterfaceIndex][orientation][1] = tdofsBdry[sd1].size();
    rowTrueOffsetsIFBR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
    
    rowTrueOffsetsIFBR[localInterfaceIndex][orientation].PartialSum();
    
    colTrueOffsetsIFBR[localInterfaceIndex][orientation] = 0;
    colTrueOffsetsIFBR[localInterfaceIndex][orientation][1] = tdofsBdry[sd1].size();
    colTrueOffsetsIFBR[localInterfaceIndex][orientation][2] = sd1os;
    colTrueOffsetsIFBR[localInterfaceIndex][orientation][3] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsetsIFBR[localInterfaceIndex][orientation][4] = sd1osComp;
    
    colTrueOffsetsIFBR[localInterfaceIndex][orientation].PartialSum();
    
    BlockOperator *blockInjectionRight = new BlockOperator(rowTrueOffsetsIFBR[localInterfaceIndex][orientation], colTrueOffsetsIFBR[localInterfaceIndex][orientation]);

    blockInjectionRight->SetBlock(0, 0, new IdentityOperator(tdofsBdry[sd1].size()));
    blockInjectionRight->SetBlock(1, 2, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize()));

    return new TripleProductOperator(blockInjectionLeft, CijS, blockInjectionRight, false, false, false);
  }

  void CreateSubdomainMatrices(const int subdomain)
  {
    ConstantCoefficient one(1.0);
    ConstantCoefficient minusk2(-k2);

    //fespace[subdomain]->GetComm()
    bf_sdND[subdomain] = new ParBilinearForm(fespace[subdomain]);  // TODO: make this a class member and delete at the end.
    bf_sdND[subdomain]->AddDomainIntegrator(new CurlCurlIntegrator(one));
    bf_sdND[subdomain]->AddDomainIntegrator(new VectorFEMassIntegrator(minusk2));

    bf_sdND[subdomain]->Assemble();

    sdND[subdomain] = new HypreParMatrix();

    Array<int> ess_tdof_list;  // empty
    bf_sdND[subdomain]->FormSystemMatrix(ess_tdof_list, *(sdND[subdomain]));

    /*
    {
      Vector zero(3);
      zero = 0.0;
      VectorConstantCoefficient vcc(zero);
      ParLinearForm *b = new ParLinearForm(fespace[subdomain]);
      b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(vcc));
      b->Assemble();
      
      ParGridFunction xsd(fespace[subdomain]);
      xsd.ProjectCoefficient(vcc);
      
      Vector sdB, sdX;
      a.FormLinearSystem(ess_tdof_list, xsd, *b, *(sdND[subdomain]), sdX, sdB);
      delete b;
    }
    */
    
    // Add sum over all interfaces of 
    // -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m)>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{mn}(u_m)>_{S_{mn}}
    
    //MFEM_VERIFY(false, "TODO: add boundary terms");
  }

  STRUMPACKSolver* CreateStrumpackSolver(Operator *Arow, MPI_Comm comm)
  {
    //STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, comm);
    STRUMPACKSolver * strumpack = new STRUMPACKSolver(0, NULL, comm);
    strumpack->SetPrintFactorStatistics(true);
    strumpack->SetPrintSolveStatistics(false);
    strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
    strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
    strumpack->SetOperator(*Arow);
    strumpack->SetFromCommandLine();
    return strumpack;
  }
  
  Solver* CreateSubdomainPreconditionerStrumpack(const int subdomain)
  {
    const bool sdNull = (fespace[subdomain] == NULL);

    if (sdNull)
      {
	sdNDinv[subdomain] = NULL;
      }
    else
      {
	Operator *A_subdomain = new STRUMPACKRowLocMatrix(*(sdND[subdomain]));
	sdNDinv[subdomain] = CreateStrumpackSolver(A_subdomain, fespace[subdomain]->GetComm());
      }
    
    BlockDiagonalPreconditioner *op = new BlockDiagonalPreconditioner(trueOffsetsSD[subdomain]);

    op->SetDiagonalBlock(0, sdNDinv[subdomain]);

    for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
      {
	const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

	// Diagonal blocks

	// Inverse of A_m^{FF}, which corresponds to
	// 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
	// Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
	// 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
	// This is an interface mass matrix.

	/*
	Operator *A_FF_scaled = new STRUMPACKRowLocMatrix(*(ifNDmass[interfaceIndex]));  // Factor 1/alpha is inverted separately as a scalar multiple. 
	Operator *A_FF_scaled_solver = CreateStrumpackSolver(A_FF_scaled, ifespace[interfaceIndex]->GetComm());
	ScaledOperator *A_FF_solver = new ScaledOperator(A_FF_scaled_solver, alpha);
	*/
	ScaledOperator *A_FF_solver = new ScaledOperator(ifNDmassInv[interfaceIndex], alpha);
	
	op->SetDiagonalBlock((2*i) + 1, A_FF_solver);
	
	// Inverse of A_m^{\rho\rho}, which corresponds to
	// <\psi_m, \rho_m>_{S_{mn}}
	// This is an interface H^1 mass matrix.

	/*
	Operator *A_rr = new STRUMPACKRowLocMatrix(*(ifH1mass[interfaceIndex]));
	Operator *A_rr_solver = CreateStrumpackSolver(A_rr, iH1fespace[interfaceIndex]->GetComm());

	op->SetDiagonalBlock((2*i) + 2, A_rr_solver);
	*/
	
	op->SetDiagonalBlock((2*i) + 2, ifH1massInv[interfaceIndex]);
      }

    return op;
  }

  void SetOffsetsSD(const int subdomain)
  {
    const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).
    trueOffsetsSD[subdomain].SetSize(numBlocks + 1);  // Number of blocks + 1
    //for (int i=0; i<numBlocks + 1; ++i)
    //trueOffsetsSD[subdomain][i] = 0;

    const bool sdNull = (fespace[subdomain] == NULL);
    
    trueOffsetsSD[subdomain] = 0;
    trueOffsetsSD[subdomain][1] = (!sdNull) ? fespace[subdomain]->GetTrueVSize() : 0;
    
    for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
      {
	const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];
	
	MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
	MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");
	
	trueOffsetsSD[subdomain][(2*i) + 2] += ifespace[interfaceIndex]->GetTrueVSize();
	trueOffsetsSD[subdomain][(2*i) + 3] += iH1fespace[interfaceIndex]->GetTrueVSize();
      }
    
    trueOffsetsSD[subdomain].PartialSum();
  }
  
  //#define SCHURCOMPSD
  
  // Create operator A_m for subdomain m, in the block space corresponding to [u_m, f_m^s, \rho_m^s].
  // We use mappings between interface and subdomain boundary DOF's, so there is no need for interior and surface blocks on each subdomain.
  Operator* CreateSubdomainOperator(const int subdomain)
  {
    BlockOperator *op = new BlockOperator(trueOffsetsSD[subdomain]);
    op->SetBlock(0, 0, sdND[subdomain]);

    for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
      {
	const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

	// In PengLee2012 A_m^{SF} corresponds to
	// -<\pi_{mn}(v_m), -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
	// Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{SF} block is 0.  TODO: verify this. The paper does not say this block is 0.
	
	// op->SetBlock(0, (2*i) + 1, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDmass[interfaceIndex], false, false), 1.0 / alpha);

	// In PengLee2012 A_m^{S\rho} corresponds to
	// -\gamma <\pi_{mn}(v_m), \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
	// Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{S\rho} block is the part
	// -\gamma <\pi_{mn}(v_m), \nabla_\tau \rho_m >_{S_{mn}}
	// The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
	
#ifdef SCHURCOMPSD
	// Modify the A_m^{SF} block by subtracting A_m^{S\rho} A_m^{\rho\rho}^{-1} A_m^{\rho F}; drop the A_m^{S\rho} block
	op->SetBlock(0, (2*i) + 1, new TripleProductOperator(new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDH1grad[interfaceIndex],
										 false, false),
							     ifH1massInv[interfaceIndex], ifNDH1gradT[interfaceIndex], false, false, false), gamma);
#else
	op->SetBlock(0, (2*i) + 2, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDH1grad[interfaceIndex], false, false), -gamma);
#endif
	
	// In PengLee2012 A_m^{F\rho} corresponds to
	// gamma / alpha <w_m^s, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
	// Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{F\rho} block is the part
	// gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
	// The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.

#ifdef SCHURCOMPSD
	// Modify the A_m^{FF} block by subtracting A_m^{F\rho} A_m^{\rho\rho}^{-1} A_m^{\rho F}; drop the A_m^{F\rho} block
#else
	op->SetBlock((2*i) + 1, (2*i) + 2, ifNDH1grad[interfaceIndex], gamma / alpha);
#endif
	
	// In PengLee2012 A_m^{FS} corresponds to
	// <w_m^s, [[u]]_{mn}>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau [[u]]_{mn}>_{S_{mn}}
	// Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the A_m^{FS} block is the part
	// <w_m, \pi_{mn}(u_m)>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau \pi_{mn}(u_m)>_{S_{mn}}
	// This is an interface mass plus curl-curl stiffness matrix.

	op->SetBlock((2*i) + 1, 0, new ProductOperator(new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, 1.0, beta / alpha), new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]), false, false));

	// In PengLee2012 A_m^{\rho F} corresponds to
	// <\nabla_\tau \psi_m, \mu_{rm}^{-1} f_{mn}>_{S_{mn}}
	// The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
	//op->SetBlock((2*i) + 2, (2*i) + 1, new TransposeOperator(ifNDH1grad[interfaceIndex]));  // TODO: Without this block, the block diagonal preconditioner works very well!
	op->SetBlock((2*i) + 2, (2*i) + 1, ifNDH1gradT[interfaceIndex]);
	
	// Diagonal blocks

	// In PengLee2012 A_m^{FF} corresponds to
	// 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
	// Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
	// 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
	// This is an interface mass matrix.
#ifdef SCHURCOMPSD
	// Modify the A_m^{FF} block by subtracting A_m^{F\rho} A_m^{\rho\rho}^{-1} A_m^{\rho F}; drop the A_m^{F\rho} block
	op->SetBlock((2*i) + 1, (2*i) + 1, new SumOperator(ifNDmass[interfaceIndex],
							   new TripleProductOperator(ifNDH1grad[interfaceIndex], 
										     ifH1massInv[interfaceIndex],
										     ifNDH1gradT[interfaceIndex],
										     false, false, false),
							   false, false, 1.0 / alpha, -gamma / alpha));
#else
	op->SetBlock((2*i) + 1, (2*i) + 1, ifNDmass[interfaceIndex], 1.0 / alpha);
#endif
	
	// In PengLee2012 A_m^{\rho\rho} corresponds to
	// <\psi_m, \rho_m>_{S_{mn}}
	// This is an interface H^1 mass matrix.

	op->SetBlock((2*i) + 2, (2*i) + 2, ifH1mass[interfaceIndex]);
	
	// TODO: should we equate redundant corner DOF's for f and \rho?
      }
    
    return op;
  }
  
  // This is the same operator as CreateSubdomainOperator, except it is stored as a strumpack matrix rather than a block operator. 
  Operator* CreateSubdomainOperatorStrumpack(const int subdomain)
  {
    const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).

    /*
    const int num_loc_rows = trueOffsetsSD[subdomain][numBlocks];

    int nprocs, rank;
    MPI_Comm_rank(fespace[subdomain]->GetComm(), &rank);
    MPI_Comm_size(fespace[subdomain]->GetComm(), &nprocs);

    int *all_num_loc_rows = new int[nprocs];

    MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows, 1, MPI_INT, fespace[subdomain]->GetComm());

    int first_loc_row = 0;
    int glob_nrows = 0;
    for (int i=0; i<nprocs; ++i)
      {
	glob_nrows += all_num_loc_rows[i];
	if (i < rank)
	  first_loc_row += all_num_loc_rows[i];
      }

    delete all_num_loc_rows;
    
    const int glob_ncols = glob_nrows;
    */
    
    Array2D<HypreParMatrix*> blocks(numBlocks, numBlocks);
    Array2D<std::vector<int>*> blockLeftInjection(numBlocks, numBlocks);
    Array2D<std::vector<int>*> blockRightInjection(numBlocks, numBlocks);
    Array2D<double> blockCoefficient(numBlocks, numBlocks);

    for (int i=0; i<numBlocks; ++i)
      {
	for (int j=0; j<numBlocks; ++j)
	  {
	    blocks(i, j) = NULL;
	    blockLeftInjection(i, j) = NULL;
	    blockRightInjection(i, j) = NULL;
	    blockCoefficient(i, j) = 1.0;
	  }
      }

    // Set blocks
    blocks(0, 0) = sdND[subdomain];

    for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
      {
	const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

	// In PengLee2012 A_m^{SF} corresponds to
	// -<\pi_{mn}(v_m), -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
	// Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{SF} block is 0.  TODO: verify this. The paper does not say this block is 0.
	
	// op->SetBlock(0, (2*i) + 1, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDmass[interfaceIndex], false, false), 1.0 / alpha);

	// In PengLee2012 A_m^{S\rho} corresponds to
	// -\gamma <\pi_{mn}(v_m), \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
	// Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{S\rho} block is the part
	// -\gamma <\pi_{mn}(v_m), \nabla_\tau \rho_m >_{S_{mn}}
	// The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
    
	// op->SetBlock(0, (2*i) + 2, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDH1grad[interfaceIndex], false, false), -gamma);
	//blocks(0, (2*i) + 2) = sdifNDH1grad[interfaceIndex];
	blockCoefficient(0, (2*i) + 2) = -gamma;
	//blockLeftInjection(0, (2*i) + 2) = &(InterfaceToSurfaceInjectionData[subdomain][i]);

	// TODO: set factor and permutation and transpose

	//MFEM_VERIFY(false, "TODO");

	
	// In PengLee2012 A_m^{F\rho} corresponds to
	// gamma / alpha <w_m^s, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
	// Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{F\rho} block is the part
	// gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
	// The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.

	// op->SetBlock((2*i) + 1, (2*i) + 2, ifNDH1grad[interfaceIndex], gamma / alpha);
	blocks((2*i) + 1, (2*i) + 2) = ifNDH1grad[interfaceIndex];
	blockCoefficient((2*i) + 1, (2*i) + 2) = gamma / alpha;
	
	// In PengLee2012 A_m^{FS} corresponds to
	// <w_m^s, [[u]]_{mn}>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau [[u]]_{mn}>_{S_{mn}}
	// Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the A_m^{FS} block is the part
	// <w_m, \pi_{mn}(u_m)>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau \pi_{mn}(u_m)>_{S_{mn}}
	// This is an interface mass plus curl-curl stiffness matrix.

	// op->SetBlock((2*i) + 1, 0, new ProductOperator(new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, 1.0, beta / alpha), new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]), false, false));

	blocks((2*i) + 1, 0) = ifND_FS[interfaceIndex];

	// In PengLee2012 A_m^{\rho F} corresponds to
	// <\nabla_\tau \psi_m, \mu_{rm}^{-1} f_{mn}>_{S_{mn}}
	// The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
	// op->SetBlock((2*i) + 2, (2*i) + 1, new TransposeOperator(ifNDH1grad[interfaceIndex]));
	
	blocks((2*i) + 2, (2*i) + 1) = ifNDH1gradT[interfaceIndex];
	
	// Diagonal blocks

	// In PengLee2012 A_m^{FF} corresponds to
	// 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
	// Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
	// 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
	// This is an interface mass matrix.

	//op->SetBlock((2*i) + 1, (2*i) + 1, ifNDmass[interfaceIndex], 1.0 / alpha);
	blocks((2*i) + 1, (2*i) + 1) = ifNDmass[interfaceIndex];
	blockCoefficient((2*i) + 1, (2*i) + 1) = 1.0 / alpha;
	
	// In PengLee2012 A_m^{\rho\rho} corresponds to
	// <\psi_m, \rho_m>_{S_{mn}}
	// This is an interface H^1 mass matrix.

	//op->SetBlock((2*i) + 2, (2*i) + 2, ifH1mass[interfaceIndex]);
	blocks((2*i) + 2, (2*i) + 2) = ifH1mass[interfaceIndex];
	
	// TODO: should we equate redundant corner DOF's for f and \rho?
      }

    return CreateStrumpackMatrixFromHypreBlocks(fespace[subdomain]->GetComm(), trueOffsetsSD[subdomain],
						blocks, blockLeftInjection, blockRightInjection, blockCoefficient);
  }
  
};
  
#endif  // DDOPER_HPP
