#ifndef DDOPER_HPP
#define DDOPER_HPP

#include "mfem.hpp"

using namespace mfem;
using namespace std;

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
      MFEM_VERIFY(ite != pmeshFaceToElem.end(), "");
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
	  ifMesh->GetFaceEdges(i, ifEdge, ifOri);
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
      const int nf = fec->DofForGeometry(ifMesh->GetFaceGeometryType(0));
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

class InjectionOperator : public Operator
{
private:
  int *id;  // Size should be width.
  mutable ParGridFunction gf;
  int fullWidth;
  
public:
  InjectionOperator(const int height, ParFiniteElementSpace *domainSpace, int *a) : Operator(height, domainSpace->GetTrueVSize()),
										    fullWidth(domainSpace->GetVSize()), id(a), gf(domainSpace)
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
      y[id[i]] = gf[i];
  }

  virtual void MultTranspose(const Vector &x, Vector &y) const
  {
    for (int i=0; i<fullWidth; ++i)
      gf[i] = x[id[i]];

    gf.GetTrueDofs(y);
  }
};

class DDMInterfaceOperator : public Operator
{
public:
  DDMInterfaceOperator(const int numSubdomains_, const int numInterfaces_, ParMesh *pmesh, ParMesh **pmeshSD_, ParMesh **pmeshIF_,
		       const int orderND, const int spaceDim, std::vector<SubdomainInterface> *localInterfaces_,
		       std::vector<int> *interfaceLocalIndex_) :
    numSubdomains(numSubdomains_), numInterfaces(numInterfaces_), pmeshSD(pmeshSD_), pmeshIF(pmeshIF_), fec(orderND, spaceDim),
    fecbdry(orderND, spaceDim-1), fecbdryH1(orderND, spaceDim-1), localInterfaces(localInterfaces_), interfaceLocalIndex(interfaceLocalIndex_),
    subdomainLocalInterfaces(numSubdomains_),
    k2(250.0),
    alpha(1.0), beta(1.0), gamma(1.0)  // TODO: set these to the right values
  {
    MFEM_VERIFY(numSubdomains > 0, "");
    MFEM_VERIFY(interfaceLocalIndex->size() == numInterfaces, "");
    
    fespace = new ParFiniteElementSpace*[numSubdomains];
    Asd = new Operator*[numSubdomains];
    sdND = new HypreParMatrix*[numSubdomains];
    ifespace = numInterfaces > 0 ? new ParFiniteElementSpace*[numInterfaces] : NULL;
    iH1fespace = numInterfaces > 0 ? new ParFiniteElementSpace*[numInterfaces] : NULL;

    ifNDmass = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
    ifH1mass = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
    ifNDcurlcurl = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
    ifNDH1grad = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
    
    numLocalInterfaces = localInterfaces->size();
    globalInterfaceIndex.resize(numLocalInterfaces);
    globalInterfaceIndex.assign(numLocalInterfaces, -1);
    
    block_trueOffsets.SetSize(numSubdomains + 1); // number of blocks + 1
    block_trueOffsets = 0;

    int size = 0;

    tdofsBdry.resize(numSubdomains);

    // For each subdomain parallel finite element space, determine all the true DOF's on the entire boundary. Also for each interface parallel finite element space, determine the number of true DOF's. Note that a true DOF on the boundary of a subdomain may coincide with an interface DOF that is not necessarily a true DOF on the corresponding interface mesh. The size of DDMInterfaceOperator will be the sum of the numbers of true DOF's on the subdomain mesh boundaries and interfaces. 
    
    for (int m=0; m<numSubdomains; ++m)
      {
	if (pmeshSD[m] == NULL)
	  {
	    fespace[m] = NULL;
	    Asd[m] = NULL;
	  }
	else
	  {
	    fespace[m] = new ParFiniteElementSpace(pmeshSD[m], &fec);  // Nedelec space for u_m
	    
	    FindBoundaryTrueDOFs(fespace[m], tdofsBdry[m]);  // Determine all true DOF's of fespace[m] on the boundary of pmeshSD[m], representing u_m^s.
	    size += tdofsBdry[m].size();
	    block_trueOffsets[m+1] += tdofsBdry[m].size();

	    CreateSubdomainMatrices(m);
	    Asd[m] = CreateSubdomainOperator(m);
	  }
      }

    for (int i=0; i<numInterfaces; ++i)
      {
	if (pmeshIF[i] == NULL)
	  {
	    ifespace[i] = NULL;
	    iH1fespace[i] = NULL;

	    ifNDmass[i] = NULL;
	    ifNDcurlcurl[i] = NULL;
	    ifNDH1grad[i] = NULL;
	    ifH1mass[i] = NULL;
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
    
    InterfaceToSurfaceInjection.resize(numSubdomains);
    InterfaceToSurfaceInjectionData.resize(numSubdomains);
    
    for (int m=0; m<numSubdomains; ++m)
      {
	InterfaceToSurfaceInjection[m].resize(subdomainLocalInterfaces[m].size());
	InterfaceToSurfaceInjectionData[m].resize(subdomainLocalInterfaces[m].size());
	
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];
	    
	    size += ifespace[interfaceIndex]->GetTrueVSize();
	    size += iH1fespace[interfaceIndex]->GetTrueVSize();

	    block_trueOffsets[m+1] += ifespace[interfaceIndex]->GetTrueVSize();
	    block_trueOffsets[m+1] += iH1fespace[interfaceIndex]->GetTrueVSize();

	    const int ifli = (*interfaceLocalIndex)[interfaceIndex];
	    MFEM_VERIFY(ifli >= 0, "");
	    
	    SetInterfaceToSurfaceDOFMap(ifespace[interfaceIndex], fespace[m], pmesh, m+1, (*localInterfaces)[ifli].faces, &fecbdry,
					InterfaceToSurfaceInjectionData[m][i]);
	    
	    InterfaceToSurfaceInjection[m][i] = new InjectionOperator(fespace[m]->GetTrueVSize(), ifespace[interfaceIndex],
								      &(InterfaceToSurfaceInjectionData[m][i][0]));
	  }
      }

    height = size;
    width = size;

    block_trueOffsets.PartialSum();
    MFEM_VERIFY(block_trueOffsets.Last() == size, "");

    globalOp = new BlockOperator(block_trueOffsets);

    for (int ili=0; ili<numLocalInterfaces; ++ili)
      {
	const int sd0 = (*localInterfaces)[ili].FirstSubdomain();
	const int sd1 = (*localInterfaces)[ili].SecondSubdomain();

	MFEM_VERIFY(sd0 < sd1, "");

	// Create operators for interface between subdomains sd0 and sd1, namely R_{sd0} A_{sd1}^{-1} C_{sd0,sd1} R_{sd1}^T and the other.
	globalOp->SetBlock(sd0, sd1, CreateInterfaceOperator(ili, 0));
	globalOp->SetBlock(sd1, sd0, CreateInterfaceOperator(ili, 1));
      }
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

    
  }

  
private:

  double k2;
  
  const int numSubdomains;
  int numInterfaces, numLocalInterfaces;
  
  ParMesh **pmeshSD;  // Subdomain meshes
  ParMesh **pmeshIF;  // Interface meshes
  ND_FECollection fec, fecbdry;
  H1_FECollection fecbdryH1;
  
  ParFiniteElementSpace **fespace, **ifespace, **iH1fespace;
  HypreParMatrix **ifNDmass, **ifNDcurlcurl, **ifNDH1grad, **ifH1mass;
  HypreParMatrix **sdND;
  Operator **Asd;
  
  std::vector<SubdomainInterface> *localInterfaces;
  std::vector<int> *interfaceLocalIndex;
  std::vector<int> globalInterfaceIndex;
  std::vector<std::vector<int> > subdomainLocalInterfaces;

  BlockOperator *globalOp;  // Operator for all global subdomains (blocks corresponding to non-local subdomains will be NULL).
  Array<int> block_trueOffsets;  // Offsets used in globalOp

  vector<set<int> > tdofsBdry;

  double alpha, beta, gamma;
  
  std::vector<std::vector<InjectionOperator*> > InterfaceToSurfaceInjection;
  std::vector<std::vector<std::vector<int> > > InterfaceToSurfaceInjectionData;
  
  // TODO: if the number of subdomains gets large, it may be better to define a local block operator only for local subdomains.

  void CreateInterfaceMatrices(const int interfaceIndex)
  {
    int num_procs, myid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    ConstantCoefficient one(1.0);
    Array<int> ess_tdof_list;  // empty

    // Nedelec interface operators

    ParBilinearForm NDmass(ifespace[interfaceIndex]);
    NDmass.AddDomainIntegrator(new VectorFEMassIntegrator(one));
    NDmass.Assemble();

    ParBilinearForm NDcurlcurl(ifespace[interfaceIndex]);
    NDcurlcurl.AddDomainIntegrator(new CurlCurlIntegrator(one));
    NDcurlcurl.Assemble();
    
    ifNDmass[interfaceIndex] = new HypreParMatrix();
    ifNDcurlcurl[interfaceIndex] = new HypreParMatrix();

    NDmass.FormSystemMatrix(ess_tdof_list, *(ifNDmass[interfaceIndex]));
    NDcurlcurl.FormSystemMatrix(ess_tdof_list, *(ifNDcurlcurl[interfaceIndex]));

    cout << myid << ": interface " << interfaceIndex << ", ND true size " << ifespace[interfaceIndex]->GetTrueVSize() << ", mass height "
	 << ifNDmass[interfaceIndex]->Height() << ", width " << ifNDmass[interfaceIndex]->Width() << ", ND V size "
	 << ifespace[interfaceIndex]->GetVSize() << endl;

    // H^1 interface operators

    ParBilinearForm H1mass(iH1fespace[interfaceIndex]);
    H1mass.AddDomainIntegrator(new MassIntegrator(one));
    H1mass.Assemble();

    ifH1mass[interfaceIndex] = new HypreParMatrix();
    
    H1mass.FormSystemMatrix(ess_tdof_list, *(ifH1mass[interfaceIndex]));

    // Mixed interface operator
    ParMixedBilinearForm NDH1grad(iH1fespace[interfaceIndex], ifespace[interfaceIndex]);
    NDH1grad.AddDomainIntegrator(new MixedVectorGradientIntegrator(one));
    NDH1grad.Assemble();
    NDH1grad.Finalize();

    //ifNDH1grad[interfaceIndex] = new HypreParMatrix();
    //NDH1grad.FormSystemMatrix(ess_tdof_list, *(ifNDH1grad[interfaceIndex]));
    ifNDH1grad[interfaceIndex] = NDH1grad.ParallelAssemble();

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
    
    Array<int> rowTrueOffsets(4);  // Number of blocks + 1
    Array<int> colTrueOffsets(4);  // Number of blocks + 1
    
    rowTrueOffsets[0] = 0;
    colTrueOffsets[0] = 0;

    // This is larger than it needs to be for this interface, because the solution space has DOF's on the entire subdomain boundaries.
    rowTrueOffsets[1] = tdofsBdry[sd0].size();
    colTrueOffsets[1] = tdofsBdry[sd1].size();

    rowTrueOffsets[2] = ifespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsets[2] = ifespace[interfaceIndex]->GetTrueVSize();

    rowTrueOffsets[3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsets[3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    
    rowTrueOffsets.PartialSum();
    colTrueOffsets.PartialSum();
    
    BlockOperator *op = new BlockOperator(rowTrueOffsets, colTrueOffsets);

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
  
  // Create operator R_{sd0} A_{sd1}^{-1} C_{sd0,sd1} R_{sd1}^T
  Operator* CreateInterfaceOperator(const int localInterfaceIndex, const int orientation)
  {
    Operator *Cij = CreateCij(localInterfaceIndex, orientation);
    
    Array<int> trueOffsets;
    BlockOperator *op = new BlockOperator(trueOffsets);
    return op;
  }

  void CreateSubdomainMatrices(const int subdomain)
  {
    ConstantCoefficient one(1.0);
    ConstantCoefficient minusk2(-k2);
    
    ParBilinearForm a(fespace[subdomain]);
    a.AddDomainIntegrator(new CurlCurlIntegrator(one));
    a.AddDomainIntegrator(new VectorFEMassIntegrator(minusk2));

    a.Assemble();

    sdND[subdomain] = new HypreParMatrix();

    Array<int> ess_tdof_list;  // empty
    a.FormSystemMatrix(ess_tdof_list, *(sdND[subdomain]));

    // Add sum over all interfaces of 
    // -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m)>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{mn}(u_m)>_{S_{mn}}
    
    //MFEM_VERIFY(false, "TODO: add boundary terms");
  }
  
  // Create operator A_m for subdomain m, in the block space corresponding to [u_m, f_m^s, \rho_m^s].
  // We use mappings between interface and subdomain boundary DOF's, so there is no need for interior and surface blocks on each subdomain.
  Operator* CreateSubdomainOperator(const int subdomain)
  {
    const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).
    Array<int> trueOffsets(numBlocks + 1);  // Number of blocks + 1

    trueOffsets = 0;
    trueOffsets[1] = fespace[subdomain]->GetTrueVSize();
    
    for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
      {
	const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];
	
	MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
	MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");
	
	trueOffsets[(2*i) + 2] += ifespace[interfaceIndex]->GetTrueVSize();
	trueOffsets[(2*i) + 3] += iH1fespace[interfaceIndex]->GetTrueVSize();
      }
    
    trueOffsets.PartialSum();
    
    BlockOperator *op = new BlockOperator(trueOffsets);

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
    
	op->SetBlock(0, (2*i) + 2, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDH1grad[interfaceIndex], false, false), -gamma);
	
	// In PengLee2012 A_m^{F\rho} corresponds to
	// gamma / alpha <w_m^s, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
	// Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{F\rho} block is the part
	// gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
	// The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.

	op->SetBlock((2*i) + 1, (2*i) + 2, ifNDH1grad[interfaceIndex], gamma / alpha);

	// In PengLee2012 A_m^{FS} corresponds to
	// <w_m^s, [[u]]_{mn}>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau [[u]]_{mn}>_{S_{mn}}
	// Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the A_m^{FS} block is the part
	// <w_m, \pi_{mn}(u_m)>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau \pi_{mn}(u_m)>_{S_{mn}}
	// This is an interface mass plus curl-curl stiffness matrix.

	op->SetBlock((2*i) + 1, 0, new ProductOperator(new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, 1.0, beta / alpha), new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]), false, false));

	// In PengLee2012 A_m^{\rho F} corresponds to
	// <\nabla_\tau \psi_m, \mu_{rm}^{-1} f_{mn}>_{S_{mn}}
	// The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
	op->SetBlock((2*i) + 2, (2*i) + 1, new TransposeOperator(ifNDH1grad[interfaceIndex]));

	// Diagonal blocks

	// In PengLee2012 A_m^{FF} corresponds to
	// 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
	// Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
	// 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
	// This is an interface mass matrix.

	op->SetBlock((2*i) + 1, (2*i) + 1, ifNDmass[interfaceIndex], 1.0 / alpha);

	// In PengLee2012 A_m^{\rho\rho} corresponds to
	// <\psi_m, \rho_m>_{S_{mn}}
	// This is an interface H^1 mass matrix.

	op->SetBlock((2*i) + 2, (2*i) + 2, ifH1mass[interfaceIndex]);
	
	// TODO: should we equate redundant corner DOF's for f and \rho?
      }
    
    return op;
  }
  
};
  
#endif  // DDOPER_HPP
