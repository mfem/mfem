#ifndef DDMESH_HPP
#define DDMESH_HPP

#include "mfem.hpp"

using namespace mfem;
using namespace std;

class SubdomainParMeshGenerator
{
private:
  int numSubdomains;
  const ParMesh *pmesh;  // global mesh

  int num_procs = -1;
  int myid = -1;
  int *cts, *offsets;
  int *sdPartition;
  int sdPartitionAlloc;
  const int d;  // Mesh dimension
  int numElVert;
  
  vector<int> procNumElems;
  vector<int> sdProcId;
  vector<int> element_vgid;
  vector<double> element_coords;
  
  H1_FECollection *h1_coll;
  ParFiniteElementSpace *H1_space;
  
  int ChooseRootForSubdomain(const int attribute, int& numElements)
  {
    numElements = 0;  // Number of elements on this process (in pmesh) in the subdomain (i.e. with the given attribute).
    for (int i=0; i<pmesh->GetNE(); ++i)
      {
	if (pmesh->GetAttribute(i) == attribute)
	  numElements++;
      }

    int idTouching = (numElements > 0) ? myid : -1;
    int maxId = -1;
    MPI_Allreduce(&idTouching, &maxId, 1, MPI_INT, MPI_MAX, pmesh->GetComm());

    MFEM_VERIFY(maxId >= 0, "");
    
    return maxId;
  }
  
  // Gather subdomain mesh data to all processes.
  void GatherSubdomainMeshData(const int root, const int attribute, const int numLocalElements,
			       int& subdomainNumElements)
  {
    MPI_Allgather(&numLocalElements, 1, MPI_INT, cts, 1, MPI_INT, MPI_COMM_WORLD);

    subdomainNumElements = 0;
    offsets[0] = 0;
    for (int i = 0; i < num_procs; ++i) {
      subdomainNumElements += cts[i];
      procNumElems[i] = cts[i];
      cts[i] *= 2;
      if (i > 0)
	offsets[i] = offsets[i-1] + cts[i-1];
    }

    MFEM_VERIFY(subdomainNumElements > 0, "");
    
    // Assumption: all elements are of the same geometric type.
    Array<int> elVert;
    pmesh->GetElementVertices(0, elVert);
    numElVert = elVert.Size();  // number of vertices per element
    MFEM_VERIFY(numElVert > 0, "");
    
    vector<int> my_element_vgid(std::max(numElVert*numLocalElements, 1));  // vertex global indices, for each element
    vector<double> my_element_coords(std::max(d*numElVert*numLocalElements, 1));

    int conn_idx = 0;
    int coords_idx = 0;
    
    for (int elId=0; elId<pmesh->GetNE(); ++elId)
      {
	if (pmesh->GetAttribute(elId) == attribute)
	  {
	    pmesh->GetElementVertices(elId, elVert);
	    MFEM_VERIFY(numElVert == elVert.Size(), "");  // Assuming a uniform element type in the mesh.
	    // NOTE: to be very careful, it should be verified that this is the same across all processes.

	    Array<int> dofs;
	    H1_space->GetElementDofs(elId, dofs);
	    MFEM_VERIFY(numElVert == dofs.Size(), "");  // Assuming a bijection between vertices and H1 dummy space DOF's.
    
	    for (int i = 0; i < numElVert; ++i)
	      {
		my_element_vgid[conn_idx++] = H1_space->GetGlobalTDofNumber(dofs[i]);
		const double* coords = pmesh->GetVertex(elVert[i]);
		for (int j=0; j<d; ++j)
		  my_element_coords[coords_idx++] = coords[j];
	      }
	  }
      }

    MFEM_VERIFY(coords_idx == d*numElVert*numLocalElements, "");
    MFEM_VERIFY(conn_idx == numElVert*numLocalElements, "");

    // Gather all the element connectivities from all processors.
    offsets[0] = 0;
    cts[0] = numElVert*procNumElems[0];
    for (int i = 1; i < num_procs; ++i) {
      cts[i] = numElVert*procNumElems[i];
      offsets[i] = offsets[i-1] + cts[i-1];
    }

    element_vgid.resize(numElVert*subdomainNumElements);
    element_coords.resize(d*numElVert*subdomainNumElements);

    int sendSize = numElVert*numLocalElements;
    
    MPI_Allgatherv(&(my_element_vgid[0]), sendSize, MPI_INT,
		   &(element_vgid[0]), cts, offsets, MPI_INT, MPI_COMM_WORLD);
    
    // Gather all the element coordinates from all processors.
    offsets[0] = 0;
    cts[0] = d*cts[0];
    for (int i = 1; i < num_procs; ++i) {
      cts[i] = d*cts[i];
      offsets[i] = offsets[i-1] + cts[i-1];
    }

    sendSize = d*numElVert*numLocalElements;
    
    int res = MPI_Allgatherv(&(my_element_coords[0]), sendSize, MPI_DOUBLE,
			     &(element_coords[0]), cts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
  }

  // This is a serial function, which should be called only processes touching the subdomain.
  Mesh* BuildSerialMesh(const int attribute, const int subdomainNumElements)
  {
    // element_vgid holds vertices as global ids.  Vertices may be shared
    // between elements so we don't know the number of unique vertices in the
    // subdomain mesh.  Find all the unique vertices and construct the map of
    // global dof ids to local dof ids (vertices).  Keep track of the number of
    // unique vertices.

    set<int> unique_gdofs;
    map<int, int> unique_gdofs_first_appearance;
    for (int i = 0; i < numElVert*subdomainNumElements; ++i) {
      int gdof = element_vgid[i];
      if (unique_gdofs.insert(gdof).second) {
	unique_gdofs_first_appearance.insert(make_pair(gdof, i));
      }
    }
    
    int subdomainNumVertices = unique_gdofs.size();
    map<int, int> unique_gdofs_2_vertex;
    map<int, int> vertex_2_unique_gdofs;
    int idx = 0;
    for (set<int>::iterator it = unique_gdofs.begin();
	 it != unique_gdofs.end(); ++it) {
      unique_gdofs_2_vertex.insert(make_pair(*it, idx));
      vertex_2_unique_gdofs.insert(make_pair(idx, *it));
      ++idx;
    }

    Mesh *sdmesh = new Mesh(d, subdomainNumVertices, subdomainNumElements);
    
    // For each unique vertex, add its coordinates to the subdomain mesh.
    for (int vert = 0; vert < subdomainNumVertices; ++vert)
      {
	int unique_gdof = vertex_2_unique_gdofs[vert];
	int first_conn_ref = unique_gdofs_first_appearance[unique_gdof];
	int coord_loc = d*first_conn_ref;
	sdmesh->AddVertex(&element_coords[coord_loc]);
      }

    if (subdomainNumElements > sdPartitionAlloc)
      {
	if (sdPartition)
	  delete [] sdPartition;

	sdPartitionAlloc = subdomainNumElements;

	sdPartition = new int[sdPartitionAlloc];
      }

    // Set the subdomain mesh communicator process indices.
    int sdProcCount = 0;
    for (int p=0; p<num_procs; ++p)
      {
	if (procNumElems[p] > 0)
	  {
	    sdProcId[p] = sdProcCount;
	    sdProcCount++;
	  }
	else
	  sdProcId[p] = -1;
      }
    
    // Now add each element and give it its attributes and connectivity.
    const int elGeom = pmesh->GetElementBaseGeometry(0);
    idx = 0;
    int ielem = 0;
    for (int p=0; p<num_procs; ++p)
      {
	for (int i=0; i<procNumElems[p]; ++i, ++ielem)
	  {
	    Element* sel = sdmesh->NewElement(elGeom);
	    sel->SetAttribute(attribute);
	  
	    Array<int> sv(numElVert);
	    for (int vert = 0; vert < numElVert; ++vert) {
	      sv[vert] = unique_gdofs_2_vertex[element_vgid[idx++]];
	    }
	    sel->SetVertices(sv);

	    sdmesh->AddElement(sel);

	    sdPartition[ielem] = sdProcId[p];
	  }
      }

    MFEM_VERIFY(ielem == subdomainNumElements, "");
    MFEM_VERIFY(idx == numElVert*subdomainNumElements, "");
    
    sdmesh->FinalizeTopology();
  
    return sdmesh;
  }
  
  Mesh* CreateSerialSubdomainMesh(const int attribute)
  {
    int numLocalElements = 0;
    const int root = ChooseRootForSubdomain(attribute, numLocalElements);
    MFEM_VERIFY(root >= 0, "");

    int subdomainNumElements = 0;
    GatherSubdomainMeshData(root, attribute, numLocalElements, subdomainNumElements);

    // Now we have enough data to build the subdomain mesh.
    Mesh *serialMesh = NULL;

    if (numLocalElements > 0)
      serialMesh = BuildSerialMesh(attribute, subdomainNumElements);
    
    return serialMesh;
  }
  
public:
  SubdomainParMeshGenerator(const int numSubdomains_, ParMesh *pmesh_) : numSubdomains(numSubdomains_), pmesh(pmesh_),
									 d(pmesh_->Dimension())
  {
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    cts = new int [num_procs];
    offsets = new int [num_procs];

    sdPartition = NULL;
    sdPartitionAlloc = 0;
    
    procNumElems.resize(num_procs);
    sdProcId.resize(num_procs);
    element_vgid.resize(1);  // Just to make it nonempty
    element_coords.resize(1);  // Just to make it nonempty

    // Define a superfluous finite element space, merely to get global vertex indices for the sample mesh construction.
    h1_coll = new H1_FECollection(1, d);  // Must be first order, to get a bijection between vertices and DOF's.
    H1_space = new ParFiniteElementSpace(pmesh_, h1_coll);  // This constructor effectively sets vertex (DOF) global indices.
  }
  
  ~SubdomainParMeshGenerator()
  {
    delete [] cts;
    delete [] offsets;
    
    if (sdPartition)
      delete [] sdPartition;
    
    delete h1_coll;
    delete H1_space;
  }
  
  ParMesh** CreateParallelSubdomainMeshes()
  {
    ParMesh **pmeshSD = new ParMesh*[numSubdomains];
    
    for (int s=0; s<numSubdomains; ++s)  // Loop over subdomains
      {
	Mesh *sdmesh = CreateSerialSubdomainMesh(s+1);

	MPI_Comm sd_com;
	int color = (sdmesh == NULL);
	const int status = MPI_Comm_split(MPI_COMM_WORLD, color, myid, &sd_com);
	MFEM_VERIFY(status == MPI_SUCCESS,
		    "Construction of hyperreduction comm failed");

	if (sdmesh != NULL)
	  {
	    pmeshSD[s] = new ParMesh(sd_com, *sdmesh, sdPartition);
	    delete sdmesh;
	  }
	else
	  pmeshSD[s] = NULL;
      }
    
    return pmeshSD;
  }
};

#endif // DDMESH_HPP
