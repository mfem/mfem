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

class DDMInterfaceOperator : public Operator
{
public:
  DDMInterfaceOperator(const int numSubdomains_, const int numInterfaces_, ParMesh **pmeshSD_, ParMesh **pmeshIF_,
		       const int orderND, const int spaceDim, std::vector<SubdomainInterface> *localInterfaces_,
		       std::vector<int> *interfaceLocalIndex_) :
    numSubdomains(numSubdomains_), numInterfaces(numInterfaces_), pmeshSD(pmeshSD_), pmeshIF(pmeshIF_), fec(orderND, spaceDim),
    fecbdry(orderND, spaceDim-1), fecbdryH1(orderND, spaceDim-1), localInterfaces(localInterfaces_), interfaceLocalIndex(interfaceLocalIndex_),
    subdomainLocalInterfaces(numSubdomains_),
    alpha(1.0), beta(1.0), gamma(1.0)  // TODO: set these to the right values
  {
    MFEM_VERIFY(numSubdomains > 0, "");
    MFEM_VERIFY(interfaceLocalIndex->size() == numInterfaces, "");
    
    fespace = new ParFiniteElementSpace*[numSubdomains];
    ifespace = numInterfaces > 0 ? new ParFiniteElementSpace*[numInterfaces] : NULL;
    iH1fespace = numInterfaces > 0 ? new ParFiniteElementSpace*[numInterfaces] : NULL;

    ifNDmass = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
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
	  fespace[m] = NULL;
	else
	  {
	    fespace[m] = new ParFiniteElementSpace(pmeshSD[m], &fec);  // Nedelec space for u_m
	    
	    FindBoundaryTrueDOFs(fespace[m], tdofsBdry[m]);  // Determine all true DOF's of fespace[m] on the boundary of pmeshSD[m], representing u_m^s.
	    size += tdofsBdry[m].size();
	    block_trueOffsets[m+1] += tdofsBdry[m].size();
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

    for (int m=0; m<numSubdomains; ++m)
      {
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];
	    
	    size += ifespace[interfaceIndex]->GetTrueVSize();
	    size += iH1fespace[interfaceIndex]->GetTrueVSize();

	    block_trueOffsets[m+1] += ifespace[interfaceIndex]->GetTrueVSize();
	    block_trueOffsets[m+1] += iH1fespace[interfaceIndex]->GetTrueVSize();
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
  
  const int numSubdomains;
  int numInterfaces, numLocalInterfaces;
  
  ParMesh **pmeshSD;  // Subdomain meshes
  ParMesh **pmeshIF;  // Interface meshes
  ND_FECollection fec, fecbdry;
  H1_FECollection fecbdryH1;
  
  ParFiniteElementSpace **fespace, **ifespace, **iH1fespace;
  HypreParMatrix **ifNDmass, **ifNDcurlcurl, **ifNDH1grad;

  std::vector<SubdomainInterface> *localInterfaces;
  std::vector<int> *interfaceLocalIndex;
  std::vector<int> globalInterfaceIndex;
  std::vector<std::vector<int> > subdomainLocalInterfaces;

  BlockOperator *globalOp;  // Operator for all global subdomains (blocks corresponding to non-local subdomains will be NULL).
  Array<int> block_trueOffsets;  // Offsets used in globalOp

  vector<set<int> > tdofsBdry;

  double alpha, beta, gamma;
  
  // TODO: if the number of subdomains gets large, it may be better to define a local block operator only for local subdomains.

  void CreateInterfaceMatrices(const int interfaceIndex)
  {
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

    // Mixed interface operator
    ParMixedBilinearForm NDH1grad(ifespace[interfaceIndex], iH1fespace[interfaceIndex]);
    NDH1grad.AddDomainIntegrator(new MixedVectorGradientIntegrator(one));
    NDH1grad.Assemble();
    NDH1grad.Finalize();

    //ifNDH1grad[interfaceIndex] = new HypreParMatrix();
    //NDH1grad.FormSystemMatrix(ess_tdof_list, *(ifNDH1grad[interfaceIndex]));
    ifNDH1grad[interfaceIndex] = NDH1grad.ParallelAssemble();
  }
  
  // Create operator C_{sd0,sd1} in the block space corresponding to [u_m^s, f_i, \rho_i]. Note that the u_m^I block is omitted (just zeros).
  Operator* CreateCij(const int localInterfaceIndex, const int orientation)
  {
    const int sd0 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].FirstSubdomain() : (*localInterfaces)[localInterfaceIndex].SecondSubdomain();
    const int sd1 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].SecondSubdomain() : (*localInterfaces)[localInterfaceIndex].FirstSubdomain();

    const int interfaceIndex = globalInterfaceIndex[localInterfaceIndex];
    
    // Nedelec interface operators
    ParBilinearForm *a = new ParBilinearForm(ifespace[interfaceIndex]);

    ConstantCoefficient one(1.0);

    a->AddDomainIntegrator(new CurlCurlIntegrator(one));
    a->AddDomainIntegrator(new VectorFEMassIntegrator(one));
    a->Assemble();
    
    HypreParMatrix A;
    Array<int> ess_tdof_list;  // empty
    a->FormSystemMatrix(ess_tdof_list, A);
   
    Array<int> rowTrueOffsets(4);  // Number of blocks + 1
    Array<int> colTrueOffsets(4);  // Number of blocks + 1
    
    rowTrueOffsets[0] = 0;
    colTrueOffsets[0] = 0;
    
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
  

};
  
#endif  // DDOPER_HPP
