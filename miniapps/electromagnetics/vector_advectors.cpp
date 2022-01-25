#include "vector_advectors.h"

//notes
//Look at DivergenceFreeProjector to pull th divergence out of a vecrot in Hcurl
// Matrix-Vector Multiplication AddMult(x,y,val):  y = y + val*A*x, Mult(x, y): y = A*x

BFieldAdvector::BFieldAdvector(ParMesh *pmesh_old, ParMesh *pmesh_new, int order_) :
   order(order_).
   pmeshOld(nullptr),
   pmeshNew(nullptr),
   grad(nullptr),
   curl(nullptr),
   weakCurl(nullptr),
   curlCurl(nullptr),
   divFreeProj(nullptr),
   a(nullptr),
   curl_b(nullptr),
   clean_curl_b(nullptr),
   recon_b(nullptr)
{
   SetMeshes(pmesh_old, pmesh_new);
}


BFieldAdvector::SetMeshes(ParMesh *pmesh_old, ParMesh *pmesh_new)
{
   CleanInternals();

   pmeshOld = pmesh_old;
   pmeshNew = pmesh_new;

   //Set up the various spaces on the meshes
   H1FESpaceOld    = new H1_ParFESpace(pmeshOld,order,pmeshOld->Dimension());
   HCurlFESpaceOld = new ND_ParFESpace(pmeshOld,order,pmeshOld->Dimension());
   HDivFESpaceOld  = new RT_ParFESpace(pmeshOld,order,pmeshOld->Dimension());
   L2FESpaceOld    = new L2_ParFESpace(pmeshOld,order,pmeshOld->Dimension());
   H1FESpaceNew    = new H1_ParFESpace(pmeshNew,order,pmeshNew->Dimension());
   HCurlFESpaceNew = new ND_ParFESpace(pmeshNew,order,pmeshNew->Dimension());
   HDivFESpaceNew  = new RT_ParFESpace(pmeshNew,order,pmeshNew->Dimension());
   L2FESpaceNew    = new L2_ParFESpace(pmeshNew,order,pmeshNew->Dimension());

   //Discrete Differential Operators
   grad = new ParDiscreteGradOperator(H1FESpace, HCurlFESpace);
   curl = new ParDiscreteCurlOperator(HCurlFESpace, HDivFESpace);

   //Weak curl operator for taking the curl of B living in Hdiv
   ConstantCoefficient oneCoef(1.0);
   weakCurl = new ParMixedBilinearForm(HDivFESpaceOld, HCurlFESpaceOld);
   weakCurl->AddDomainIntegrator(new VectorFECurlIntegrator(oneCoef));

   //CurlCurl operator
   curlCurl  = new ParBilinearForm(HCurlFESpaceOld);
   curlCurl->AddDomainIntegrator(new CurlCurlIntegrator(*oneCoef));

   //Projector to clean the divergence out of vectors in Hcurl
   int irOrder = H1FESpaceOld->GetElementTransformation(0)->OrderW()+ 2 * order;   
   divFreeProj = new DivergenceFreeProjector(*H1FESpaceOld, *HCurlFESpaceOld,
                                              irOrder, NULL, NULL, grad);

   // Build grid internal functions on the spaces
   a  = new ParGridFunction(HCurlFESpacOld);             //Vector potential Ain HCurl
   curl_b = new ParGridFunction(HCurlFESpaceOld);        //curl B in Hcurl from the weak curl
   clean_curl_b = new ParGridFunction(HCurlFESpaceOld);  //B in Hcurl
   recon_b = new ParGridFunction(HDivFESpaceOld);        //Reconstructed B from A
}


BFieldAdvector::CleanInternals()
{
   if (H1FESpaceOld != nullptr) delete H1FESpaceOld;
   if (HCurlFESpaceOld != nullptr) delete HCurlFESpaceOld;
   if (HDivFESpaceOld != nullptr) delete HDivFESpaceOld;
   if (L2FESpaceOld != nullptr) delete L2FESpaceOld;
   if (H1FESpaceNew != nullptr) delete H1FESpaceNew;
   if (HCurlFESpaceNew != nullptr) delete HCurlFESpaceNew;
   if (HDivFESpaceNew != nullptr) delete HDivFESpaceNew;
   if (L2FESpaceNew != nullptr) delete L2FESpaceNew;

   if (grad != nullptr) delete grad;
   if (curl != nullptr) delete curl;

   if (weakCurl != nullptr) delete weakCurl;
   if (divFreeProj != nullptr) delete divFreeProj;
   if (curlCurl != nullptr) delete curlCurl;

   if (a != nullptr) delete a;
   if (curl_b != nullptr) delete curl_b;
   if (clean_curl_b != nullptr) delete clean_curl_b;
   if (recon_b != nullptr) delete recon_b;
}


BFieldAdvector::Advect(ParGridFunction* b_old, ParGridFunction* b_new)
{
   computeA(b_old);
}


//Given b_ in Hdiv compute the curl of b_ in Hcurl
//and then clean any divergence out of it
BFieldAdvector::ComputeCleanCurlB(ParGridFunction* b)
{
   weakCurl->Mult(*b, *curl_b);
   divFreeProj->Mult(*curl_b, *clean_curl_b);
}


//Solve Curl Curl A = Curl B for A using AMS
BfieldAdvector::ComputeA(ParGridFunction* b)
{
   Array<int> ess_bdr;
   Array<int> ess_bdr_tdofs;
   HCurlFESpaceOld->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);

   //Get the divergence cleaned curl of b
   computeCleanCurlB(b);

   // Apply Dirichlet BCs to matrix and right hand side and otherwise
   // prepare the linear system
   HypreParMatrix M;
   HypreParVector A(HCurlFESpacOld);
   HypreParVector RHS(HCurlFESpaceOld);

   curlCurl->FormLinearSystem(ess_bdr_tdofs, *a, *clean_curl_b, M, A, RHS);

   // Define and apply a parallel PCG solver for AX=B with the AMS
   // preconditioner from hypre.
   HypreAMS ams(M, HCurlFESpaceOld);
   ams.SetSingularProblem();

   HyprePCG pcg(M);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(50);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(ams);
   pcg.Mult(RHS, A);

   // Extract the parallel grid function corresponding to the finite
   // element approximation A. This is the local solution on each
   // processor.
   curlMuInvCurl->RecoverFEMSolution(A, clean_curl_b, *a);

   //Compute the reconstructed b field for comparison
   curl->Mult(a, recon_b);

   //
   Vector diff(*b);
   diff -= *recon_b;    //diff = b - recon_b
   std::cout << "L2 Error in reconstructed B field:  " << diff.Norl2() << std::endl;




}



