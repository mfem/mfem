#include "linear_elasticity_solver.hpp"
#include "sbm_aux.hpp"

namespace mfem {

  LinearElasticitySolver::LinearElasticitySolver(mfem::ParMesh* mesh_, int vorder, bool useEmb, int gS, int nT, int nStrainTerms, double ghostPenCoeff, bool useMumps, bool useAS, bool vis): pmesh(mesh_), displacementOrder(vorder), useEmbedded(useEmb), geometricShape(gS), nTerms(nT), numberStrainTerms(nStrainTerms), ghostPenaltyCoefficient(ghostPenCoeff), mumps_solver(useMumps), useAnalyticalShape(useAS), visualization(vis), fdisplacement(NULL), volforce(NULL), prec(NULL),  ns(NULL), vfes(NULL), vfec(NULL), alpha_fes(NULL), alpha_fec(NULL), shearMod(NULL), bulkMod(NULL), exactDisplacement(NULL), ess_elem(pmesh->attributes.Max()), C_I(0.0), marker(NULL), neumann_dist_coef(NULL), distance_vec_space(NULL), distance(NULL), normal_vec_space(NULL), normal(NULL), level_set_gf(NULL), dist_vec(NULL), normal_vec(NULL), alphaCut(NULL)
  {
    int dim=pmesh->Dimension();
    // 5a. Define the FECollection and FESpace for the displacement and then create it  
    vfec=new H1_FECollection(vorder,dim);
    vfes=new mfem::ParFiniteElementSpace(pmesh,vfec,dim);
    vfes->ExchangeFaceNbrData();
    fdisplacement = new ParGridFunction(vfes);
    *fdisplacement=0.0;

    // 5b. Define the FECollection and FESpace for the level-set      
    mfem::FiniteElementCollection* lsvec = new H1_FECollection(vorder+2,dim);
    mfem::ParFiniteElementSpace* lsfes = new mfem::ParFiniteElementSpace(pmesh,lsvec);
    lsfes->ExchangeFaceNbrData();

    // 5c. Define the FECollection and FESpace for the volume fractions, create it and setting it to 1
    alpha_fec = new L2_FECollection(0, pmesh->Dimension());
    alpha_fes = new ParFiniteElementSpace(pmesh, alpha_fec);
    alpha_fes->ExchangeFaceNbrData();
    alphaCut = new ParGridFunction(alpha_fes);
    alphaCut->ExchangeFaceNbrData();  
    *alphaCut = 1;
    
   if (useEmbedded){
     // 5.d Create the level set grid function and the marking class
     level_set_gf =  new ParGridFunction(lsfes);
     level_set_gf->ExchangeFaceNbrData();
     marker = new AttributeShiftedFaceMarker(*pmesh, *vfes, *alpha_fes, 1);

     if (useAnalyticalShape){
       // 5.e if we are using an analytical surface to describe the geometry
       //     Dist_Coefficient just returns +1 or -1 for inside or outside domain 
       neumann_dist_coef = new Dist_Coefficient(geometricShape);
       // 5.f project the neumann dist coef to the level set gf 
       level_set_gf->ProjectCoefficient(*neumann_dist_coef);
       // Exchange information for ghost elements i.e. elements that share a face
       // with element on the current processor, but belong to another processor.
       level_set_gf->ExchangeFaceNbrData();
       // Setup the class to mark all elements based on whether they are located
       // inside or outside the true domain, or intersected by the true boundary.
       // In addition, we also mark the surrogate edges and the ghost penalty ones
       marker->MarkElements(*level_set_gf);

       // 5.g Create the distance and normal vectors 
       dist_vec = new Dist_Vector_Coefficient(dim, geometricShape);
       normal_vec = new Normal_Vector_Coefficient(dim, geometricShape);
     }
     else{
       // 5.e if we are using a level set to describe the geometry
       //     Dist_Level_Set_Coefficient just returns +1 or -1 for inside or outside domain     
       neumann_dist_coef = new Dist_Level_Set_Coefficient(geometricShape);

       // 5.f Define the FECollection and FESpace for the distance and normal vectors
       //     Create and set them to 1 
       mfem::FiniteElementCollection* d_vec = new H1_FECollection(vorder+2,dim);       
       distance_vec_space = new ParFiniteElementSpace(pmesh, d_vec, dim);
       normal_vec_space = new ParFiniteElementSpace(pmesh, d_vec, dim);
       distance_vec_space->ExchangeFaceNbrData();
       normal_vec_space->ExchangeFaceNbrData();   
       distance = new ParGridFunction(distance_vec_space);
       *distance = 0.0;
       normal = new ParGridFunction(normal_vec_space);
       *normal = 0.0;

       // 5.g project the neumann dist coef to the level set gf and smooth it.
       //    To smooth it, code is using DiffuseH1, but one can also use the PDEFilter
       //    Just uncomment the filter lines and comment out the DiffuseH1
       level_set_gf->ProjectCoefficient(*neumann_dist_coef);
       // PDEFilter filter(*pmesh, 0.1);
       //  filter.Filter(*neumann_dist_coef, *level_set_gf);
       DiffuseH1(*level_set_gf, 1.0);
       // Exchange information for ghost elements i.e. elements that share a face
       // with element on the current processor, but belong to another processor.      
       level_set_gf->ExchangeFaceNbrData();     
       // Setup the class to mark all elements based on whether they are located
       // inside or outside the true domain, or intersected by the true boundary.
       marker->MarkElements(*level_set_gf);

       // 5.h Create a grid function coefficient from the smoothed level-set
       //     Use it to populate the distance and normal grid functions
       //     Create the distance and normal vector grid function coefficients
       GridFunctionCoefficient ls_filt_coeff(level_set_gf);       
       NormalizationDistanceSolver dist_solver;
       dist_solver.ComputeVectorDistance(ls_filt_coeff, *distance); 
       dist_solver.ComputeVectorNormal(ls_filt_coeff, *distance, *normal);
       distance->ExchangeFaceNbrData();
       normal->ExchangeFaceNbrData();       
       dist_vec = new VectorGridFunctionCoefficient(distance);
       normal_vec = new VectorGridFunctionCoefficient(normal);
     }

     // 5.i Populate the volume fractions using the level_set grid function
     UpdateAlpha(*alphaCut, *vfes, *level_set_gf, geometricShape);
     alphaCut->ExchangeFaceNbrData();

     // 5.j Get the inactive dofs to remove them from the linear systerm
     Array<int> ess_tdofs;
     Array<int> ess_inactive_dofs = marker->GetEss_Vdofs();
     vfes->GetRestrictionMatrix()->BooleanMult(ess_inactive_dofs, ess_tdofs);
     vfes->MarkerToList(ess_tdofs, ess_vdofs);
       
   }
   // 5.k Create the shear and bulk moduli classes  
   shearMod = new ShearModulus(pmesh);
   bulkMod = new BulkModulus(pmesh);

   // 5.l Create a vector of size equal to the highes number attribute and set it to 1     
   const int max_elem_attr = pmesh->attributes.Max();
   ess_elem.SetSize(max_elem_attr);
   ess_elem = 1;
   //    Set the entry corresponding to the inactive attribute to 0 
   if (useEmbedded && (max_elem_attr >= 2)){
     ess_elem[AttributeShiftedFaceMarker::SBElementType::OUTSIDE-1] = 0;
   }

   switch (pmesh->GetElementBaseGeometry(0))
    {
    case Geometry::TRIANGLE:
    case Geometry::TETRAHEDRON:{
      C_I = (displacementOrder)*(displacementOrder+1)/dim;
      break;
    }
    case Geometry::SQUARE: 
    case Geometry::CUBE:{
      C_I = displacementOrder*displacementOrder;
      break;
    }
    default: MFEM_ABORT("Unknown zone type!");
    }
  }

  LinearElasticitySolver::~LinearElasticitySolver()
  {
    delete ns;
    delete prec;
    delete vfes;
    delete vfec;
    delete pmesh;  
    delete shearMod;
    delete bulkMod;
    delete fdisplacement;
    delete alphaCut;
     if (useEmbedded){
      delete marker;
      delete distance_vec_space;
      delete distance;
      delete normal_vec_space;
      delete normal;
      delete level_set_gf;
      delete dist_vec;
      delete normal_vec;
      delete neumann_dist_coef;
    }

    surf_loads.clear();
    displacement_BC.clear();
    
  }

  void LinearElasticitySolver::SetNewtonSolver(double rtol, double atol,int miter, int prt_level)
  {
    rel_tol=rtol;
    abs_tol=atol;
    max_iter=miter;
    print_level=prt_level;
  }

  void LinearElasticitySolver::SetVolForce(mfem::VectorCoefficient& fv)
  {
    volforce=&fv;
  }

  void LinearElasticitySolver::SetExactDisplacementSolution(mfem::VectorCoefficient& fv)
  {
    exactDisplacement=&fv;
  }

  void LinearElasticitySolver::FSolve()
  {
    // Set the BC
    int dim=pmesh->Dimension();
    
    double penaltyParameter_bf = 10.0*2*C_I*2;

    ParLinearForm *fform(new ParLinearForm(vfes));
    fform->AddDomainIntegrator(new WeightedVectorForceIntegrator(*alphaCut, *volforce), ess_elem);     
    for(auto it=surf_loads.begin();it!=surf_loads.end();it++)
    {
      fform->AddBdrFaceIntegrator(new WeightedTractionBCIntegrator(*alphaCut, *(it->second)),*(it->first));
    }
   
    for(auto it=displacement_BC.begin();it!=displacement_BC.end();it++)
    {    
      // Nitsche
      fform->AddBdrFaceIntegrator(new WeightedStressNitscheBCForceIntegrator(*alphaCut, *shearMod, *bulkMod, *(it->second)),*(it->first));
      // Normal Penalty
      fform->AddBdrFaceIntegrator(new WeightedNormalDisplacementBCPenaltyIntegrator(*alphaCut, penaltyParameter_bf, *bulkMod, *(it->second)), *(it->first));
      // Tangential Penalty
      fform->AddBdrFaceIntegrator(new WeightedTangentialDisplacementBCPenaltyIntegrator(*alphaCut, penaltyParameter_bf, *shearMod, *(it->second)), *(it->first));
    }
    if (useEmbedded){
      // Nitsche
      fform->AddInteriorFaceIntegrator(new WeightedShiftedStressNitscheBCForceIntegrator(pmesh, *alphaCut, *shifted_traction_BC, dist_vec, normal_vec, 1));
    }
    fform->Assemble();    
    fform->ParallelAssemble();
   
    ParBilinearForm *mVarf(new ParBilinearForm(vfes));
    mVarf->AddDomainIntegrator(new WeightedStressForceIntegrator(*alphaCut, *shearMod, *bulkMod),ess_elem);
    for(auto it=displacement_BC.begin();it!=displacement_BC.end();it++)
    {
      // Nitsche
      mVarf->AddBdrFaceIntegrator(new WeightedStressBoundaryForceIntegrator(*alphaCut, *shearMod, *bulkMod),*(it->first));
      // IP
      mVarf->AddBdrFaceIntegrator(new WeightedStressBoundaryForceTransposeIntegrator(*alphaCut, *shearMod, *bulkMod),*(it->first));
      // Normal Penalty
      mVarf->AddBdrFaceIntegrator(new WeightedNormalDisplacementPenaltyIntegrator(*alphaCut, penaltyParameter_bf, *bulkMod),*(it->first));
      // Tangential Penalty
      mVarf->AddBdrFaceIntegrator(new WeightedTangentialDisplacementPenaltyIntegrator(*alphaCut, penaltyParameter_bf, *shearMod),*(it->first));
    }
    if (useEmbedded){     
      // Nitsche
      mVarf->AddInteriorFaceIntegrator(new WeightedShiftedStressBoundaryForceIntegrator(pmesh, *alphaCut, *shearMod, *bulkMod, dist_vec, normal_vec, nTerms, 1));
      // IP
      mVarf->AddInteriorFaceIntegrator(new WeightedShiftedStressBoundaryForceTransposeIntegrator(pmesh, *alphaCut, *shearMod, *bulkMod, 1));
      // ghost penalty
      mVarf->AddInteriorFaceIntegrator(new GhostStressFullGradPenaltyIntegrator(pmesh, *shearMod, *bulkMod, ghostPenaltyCoefficient, numberStrainTerms));      
    }
  
    mVarf->Assemble();   
    mVarf->Finalize();
  
    mVarf->ParallelAssemble();  
    //  HypreParMatrix *displ_Mass = NULL;
    //  displ_Mass = mVarf->ParallelAssemble();
    
    std::cout << " Done " << std::endl;
     
    // Form the linear system and solve it.
    OperatorPtr A;
    Vector B, X;
    mVarf->FormLinearSystem(ess_vdofs, *fdisplacement, *fform, A, X, B);
     
    if (mumps_solver){
      MUMPSSolver mumps;
      mumps.SetPrintLevel(1);
      mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      mumps.SetOperator(*A);
      X = 0.0;
      mumps.Mult(B, X);
    }
    else{    
      //allocate the solvers
      if(ns==nullptr)
	{
	  ns=new mfem::GMRESSolver(pmesh->GetComm());
	}
     
      // PRECONDITIONER
      ParGridFunction * UnitVal = new ParGridFunction(alpha_fes);
      *UnitVal = 1;      
      ParBilinearForm *displMass(new ParBilinearForm(vfes));
      displMass->AddDomainIntegrator(new WeightedStressForceIntegrator(*UnitVal, *shearMod, *bulkMod),ess_elem);
      for(auto it=displacement_BC.begin();it!=displacement_BC.end();it++)
	{
	  // Normal Penalty
	  displMass->AddBdrFaceIntegrator(new WeightedNormalDisplacementPenaltyIntegrator(*UnitVal, penaltyParameter_bf, *bulkMod),*(it->first));
	  // Tangential Penalty
	  displMass->AddBdrFaceIntegrator(new WeightedTangentialDisplacementPenaltyIntegrator(*UnitVal, penaltyParameter_bf, *shearMod),*(it->first));
	}
      if (useEmbedded){     
	// ghost penalty
	displMass->AddInteriorFaceIntegrator(new GhostStressFullGradPenaltyIntegrator(pmesh, *shearMod, *bulkMod, ghostPenaltyCoefficient, numberStrainTerms));
      }
      displMass->Assemble();
      displMass->Finalize();

      HypreParMatrix *displ_Mass = NULL;
      displ_Mass = displMass->ParallelAssemble();
      
      HypreParMatrix *DMe = NULL;
      DMe = displ_Mass->EliminateRowsCols(ess_vdofs);
      prec = new HypreBoomerAMG(*displ_Mass);
      prec->SetSystemsOptions(dim);
      prec->SetElasticityOptions(vfes);
      
      //set the parameters
      ns->SetPrintLevel(print_level);
      ns->SetRelTol(rel_tol);
      ns->SetAbsTol(abs_tol);
      ns->SetPreconditioner(*prec);   
      ns->SetMaxIter(max_iter);
      ns->SetOperator(*A);
      X = 0.0;
      //solve the problem
      ns->Mult(B, X);
    }
    mVarf->RecoverFEMSolution(X, *fform, *fdisplacement);
  }

  void LinearElasticitySolver::ComputeL2Errors(){
    int order_quad = max(2, 6*displacementOrder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
      {
	irs[i] = &(IntRules.Get(i, order_quad));
      }
    double err_u  = fdisplacement->ComputeL2Error(*exactDisplacement,irs);
    MPI_Comm comm = pmesh->GetComm();
    if (useEmbedded){
      Array<int> elem_marker(pmesh->GetNE());
      elem_marker = 0;
      for (int e = 0; e < vfes->GetNE(); e++)
	{
	  if ( (pmesh->GetAttribute(e) == AttributeShiftedFaceMarker::SBElementType::INSIDE) || (pmesh->GetAttribute(e) == AttributeShiftedFaceMarker::SBElementType::CUT)) {
	    elem_marker[e] = 1;
	  }	 
	}
      err_u  = fdisplacement->ComputeL2Error(*exactDisplacement, irs, &elem_marker);
    }
    else{
      err_u  = fdisplacement->ComputeL2Error(*exactDisplacement, irs);
    }
 
    int myid;
    MPI_Comm_rank(comm, &myid);

    if (myid == 0){
      std::cout << "|| u_h - u_ex || = " << err_u << "\n";
    }
  }

  void LinearElasticitySolver::VisualizeFields(){
    if (visualization){
      ParaViewDataCollection paraview_dc("Example5_pmesh3", pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(4);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("displacement",fdisplacement);
      if (useEmbedded && !useAnalyticalShape){
	paraview_dc.RegisterField("level_set_gf",level_set_gf);
      }
      paraview_dc.Save();
    }
  }
double AvgElementSize(ParMesh &pmesh)
{
   // Compute average mesh size (assumes similar cells).
   double dx, loc_area = 0.0;
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      loc_area += pmesh.GetElementVolume(i);
   }
   double glob_area;
   MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE,
                 MPI_SUM, pmesh.GetComm());
   const int glob_zones = pmesh.GetGlobalNE();
   switch (pmesh.GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT:
         dx = glob_area / glob_zones; break;
      case Geometry::SQUARE:
         dx = sqrt(glob_area / glob_zones); break;
      case Geometry::TRIANGLE:
         dx = sqrt(2.0 * glob_area / glob_zones); break;
      case Geometry::CUBE:
         dx = pow(glob_area / glob_zones, 1.0/3.0); break;
      case Geometry::TETRAHEDRON:
         dx = pow(6.0 * glob_area / glob_zones, 1.0/3.0); break;
      default: MFEM_ABORT("Unknown zone type!"); dx = 0.0;
   }

   return dx;
}

}
