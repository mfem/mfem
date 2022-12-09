#include "linear_elasticity_solver.hpp"
#include "sbm_aux.hpp"

namespace mfem {

  LinearElasticitySolver::LinearElasticitySolver(mfem::ParMesh* mesh_, int vorder, bool useEmb, int gS, int nT, int nStrainTerms, double ghostPenCoeff, bool useMumps, bool useAS, bool vis): pmesh(mesh_), displacementOrder(vorder), useEmbedded(useEmb), geometricShape(gS), nTerms(nT), numberStrainTerms(nStrainTerms), ghostPenaltyCoefficient(ghostPenCoeff), mumps_solver(useMumps), useAnalyticalShape(useAS), visualization(vis), fdisplacement(NULL), volforce(NULL), prec(NULL),  ns(NULL), vfes(NULL), vfec(NULL), shearMod(NULL), bulkMod(NULL), exactDisplacement(NULL), ess_elem(pmesh->attributes.Max()), C_I(0.0), analyticalSurface(NULL), neumann_dist_coef(NULL), combo_dist_coef(NULL), distance_vec_space(NULL), distance(NULL), normal_vec_space(NULL), normal(NULL), ls_func(NULL), level_set_gf(NULL), dist_vec(NULL), normal_vec(NULL), filt_gf(NULL)
  {
    int dim=pmesh->Dimension();
    vfec=new H1_FECollection(vorder,dim);
    
    vfes=new mfem::ParFiniteElementSpace(pmesh,vfec,dim);
    vfes->ExchangeFaceNbrData();

    fdisplacement = new ParGridFunction(vfes);

    *fdisplacement=0.0;

    mfem::FiniteElementCollection* lsvec = new H1_FECollection(vorder+2,dim);
    mfem::ParFiniteElementSpace* lsfes = new mfem::ParFiniteElementSpace(pmesh,lsvec);
    lsfes->ExchangeFaceNbrData();
    
    // Weak Boundary condition imposition: all tests use v.n = 0 on the boundary
   // We need to define ess_tdofs and ess_vdofs, but they will be kept empty
   Array<int> ess_tdofs;
   level_set_gf =  new ParGridFunction(lsfes);

   if (useEmbedded){
     analyticalSurface = new ShiftedFaceMarker(*pmesh, *vfes, 1);
     neumann_dist_coef = new Dist_Level_Set_Coefficient(geometricShape);
     combo_dist_coef = new Combo_Level_Set_Coefficient;
     
     level_set_gf->ProjectCoefficient(*neumann_dist_coef);
     // Exchange information for ghost elements i.e. elements that share a face
     // with element on the current processor, but belong to another processor.
     level_set_gf->ExchangeFaceNbrData();
     // Setup the class to mark all elements based on whether they are located
     // inside or outside the true domain, or intersected by the true boundary.
     analyticalSurface->MarkElements(*level_set_gf);
     combo_dist_coef->Add_Level_Set_Coefficient(*neumann_dist_coef);
     Array<int> ess_inactive_dofs = analyticalSurface->GetEss_Vdofs();
     vfes->GetRestrictionMatrix()->BooleanMult(ess_inactive_dofs, ess_tdofs);
     vfes->MarkerToList(ess_tdofs, ess_vdofs);
     if (useAnalyticalShape){
       dist_vec = new Dist_Vector_Coefficient(dim, geometricShape);
       normal_vec = new Normal_Vector_Coefficient(dim, geometricShape);
     }
     else{
       mfem::FiniteElementCollection* d_vec = new H1_FECollection(vorder+2,dim);

       distance_vec_space = new ParFiniteElementSpace(pmesh, d_vec, dim);
       normal_vec_space = new ParFiniteElementSpace(pmesh, d_vec, dim);

       distance = new ParGridFunction(distance_vec_space);
       *distance = 0.0;
       normal = new ParGridFunction(normal_vec_space);
       *normal = 0.0;
       
       //       ls_func = new ParGridFunction(lsfes);

       /* double dx = AvgElementSize(*pmesh);
       filt_gf =  new ParGridFunction(lsfes);
       PDEFilter filter(*pmesh, 2.0 * dx);
       filter.Filter(*combo_dist_coef, *filt_gf);
       GridFunctionCoefficient ls_filt_coeff(filt_gf);
       */
       /*   PLapDistanceSolver dist_solver(10);
	    dist_solver.print_level = 0;
	    dist_solver.ComputeScalarDistance(*combo_dist_coef, *ls_func);
     
	    dist_solver.ComputeVectorDistance(*combo_dist_coef, *distance); 
	    dist_solver.ComputeVectorNormal(*combo_dist_coef, *distance, *normal); */
              //    dist_solver.ComputeVectorDistance(ls_filt_coeff, *distance); 
       //     dist_solver.ComputeVectorNormal(ls_filt_coeff, *distance, *normal); 
	    
       NormalizationDistanceSolver dist_solver;
       dist_solver.ComputeVectorDistance(*combo_dist_coef, *distance); 
       dist_solver.ComputeVectorNormal(*combo_dist_coef, *distance, *normal);

       dist_vec = new VectorGridFunctionCoefficient(distance);
       normal_vec = new VectorGridFunctionCoefficient(normal);
     }
   }
   
   shearMod = new ShearModulus(pmesh);
   bulkMod = new BulkModulus(pmesh);
 
   const int max_elem_attr = pmesh->attributes.Max();
   ess_elem.SetSize(max_elem_attr);
   ess_elem = 1;
   if (useEmbedded && (max_elem_attr >= 2)){
     ess_elem[max_elem_attr-1] = 0;
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
     if (useEmbedded){
      delete analyticalSurface;
      delete distance_vec_space;
      delete distance;
      delete normal_vec_space;
      delete normal;
 
      delete dist_vec;
      delete normal_vec;
      delete combo_dist_coef;
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
    // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
    // gamma values are projected on function that's constant on the moving mesh.
    L2_FECollection alpha_fec(0, pmesh->Dimension());
    ParFiniteElementSpace alpha_fes(pmesh, &alpha_fec);
    ParGridFunction alphaCut(&alpha_fes);
    alphaCut = 1;
    if (useEmbedded){
      UpdateAlpha(*analyticalSurface,alphaCut, *vfes, geometricShape);
      alphaCut.ExchangeFaceNbrData();
    }

    // Set the BC
    int dim=pmesh->Dimension();
    
    double penaltyParameter_bf = 10.0*2*C_I*2;

    ParLinearForm *fform(new ParLinearForm(vfes));
    fform->AddDomainIntegrator(new WeightedVectorForceIntegrator(alphaCut, *volforce), ess_elem);
    for(auto it=surf_loads.begin();it!=surf_loads.end();it++)
    {
      fform->AddBdrFaceIntegrator(new WeightedTractionBCIntegrator(alphaCut, *(it->second)),*(it->first));
    }
    for(auto it=displacement_BC.begin();it!=displacement_BC.end();it++)
    {
      // Nitsche
      fform->AddBdrFaceIntegrator(new WeightedStressNitscheBCForceIntegrator(alphaCut, *shearMod, *bulkMod, *(it->second)),*(it->first));
      // Normal Penalty
      fform->AddBdrFaceIntegrator(new WeightedNormalDisplacementBCPenaltyIntegrator(alphaCut, penaltyParameter_bf, *bulkMod, *(it->second)), *(it->first));
      // Tangential Penalty
      fform->AddBdrFaceIntegrator(new WeightedTangentialDisplacementBCPenaltyIntegrator(alphaCut, penaltyParameter_bf, *shearMod, *(it->second)), *(it->first));
    }
    if (useEmbedded){
      // Nitsche
      fform->AddInteriorFaceIntegrator(new WeightedShiftedStressNitscheBCForceIntegrator(pmesh, alphaCut, *shifted_traction_BC, dist_vec, normal_vec, analyticalSurface, 1));
    }
    fform->Assemble();
    fform->ParallelAssemble();
  
    ParBilinearForm *mVarf(new ParBilinearForm(vfes));
    mVarf->AddDomainIntegrator(new WeightedStressForceIntegrator(alphaCut, *shearMod, *bulkMod),ess_elem);
    for(auto it=displacement_BC.begin();it!=displacement_BC.end();it++)
    {
      // Nitsche
      mVarf->AddBdrFaceIntegrator(new WeightedStressBoundaryForceIntegrator(alphaCut, *shearMod, *bulkMod),*(it->first));
      // IP
      mVarf->AddBdrFaceIntegrator(new WeightedStressBoundaryForceTransposeIntegrator(alphaCut, *shearMod, *bulkMod),*(it->first));
      // Normal Penalty
      mVarf->AddBdrFaceIntegrator(new WeightedNormalDisplacementPenaltyIntegrator(alphaCut, penaltyParameter_bf, *bulkMod),*(it->first));
      // Tangential Penalty
      mVarf->AddBdrFaceIntegrator(new WeightedTangentialDisplacementPenaltyIntegrator(alphaCut, penaltyParameter_bf, *shearMod),*(it->first));
    }
    if (useEmbedded){     
      // Nitsche
      mVarf->AddInteriorFaceIntegrator(new WeightedShiftedStressBoundaryForceIntegrator(pmesh, alphaCut, *shearMod, *bulkMod, dist_vec, normal_vec, analyticalSurface, nTerms, 1));
      // IP
      mVarf->AddInteriorFaceIntegrator(new WeightedShiftedStressBoundaryForceTransposeIntegrator(pmesh, alphaCut, *shearMod, *bulkMod, analyticalSurface, 1));
      // ghost penalty
      mVarf->AddInteriorFaceIntegrator(new GhostStressPenaltyIntegrator(pmesh, *shearMod, *bulkMod, alphaCut, ghostPenaltyCoefficient, analyticalSurface, 1));
      for (int i = 2; i <= numberStrainTerms; i++){
	// best to use 1.0 / i!
	double factorial = 1.0;	
	for (int s = 1; s <= i; s++){
	  factorial = factorial*s;
	}
	mVarf->AddInteriorFaceIntegrator(new GhostStressFullGradPenaltyIntegrator(pmesh, *shearMod, *bulkMod, ghostPenaltyCoefficient/factorial, analyticalSurface, i));
      }
    }
    mVarf->Assemble();
    mVarf->Finalize();
    mVarf->ParallelAssemble();
    
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
	  prec = new HypreBoomerAMG;
	}
      
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
    // 12. Create the grid functions u and p. Compute the L2 error norms.
    //  p.MakeRef(pfes, x.GetBlock(1), 0);
    // 13. Compute the L2 error norms.
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
      Array<int> &elemStatus = analyticalSurface->GetElement_Status();
      for (int e = 0; e < vfes->GetNE(); e++)
	{
	  if ( (elemStatus[e] == AnalyticalGeometricShape::SBElementType::INSIDE) || (elemStatus[e] == AnalyticalGeometricShape::SBElementType::CUT)) {
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
      paraview_dc.RegisterField("distance",distance);
      paraview_dc.RegisterField("normal",normal);
      paraview_dc.RegisterField("level_set_gf",level_set_gf);
      //  paraview_dc.RegisterField("ls_func",ls_func);
      //      paraview_dc.RegisterField("filt_gf",filt_gf);
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
