#include "MeshOptSolver.hpp"

// extern topopt::Profiler profile;

namespace mfem
{
MeshMorphingSolver::MeshMorphingSolver(
    mfem::ParMesh* pmesh,
    const int quadOrder,
    const int targetId,
    const int metricId,
    const int newtonIter,
    const bool moveBnd,
    const int verbosityLevel,
    const double surfaceFit) :
    pmesh_(pmesh), quadOrder_(quadOrder), targetId_(targetId),
    metricId_(metricId), newtonIter_(newtonIter), moveBnd_(moveBnd),
    verbosityLevel_(verbosityLevel), surfaceFit_(surfaceFit) { }

// Get the solution
int MeshMorphingSolver::Solve(
    mfem::ParGridFunction & LSF_OptStep,
    bool relax)
{
    const int dim = pmesh_->Dimension();
    mfem::H1_FECollection dispColl(pmesh_->GetNodalFESpace()->GetElementOrder(0), dim);
    mfem::ParFiniteElementSpace dispFESpace(pmesh_, &dispColl, pmesh_->GetNodalFESpace()->GetVDim(),
                                            pmesh_->GetNodalFESpace()->GetOrdering());

    mfem::ParGridFunction nodeDisp(&dispFESpace);

    // Get nodal displacements
    int retVal = Solve(LSF_OptStep, nodeDisp, relax);

    // Update coordinates
    mfem::ParGridFunction *X = dynamic_cast<mfem::ParGridFunction *>(pmesh_->GetNodes());
    *X += nodeDisp;

    // Compute the minimum det(J) of the input mesh.
    // Setup the quadrature rules for the TMOP integrator.
    mfem::IntegrationRules *irules = nullptr;
    mfem::IntegrationRules IntRulesLo(0, mfem::Quadrature1D::GaussLobatto);
    irules = &IntRulesLo;

    const int MyRank = pmesh_->GetMyRank();

    double minJacDet = mfem::infinity();
    const int NE = pmesh_->GetNE();
    for (int i = 0; i < NE; i++)
    {
        const mfem::IntegrationRule &ir =
            irules->Get(dispFESpace.GetFE(i)->GetGeomType(), quadOrder_);
        mfem::ElementTransformation *transf = pmesh_->GetElementTransformation(i);
        for (int j = 0; j < ir.GetNPoints(); j++)
        {
            transf->SetIntPoint(&ir.IntPoint(j));
            minJacDet = std::min(minJacDet, transf->Jacobian().Det());
        }
    }
    double minJ0;
    MPI_Allreduce(&minJacDet, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    minJacDet = minJ0;

    if (MyRank == 0) { std::cout << "Minimum det(J) of the final mesh is " << minJacDet << std::endl; }

    return retVal;
}

// Get the solution
int MeshMorphingSolver::Solve(
    mfem::ParGridFunction & LSF_OptStep,
    mfem::ParGridFunction & nodeDisp,
    bool relax)
{
    bool skip(false);

    if (surfaceFit_ > 0.0 && marker_.Size() != LSF_OptStep.Size())
    {
        const int dim = pmesh_->Dimension();
        mfem::L2_FECollection matColl(0, dim);
        mfem::ParFiniteElementSpace matFes(pmesh_, &matColl);
        mfem::ParGridFunction mat(&matFes);

        bool resetMarker(true);
        if ( resetMarker )
        {
            int polinomialOrder = 1;
            ::mfem::L2_FECollection FECol_L2(polinomialOrder, dim);
            ::mfem::ParFiniteElementSpace FESpace_L2(pmesh_, &FECol_L2, 1);

            ::mfem::GridFunctionCoefficient finalLSFCoeff(&LSF_OptStep);
            for (int e = 0; e < pmesh_->GetNE(); e++)
            {
                ::mfem::Vector eval(FESpace_L2.GetFE(e)->GetDof());
                FESpace_L2.GetFE(e)->Project(finalLSFCoeff, *(FESpace_L2.GetElementTransformation(e)), eval);
                int attr = eval.Sum() > 0.0 ? 1 : 2;
                pmesh_->GetElement(e)->SetAttribute(attr);
            }
            pmesh_->SetAttributes();
        }

        for (int i = 0; i < pmesh_->GetNE(); i++)
        {
            mat(i) = pmesh_->GetAttribute(i)-1;
        }

        marker_.SetSize(LSF_OptStep.Size());
        mfem::GridFunctionCoefficient coeffMat(&mat);
        markerGF.SetSpace(LSF_OptStep.FESpace());
        markerGF = LSF_OptStep;

        markerGF.ProjectDiscCoefficient(coeffMat, mfem::GridFunction::ARITHMETIC);
        for (int j = 0; j < marker_.Size(); j++)
        {
            if (markerGF(j) > 0.1 && markerGF(j) < 0.9)
            {
                marker_[j] = true;
                markerGF(j) = 1.0;
            }
            else
            {
                marker_[j] = false;
                markerGF(j) = 0.0;
            }
        }
    }

    if (skip) { return 0; }

    const int dim = pmesh_->Dimension();
    double minJacDet = mfem::infinity();
    const int MyRank = pmesh_->GetMyRank();

    // Get node initial coordinates and displacement grid function with same data
    // mfem::ParGridFunction *x = dynamic_cast<mfem::ParGridFunction *>(pmesh_->GetNodes());
    mfem::GridFunction x0 = *(pmesh_->GetNodes());
    nodeDisp = *(pmesh_->GetNodes());
    mfem::ParFiniteElementSpace *pfespace = nodeDisp.ParFESpace();

    // Setup the mesh quality metric
    mfem::TMOP_QualityMetric *metric = nullptr;
    switch (metricId_)
    {
        // T-metrics
        case 1: metric = new mfem::TMOP_Metric_001; break;
        case 2: metric = new mfem::TMOP_Metric_002; break;
        case 7: metric = new mfem::TMOP_Metric_007; break;
        case 9: metric = new mfem::TMOP_Metric_009; break;
        case 22: metric = new mfem::TMOP_Metric_022(minJacDet); break;
        case 50: metric = new mfem::TMOP_Metric_050; break;
        case 55: metric = new mfem::TMOP_Metric_055; break;
        case 56: metric = new mfem::TMOP_Metric_056; break;
        case 58: metric = new mfem::TMOP_Metric_058; break;
        case 77: metric = new mfem::TMOP_Metric_077; break;
        case 80: metric = new mfem::TMOP_Metric_080(0.5); break;
        case 301: metric = new mfem::TMOP_Metric_301; break;
        case 302: metric = new mfem::TMOP_Metric_302; break;
        case 303: metric = new mfem::TMOP_Metric_303; break;
        case 313: metric = new mfem::TMOP_Metric_313(minJacDet); break;
        case 315: metric = new mfem::TMOP_Metric_315; break;
        case 316: metric = new mfem::TMOP_Metric_316; break;
        case 321: metric = new mfem::TMOP_Metric_321; break;
        default:
            std::cout << "Unknown metricId_: " << metricId_ << std::endl;
            return 3;
    }
    mfem::TMOP_QualityMetric *metricRelax = nullptr;
    if (relax)
    {
        int metricIDRelax = metricIDRelax_ < 0 ? (dim == 2 ? 1 : 301) : metricIDRelax_;
        switch (metricIDRelax)
        {
            // T-metrics
            case 1: metricRelax = new mfem::TMOP_Metric_001; break;
            case 2: metricRelax = new mfem::TMOP_Metric_002; break;
            case 7: metricRelax = new mfem::TMOP_Metric_007; break;
            case 9: metricRelax = new mfem::TMOP_Metric_009; break;
            case 22: metricRelax = new mfem::TMOP_Metric_022(minJacDet); break;
            case 50: metricRelax = new mfem::TMOP_Metric_050; break;
            case 55: metricRelax = new mfem::TMOP_Metric_055; break;
            case 56: metricRelax = new mfem::TMOP_Metric_056; break;
            case 58: metricRelax = new mfem::TMOP_Metric_058; break;
            case 77: metricRelax = new mfem::TMOP_Metric_077; break;
            case 80: metricRelax = new mfem::TMOP_Metric_080(0.5); break;
            case 301: metricRelax = new mfem::TMOP_Metric_301; break;
            case 302: metricRelax = new mfem::TMOP_Metric_302; break;
            case 303: metricRelax = new mfem::TMOP_Metric_303; break;
            case 313: metricRelax = new mfem::TMOP_Metric_313(minJacDet); break;
            case 315: metricRelax = new mfem::TMOP_Metric_315; break;
            case 316: metricRelax = new mfem::TMOP_Metric_316; break;
            case 321: metricRelax = new mfem::TMOP_Metric_321; break;
            default:
                std::cout << "Unknown metricIdRelax_: " << metricIDRelax_ << std::endl;
                return 3;
        }
    }

    // Setup the target type for mesh optimization
    mfem::TargetConstructor::TargetType targetT;
    mfem::TargetConstructor *targetC = nullptr;
    switch (targetId_)
    {
        case 1: targetT = mfem::TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
        case 2: targetT = mfem::TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
        case 3: targetT = mfem::TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
        default:
            if (MyRank == 0) { std::cout << "Unknown targetId: " << targetId_ << "\n"; }
            return 3;
    }
    if (nullptr == targetC)
    {
        targetC = new mfem::TargetConstructor(targetT, MPI_COMM_WORLD);
    }
    targetC->SetNodes(x0);
    mfem::TMOP_Integrator *TMOPInteg = new mfem::TMOP_Integrator(metric, targetC);
    mfem::TMOP_Integrator *TMOPIntegRelax = NULL;
    if (relax)
    {
        TMOPIntegRelax = new mfem::TMOP_Integrator(metricRelax, targetC);
    }

    // Setup the quadrature rules for the TMOP integrator.
    mfem::IntegrationRules *irules = nullptr;
    mfem::IntegrationRules IntRulesLo(0, mfem::Quadrature1D::GaussLobatto);
    irules = &IntRulesLo;
    TMOPInteg->SetIntegrationRules(*irules, quadOrder_);
    if (relax)
    {
        TMOPIntegRelax->SetIntegrationRules(*irules, quadOrder_);
    }

    const int adaptEval = 0;
    mfem::ConstantCoefficient coefLS(surfaceFit_);
    mfem::AdaptivityEvaluator *adaptSurface = nullptr;

    if (surfaceFit_ > 0.0)
    {
        if (adaptEval == 0) { adaptSurface = new mfem::AdvectorCG; }
        else if (adaptEval == 1)
        {
#ifdef MFEM_USE_GSLIB
            adaptSurface = new mfem::InterpolatorFP;
#else
            MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif
        }
        else { MFEM_ABORT("Bad interpolation option."); }

        TMOPInteg->EnableSurfaceFitting(LSF_OptStep, marker_, coefLS, *adaptSurface);
    }

    // Set up an empty right-hand side vector b, which is equivalent to b=0.
    mfem::Vector b(0);
    mfem::ParNonlinearForm a(pfespace);
    a.AddDomainIntegrator(TMOPInteg);
    mfem::ParNonlinearForm *aRelax = nullptr;
    if (relax)
    {
        aRelax = new mfem::ParNonlinearForm(pfespace);
        aRelax->AddDomainIntegrator(TMOPIntegRelax);
    }

    // Compute the minimum det(J) of the input mesh.
    const int NE = pmesh_->GetNE();
    for (int i = 0; i < NE; i++)
    {
        const mfem::IntegrationRule &ir =
            irules->Get(pfespace->GetFE(i)->GetGeomType(), quadOrder_);
        mfem::ElementTransformation *transf = pmesh_->GetElementTransformation(i);
        for (int j = 0; j < ir.GetNPoints(); j++)
        {
            transf->SetIntPoint(&ir.IntPoint(j));
            minJacDet = std::min(minJacDet, transf->Jacobian().Det());
        }
    }
    double minJ0;
    MPI_Allreduce(&minJacDet, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    minJacDet = minJ0;

    if (MyRank == 0) { std::cout << "Minimum det(J) of the original mesh is " << minJacDet << std::endl; }

    if (minJacDet < 0.0 && metricId_ != 22 && metricId_ != 211 && metricId_ != 252
        && metricId_ != 311 && metricId_ != 313 && metricId_ != 352)
    {
        MFEM_ABORT("The input mesh is inverted! Try an untangling metric.");
    }

    initTMOPEnergy = a.GetParGridFunctionEnergy(nodeDisp);
    const bool modBndrAttr = true;
    mfem::Array<int> originalBndrAttr;

    if (modBndrAttr)
    {
        for (int i = 0; i < pmesh_->GetNBE(); i++)
        {
            mfem::Array<int> dofs;
            pfespace->GetBdrElementDofs(i, dofs);
            mfem::Vector bdr_xy_data;
            mfem::Vector dof_xyz(dim);
            mfem::Vector dof_xyz_compare;
            mfem::Array<int> xyz_check(dim);
            for (int j = 0; j < dofs.Size(); j++)
            {
                for (int d = 0; d < dim; d++)
                {
                    dof_xyz(d) = nodeDisp(pfespace->DofToVDof(dofs[j], d));
                }
                if (j == 0)
                {
                    dof_xyz_compare = dof_xyz;
                    xyz_check = 1;
                }
                else
                {
                    for (int d = 0; d < dim; d++)
                    {
                        if (std::fabs(dof_xyz(d)-dof_xyz_compare(d)) < 1.e-10)
                        {
                            xyz_check[d] += 1;
                        }
                    }
                }
            }
            if (dim == 2)
            {
                if (xyz_check[0] == dofs.Size())
                {
                    pfespace->GetMesh()->SetBdrAttribute(i, 1);
                }
                else if (xyz_check[1] == dofs.Size())
                {
                    pfespace->GetMesh()->SetBdrAttribute(i, 2);
                }
                else
                {
                    pfespace->GetMesh()->SetBdrAttribute(i, 4);
                }
            }
            else if (dim == 3)
            {
                if (xyz_check[0] == dofs.Size())
                {
                    pfespace->GetMesh()->SetBdrAttribute(i, 1);
                }
                else if (xyz_check[1] == dofs.Size())
                {
                    pfespace->GetMesh()->SetBdrAttribute(i, 2);
                }
                else if (xyz_check[2] == dofs.Size())
                {
                    pfespace->GetMesh()->SetBdrAttribute(i, 3);
                }
                else
                {
                    pfespace->GetMesh()->SetBdrAttribute(i, 4);
                }
            }
            originalBndrAttr.Append(pfespace->GetMesh()->GetBdrAttribute(i));
        }
    }

    if (moveBnd_ == false)
    {
        mfem::Array<int> essBdr(pmesh_->bdr_attributes.Max());
        essBdr = 1;
        a.SetEssentialBC(essBdr);
        if (relax) { aRelax->SetEssentialBC(essBdr); }
    }
    else
    {
        int n = 0;
        for (int i = 0; i < pmesh_->GetNBE(); i++)
        {
            const int nd = pfespace->GetBE(i)->GetDof();
            const int attr = pmesh_->GetBdrElement(i)->GetAttribute();
            MFEM_VERIFY(!(dim == 2 && attr == 3),
                        "Boundary attribute 3 must be used only for 3D meshes. "
                        "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                        "components, rest for free nodes), or use -fix-bnd.");
            if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
            if (attr == 4) { n += nd * dim; }
        }
        mfem::Array<int> essVdofs(n), vdofs;
        n = 0;
        for (int i = 0; i < pmesh_->GetNBE(); i++)
        {
            const int nd = pfespace->GetBE(i)->GetDof();
            const int attr = pmesh_->GetBdrElement(i)->GetAttribute();
            pfespace->GetBdrElementVDofs(i, vdofs);
            if (attr == 1) // Fix x components.
            {
                for (int j = 0; j < nd; j++)
                { essVdofs[n++] = vdofs[j]; }
            }
            else if (attr == 2) // Fix y components.
            {
                for (int j = 0; j < nd; j++)
                { essVdofs[n++] = vdofs[j+nd]; }
            }
            else if (attr == 3) // Fix z components.
            {
                for (int j = 0; j < nd; j++)
                { essVdofs[n++] = vdofs[j+2*nd]; }
            }
            else if (attr == 4) // Fix all components.
            {
                for (int j = 0; j < vdofs.Size(); j++)
                { essVdofs[n++] = vdofs[j]; }
            }
        }
        a.SetEssentialVDofs(essVdofs);
        if (relax) { aRelax->SetEssentialVDofs(essVdofs); }
    }

    // Restore original boundary attributes
    if (originalBndrAttr.Size() > 0)
    {
        for (int i = 0; i < pmesh_->GetNBE(); i++)
        {
            pfespace->GetMesh()->SetBdrAttribute(i, originalBndrAttr[i]);
        }
    }

    int solverType       = 0;
    double solverRTol    = 1e-5;
    int maxLinIter       = 50;

    mfem::Solver *S = nullptr;
    const double linsol_rtol = 1e-12;
    mfem::MINRESSolver *minres = new mfem::MINRESSolver(MPI_COMM_WORLD);
    minres->SetMaxIter(maxLinIter);
    minres->SetRelTol(linsol_rtol);
    minres->SetAbsTol(0.0);
    if (verbosityLevel_ > 2) { minres->SetPrintLevel(1); }
    else { minres->SetPrintLevel(verbosityLevel_ == 2 ? 3 : -1); }
    S = minres;

    const mfem::IntegrationRule &ir =
        irules->Get(pfespace->GetFE(0)->GetGeomType(), quadOrder_);
    mfem::TMOPNewtonSolver solver(pfespace->GetComm(), ir, solverType);
    solver.SetIntegrationRules(*irules, quadOrder_);
    solver.SetPreconditioner(*S);
    solver.SetRelTol(solverRTol);
    solver.SetAbsTol(0.0);
    solver.SetPrintLevel(verbosityLevel_ >= 1 ? 1 : -1);

    if (relax)
    {
        solver.SetMaxIter(newtonIterRelax_);
        solver.SetOperator(*aRelax);
        nodeDisp.SetTrueVector();
        solver.Mult(b, nodeDisp.GetTrueVector());
        nodeDisp.SetFromTrueVector();
        b *= 0.0;
    }

    solver.SetOperator(a);
    solver.SetMaxIter(newtonIter_);
    if (surfaceFit_ > 0.0)
    {
        if( surface_fit_adapt )
        {
            solver.EnableAdaptiveSurfaceFitting();
        }
        solver.SetTerminationWithMaxSurfaceFittingError(1e-7);
    }
    nodeDisp.SetTrueVector();
    solver.Mult(b, nodeDisp.GetTrueVector());
    nodeDisp.SetFromTrueVector();

    finalTMOPEnergy = a.GetParGridFunctionEnergy(nodeDisp);

    if (surfaceFit_ > 0.0)
    {
        TMOPInteg->GetSurfaceFittingErrors(interfaceFittingErrorAvg, interfaceFittingErrorMax);
        if (MyRank == 0)
        {
            std::cout << "Avg fitting error: " << interfaceFittingErrorAvg << std::endl
                      << "Max fitting error: " << interfaceFittingErrorMax << std::endl;
        }
    }

    if (MyRank == 0)
    {
        std::cout << "Final TMOP Energy: " << finalTMOPEnergy << std::endl;
    }

    nodeDisp -= x0;

    return 0;
}

}
