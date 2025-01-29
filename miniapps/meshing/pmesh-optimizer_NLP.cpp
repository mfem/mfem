// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//    ---------------------------------------------------------------------
//    Mesh Optimizer NLP Miniapp: Optimize high-order meshes - Parallel Version
//    ---------------------------------------------------------------------
//
// This miniapp performs mesh optimization using the Target-Matrix Optimization
// Paradigm (TMOP) by P.Knupp et al., it used non-linear programming techniques
// to solve the proble,
//
// Compile with: make pmesh-optimizer_NLP
// mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-4 -ni 100 -ft 2 -w1 5e1 -w2 1e-2
// mpirun -np 10 pmesh-optimizer_NLP -met 1 -ch 2e-3 -ni 200 -ft 2 --qtype 3 -w1 5e3 -w2 1e-2
// WIP mpirun -np 10 pmesh-optimizer_NLP -met 1 -ch 2e-3 -ni 200 -ft 2 --qtype 4 -w1 1e-4 -w2 1e-2


// K10 -  TMOP solver based run
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 200 -ft 2 --qtype 3 -w1 5e3 -w2 1e-2 -m square01.mesh -rs 4
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 100 -ft 2 --qtype 3 -w1 5e3 -w2 1e-2 -m cube.mesh -o 1 -rs 4 -mid 303


#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer_using_NLP.hpp"
#include "MMA.hpp"
#include "mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

int ftype = 1;
double kw = 10.0;
double alphaw = 50;

double trueSolFunc(const Vector & x)
{
  if (ftype == 0)
  {
    double val = std::sin( M_PI *x[0] )*std::sin(2.0*M_PI*x[1]);
    return val;
  }
  else if (ftype == 1) // circular wave centered in domain
  {
    // double k_w = 5.0;
    double k_w = kw;
    double k_t = 0.5;
    double T_ref = 1.0;

    double val = 0.5+0.5*std::tanh(k_w*((std::sin( M_PI *x[0] )*std::sin(M_PI *x[1]))-k_t*T_ref));
    return val;
  }
  else if (ftype == 2) // circular shock wave front centered at origin
  {
    double xc = -0.05,
           yc = -0.05,
           zc = -0.05,
           rc = 0.7,
           alpha = alphaw;
    double dx = (x[0]-xc),
           dy = x[1]-yc,
           dz = 0.0;
    double val = dx*dx + dy*dy;
    if (x.Size() == 3)
    {
      dz = x[2]-zc;
      val += dz*dz;
    }
    if (val > 0.0) { val = std::sqrt(val); }
    val -= rc;
    val *= alpha;
    return std::atan(val);
  }
  return 0.0;
  //--------------------------------------------------------------
    // double k_w = kw;
    // // double k_t = 0.5;
    // // double T_ref = 1.0;

    // double starval = 0.5+0.1*std::sin(5.0*std::atan2(x[0],x[1])) - std::sqrt(x[0]*x[0]+x[1]*x[1]);

    // double val = 0.5+0.5*std::tanh(k_w*starval);
    // return val;
};

void trueSolGradFunc(const Vector & x,Vector & grad)
{
  if (ftype == 0)
  {
    grad[0] = std::cos( M_PI *x[0] )*std::sin(2.0*M_PI*x[1]);
    grad[1] = std::sin( M_PI *x[0] )*std::cos(2.0*M_PI*x[1]);
  }
  else if (ftype == 1) // circular wave centered in domain
  {
    // double k_w = 5.0;
    double k_w = kw;

    grad[0]= 1.5708 * k_w * std::cos(M_PI * x[0]) / std::pow(std::cosh(k_w*( std::sin(M_PI * x[0]) * std::sin(M_PI * x[1])-0.5)) , 2) * std::sin(M_PI * x[1]);
    grad[1]= 1.5708 * k_w * std::cos(M_PI * x[1]) / std::pow(std::cosh(k_w*( std::sin(M_PI * x[0]) * std::sin(M_PI * x[1])-0.5)) , 2) * std::sin(M_PI * x[0]);

    // double k_t = 0.5;
    // double T_ref = 1.0;

    // double val = 0.5+0.5*std::tanh(k_w*((std::sin( M_PI *x[0] )*std::sin(M_PI *x[1]))-k_t*T_ref));
  }
  else if (ftype == 2) // circular shock wave front centered at origin
  {
    double xc = -0.05,
        yc = -0.05,
        zc = -0.05,
        rc = 0.7,
        alpha = alphaw;
    double dx = (x[0]-xc),
           dy = x[1]-yc,
           dz = 0.0;
    double val = dx*dx + dy*dy;
    if (x.Size() == 3)
    {
      dz = x[2]-zc;
      val += dz*dz;
    }
    if (val > 0.0) { val = std::sqrt(val); }
    double valo = val;
    val -= rc;
    val *= alpha;
    // return std::atan(val);

    double den1 = (1.0+val*val)*(valo);
    grad[0] = alpha*dx/den1;
    grad[1] = alpha*dy/den1;
    if (x.Size() == 3)
    {
      grad[2] = alpha*dz/den1;
    }
    // mfem_error("ftype 2 not implemented");
  }
};

double loadFunc(const Vector & x)
{
  if (ftype == 0)
  {
    double val = 5.0*M_PI*M_PI * std::sin( M_PI *x[0] )*std::sin(2.0*M_PI *x[1]);
    return val;
  }
  else if (ftype == 1)
  {
    // double k_w =5.0;
    double k_w = kw;
    double k_t = 0.5;
    double T_ref = 1.0;

    double bt = k_w*M_PI*M_PI*std::sin( M_PI *x[0] )*std::sin(M_PI *x[1]);
    double bx = k_w*M_PI*std::cos( M_PI *x[0] )*std::sin(M_PI *x[1]);
    double by = k_w*M_PI*std::sin( M_PI *x[0] )*std::cos(M_PI *x[1]);
    double sh = std::tanh(k_w*((std::sin( M_PI *x[0] )*std::sin(M_PI *x[1]))-k_t*T_ref));


    double val = -1.0*( 0.5*( - 1.0*bt - (2.0*sh*(1 - sh*sh))*bx*bx + sh*sh*bt) +
                        0.5*( - 1.0*bt - (2.0*sh*(1 - sh*sh))*by*by + sh*sh*bt) );
    return val;
  }
  else if (ftype == 2)
  {
    double xc = -0.05,
           yc = -0.05,
           zc = -0.05,
           rc = 0.7,
           alpha = alphaw;
    double dx = (x[0]-xc),
           dy = x[1]-yc,
           dz = 0.0;
    double val = dx*dx + dy*dy;
    if (x.Size() == 3)
    {
      dz = x[2]-zc;
      val += dz*dz;
    }
    if (val > 0.0) { val = std::sqrt(val); }
    double num1 = 2.0*alpha*alpha*alpha*(val-rc);
    double den1 = std::pow((1.0 + alpha*alpha*(val-rc)*(val-rc)),2.0);

    double num2 = 2.0*alpha;
    double den2 = val*((1.0 + alpha*alpha*(val-rc)*(val-rc)));

    double f = num1/den1 - num2/den2;
    return f;


    // double lambda = val;
    // double numerator = alpha - std::pow(alpha, 3) * (std::pow(lambda, 2) - std::pow(r, 2));

    // // Denominator components
    // double term1 = std::pow(alpha, 2) * std::pow(xc, 2);
    // double term2 = -2 * std::pow(alpha, 2) * r * lambda;
    // double term3 = -2 * xc * alpha * alpha * x[0];
    // double term4 = std::pow(alpha, 2) * std::pow(yc, 2);
    // double term5 = -2 * yc * alpha * alpha * x[1];
    // double term6 = std::pow(alpha, 2) * std::pow(r, 2);
    // double term7 = std::pow(alpha, 2) * std::pow(x[0], 2);
    // double term8 = std::pow(alpha, 2) * std::pow(x[1], 2);

    // // Denominator
    // double denominator = lambda * std::pow(
    //     term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + 1, 2);

    // // Compute f
    // double f = numerator / denominator;
    // return f;
  }
  return 0.0;
};

int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

#ifdef MFEM_USE_PETSC
   const char *petscrc_file = "";
   MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
#endif

  int qoitype = static_cast<int>(QoIType::H1_ERROR);
  bool perturbMesh = false;
  double epsilon_pert =  0.006;
  int ref_ser = 2;
  int mesh_node_ordering = 0;
  int max_it = 100;
  double max_ch=0.002; //max design change
  double weight_1 = 1e4; //1e7; // 5e2;
  double weight_tmop = 1e-2;
  int metric_id   = 2;
  int target_id   = 1;
  int quad_type         = 1;
  int quad_order        = 8;
  srand(9898975);
  bool visualization = true;
  int method = 0;
  int mesh_poly_deg     = 1;
  int nx                = 20;
  const char *mesh_file = "null.mesh";

  OptionsParser args(argc, argv);
  args.AddOption(&ref_ser, "-rs", "--refine-serial",
                 "Number of times to refine the mesh uniformly in serial.");
  args.AddOption(&metric_id, "-mid", "--metric-id",
                "Mesh optimization metric:\n\t"
                "T-metrics\n\t"
                "1  : |T|^2                          -- 2D no type\n\t"
                "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                );
  args.AddOption(&target_id, "-tid", "--target-id",
                "Target (ideal element) type:\n\t"
                "1: Ideal shape, unit size\n\t"
                "2: Ideal shape, equal size\n\t"
                "3: Ideal shape, initial size\n\t"
                "4: Given full analytic Jacobian (in physical space)\n\t"
                "5: Ideal shape, given size (in physical space)");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&method, "-met", "--method",
                  "0(Defaults to TMOP_MMA), 1 - MS");
   args.AddOption(&max_ch, "-ch", "--max-ch",
                  "max node movement");
   args.AddOption(&max_it, "-ni", "--newton-oter",
                  "number of iters");
   args.AddOption(&ftype, "-ft", "--ftype",
                  "function type");
   args.AddOption(&qoitype, "-qoit", "--qtype",
                  "Quantity of interest type");
   args.AddOption(&weight_1, "-w1", "--weight1",
                  "Quantity of interest weight");
   args.AddOption(&weight_tmop, "-w2", "--weight2",
                  "Mesh quality weight type");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

  enum QoIType qoiType  = static_cast<enum QoIType>(qoitype);
  bool dQduFD =false;
  bool dQdxFD =false;
  bool BreakAfterFirstIt = false;


  // Create mesh
  Mesh *des_mesh = nullptr;
  if (strcmp(mesh_file, "null.mesh") == 0)
  {
     des_mesh = new Mesh(Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL,
                                        true, 1.0, 1.0));
  }
  else
  {
    des_mesh = new Mesh(mesh_file, 1, 1, false);
  }

  if(perturbMesh)
  {
     int tNumVertices  = des_mesh->GetNV();
     for (int i = 0; i < tNumVertices; ++i) {
        double * Coords = des_mesh->GetVertex(i);
        if (Coords[ 0 ] != 0.0 && Coords[ 0 ] != 1.0 && Coords[ 1 ] != 0.0 && Coords[ 1 ] != 1.0) {
           Coords[ 0 ] = Coords[ 0 ] + ((rand() / double(RAND_MAX)* 2.0 - 1.0)* epsilon_pert);
           Coords[ 1 ] = Coords[ 1 ] + ((rand() / double(RAND_MAX)* 2.0 - 1.0)* epsilon_pert);
        }
     }
  }

  // Refine mesh in serial
  for (int lev = 0; lev < ref_ser; lev++) { des_mesh->UniformRefinement(); }

  auto PMesh = new ParMesh(MPI_COMM_WORLD, *des_mesh);

  int spatialDimension = PMesh->SpaceDimension();

  // -----------------------
  // Remaining mesh settings
  // -----------------------

  // Nodes are only active for higher order meshes, and share locations with
  // the vertices, plus all the higher- order control points within  the
  // element and along the edges and on the faces.
  if (nullptr == PMesh->GetNodes())
  {
    PMesh->SetCurvature(mesh_poly_deg, false, -1, 0);
  }

  // int mesh_poly_deg = PMesh->GetNodes()->FESpace()->GetElementOrder(0);

  // Create finite Element Spaces for analysis mesh
  // if ( spatialDimension != 2 ) {
  //   mfem_error("... This example only supports 2D meshes");
  // }

  // 4. Define a finite element space on the mesh. Here we use vector finite
  //    elements which are tensor products of quadratic finite elements. The
  //    number of components in the vector finite element space is specified by
  //    the last parameter of the FiniteElementSpace constructor.
  FiniteElementCollection *fec;
  if (mesh_poly_deg <= 0)
  {
    fec = new QuadraticPosFECollection;
    mesh_poly_deg = 2;
  }
  else { fec = new H1_FECollection(mesh_poly_deg, spatialDimension); }
  ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(PMesh, fec, spatialDimension,
                                                               mesh_node_ordering);
  auto fespace_scalar = new ParFiniteElementSpace(PMesh, fec, 1);
  ParFiniteElementSpace pfespace_gf(PMesh, fec);
  ParGridFunction x_gf(&pfespace_gf);

  // 5. Make the mesh curved based on the above finite element space. This
  //    means that we define the mesh elements through a fespace-based
  //    transformation of the reference element.
  PMesh->SetNodalFESpace(pfespace);

  // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
  //    element space) as a finite element grid function in fespace. Note that
  //    changing x automatically changes the shapes of the mesh elements.
  ParGridFunction x(pfespace);
  PMesh->SetNodalGridFunction(&x);
  ParGridFunction x0 = x;
  int numOptVars = pfespace->GetTrueVSize();

  // TMOP Integrator setup
     TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 4: metric = new TMOP_Metric_004; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 303: metric = new TMOP_Metric_303; break;
      default:
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 3;
   }

   TargetConstructor::TargetType target_t;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      default:
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
   }
   TargetConstructor *target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   target_c->SetNodes(x);

   IntegrationRules *irules = NULL;
   switch (quad_type)
   {
      case 1: irules = &IntRulesLo; break;
      case 2: irules = &IntRules; break;
      case 3: irules = &IntRulesCU; break;
      default:
         if (myid == 0) { cout << "Unknown quad_type: " << quad_type << endl; }
         return 3;
   }

   auto tmop_integ = new TMOP_Integrator(metric, target_c);
   tmop_integ->SetIntegrationRules(*irules, quad_order);

   ConstantCoefficient metric_w(weight_tmop);
   tmop_integ->SetCoefficient(metric_w);

  // set esing variable bounds
  Vector objgrad(numOptVars); objgrad=0.0;
  Vector volgrad(numOptVars); volgrad=1.0;
  Vector xxmax(numOptVars);   xxmax=  0.001;
  Vector xxmin(numOptVars);   xxmin= -0.001;

  ParGridFunction gridfuncOptVar(pfespace);
  gridfuncOptVar = 0.0;
  ParGridFunction gridfuncLSBoundIndicator(pfespace);
  gridfuncLSBoundIndicator = 0.0;
  Array<int> vdofs;

  // Identify coordinate dofs perpendicular to BE
  if (strcmp(mesh_file, "null.mesh") == 0)
  {
    for (int i = 0; i < PMesh->GetNBE(); i++)
    {
      Element * tEle = PMesh->GetBdrElement(i);
      int attribute = tEle->GetAttribute();
      pfespace->GetBdrElementVDofs(i, vdofs);
      const int nd = pfespace->GetBE(i)->GetDof();

      if (attribute == 1 || attribute == 3) // zero out motion in y
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
        }
      }
      else if (attribute == 2 || attribute == 4) // zero out in x
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
        }
      }
    }
  }
  else
  {
    for (int i = 0; i < PMesh->GetNBE(); i++)
    {
      Element * tEle = PMesh->GetBdrElement(i);
      int attribute = tEle->GetAttribute();
      pfespace->GetBdrElementVDofs(i, vdofs);
      const int nd = pfespace->GetBE(i)->GetDof();

      if (attribute == 2) // zero out motion in y
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
        }
      }
      else if (attribute == 1) // zero out in x
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
        }
      }
      else if (attribute == 3) // zero out in z
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j+2*nd] ] = 1.0;
        }
      }
    }
  }

  gridfuncOptVar.SetTrueVector();
  gridfuncLSBoundIndicator.SetTrueVector();

  Vector & trueOptvar = gridfuncOptVar.GetTrueVector();

  const int nbattr = PMesh->bdr_attributes.Max();
  std::vector<std::pair<int, double>> essentialBC(nbattr);
  for (int i = 0; i < nbattr; i++)
  {
    essentialBC[i] = {i+1, 0};
  }
  // essentialBC[0] = {1, 0};
  // essentialBC[1] = {2, 0};
  // if (strcmp(mesh_file, "null.mesh") == 0)
  // {
    // essentialBC[2] = {3, 0};
    // essentialBC[3] = {4, 0};
  // }

  const IntegrationRule &ir =
      irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
  mfem::MMAOpt* mma = nullptr;
#ifdef MFEM_USE_PETSC
  mfem::NativeMMA* mmaPetsc = nullptr;
#endif
    // mfem::NativeMMA* mma = nullptr;
  TMOP_MMA *tmma = new TMOP_MMA(MPI_COMM_WORLD, trueOptvar.Size(), 0,
                                 trueOptvar, ir);
  {
#ifdef MFEM_USE_PETSC
    double a=0.0;
    double c=1000.0;
    double d=0.0;
    mmaPetsc=new mfem::NativeMMA(MPI_COMM_WORLD,1, objgrad,&a,&c,&d);
#else
    mma=new mfem::MMAOpt(MPI_COMM_WORLD, trueOptvar.Size(), 0, trueOptvar);
#endif
  }

if (myid == 0) {
  switch (qoiType) {
  case 0:
     std::cout<<" L2 Error"<<std::endl;
     break;
  case 1:
  std::cout<<" H1 Error"<<std::endl;
    break;
  case 2:
  std::cout<<" ZZ Error"<<std::endl;
    break;
  case 3:
  std::cout<<" Avg Error"<<std::endl;;
    break;
  case 4:
  std::cout<<" Energy"<<std::endl;;
    break;
  default:
    std::cout << "Unknown Error Coeff: " << qoiType << std::endl;
  }
}


  Diffusion_Solver solver(PMesh, essentialBC, 1);
  QuantityOfInterest QoIEvaluator(PMesh, qoiType, 1);
  NodeAwareTMOPQuality MeshQualityEvaluator(PMesh, 1);

  Coefficient *QCoef = new FunctionCoefficient(loadFunc);
  solver.SetManufacturedSolution(QCoef);
  Coefficient *trueSolution = new FunctionCoefficient(trueSolFunc);
  VectorCoefficient *trueSolutionGrad = new VectorFunctionCoefficient(spatialDimension,trueSolGradFunc);
  QoIEvaluator.setTrueSolCoeff( trueSolution );
  if(qoiType == QoIType::ENERGY){QoIEvaluator.setTrueSolCoeff( QCoef );}
  QoIEvaluator.setTrueSolGradCoeff(trueSolutionGrad);
  x_gf.ProjectCoefficient(*trueSolution);

  ParaViewDataCollection paraview_dc("MeshOptimizer", PMesh);
  paraview_dc.SetLevelsOfDetail(1);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);

   if (visualization)
   {
      socketstream vis;
      common::VisualizeMesh(vis, "localhost", 19916, *PMesh, "Initial Mesh",
                            0, 0, 500, 500, "m");
   }


  x.SetTrueVector();
  if (method == 0)
  {
    ParNonlinearForm a(pfespace);
    a.AddDomainIntegrator(tmop_integ);
    {
      Array<int> ess_bdr(PMesh->bdr_attributes.Max());
      ess_bdr = 1;
      //a.SetEssentialBC(ess_bdr);
    }
    double init_energy = a.GetParGridFunctionEnergy(x);
    IterativeSolver::PrintLevel newton_print;
    newton_print.Errors().Warnings().Iterations();
    // set the TMOP Integrator
    tmma->SetOperator(a);
    // Set change limits on dx
    tmma->SetUpperBound(max_ch);
    tmma->SetLowerBound(max_ch);
    // Set true vector so that it can be zeroed out
    {
      Vector & trueBounds = gridfuncLSBoundIndicator.GetTrueVector();
      tmma->SetTrueDofs(trueBounds);
    }
    // Set QoI and Solver and weight
    tmma->SetQuantityOfInterest(&QoIEvaluator);
    tmma->SetDiffusionSolver(&solver);
    tmma->SetQoIWeight(weight_1);

    // Set max # iterations
    tmma->SetMaxIter(max_it);
    tmma->SetPrintLevel(newton_print);

    tmma->Mult(x.GetTrueVector());
    x.SetFromTrueVector();

    // Visualize the mesh displacement.
    if (visualization)
    {
      x0 -= x;
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, x0,
                              "Displacements", 600, 000, 500, 500, "jRmclAppppppppppppp");

      ParaViewDataCollection paraview_dc("NativeMeshOptimizer", PMesh);
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(1.0);
      //paraview_dc.RegisterField("Solution",&x_gf);
      paraview_dc.Save();
    }

    {
      ostringstream mesh_name;
      mesh_name << "optimized.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      PMesh->PrintAsOne(mesh_ofs);
    }
  }
  else
  {
    for(int i=1;i<max_it;i++)
    {
      solver.SetDesign( gridfuncOptVar );
      solver.FSolve();

      ParGridFunction & discretSol = solver.GetSolution();

      QoIEvaluator.SetDesign( gridfuncOptVar );
      MeshQualityEvaluator.SetDesign( gridfuncOptVar );

      QoIEvaluator.SetDiscreteSol( discretSol );

      double ObjVal = QoIEvaluator.EvalQoI();
      double meshQualityVal = MeshQualityEvaluator.EvalQoI();

      double val = weight_1 * ObjVal+ weight_tmop * meshQualityVal;

      QoIEvaluator.EvalQoIGrad();
      MeshQualityEvaluator.EvalQoIGrad();

      ParLinearForm * dQdu = QoIEvaluator.GetDQDu();
      ParLinearForm * dQdxExpl = QoIEvaluator.GetDQDx();
      ParLinearForm * dMeshQdxExpl = MeshQualityEvaluator.GetDQDx();

      solver.ASolve( *dQdu );

      ParLinearForm * dQdxImpl = solver.GetImplicitDqDx();

      ParLinearForm dQdx(pfespace); dQdx = 0.0;
      dQdx.Add(weight_1, *dQdxExpl);
      dQdx.Add(weight_1, *dQdxImpl);
      dQdx.Add(weight_tmop, *dMeshQdxExpl);

      HypreParVector *truedQdx = dQdx.ParallelAssemble();

      objgrad = *truedQdx;

      //----------------------------------------------------------------------------------------------------------

      if(dQduFD)
      {
        double epsilon = 1e-8;
        mfem::ParGridFunction tFD_sens(fespace_scalar); tFD_sens = 0.0;
        for( int Ia = 0; Ia<discretSol.Size(); Ia++)
        {
          if (myid == 0)
          {
            std::cout<<"iter: "<< Ia<< " out of: "<<discretSol.Size() <<std::endl;
          }
          discretSol[Ia] +=epsilon;

          QuantityOfInterest QoIEvaluator_FD1(PMesh, qoiType, 1);
          QoIEvaluator_FD1.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD1.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD1.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD1.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD1.SetDiscreteSol( discretSol );
          QoIEvaluator_FD1.SetNodes(x0);

          double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

          discretSol[Ia] -=2.0*epsilon;

          QuantityOfInterest QoIEvaluator_FD2(PMesh, qoiType, 1);
          QoIEvaluator_FD2.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD2.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD2.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD2.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD2.SetDiscreteSol( discretSol );
          QoIEvaluator_FD2.SetNodes(x0);

          double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

          discretSol[Ia] +=epsilon;

          tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
        }
        dQdu->Print();
        std::cout<<"  ----------  FD Diff ------------"<<std::endl;
        tFD_sens.Print();

        std::cout<<"  ---------- dQdu Analytic - FD Diff ------------"<<std::endl;
        mfem::ParGridFunction tFD_diff(fespace_scalar); tFD_diff = 0.0;
        tFD_diff = *dQdu;
        tFD_diff -=tFD_sens;
        //tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
      }

      if(dQdxFD)
      {
        double epsilon = 1e-8;
        mfem::ParGridFunction tFD_sens(pfespace); tFD_sens = 0.0;
        for( int Ia = 0; Ia<gridfuncOptVar.Size(); Ia++)
        {
          std::cout<<"iter: "<< Ia<< " out of: "<<gridfuncOptVar.Size() <<std::endl;
          gridfuncOptVar[Ia] +=epsilon;

          QuantityOfInterest QoIEvaluator_FD1(PMesh, qoiType, 1);
          QoIEvaluator_FD1.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD1.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD1.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD1.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD1.SetDiscreteSol( discretSol );
          QoIEvaluator_FD1.SetNodes(x0);

          double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

          gridfuncOptVar[Ia] -=2.0*epsilon;

          QuantityOfInterest QoIEvaluator_FD2(PMesh, qoiType, 1);
          QoIEvaluator_FD2.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD2.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD2.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD2.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD2.SetDiscreteSol( discretSol );
          QoIEvaluator_FD2.SetNodes(x0);

          double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

          gridfuncOptVar[Ia] +=epsilon;

          tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
        }

        dQdxExpl->Print();
        std::cout<<"  ----------  FD Diff ------------"<<std::endl;
        tFD_sens.Print();

        std::cout<<"  ---------- dQdx Analytic - FD Diff ------------"<<std::endl;
        mfem::ParGridFunction tFD_diff(pfespace); tFD_diff = 0.0;
        tFD_diff = *dQdxExpl;
        tFD_diff -=tFD_sens;
        tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
      }

      if( BreakAfterFirstIt )
      {
        mfem::mfem_error("break before update");
      }

      //----------------------------------------------------------------------------------------------------------
      gridfuncOptVar.SetTrueVector();
      Vector & trueBounds = gridfuncLSBoundIndicator.GetTrueVector();

      // impose desing variable bounds - set xxmin and xxmax
      xxmin=trueOptvar; xxmin-=max_ch;
      xxmax=trueOptvar; xxmax+=max_ch;
      for(int li=0;li<xxmin.Size();li++){
        if( trueBounds[li] ==1.0)
        {
          xxmin[li] = -1e-8;
          xxmax[li] =  1e-8;
        }
      }

      x_gf.ProjectCoefficient(*trueSolution);
      //ParGridFunction objGradGF(pfespace); objGradGF = objgrad;
      paraview_dc.SetCycle(i);
      paraview_dc.SetTime(i*1.0);
      //paraview_dc.RegisterField("ObjGrad",&objGradGF);
      paraview_dc.RegisterField("Solution",&x_gf);
      paraview_dc.Save();

      double localGradNormSquared = std::pow(objgrad.Norml2(), 2);
      double globGradNorm;
  #ifdef MFEM_USE_MPI
    MPI_Allreduce(&localGradNormSquared, &globGradNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  #endif
    globGradNorm = std::sqrt(globGradNorm);

    if (myid == 0)
    {
      std:cout<<"Iter: "<<i<<" obj: "<<val<<" with: "<<ObjVal<<" | "<<meshQualityVal<<" objGrad_Norm: "<<globGradNorm<<std::endl;
    }

  #ifdef MFEM_USE_PETSC
      double  conDummy = -0.1;
      mmaPetsc->Update(trueOptvar,objgrad,&conDummy,&volgrad,xxmin,xxmax);
  #else
      mfem:Vector conDummy(1);  conDummy= -0.1;
      mma->Update(i, objgrad, conDummy, volgrad, xxmin,xxmax, trueOptvar);
  #endif

      gridfuncOptVar.SetFromTrueVector();

      // std::string tDesingName = "DesingVarVec";
      // desingVarVec.Save( tDesingName.c_str() );

      // std::string tFieldName = "FieldVec";
      // tPreassureGF.Save( tFieldName.c_str() );
    }

    if (visualization)
    {
        x0 -= x;
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, x0,
                              "Displacements", 400, 400, 300, 300, "jRmclA");
    }

      {
        ostringstream mesh_name;
        mesh_name << "optimized.mesh";
        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        PMesh->PrintAsOne(mesh_ofs);
    }
  }

  delete PMesh;

  return 0;
}
