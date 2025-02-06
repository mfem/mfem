// Sample runs:

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "gslib.h"
#include "bounds.hpp"

using namespace std;
using namespace mfem;

GridFunction *GetDetJacobian(Mesh *mesh)
{
   int mesh_order = mesh->GetNodalFESpace()->GetMaxElementOrder();
   int det_order = 2*mesh_order-1;
   int dim = mesh->Dimension();
   L2_FECollection *fec = new L2_FECollection(det_order, dim, BasisType::GaussLobatto);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   GridFunction *detgf = new GridFunction(fespace);
   detgf->MakeOwner(fec);
   Array<int> dofs;
   mesh->DeleteGeometricFactors();

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      // std::cout << e << " k10e\n";
      const FiniteElement *fe = fespace->GetFE(e);
      const IntegrationRule ir = fe->GetNodes();
      ElementTransformation *transf = mesh->GetElementTransformation(e);
      DenseMatrix Jac(fe->GetDim());
      const NodalFiniteElement *nfe = dynamic_cast<const NodalFiniteElement*>
                                      (fe);
      const Array<int> &irordering = nfe->GetLexicographicOrdering();
      IntegrationRule ir2 = irordering.Size() ?
                            ir.Permute(irordering) :
                            ir;

      Vector detvals(ir2.GetNPoints());
      Vector loc(dim);
      for (int q = 0; q < ir2.GetNPoints(); q++)
      {
         IntegrationPoint ip = ir2.IntPoint(q);
         transf->SetIntPoint(&ip);
         transf->Transform(ip, loc);
         Jac = transf->Jacobian();
         detvals(q) = Jac.Det();
      }

      fespace->GetElementDofs(e, dofs);
      if (irordering.Size())
      {
         for (int i = 0; i < dofs.Size(); i++)
         {
            (*detgf)(dofs[i]) = detvals(irordering[i]);
         }
      }
      else
      {
         detgf->SetSubVector(dofs, detvals);
      }
   }
   return detgf;
}

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   int order = 3;
   int mr = 8;
   int nbrute = 1000;
   int seed = 5;
   int outsuffix = 0;
   string mesh_file = "triplept2d.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&mr, "-mr", "--mr", "Finite element polynomial degree");
   args.AddOption(&nbrute, "-nh", "--nh", "Finite element polynomial degree");
   args.AddOption(&seed, "-seed", "--seed", "Seed");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&outsuffix, "-out", "--out", "out suffix");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   GridFunction *x = mesh.GetNodes();
   if (!x)
   {
      mesh.SetCurvature(order, false, -1, Ordering::byNODES);
      x = mesh.GetNodes();
   }
   int mesh_order = x->FESpace()->GetOrder(0);
   std::cout << "The mesh order is: " << mesh_order << std::endl;

   const int dim = mesh.Dimension();
   int det_order = dim*mesh_order - 1;
   // int det_order = 2*mesh_order;
   std::cout << "The determinant order is: "  << det_order << std::endl;

   int n1D = det_order + 1;
   int b_type = BasisType::GaussLobatto;
   Poly_1D poly1d;
   Poly_1D::Basis basis1d(poly1d.GetBasis(det_order, b_type));
   std::cout << "N and M are: " << n1D << " " << mr << std::endl;

   Vector ref_nodes(n1D);
   const double *temp = poly1d.GetPoints(det_order, b_type);
   ref_nodes = temp;

   Vector chebX = GetChebyshevNodes(mr); // [-1, 1]
   ScaleNodes(chebX, 0.0, 1.0, chebX); // [0, 1]

   DenseMatrix lbound, ubound;
   GetGSLIBBasisBounds(n1D, mr, lbound, ubound);

   IntegrationRule irule(n1D);
   QuadratureFunctions1D::GaussLobatto(n1D, &irule);
   Vector gllW(n1D), gllX(n1D);

   for (int i = 0; i < n1D; i++)
   {
      gllW(i) = irule.IntPoint(i).weight;
      gllX(i) = irule.IntPoint(i).x;
   }

   std::string filename = "../scripts/bnddata_spts_lobatto_" + std::to_string(n1D) + "_bpts_opt_" + std::to_string(mr) + ".txt";
   // std::string filename = "bnddata_" + std::to_string(n1D) + "_" + std::to_string(mr) + "_opt.txt";

   GridFunction *detgf = GetDetJacobian(&mesh);

   L2_FECollection fecp(0, dim);
   FiniteElementSpace fesp(&mesh, &fecp);
   GridFunction cusboundmin(&fesp), cusboundmax(&fesp),
                bruteboundmin(&fesp), bruteboundmax(&fesp),
                bernboundmin(&fesp), bernboundmax(&fesp),
                bernerrmin(&fesp), bernerrmax(&fesp),
                cuserrmin(&fesp), cuserrmax(&fesp),
                impprovemin(&fesp), impprovemax(&fesp);

   Array<int> dofs;
   Vector detvals;

   DenseMatrix lboundT, uboundT;
   Vector gllT, intT;
   ReadCustomBounds(gllT, intT, lboundT, uboundT, filename);

   for (int e = 0; e < mesh.GetNE(); e++)
   {
      const FiniteElementSpace *fespace = detgf->FESpace();
      const FiniteElement *fe = fespace->GetFE(e);
      fespace->GetElementDofs(e, dofs);
      detgf->GetSubVector(dofs, detvals);
      Vector qpminCus, qpmaxCus;
      if (intT.Size() > 0)
      {
         Get2DBounds(gllX, intT, gllW, lboundT, uboundT, detvals, qpminCus, qpmaxCus, true);
      }
      else
      {
         // my implementation of gslib
         // Get2DBounds(gllX, chebX, gllW, lbound, ubound, detvals, qpminCus, qpmaxCus, true);
      }
      cusboundmin(e) = qpminCus.Min();
      cusboundmax(e) = qpminCus.Max();


      // Brute force also
      IntegrationPoint ip;
      double brute_el_min = std::numeric_limits<double>::infinity();
      double brute_el_max = -std::numeric_limits<double>::infinity();
      ElementTransformation *transf = mesh.GetElementTransformation(e);
      DenseMatrix Jac(fe->GetDim());

      for (int j = 0; j < nbrute; j++)
      {
         for (int i = 0; i < nbrute; i++)
         {
            ip.x = i/(nbrute-1.0);
            ip.y = j/(nbrute-1.0);
            transf->SetIntPoint(&ip);
            Jac = transf->Jacobian();
            double detj = Jac.Det();
            brute_el_min = std::min(brute_el_min, detj);
            brute_el_max = std::max(brute_el_max, detj);
         }
      }
      bruteboundmin(e) = brute_el_min;
      bruteboundmax(e) = brute_el_max;
      cuserrmin(e) = std::fabs(bruteboundmin(e)-cusboundmin(e));
      cuserrmax(e) = std::fabs(bruteboundmax(e)-cusboundmax(e));
   }

   // Bernstein based bounds
   DG_FECollection fec_bern(det_order, dim, BasisType::Positive);
   FiniteElementSpace fes_bern(&mesh, &fec_bern);
   GridFunction detgf_pos(&fes_bern);
   detgf_pos.ProjectGridFunction(*detgf);

   for (int e = 0; e < mesh.GetNE(); e++)
   {
      fes_bern.GetElementDofs(e, dofs);
      detgf_pos.GetSubVector(dofs, detvals);
      double minv = detvals.Min();
      double maxv = detvals.Max();
      bernboundmin(e) = minv;
      bernboundmax(e) = maxv;
      bernerrormin(e) = std::fabs(minv-bruteboundmin(e));
      bernerrormax(e) = std::fabs(maxv-bruteboundmax(e));
   }

   // Do error plots
   {

   }

   ParaViewDataCollection *pdber = NULL;
   {
      pdber = new ParaViewDataCollection("mesh_detj_bounds_comparison", &mesh);
      pdber->SetPrefixPath("ParaView");
      pdber->RegisterField("cusboundmin", &cusboundmin);
      pdber->RegisterField("cusboundmax", &cusboundmax);
      pdber->RegisterField("bernboundmin", &bernboundmin);
      pdber->RegisterField("bernboundmax", &bernboundmax);
      pdber->RegisterField("bruteboundmin", &bruteboundmin);
      pdber->RegisterField("bruteboundmax", &bruteboundmax);
      pdber->SetLevelsOfDetail(det_order);
      pdber->SetDataFormat(VTKFormat::BINARY);
      pdber->SetHighOrderOutput(true);
      pdber->SetCycle(0);
      pdber->SetTime(0.0);
      pdber->Save();
   }

   return 0;
}
