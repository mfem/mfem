//                                MFEM Example 0
//
// Compile with: make getdetjelem
//
// Sample runs:  make getdetbound -j4 && ./getdetbound -nfac 32 -nh 40 -m semi-invert.mesh
// grep -i k10jac info.out | awk '{print $1 " " $2 " " $3 " " $4}' > detJnodes.out
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "gslib.h"

using namespace std;
using namespace mfem;

static IntegrationRule PermuteIR(const IntegrationRule *irule,
                                 const Array<int> &perm)
{
   const int np = irule->GetNPoints();
   MFEM_VERIFY(np == perm.Size(), "Invalid permutation size");
   IntegrationRule ir(np);
   ir.SetOrder(irule->GetOrder());

   for (int i = 0; i < np; i++)
   {
      IntegrationPoint &ip_new = ir.IntPoint(i);
      const IntegrationPoint &ip_old = irule->IntPoint(perm[i]);
      ip_new.Set(ip_old.x, ip_old.y, ip_old.z, ip_old.weight);
   }

   return ir;
}

class GSLIBBound
{
private:
   int nr = 0;
   int mr = 0;
   int mrfac = 0;
   Vector data_r;

public:
   GSLIBBound(int nr_, int mrfac_) : mrfac(mrfac_)
   {
      SetupWorkArrays(nr, mrfac*nr);
   }

   ~GSLIBBound() {};

   void SetupWorkArrays(int nrnew, int mrnew);

   void GetGridFunctionBounds(GridFunction &gf,
                              Vector &qpmin, Vector &qpmax,
                              Vector &elminmax, Vector &elminmin,
                              Vector &elmin, Vector &elmax);

   double GetDetJBounds(Mesh *mesh,
                        Vector &qpmin, Vector &qpmax,
                        Vector &elminmax, Vector &elminmin,
                        Vector &elmin, Vector &elmax);

   // write setters for nr, mrfac and getters as well
   void Setnr(int nr_) { nr = nr_; }
   void Setmrfac(int mrfac_) { mrfac = mrfac_; }
   int Getnr() { return nr; }
   int Getmr() { return mr; }
   int Getmrfac() { return mrfac; }
};

void GSLIBBound::SetupWorkArrays(int nrnew, int mrnew)
{
   MFEM_VERIFY(mrnew >= 2*nrnew,"mrnew must be at least 2*nrnew");

   if (nrnew != nr || mrnew != mr)
   {
      nr = nrnew;
      mr = mrnew;
      data_r.SetSize(lob_bnd_size(nr, mr));
      lob_bnd_setup(data_r.GetData(), nr,mr);
   }
}

void GSLIBBound::GetGridFunctionBounds(GridFunction &gf,
                                       Vector &qpmin, Vector &qpmax,
                                       Vector &elminmax, Vector &elmaxmin,
                                       Vector &elmin, Vector &elmax)
{
   const FiniteElementSpace *fespace = gf.FESpace();
   const Mesh *mesh = fespace->GetMesh();
   const int dim = mesh->Dimension();
   Array<int> dofs;
   Vector vect;
   int nelem = mesh->GetNE();
   int maxorder = fespace->GetMaxElementOrder();
   int mrs = mrfac*(maxorder+1);
   int nqpel = dim == 1 ? mrs : (dim == 2 ? mrs*mrs : mrs*mrs*mrs);
   int nqpts = nelem*nqpel;

   elmin.SetSize(nelem);
   elmax.SetSize(nelem);
   elminmax.SetSize(nelem);
   elmaxmin.SetSize(nelem);
   qpmin.SetSize(nqpts);
   qpmax.SetSize(nqpts);
   elmin = 0.0;
   elmax = 0.0;
   elmaxmin = 0.0;
   elminmax = 0.0;
   qpmin = 0.0;
   qpmax = 0.0;

   int n = 0;
   for (int e = 0; e < nelem; e++)
   {
      fespace->GetElementDofs(e, dofs);
      gf.GetSubVector(dofs, vect);
      int order = fespace->GetOrder(e);
      SetupWorkArrays(order+1, mrfac*(order+1));
      int wrksize = 2*mr;
      if (dim == 2)
      {
         wrksize = 2*mr*(nr+mr+1);
      }
      else if (dim == 3)
      {
         wrksize = 2*mr*mr*(nr+mr+1);
      }

      Vector work(wrksize);
      struct dbl_range bound;
      if (dim == 1)
      {
         bound = lob_bnd_1(data_r.GetData(),nr,mr, vect.GetData(),
                           work.GetData());
      }
      else if (dim == 2)
      {
         bound = lob_bnd_2(data_r.GetData(),nr,mr,
                           data_r.GetData(),nr,mr,
                           vect.GetData(),
                           work.GetData()); // compute bounds on u2
      }
      else
      {
         bound = lob_bnd_3(data_r.GetData(),nr,mr,
                           data_r.GetData(),nr,mr,
                           data_r.GetData(),nr,mr,
                           vect.GetData(),
                           work.GetData()); // compute bounds on u2
      }
      elmin(e) = bound.min;
      elmax(e) = bound.max;
      double min_max_bound = std::numeric_limits<double>::infinity();
      double max_min_bound = -std::numeric_limits<double>::infinity();
      for (int i = 0; i < nqpel; i++)
      {
         qpmin(n) = work[2*i];
         qpmax(n) = work[2*i+1];
         min_max_bound = std::min(min_max_bound, work[2*i+1]);
         max_min_bound = std::max(min_max_bound, work[2*i+0]);
         n++;
      }
      elminmax(e) = min_max_bound;
      elmaxmin(e) = max_min_bound;
   }
   qpmin.SetSize(n);
   qpmax.SetSize(n);
}

double GSLIBBound::GetDetJBounds(Mesh *mesh,
                                 Vector &qpmin, Vector &qpmax,
                                 Vector &elminmax, Vector &elmaxmin,
                                 Vector &elmin, Vector &elmax)
{
   int mesh_order = mesh->GetNodalFESpace()->GetMaxElementOrder();
   int det_order = 2*mesh_order;
   int dim = mesh->Dimension();
   L2_FECollection fec(det_order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(mesh, &fec);
   GridFunction detgf(&fespace);
   Array<int> dofs;

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      const FiniteElement *fe = fespace.GetFE(e);
      const IntegrationRule ir = fe->GetNodes();
      ElementTransformation *transf = mesh->GetElementTransformation(e);
      DenseMatrix Jac(fe->GetDim());
      const NodalFiniteElement *nfe = dynamic_cast<const NodalFiniteElement*>
                                      (fe);
      const Array<int> &irordering = nfe->GetLexicographicOrdering();
      IntegrationRule ir2 = irordering.Size() ?
                            PermuteIR(&ir, irordering) :
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

      fespace.GetElementDofs(e, dofs);
      if (irordering.Size())
      {
         for (int i = 0; i < dofs.Size(); i++)
         {
            detgf(dofs[i]) = detvals(irordering[i]);
         }
      }
      else
      {
         detgf.SetSubVector(dofs, detvals);
      }
   }

   GetGridFunctionBounds(detgf, qpmin, qpmax, elminmax, elmaxmin, elmin, elmax);
   return elminmax.Min();
}

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   int order = 1;
   int nfac = 4;
   int nbrute = 1000;
   string mesh_file = "semi-invert.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&nfac, "-nfac", "--seed", "Finite element polynomial degree");
   args.AddOption(&nbrute, "-nh", "--seed", "Finite element polynomial degree");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
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

   int det_order = 2*mesh_order;
   const int dim = mesh.Dimension();
   std::cout << "The determinant order is: " << det_order << std::endl;

   // Mesh meshcopy(mesh);
   L2_FECollection fec(det_order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction detgf(&fespace);

   H1Pos_FECollection fecpos(det_order, dim);
   FiniteElementSpace fespacepos(&mesh, &fecpos);
   GridFunction detgfpos(&fespacepos);

   Array<int> dofs;
   // int nr = det_order+1;
   // int mr = nfac*nr;
   // int ns = nr;
   // int ms = mr;
   // Vector data_r(lob_bnd_size(nr,mr));
   // lob_bnd_setup(data_r.GetData(), nr,mr);

   GSLIBBound detb(det_order+1, nfac);
   Vector qpmin, qpmax, elminmax, elmaxmin, elmin, elmax;
   double minmaxdetj = detb.GetDetJBounds(&mesh, qpmin, qpmax, elminmax, elmaxmin,
                                          elmin, elmax);

   double brute_min_det = std::numeric_limits<double>::infinity();
   double brute_max_det = -std::numeric_limits<double>::infinity();
   Array<int> gsl_inv_maybe, gsl_inv_sure, brute_inv_sure;

   for (int e = 0; e < mesh.GetNE(); e++)
   {
      const FiniteElement *fe = fespace.GetFE(e);
      const IntegrationRule ir = fe->GetNodes();
      ElementTransformation *transf = mesh.GetElementTransformation(e);
      DenseMatrix Jac(fe->GetDim());
      // const NodalFiniteElement *nfe = dynamic_cast<const NodalFiniteElement*>
      //                               (fe);
      // const Array<int> &irordering = nfe->GetLexicographicOrdering();
      // IntegrationRule ir2 = irordering.Size() ?
      //                       PermuteIR(&ir, irordering) :
      //                       ir;
      // Vector detvals(ir2.GetNPoints());
      // Vector loc(dim);
      // for (int q = 0; q < ir2.GetNPoints(); q++)
      // {
      //    IntegrationPoint ip = ir2.IntPoint(q);
      //    transf->SetIntPoint(&ip);
      //    transf->Transform(ip, loc);
      //    Jac = transf->Jacobian();
      //    detvals(q) = Jac.Det();
      // }

      // fespace.GetElementDofs(e, dofs);
      // if (irordering.Size())
      // {
      //    for (int i = 0; i < dofs.Size(); i++)
      //    {
      //       detgf(dofs[i]) = detvals(irordering[i]);
      //    }
      // }
      // else
      // {
      //    detgf.SetSubVector(dofs, detvals);
      // }

      // // GSLIB work
      // double work2[2*mr*(ns+ms+1)];
      // struct dbl_range bound;

      // bound = lob_bnd_2(data_r.GetData(),nr,mr,
      //                   data_r.GetData(),nr,mr,
      //                   detvals.GetData(), work2); // compute bounds on u2
      // gsl_min_det = std::min(gsl_min_det, bound.min);
      // gsl_max_det = std::max(gsl_max_det, bound.max);

      // double min_max_bound = std::numeric_limits<double>::infinity();
      // for (int i = 0; i < mr*ms; i++)
      // {
      //    min_max_bound = std::min(min_max_bound, work2[2*i+1]);
      // }
      // if (min_max_bound < 0)
      // {
      //    gsl_inv_sure.Append(e);
      // }

      IntegrationPoint ip;
      double brute_el_min = std::numeric_limits<double>::infinity();
      double brute_el_max = -std::numeric_limits<double>::infinity();

      for (int i = 0; i < nbrute; i++)
      {
         for (int j = 0; j < nbrute; j++)
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
      brute_min_det = std::min(brute_el_min, brute_min_det);
      brute_max_det = std::max(brute_el_max, brute_max_det);
      if (brute_el_min < 0)
      {
         brute_inv_sure.Append(e);
      }
   }

   detgf.Save("detJ.gf");
   mesh.Save("detJmesh.mesh");
   detgfpos.Save("detJpos.gf");

   std::cout << "Minimum determinant detected by gslib vs brute force: " <<
             qpmin.Min() << " " << brute_min_det << std::endl;
   std::cout << "Maximum determinant detected by gslib vs brute force: " <<
             qpmax.Max() << " " << brute_max_det << std::endl;
   std::cout << "The minimum determinant is at-least " << minmaxdetj << std::endl;

   return 0;
}
