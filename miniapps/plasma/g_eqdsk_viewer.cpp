// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "g_eqdsk_data.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::plasma;

void ShiftMesh(double x0, double y0, Mesh &mesh);

int main(int argc, char *argv[])
{
   const char *eqdsk_file = "";

   int order = 1;
   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&eqdsk_file, "-eqdsk", "--eqdsk-file",
                  "G EQDSK input file.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   named_ifgzstream ieqdsk(eqdsk_file);
   if (!ieqdsk)
   {
      return 1;
   }

   G_EQDSK_Data eqdsk(ieqdsk);
   eqdsk.PrintInfo();
   eqdsk.DumpGnuPlotData("gnuplot_eqdsk");

   G_EQDSK_Psi_Coefficient psiCoef(eqdsk);
   G_EQDSK_FPol_Coefficient fPolCoef(eqdsk);
   G_EQDSK_Pres_Coefficient presCoef(eqdsk);
   G_EQDSK_Q_Coefficient qCoef(eqdsk);
   G_EQDSK_NxGradPsi_Coefficient nxGradPsiCoef(eqdsk);
   G_EQDSK_BPol_Coefficient BPolCoef(eqdsk);
   G_EQDSK_BTor_Coefficient BTorCoef(eqdsk);
   G_EQDSK_JTor_Coefficient JTorCoef(eqdsk);

   Mesh mesh = Mesh::MakeCartesian2D(eqdsk.GetNumPtsR(),
                                     eqdsk.GetNumPtsZ(),
                                     Element::QUADRILATERAL,
                                     false,
                                     eqdsk.GetRExtent(),
                                     eqdsk.GetZExtent());

   ShiftMesh(eqdsk.GetRMin(), eqdsk.GetZMid() - eqdsk.GetZExtent()/2.0, mesh);

   H1_FECollection fec_h1(order, 2);
   FiniteElementSpace fes_h1(&mesh, &fec_h1);
   FiniteElementSpace fes_h1v(&mesh, &fec_h1, 2);

   GridFunction psi(&fes_h1);
   psi.ProjectCoefficient(psiCoef);

   GridFunction nxGradPsi(&fes_h1v);
   nxGradPsi.ProjectCoefficient(nxGradPsiCoef);

   GridFunction fPol(&fes_h1);
   fPol.ProjectCoefficient(fPolCoef);

   GridFunction pres(&fes_h1);
   pres.ProjectCoefficient(presCoef);

   GridFunction q(&fes_h1);
   q.ProjectCoefficient(qCoef);

   GridFunction BPol(&fes_h1v);
   BPol.ProjectCoefficient(BPolCoef);

   GridFunction BTor(&fes_h1);
   BTor.ProjectCoefficient(BTorCoef);

   GridFunction JTor(&fes_h1);
   JTor.ProjectCoefficient(JTorCoef);

   {
      VisItDataCollection visit_dc("G_EQDSK_Viewer", &mesh);

      visit_dc.RegisterField("Psi", &psi);
      visit_dc.RegisterField("FPol", &fPol);
      visit_dc.RegisterField("Pres", &pres);
      visit_dc.RegisterField("Q", &q);
      visit_dc.RegisterField("nxGradPsi", &nxGradPsi);
      visit_dc.RegisterField("BPol", &BPol);
      visit_dc.RegisterField("BTor", &BTor);
      visit_dc.RegisterField("JTor", &JTor);

      visit_dc.Save();
   }

   {
      int nbdr = eqdsk.GetNumBoundaryPts();
      const vector<double> &r = eqdsk.GetBoundaryRVals();
      const vector<double> &z = eqdsk.GetBoundaryZVals();

      Mesh bdr(1, nbdr, nbdr-1, 2, 2);

      for (int i=0; i<nbdr; i++)
      {
         bdr.AddVertex(r[i], z[i]);
      }

      for (int i=1; i<nbdr; i++)
      {
         bdr.AddSegment(i-1, i);
      }

      bdr.AddBdrPoint(0);
      bdr.AddBdrPoint(nbdr-1);

      bdr.FinalizeMesh();

      VisItDataCollection visit_dc("G_EQDSK_Viewer_Boundary", &bdr);
      visit_dc.Save();
   }

   {
      int nlim = eqdsk.GetNumLimiterPts();
      const vector<double> &r = eqdsk.GetLimiterRVals();
      const vector<double> &z = eqdsk.GetLimiterZVals();

      Mesh lim(1, nlim, nlim-1, 2, 2);

      for (int i=0; i<nlim; i++)
      {
         lim.AddVertex(r[i], z[i]);
      }

      for (int i=1; i<nlim; i++)
      {
         lim.AddSegment(i-1, i);
      }

      lim.AddBdrPoint(0);
      lim.AddBdrPoint(nlim-1);

      lim.FinalizeMesh();

      VisItDataCollection visit_dc("G_EQDSK_Viewer_Limiter", &lim);
      visit_dc.Save();
   }
}

void ShiftMesh(double x0, double y0, Mesh &mesh)
{
   class ShiftCoef : public VectorCoefficient
   {
   private:
      double xs_, ys_;

   public:
      ShiftCoef(double xs, double ys) : VectorCoefficient(2), xs_(xs), ys_(ys) {}

      void Eval(Vector &v, ElementTransformation &T, const IntegrationPoint &ip)
      {
         T.Transform(ip, v);
         v[0] += xs_;
         v[1] += ys_;
      }
   };

   ShiftCoef shift(x0, y0);
   mesh.Transform(shift);
}
