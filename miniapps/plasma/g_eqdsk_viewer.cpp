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
#include "../common/fem_extras.hpp"
#include "g_eqdsk_data.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::plasma;

void ShiftMesh(double x0, double y0, Mesh &mesh);

int main(int argc, char *argv[])
{
   const char *eqdsk_file = "";
   const char *mesh_file = "";

   int order = 1;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool binary = false;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&eqdsk_file, "-eqdsk", "--eqdsk-file",
                  "G EQDSK input file.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

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

   int xPos = 0, yPos = 0, w = 400, h = 300, b = 30, m = 65;

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      char skeys[] = "mmaaAcjR";
      char vkeys[] = "vvvmmaaAcjR";

      socketstream sock_fpol;
      VisualizeField(sock_fpol, vishost, visport, fPol, "Current Flux",
                     xPos, yPos, w, h, skeys);
      xPos += w;

      socketstream sock_pres;
      VisualizeField(sock_pres, vishost, visport, pres, "Pressure",
                     xPos, yPos, w, h, "mmaaAcjR");
      xPos += w;

      socketstream sock_psi;
      VisualizeField(sock_psi, vishost, visport, psi, "Poloidal Flux",
                     xPos, yPos, w, h, "mmaaAcjR");
      xPos += w;

      socketstream sock_q;
      VisualizeField(sock_q, vishost, visport, q, "Safety Factor (q)",
                     xPos, yPos, w, h, "mmaaAcjR");
      xPos = 0; yPos += h + b + m;

      socketstream sock_bpol;
      VisualizeField(sock_bpol, vishost, visport, BPol, "Poloidal B",
                     xPos, yPos, w, h, vkeys, true);
      xPos += w;

      socketstream sock_btor;
      VisualizeField(sock_btor, vishost, visport, BTor, "Toroidal B",
                     xPos, yPos, w, h, "mmaaAcjR");
      xPos += w;

      socketstream sock_jtor;
      VisualizeField(sock_jtor, vishost, visport, JTor, "Toroidal J",
                     xPos, yPos, w, h, "mmaaAcjR");
      xPos = 0; yPos += h + b;
   }

   Array<DataCollection*> dc(2); dc = NULL;

   if (visit)
   {
#ifdef MFEM_USE_SIDRE
      if (binary)
      {
         dc[0] = new SidreDataCollection("G_EQDSK_Viewer", &mesh);
      }
      else
#else
      {
         dc[0] = new VisItDataCollection("G_EQDSK_Viewer", &mesh);
         dc[0]->SetPrecision(precision);
      }
#endif
      }
   if (paraview)
   {
      ParaViewDataCollection *pd =
         new ParaViewDataCollection("G_EQDSK_Viewer", &mesh);
      pd->SetPrefixPath("ParaView");
      pd->SetHighOrderOutput(true);
      if (binary) { pd->SetDataFormat(VTKFormat::BINARY); }
      dc[1] = pd;
   }

   for (int i=0; i<2; i++)
   {
      if (dc[i] == NULL) { continue; }

      dc[i]->SetCycle(0);
      dc[i]->SetTime(0.0);

      dc[i]->RegisterField("Psi", &psi);
      dc[i]->RegisterField("FPol", &fPol);
      dc[i]->RegisterField("Pres", &pres);
      dc[i]->RegisterField("Q", &q);
      dc[i]->RegisterField("nxGradPsi", &nxGradPsi);
      dc[i]->RegisterField("BPol", &BPol);
      dc[i]->RegisterField("BTor", &BTor);
      dc[i]->RegisterField("JTor", &JTor);

      dc[i]->Save();
   }
   delete dc[0];
   delete dc[1];

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

      if (visualization)
      {
         socketstream sock;
         char vishost[] = "localhost";
         int  visport   = 19916;
         VisualizeMesh(sock, vishost, visport, bdr, "Plasma Boundary",
                       xPos, yPos, w, h, "aaA");
         xPos += w;
      }
      if (visit)
      {
#ifdef MFEM_USE_SIDRE
         if (binary)
         {
            SidreDataCollection sd("G_EQDSK_Viewer_Boundary", &bdr);
            sd.Save();
         }
         else
#else
         {
            VisItDataCollection vd("G_EQDSK_Viewer_Boundary", &bdr);
            vd.SetPrecision(precision);
            vd.Save();
         }
#endif
         }
      if (paraview)
      {
         ParaViewDataCollection pd("G_EQDSK_Viewer_Boundary", &bdr);
         pd.SetPrefixPath("ParaView");
         pd.SetHighOrderOutput(true);
         if (binary) { pd.SetDataFormat(VTKFormat::BINARY); }
         pd.Save();
      }
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

      if (visualization)
      {
         socketstream sock;
         char vishost[] = "localhost";
         int  visport   = 19916;
         VisualizeMesh(sock, vishost, visport, lim, "Limiter",
                       xPos, yPos, w, h, "aaA");
         xPos += w;
      }
      if (visit)
      {
#ifdef MFEM_USE_SIDRE
         if (binary)
         {
            SidreDataCollection sd("G_EQDSK_Viewer_Limiter", &lim);
            sd.Save();
         }
         else
#else
         {
            VisItDataCollection vd("G_EQDSK_Viewer_Limiter", &lim);
            vd.SetPrecision(precision);
            vd.Save();
         }
#endif
         }
      if (paraview)
      {
         ParaViewDataCollection pd("G_EQDSK_Viewer_Limiter", &lim);
         pd.SetPrefixPath("ParaView");
         pd.SetHighOrderOutput(true);
         if (binary) { pd.SetDataFormat(VTKFormat::BINARY); }
         pd.Save();
      }
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
