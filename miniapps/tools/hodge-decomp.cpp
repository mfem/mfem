// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//    -------------------------------------------------------------------
//    Hodge Decomposition Miniapp: Split vector fields into
//    -------------------------------------------------------------------

#include "../common/pfem_extras.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;

using miniapps::H1_ParFESpace;
using miniapps::ND_ParFESpace;
using miniapps::RT_ParFESpace;
//using miniapps::DivergenceFreeNDProjector;
using miniapps::DivergenceFreeRTProjector;
using miniapps::IrrotationalNDProjector;
//using miniapps::IrrotationalFreeRTProjector;

static int nr_   = 1;
static int nphi_ = 0;

static double r_ = 0.4;
static double R_ = 1.1;

void w_exact(const Vector &, Vector &);

double a_exact(const Vector &);
void da_exact(const Vector &, Vector &);

void b_exact(const Vector &, Vector &);
void db_exact(const Vector &, Vector &);

void c_exact(const Vector &, Vector &);

int main(int argc, char ** argv)
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/toroid-hex.mesh";
   int order = 1;
   int serial_ref_levels = 2;
   int parallel_ref_levels = 0;
   bool visualization = 1;

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+3, offy = Wh+25; // window offsets

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   int par_ref_levels = parallel_ref_levels;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   H1_ParFESpace fespace_h1(&pmesh, order, pmesh.Dimension());
   ND_ParFESpace fespace_nd(&pmesh, order, pmesh.Dimension());
   RT_ParFESpace fespace_rt(&pmesh, order, pmesh.Dimension());

   ParDiscreteGradOperator Grad(&fespace_h1, &fespace_nd);
   Grad.Assemble();
   Grad.Finalize();

   ParDiscreteCurlOperator Curl(&fespace_nd, &fespace_rt);
   Curl.Assemble();
   Curl.Finalize();

   ParGridFunction a_h1(&fespace_h1);
   ParGridFunction da_nd(&fespace_nd);

   ParGridFunction b_nd(&fespace_nd);
   ParGridFunction db_rt(&fespace_rt);

   ParGridFunction w_nd(&fespace_nd);
   ParGridFunction irr_w_nd(&fespace_nd);

   ParGridFunction w_rt(&fespace_rt);
   ParGridFunction df_w_rt(&fespace_rt);
   ParGridFunction w_c_rt(&fespace_rt);

   FunctionCoefficient aCoef(a_exact);
   VectorFunctionCoefficient daCoef(pmesh.SpaceDimension(), da_exact);

   VectorFunctionCoefficient bCoef(pmesh.SpaceDimension(), b_exact);
   VectorFunctionCoefficient dbCoef(pmesh.SpaceDimension(), db_exact);

   VectorFunctionCoefficient cCoef(pmesh.SpaceDimension(), c_exact);

   VectorFunctionCoefficient wCoef(pmesh.SpaceDimension(), w_exact);

   VectorGridFunctionCoefficient irr_w_Coef(&irr_w_nd);

   a_h1.ProjectCoefficient(aCoef);
   b_nd.ProjectCoefficient(bCoef);

   Grad.Mult(a_h1, da_nd);
   Curl.Mult(b_nd, db_rt);

   double err_a_h1 = a_h1.ComputeL2Error(aCoef);
   double err_da_nd = da_nd.ComputeL2Error(daCoef);

   double err_b_nd = b_nd.ComputeL2Error(bCoef);
   double err_db_rt = db_rt.ComputeL2Error(dbCoef);

   if (myid == 0)
   {
      cout << "Error in a (H1):  " << err_a_h1 << endl;
      cout << "Error in da (ND): " << err_da_nd << endl;
      cout << "Error in b (ND):  " << err_b_nd << endl;
      cout << "Error in db (RT): " << err_db_rt << endl;
   }

   w_nd.ProjectCoefficient(wCoef);
   w_rt.ProjectCoefficient(wCoef);

   double err_w_nd = w_nd.ComputeL2Error(wCoef);
   double err_w_rt = w_rt.ComputeL2Error(wCoef);

   if (myid == 0)
   {
      cout << "Error in w (ND): " << err_w_nd << endl;
      cout << "Error in w (RT): " << err_w_rt << endl;
   }

   map<string, socketstream*> socks;
   {
      socks["w_nd"] = new socketstream;
      socks["w_nd"]->precision(8);
      VisualizeField(*socks["w_nd"], vishost, visport,
                     w_nd, "w ND", Wx, Wy, Ww, Wh);
      Wy += offy;

      socks["w_rt"] = new socketstream;
      socks["w_rt"]->precision(8);
      VisualizeField(*socks["w_rt"], vishost, visport,
                     w_rt, "w RT", Wx, Wy, Ww, Wh);
   }

   IrrotationalNDProjector irr_nd(fespace_h1, fespace_nd, 2 * order + 1);
   irr_nd.Mult(w_nd, irr_w_nd);

   double err_irr_w_nd = irr_w_nd.ComputeL2Error(daCoef);

   if (myid == 0)
   {
      cout << "Error in da (ND): " << err_da_nd << endl;
      cout << "Error in irr w (ND): " << err_irr_w_nd << endl;
   }
   {
      Wy -= offy;
      Wx += offx;

      socks["da_nd"] = new socketstream;
      socks["da_nd"]->precision(8);
      VisualizeField(*socks["da_nd"], vishost, visport,
                     irr_w_nd, "irr w ND", Wx, Wy, Ww, Wh);
   }

   DivergenceFreeRTProjector df_rt(fespace_nd, fespace_rt, 2 * order + 1);

   df_rt.Mult(w_rt, df_w_rt);

   double err_df_w_rt = df_w_rt.ComputeL2Error(dbCoef);

   if (myid == 0)
   {
      cout << "Error in df w (RT): " << err_df_w_rt << endl;
   }

   {
      Wy += offy;

      socks["db_rt"] = new socketstream;
      socks["db_rt"]->precision(8);
      VisualizeField(*socks["db_rt"], vishost, visport,
                     df_w_rt, "df w RT", Wx, Wy, Ww, Wh);
   }

   w_c_rt.ProjectCoefficient(irr_w_Coef);
   w_c_rt += df_w_rt;
   w_c_rt *= -1.0;
   w_c_rt += w_rt;

   double err_w_c_rt = w_c_rt.ComputeL2Error(cCoef);
   if (myid == 0)
   {
      cout << "Error in c (RT): " << err_w_c_rt << endl;
   }
   {
      Wx += offx;

      socks["w_c_rt"] = new socketstream;
      socks["w_c_rt"]->precision(8);
      VisualizeField(*socks["w_c_rt"], vishost, visport,
                     w_c_rt, "w c RT", Wx, Wy, Ww, Wh);
   }
}

void w_exact(const Vector &x, Vector &w)
{
   w.SetSize(3);

   double da_data[3];
   double db_data[3];
   Vector da(da_data, 3);
   Vector db(db_data, 3);

   da_exact(x, da);
   db_exact(x, db);
   c_exact(x, w);
   w += da;
   w += db;
}

double a_exact(const Vector &x)
{
   double r = sqrt(x[0] * x[0] + x[1] * x[1]);
   double phi = atan2(x[1], x[0]);

   double ar = 0.5 * M_PI * nr_ * (r - R_) / r_;
   double ap = phi * nphi_;
   double az = 0.5 * M_PI * nr_ * x[2] / r_;

   return (2.0 * r_ / (M_PI * nr_)) * cos(ar) * cos(ap) * cos(az);
}

void da_exact(const Vector &x, Vector &da)
{
   da.SetSize(3);

   double r = sqrt(x[0] * x[0] + x[1] * x[1]);
   double phi = atan2(x[1], x[0]);

   double ar = 0.5 * M_PI * nr_ * (r - R_) / r_;
   double ap = phi * nphi_;
   double az = 0.5 * M_PI * nr_ * x[2] / r_;

   double drdx = x[0] / r;
   double drdy = x[1] / r;
   double dpdx = -x[1] / (r * r);
   double dpdy =  x[0] / (r * r);

   double dardr = 0.5 * M_PI * nr_ / r_;
   double dapdp = (double)nphi_;
   double dazdz = 0.5 * M_PI * nr_ / r_;

   da(0) = -(dardr * drdx * sin(ar) * cos(ap) +
             dapdp * dpdx * cos(ar) * sin(ap)
            ) * cos(az);
   da(1) = -(dardr * drdy * sin(ar) * cos(ap) +
             dapdp * dpdy * cos(ar) * sin(ap)
            ) * cos(az);
   da(2) = -dazdz * cos(ar) * cos(ap) * sin(az);
   da *= (2.0 * r_ / (M_PI * nr_));
}

void b_exact(const Vector &x, Vector &b)
{
   b.SetSize(3);

   double r = sqrt(x[0] * x[0] + x[1] * x[1]);
   double phi = atan2(x[1], x[0]);

   double ar = 0.5 * M_PI * nr_ * (r - R_) / r_;
   double ap = phi * nphi_;
   double az = 0.5 * M_PI * nr_ * x[2] / r_;

   double cp = x[0] / r;
   double sp = x[1] / r;

   b(0) = cp * cos(ap) * cos(az);
   b(1) = sp * cos(ap) * cos(az);
   b(2) = cos(ar) * cos(ap);

   b *= r_ / (M_PI * nr_);
}

void db_exact(const Vector &x, Vector &db)
{
   db.SetSize(3);

   double r = sqrt(x[0] * x[0] + x[1] * x[1]);
   double phi = atan2(x[1], x[0]);

   double ar = 0.5 * M_PI * nr_ * (r - R_) / r_;
   double ap = phi * nphi_;
   double az = 0.5 * M_PI * nr_ * x[2] / r_;

   double cp = x[0] / r;
   double sp = x[1] / r;

   double drdx = x[0] / r;
   double drdy = x[1] / r;
   double dpdx = -x[1] / (r * r);
   double dpdy =  x[0] / (r * r);

   double dardr = 0.5 * M_PI * nr_ / r_;
   double dapdp = (double)nphi_;
   double dazdz = 0.5 * M_PI * nr_ / r_;

   double dcpdy = -drdy * cp / r;
   double dspdx = -drdx * sp / r;

   db(0) = -(dardr * drdy * sin(ar) * cos(ap) +
             dapdp * dpdy * cos(ar) * sin(ap) -
             dazdz * sp * cos(ap) * sin(az));
   db(1) = (dardr * drdx * sin(ar) * cos(ap) +
            dapdp * dpdx * cos(ar) * sin(ap) -
            dazdz * cp * cos(ap) * sin(az));
   db(2) = (dspdx * cos(ap) * cos(az) - dapdp * dpdx * sp * sin(ap)
            -dcpdy * cos(ap) * cos(az) + dapdp * dpdy * cp * sin(ap)
           ) * cos(az);

   db *= r_ / (M_PI * nr_);
}

void c_exact(const Vector &x, Vector &c)
{
   c.SetSize(3);
   c = 0.0;
   c(0) = -x[1];
   c(1) = x[0];

   double r2 = x[0] * x[0] + x[1] * x[1];
   c *= 0.7 / r2;
   // c(0) = sin(kappa * x[0]) * (cos(kappa * x[1]) - cos(kappa * x[2]));
   // c(1) = sin(kappa * x[1]) * (cos(kappa * x[2]) - cos(kappa * x[0]));
   // c(2) = sin(kappa * x[2]) * (cos(kappa * x[0]) - cos(kappa * x[1]));
}
