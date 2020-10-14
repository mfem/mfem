#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void Func_3D_lin(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] =  1.234 * x[0] - 2.357 * x[1] + 3.572 * x[2];
   v[1] =  2.537 * x[0] + 4.321 * x[1] - 1.234 * x[2];
   v[2] = -2.572 * x[0] + 1.321 * x[1] + 3.234 * x[2];
}

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int log = 1;
   int n = (int)ceil(pow(2*num_procs, 1.0 / 3.0));
   int dim = 3;
   int order = 2;
   int npts = 0;

   double tol = 1e-6;

   // std::ostringstream osslog; osslog << "log." << my_rank;
   // std::ofstream ofslog(osslog.str().c_str());

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      mfem::out << "element type " << type << std::endl;

      Mesh *mesh = new Mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);
      ParMesh pmesh(MPI_COMM_WORLD, *mesh);
      delete mesh;
      /*
      if (type == 4 && my_rank  == 0)
      {
	Array<int> fcs, cor;
	pmesh.GetElementFaces(11, fcs, cor);
	cout << " Element 11" << endl;
	for (int i=0; i<4; i++)
	  {
	    cout << fcs[i] << '\t' << cor[i] << endl;
	  }
      }
      */
      ostringstream oss; oss << "get_value_";
      if (type == 4) oss << "tet";
      else oss << "hex";
      VisItDataCollection visit_dc(oss.str(), &pmesh);
      visit_dc.Save();
      
      if (true)
      {
         std::ostringstream oss;
         std::ofstream ofs;
         oss << "test_t" << type << "_p" << my_rank << ".mesh";
         ofs.open(oss.str().c_str());
         pmesh.Print(ofs);

         ofs << std::endl << "shared face orientations" << std::endl;
         for (int sf = 0; sf < pmesh.GetNSharedFaces(); sf++)
         {
            int lf = pmesh.GetSharedFace(sf);
            int inf1, inf2;
            pmesh.GetFaceInfos(lf, &inf1, &inf2);
            ofs << sf << " " << lf << " " << inf1/64 << " " << inf1%64
		<< " " << inf2/64 << " " << inf2%64 << std::endl;
         }

         ofs << std::endl << "group" << std::endl;
         int ng = pmesh.GetNGroups();
         for (int g=1; g<ng; g++)
         {
            int nt = pmesh.GroupNTriangles(g);
            if (nt > 0)
            {
               ofs << "triangles" << std::endl;
               for (int t = 0; t<nt; t++)
               {
                  int f, o;
                  pmesh.GroupTriangle(g, t, f, o);
                  ofs << g << " " << t << " " << f << " " << o << std::endl;
               }
            }

            int nq = pmesh.GroupNQuadrilaterals(g);
            if (nq > 0)
            {
               ofs << "quads" << std::endl;
               for (int q = 0; q<nq; q++)
               {
                  int f, o;
                  pmesh.GroupQuadrilateral(g, q, f, o);
                  ofs << g << " " << q << " " << f << " " << o << std::endl;
               }
            }
         }

         ofs.close();
      }

      VectorFunctionCoefficient funcCoef(dim, Func_3D_lin);

      mfem::out << "3D GetVectorValue tests for element type " <<
	std::to_string(type) << endl;
      {
         H1_FECollection  h1_fec(order, dim);
         ND_FECollection  nd_fec(order+1, dim);
         RT_FECollection  rt_fec(order+1, dim);
         L2_FECollection  l2_fec(order, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);
         DG_FECollection dgi_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::INTEGRAL);

         ParFiniteElementSpace  h1_fespace(&pmesh,  &h1_fec, dim);
         ParFiniteElementSpace  nd_fespace(&pmesh,  &nd_fec);
         ParFiniteElementSpace  rt_fespace(&pmesh,  &rt_fec);
         ParFiniteElementSpace  l2_fespace(&pmesh,  &l2_fec, dim);
         ParFiniteElementSpace dgv_fespace(&pmesh, &dgv_fec, dim);
         ParFiniteElementSpace dgi_fespace(&pmesh, &dgi_fec, dim);

         ParGridFunction  h1_x( &h1_fespace);
         ParGridFunction  nd_x( &nd_fespace);
         ParGridFunction  rt_x( &rt_fespace);
         ParGridFunction  l2_x( &l2_fespace);
         ParGridFunction dgv_x(&dgv_fespace);
         ParGridFunction dgi_x(&dgi_fespace);

         VectorGridFunctionCoefficient  h1_xCoef( &h1_x);
         VectorGridFunctionCoefficient  nd_xCoef( &nd_x);
         VectorGridFunctionCoefficient  rt_xCoef( &rt_x);
         VectorGridFunctionCoefficient  l2_xCoef( &l2_x);
         VectorGridFunctionCoefficient dgv_xCoef(&dgv_x);
         VectorGridFunctionCoefficient dgi_xCoef(&dgi_x);

         h1_x.ProjectCoefficient(funcCoef);
         nd_x.ProjectCoefficient(funcCoef);
         rt_x.ProjectCoefficient(funcCoef);
         l2_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         h1_x.ExchangeFaceNbrData();
         nd_x.ExchangeFaceNbrData();
         rt_x.ExchangeFaceNbrData();
         l2_x.ExchangeFaceNbrData();
         dgv_x.ExchangeFaceNbrData();
         dgi_x.ExchangeFaceNbrData();

         Vector       f_val(dim);       f_val = 0.0;

         Vector  h1_gfc_val(dim);  h1_gfc_val = 0.0;
         Vector  nd_gfc_val(dim);  nd_gfc_val = 0.0;
         Vector  rt_gfc_val(dim);  rt_gfc_val = 0.0;
         Vector  l2_gfc_val(dim);  l2_gfc_val = 0.0;
         Vector dgv_gfc_val(dim); dgv_gfc_val = 0.0;
         Vector dgi_gfc_val(dim); dgi_gfc_val = 0.0;

         Vector  h1_gvv_val(dim);  h1_gvv_val = 0.0;
         Vector  nd_gvv_val(dim);  nd_gvv_val = 0.0;
         Vector  rt_gvv_val(dim);  rt_gvv_val = 0.0;
         Vector  l2_gvv_val(dim);  l2_gvv_val = 0.0;
         Vector dgv_gvv_val(dim); dgv_gvv_val = 0.0;
         Vector dgi_gvv_val(dim); dgi_gvv_val = 0.0;

	 mfem::out << "Shared Face Evaluation 3D" << endl;
         {
            std::ostringstream oss;
            oss << "gvv_" << type << "_" << my_rank << ".out";
            std::ofstream ofs(oss.str().c_str());

	    cout << my_rank << ": num shared faces "
		 << pmesh.GetNSharedFaces() << endl;
            for (int sf = 0; sf < pmesh.GetNSharedFaces(); sf++)
            {
               FaceElementTransformations *FET =
                  pmesh.GetSharedFaceTransformations(sf);
               ElementTransformation *T = &FET->GetElement2Transformation();
               int e = FET->Elem2No;
               int e_nbr = e - pmesh.GetNE();
               const FiniteElement   *fe = dgv_fespace.GetFaceNbrFE(e_nbr);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               Vector x(dim);

               double  h1_gfc_err = 0.0;
               double  nd_gfc_err = 0.0;
               double  rt_gfc_err = 0.0;
               double  l2_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double  h1_gvv_err = 0.0;
               double  nd_gvv_err = 0.0;
               double  rt_gvv_err = 0.0;
               double  l2_gvv_err = 0.0;
               double dgv_gvv_err = 0.0;
               double dgi_gvv_err = 0.0;

	       cout << my_rank << ": num integration points "
		    << ir.GetNPoints() << endl;
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, x);

                  funcCoef.Eval(f_val, *T, ip);

                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  rt_xCoef.Eval(rt_gfc_val, *T, ip);
                  l2_xCoef.Eval(l2_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gfc_val, *T, ip);

                  h1_x.GetVectorValue(e, ip, h1_gvv_val);
                  nd_x.GetVectorValue(e, ip, nd_gvv_val);
                  rt_x.GetVectorValue(e, ip, rt_gvv_val);
                  l2_x.GetVectorValue(e, ip, l2_gvv_val);
                  dgv_x.GetVectorValue(e, ip, dgv_gvv_val);
                  dgi_x.GetVectorValue(e, ip, dgi_gvv_val);
		  /*
                  Vector f_tan(f_val);
                  f_tan.Add(-(nor*f_val), nor);

                  Vector nd_tan(nd_gfc_val);
                  nd_tan.Add(-(nor*nd_gfc_val), nor);
		  */
                  double  h1_gfc_dist = Distance(f_val,  h1_gfc_val, dim);
                  double  nd_gfc_dist = Distance(f_val,  nd_gfc_val, dim);
                  double  rt_gfc_dist = Distance(f_val,  rt_gfc_val, dim);
                  double  l2_gfc_dist = Distance(f_val,  l2_gfc_val, dim);
                  double dgv_gfc_dist = Distance(f_val, dgv_gfc_val, dim);
                  double dgi_gfc_dist = Distance(f_val, dgi_gfc_val, dim);

                  double  h1_gvv_dist = Distance(f_val,  h1_gvv_val, dim);
                  double  nd_gvv_dist = Distance(f_val,  nd_gvv_val, dim);
                  double  rt_gvv_dist = Distance(f_val,  rt_gvv_val, dim);
                  double  l2_gvv_dist = Distance(f_val,  l2_gvv_val, dim);
                  double dgv_gvv_dist = Distance(f_val, dgv_gvv_val, dim);
                  double dgi_gvv_dist = Distance(f_val, dgi_gvv_val, dim);

                  h1_gfc_err  +=  h1_gfc_dist;
                  nd_gfc_err  +=  nd_gfc_dist;
                  rt_gfc_err  +=  rt_gfc_dist;
                  l2_gfc_err  +=  l2_gfc_dist;
                  dgv_gfc_err += dgv_gfc_dist;
                  dgi_gfc_err += dgi_gfc_dist;

                  h1_gvv_err  +=  h1_gvv_dist;
                  nd_gvv_err  +=  nd_gvv_dist;
                  rt_gvv_err  +=  rt_gvv_dist;
                  l2_gvv_err  +=  l2_gvv_dist;
                  dgv_gvv_err += dgv_gvv_dist;
                  dgi_gvv_err += dgi_gvv_dist;

                  if (log > 0 && h1_gfc_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " h1  gfc ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                         << h1_gfc_val[2] << ") "
                         << h1_gfc_dist << std::endl;
                  }
                  if (log > 0 && nd_gfc_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j
                         << " x = (" << x[0] << "," << x[1] << ","
                         << x[2] << ")\n nd  gfc ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ")\n vs. ("
                         << nd_gfc_val[0] << "," << nd_gfc_val[1] << ","
                         << nd_gfc_val[2] << ") "
                         << nd_gfc_dist << std::endl;
                     // std::cout
		     /*
                     ofs << "tangent ("
                         << f_tan[0] << "," << f_tan[1] << ","
                         << f_tan[2] << ") vs. ("
                         << nd_tan[0] << "," << nd_tan[1] << ","
                         << nd_tan[2] << ") "
                         << std::endl;
		     */
                  }
                  if (log > 0 && rt_gfc_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " rt  gfc ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << rt_gfc_val[0] << "," << rt_gfc_val[1] << ","
                         << rt_gfc_val[2] << ") "
                         << rt_gfc_dist << std::endl;
                  }
                  if (log > 0 && l2_gfc_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " l2  gfc ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << l2_gfc_val[0] << "," << l2_gfc_val[1] << ","
                         << l2_gfc_val[2] << ") "
                         << l2_gfc_dist << std::endl;
                  }
                  if (log > 0 && dgv_gfc_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " dgv gfc ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << dgv_gfc_val[0] << ","
                         << dgv_gfc_val[1] << ","
                         << dgv_gfc_val[2] << ") "
                         << dgv_gfc_dist << std::endl;
                  }
                  if (log > 0 && dgi_gfc_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " dgi gfc ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << dgi_gfc_val[0] << ","
                         << dgi_gfc_val[1] << ","
                         << dgi_gfc_val[2] << ") "
                         << dgi_gfc_dist << std::endl;
                  }
                  if (log > 0 && h1_gvv_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " h1  gvv ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << h1_gvv_val[0] << "," << h1_gvv_val[1] << ","
                         << h1_gvv_val[2] << ") "
                         << h1_gvv_dist << std::endl;
                  }
                  if (log > 0 && nd_gvv_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " nd  gvv ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << nd_gvv_val[0] << "," << nd_gvv_val[1] << ","
                         << nd_gvv_val[2] << ") "
                         << nd_gvv_dist << std::endl;
                  }
                  if (log > 0 && rt_gvv_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " rt  gvv ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << rt_gvv_val[0] << "," << rt_gvv_val[1] << ","
                         << rt_gvv_val[2] << ") "
                         << rt_gvv_dist << std::endl;
                  }
                  if (log > 0 && l2_gvv_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " l2  gvv ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << l2_gvv_val[0] << "," << l2_gvv_val[1] << ","
                         << l2_gvv_val[2] << ") "
                         << l2_gvv_dist << std::endl;
                  }
                  if (log > 0 && dgv_gvv_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " dgv gvv ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << dgv_gvv_val[0] << ","
                         << dgv_gvv_val[1] << ","
                         << dgv_gvv_val[2] << ") "
                         << dgv_gvv_dist << std::endl;
                  }
                  if (log > 0 && dgi_gvv_dist > tol)
                  {
                     // std::cout
                     ofs << e << ":" << j << " dgi gvv ("
                         << f_val[0] << "," << f_val[1] << ","
                         << f_val[2] << ") vs. ("
                         << dgi_gvv_val[0] << ","
                         << dgi_gvv_val[1] << ","
                         << dgi_gvv_val[2] << ") "
                         << dgi_gvv_dist << std::endl;
                  }
               }

               h1_gfc_err  /= ir.GetNPoints();
               nd_gfc_err  /= ir.GetNPoints();
               rt_gfc_err  /= ir.GetNPoints();
               l2_gfc_err  /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gvv_err  /= ir.GetNPoints();
               nd_gvv_err  /= ir.GetNPoints();
               rt_gvv_err  /= ir.GetNPoints();
               l2_gvv_err  /= ir.GetNPoints();
               dgv_gvv_err /= ir.GetNPoints();
               dgi_gvv_err /= ir.GetNPoints();

	       cout << my_rank << " H1:  " << h1_gfc_err << '\t' << h1_gvv_err << endl;
	       cout << my_rank << " ND:  " << nd_gfc_err << '\t' << nd_gvv_err << endl;
	       cout << my_rank << " RT:  " << rt_gfc_err << '\t' << rt_gvv_err << endl;
	       cout << my_rank << " L2:  " << l2_gfc_err << '\t' << l2_gvv_err << endl;
	       cout << my_rank << " DGv: " << dgv_gfc_err << '\t' << dgv_gvv_err << endl;
	       cout << my_rank << " DGi: " << dgi_gfc_err << '\t' << dgi_gvv_err << endl;
	       /*
               REQUIRE( h1_gfc_err == MFEM_Approx(0.0));
               // REQUIRE( nd_gfc_err == MFEM_Approx(0.0));
               REQUIRE( rt_gfc_err == MFEM_Approx(0.0));
               REQUIRE( l2_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE( h1_gvv_err == MFEM_Approx(0.0));
               // REQUIRE( nd_gvv_err == MFEM_Approx(0.0));
               REQUIRE( rt_gvv_err == MFEM_Approx(0.0));
               REQUIRE( l2_gvv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gvv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gvv_err == MFEM_Approx(0.0));
	       */
            }
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }

   std::cout << my_rank << ": Checked GridFunction::GetVectorValue at "
             << npts << " 3D points" << std::endl;

   MPI_Finalize();
}
