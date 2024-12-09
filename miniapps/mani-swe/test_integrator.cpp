#include "mfem.hpp"
#include "manihyp.hpp"

using namespace mfem;
void gaussian_initial(const Vector &x, Vector &u)
{
   // const real_t theta = std::acos(x[2]/std::sqrt(x*x));
   // const real_t hmin = 1;
   // const real_t hmax = 2;
   // const real_t sigma = 0.2;
   // u = 0.0;
   // u[0] = hmin + (hmax - hmin)*std::exp(-theta*theta/(2*sigma*sigma));

   const real_t hmin = 1;
   const real_t hmax = 1;
   const real_t sigma = 0.2;
   const real_t r2 = x[0]*x[0] + x[1]*x[1];
   u = 1.0;
   u[0] = hmin + (hmax - hmin)*std::exp(-r2/(2*sigma*sigma));
   if (x.Size() == 3) { u[3] = 0.0; }
}

int main(int argc, char *argv[])
{
   const int order = 4;
   const int dim = 2;
   const int sdim = 3;
   DG_FECollection dg_fec(order, dim);
   const real_t h = 0.01;


   const int nrElem = 1;
   const int nrVert = 4;
   const int nrBdr = 4;
   VectorFunctionCoefficient u0_quad(dim + 1, gaussian_initial);
   VectorFunctionCoefficient u0_mani_phys(sdim + 1, gaussian_initial);
   ManifoldStateCoefficient u0_mani(u0_mani_phys, 1, 1, dim);

   // 1. 2D Quad mesh
   Mesh quad("./data/periodic-square-2d.mesh");
   // Mesh quad(dim, nrVert, nrElem, nrBdr, dim);
   // quad.AddVertex(0, 0);
   // quad.AddVertex(1, 0);
   // quad.AddVertex(1, 1);
   // quad.AddVertex(0, 1);
   //
   // quad.AddQuad(0,1,2,3);
   // 
   // quad.AddBdrSegment(0, 1);
   // quad.AddBdrSegment(1, 2);
   // quad.AddBdrSegment(2, 3);
   // quad.AddBdrSegment(3, 0);
   //
   // quad.FinalizeQuadMesh();
   // quad.Save("quad.mesh");

   FiniteElementSpace fes_quad(&quad, &dg_fec, dim+1);
   GridFunction x_quad(&fes_quad), y_quad(&fes_quad);

   NonlinearForm form_quad(&fes_quad);
   ShallowWaterFlux swe_flux_quad(dim);

   RusanovFlux swe_quad_numer(swe_flux_quad);
   HyperbolicFormIntegrator swe_integ_quad(swe_quad_numer);

   form_quad.AddDomainIntegrator(&swe_integ_quad);
   form_quad.UseExternalIntegrators();
   x_quad.ProjectCoefficient(u0_quad);
   form_quad.Mult(x_quad, y_quad);

   // 2. 3D Quad mesh
   Mesh mani("./data/periodic-square-3d.mesh");
   // Mesh mani(dim, nrVert, nrElem, nrBdr, sdim);
   // mani.AddVertex(0, 0, 10.0);
   // mani.AddVertex(1, 0, 10.0);
   // mani.AddVertex(1, 1, 10.0);
   // mani.AddVertex(0, 1, 10.0);
   //
   // mani.AddQuad(0,1,2,3);
   //
   // mani.AddBdrSegment(0, 1);
   // mani.AddBdrSegment(1, 2);
   // mani.AddBdrSegment(2, 3);
   // mani.AddBdrSegment(3, 0);
   //
   // mani.FinalizeQuadMesh();
   // mani.SetCurvature(order, true);
   // mani.Save("mani.mesh");

   FiniteElementSpace fes_mani(&mani, &dg_fec, dim+1);
   GridFunction x_mani(&fes_mani), y_mani(&fes_mani);

   NonlinearForm form_mani(&fes_mani);
   ShallowWaterFlux swe_flux_phys(sdim);
   ManifoldCoord coord(dim, sdim);
   ManifoldFlux swe_flux_mani(swe_flux_phys, coord, 1);

   ManifoldRusanovFlux swe_mani_numer(swe_flux_mani);
   ManifoldHyperbolicFormIntegrator swe_integ_mani(swe_mani_numer);

   form_mani.AddDomainIntegrator(&swe_integ_mani);
   form_mani.UseExternalIntegrators();
   x_mani.ProjectCoefficient(u0_mani);
   form_mani.Mult(x_mani, y_mani);


   // Logging
   out << "X in quad: " << std::endl;
   x_quad.Print(out, x_quad.Size());
   out << "X in mani: " << std::endl;
   x_mani.Print(out, x_mani.Size());
   out << "Y in quad: " << std::endl;
   y_quad.Print(out, y_quad.Size());
   out << "Y in mani: " << std::endl;
   y_mani.Print(out, y_mani.Size());
   out << y_mani.DistanceTo(y_quad) << std::endl;
}
