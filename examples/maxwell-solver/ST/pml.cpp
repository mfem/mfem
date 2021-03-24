#include "pml.hpp"


CartesianPML::CartesianPML(Mesh *mesh_, Array2D<double> length_)
   : mesh(mesh_), length(length_)
{
   dim = mesh->Dimension();
   SetBoundaries();
}

void CartesianPML::SetBoundaries()
{
   comp_dom_bdr.SetSize(dim, 2);
   dom_bdr.SetSize(dim, 2);
   // initialize with any vertex
   for (int i = 0; i < dim; i++)
   {
      dom_bdr(i, 0) = mesh->GetVertex(0)[i];
      dom_bdr(i, 1) = mesh->GetVertex(0)[i];
   }

   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      Array<int> bdr_vertices;
      mesh->GetBdrElementVertices(i, bdr_vertices);
      for (int j = 0; j < bdr_vertices.Size(); j++)
      {
         for (int k = 0; k < dim; k++)
         {
            dom_bdr(k, 0) = min(dom_bdr(k, 0), mesh->GetVertex(bdr_vertices[j])[k]);
            dom_bdr(k, 1) = max(dom_bdr(k, 1), mesh->GetVertex(bdr_vertices[j])[k]);
         }
      }
   }

   for (int i = 0; i < dim; i++)
   {
      comp_dom_bdr(i, 0) = dom_bdr(i, 0) + length(i, 0);
      comp_dom_bdr(i, 1) = dom_bdr(i, 1) - length(i, 1);
   }
}

void CartesianPML::SetAttributes(Mesh *mesh_)
{
   int nrelem = mesh_->GetNE();
   elems.SetSize(nrelem);

   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = mesh_->GetElement(i);
      Array<int> vertices;
      // Initialize Attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();
      // Check if any vertex is in the pml
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         double *coords = mesh_->GetVertex(vert_idx);
         for (int comp = 0; comp < dim; ++comp)
         {
            if (coords[comp] > comp_dom_bdr(comp, 1) ||
                coords[comp] < comp_dom_bdr(comp, 0))
            {
               in_pml = true;
               break;
            }
         }
      }
      if (in_pml)
      {
         elems[i] = 0;
         el->SetAttribute(2);
      }
   }
   mesh_->SetAttributes();
}

void CartesianPML::StretchFunction(const Vector &x,
                                   vector<complex<double>> &dxs, double omega)
{
   complex<double> zi = complex<double>(0., 1.);

   double n = 2.0;
   double c = 5.0;
   double coeff;
   // Stretch in each direction independently
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_dom_bdr(i, 1))
      {
         coeff = n * c / omega / pow(length(i, 1), n);
         dxs[i] = 1.0 + zi * coeff * abs(pow(x(i) - comp_dom_bdr(i, 1), n - 1.0));
      }
      if (x(i) <= comp_dom_bdr(i, 0))
      {
         coeff = n * c / omega / pow(length(i, 0), n);
         dxs[i] = 1.0 + zi * coeff * abs(pow(x(i) - comp_dom_bdr(i, 0), n - 1.0));
      }
   }
}


double pml_detJ_Re(const Vector & x, CartesianPML * pml)
{
   int dim = pml->dim;
   double omega = pml->omega;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml->StretchFunction(x, dxs, omega);
   for (int i=0; i<dim; ++i) det *= dxs[i];
   return det.real();
}

double pml_detJ_Im(const Vector & x, CartesianPML * pml)
{
   int dim = pml->dim;
   double omega = pml->omega;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml->StretchFunction(x, dxs, omega);
   for (int i=0; i<dim; ++i) det *= dxs[i];
   return det.imag();
}

void pml_detJ_JT_J_inv_Re(const Vector & x, CartesianPML * pml , DenseMatrix & M)
{
   int dim = pml->dim;
   double omega = pml->omega;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml->StretchFunction(x, dxs, omega);

   for (int i = 0; i<dim; ++i)
   {
      det *= dxs[i];
   }

   M=0.0;
   for (int i = 0; i<dim; ++i)
   {
      M(i,i) = (det / pow(dxs[i],2)).real();
   }
}

void pml_detJ_JT_J_inv_Im(const Vector & x, CartesianPML * pml , DenseMatrix & M)
{
   int dim = pml->dim;
   double omega = pml->omega;

   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, omega);

   for (int i = 0; i<dim; ++i)
   {
      det *= dxs[i];
   }

   M=0.0;
   for (int i = 0; i<dim; ++i)
   {
      M(i,i) = (det / pow(dxs[i],2)).imag();
   }
}

