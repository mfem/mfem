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
   // initialize
   for (int i = 0; i < dim; i++)
   {
      dom_bdr(i, 0) = infinity();
      dom_bdr(i, 1) = -infinity();
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

#ifdef MFEM_USE_MPI
   ParMesh * pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh)
   {
      for (int d=0; d<dim; d++)
      {
         MPI_Allreduce(MPI_IN_PLACE,&dom_bdr(d,0),1,MPI_DOUBLE,MPI_MIN,pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE,&dom_bdr(d,1),1,MPI_DOUBLE,MPI_MAX,pmesh->GetComm());
      }
   }
#endif

   for (int i = 0; i < dim; i++)
   {
      comp_dom_bdr(i, 0) = dom_bdr(i, 0) + length(i, 0);
      comp_dom_bdr(i, 1) = dom_bdr(i, 1) - length(i, 1);
   }
}

void CartesianPML::SetAttributes(Mesh *mesh_, Array<int> * attrNonPML, Array<int> * attrPML)
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

   if (mesh_->attributes.Size())
   {
      if (attrNonPML)
      {
         attrNonPML->SetSize(mesh_->attributes.Max());
         *attrNonPML = 0; (*attrNonPML)[0] = 1;

      } 
      if (attrPML)
      {
         attrPML->SetSize(mesh_->attributes.Max());
         *attrPML = 0;
         if (mesh_->attributes.Max()>1)
         {
            (*attrPML)[1]=1;
         }
      }
   }

}

void CartesianPML::StretchFunction(const Vector &x,
                                   vector<complex<double>> &dxs)
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



// acoustics UW PML coefficients functions
// |J|
double detJ_r_function(const Vector & x, CartesianPML * pml)
{
   int dim = pml->dim;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml->StretchFunction(x, dxs);
   for (int i=0; i<dim; ++i) det *= dxs[i];
   return det.real();
}

double detJ_i_function(const Vector & x, CartesianPML * pml)
{
   int dim = pml->dim;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml->StretchFunction(x, dxs);
   for (int i=0; i<dim; ++i) det *= dxs[i];
   return det.imag();
}

double abs_detJ_2_function(const Vector & x, CartesianPML * pml)
{
   int dim = pml->dim;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml->StretchFunction(x, dxs);
   for (int i=0; i<dim; ++i) det *= dxs[i];
   return det.imag()*det.imag() + det.real()*det.real();
}

// J^T J / |J| 
void Jt_J_detJinv_r_function(const Vector & x, CartesianPML * pml , DenseMatrix & M)
{
   int dim = pml->dim;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml->StretchFunction(x, dxs);
   for (int i = 0; i<dim; ++i) det *= dxs[i];

   M=0.0;
   for (int i = 0; i<dim; ++i)
   {
      M(i,i) = (pow(dxs[i], 2)/det).real();
   }
}

void Jt_J_detJinv_i_function(const Vector & x, CartesianPML * pml , DenseMatrix & M)
{
   int dim = pml->dim;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);
   for (int i = 0; i<dim; ++i) det *= dxs[i];

   M=0.0;
   for (int i = 0; i<dim; ++i)
   {
      M(i,i) = (pow(dxs[i], 2)/det).imag();
   }
}

void abs_Jt_J_detJinv_2_function(const Vector & x, CartesianPML * pml , DenseMatrix & M)
{
   int dim = pml->dim;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);
   for (int i = 0; i<dim; ++i) det *= dxs[i];

   M=0.0;
   for (int i = 0; i<dim; ++i)
   {
      complex<double> a = pow(dxs[i], 2)/det;
      M(i,i) = a.imag() * a.imag() + a.real() * a.real();
   }
}


// Maxwell PML coeffiecients
void detJ_Jt_J_inv_r(const Vector &x, CartesianPML * pml, DenseMatrix &M)
{
   int dim = pml->dim;
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i) det *= dxs[i];

   M = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      M(i, i) = (det / pow(dxs[i], 2)).real();
   }
}

void detJ_Jt_J_inv_i(const Vector &x, CartesianPML * pml, DenseMatrix &M)
{
   int dim = pml->dim;
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i) det *= dxs[i];

   M = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      M(i, i) = (det / pow(dxs[i], 2)).imag();
   }
}

void detJ_Jt_J_inv_abs_2(const Vector &x, CartesianPML * pml, DenseMatrix &M)
{
   int dim = pml->dim;
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)  det *= dxs[i];

   M = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      complex<double> a = det / pow(dxs[i], 2);
      M(i, i) = a.real()*a.real() + a.imag()*a.imag();
   }
}

// void detJ_inv_JT_J_Re(const Vector &x, CartesianPML * pml, DenseMatrix &M)
// {
//    int dim = pml->dim;
//    vector<complex<double>> dxs(dim);
//    complex<double> det(1.0, 0.0);
//    pml->StretchFunction(x, dxs);

//    for (int i = 0; i < dim; ++i)
//    {
//       det *= dxs[i];
//    }

//    // in the 2D case the coefficient is scalar 1/det(J)
//    if (dim == 2)
//    {
//       M = (1.0 / det).real();
//    }
//    else
//    {
//       M = 0.0;
//       for (int i = 0; i < dim; ++i)
//       {
//          M(i, i) = (pow(dxs[i], 2) / det).real();
//       }
//    }
// }

// void detJ_inv_JT_J_Im(const Vector &x, CartesianPML * pml, DenseMatrix &M)
// {
//    int dim = pml->dim;
//    vector<complex<double>> dxs(dim);
//    complex<double> det = 1.0;
//    pml->StretchFunction(x, dxs);

//    for (int i = 0; i < dim; ++i)
//    {
//       det *= dxs[i];
//    }

//    if (dim == 2)
//    {
//       M = (1.0 / det).imag();
//    }
//    else
//    {
//       M = 0.0;
//       for (int i = 0; i < dim; ++i)
//       {
//          M(i, i) = (pow(dxs[i], 2) / det).imag();
//       }
//    }
// }