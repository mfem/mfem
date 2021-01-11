
#include "MeshPart.hpp"

double GetPointAngle(const Vector & pt)
{
   double x = pt(0);
   double y = pt(1);
   x = (abs(x)<1e-12) ? 0.0 : x;
   y = (abs(y)<1e-12) ? 0.0 : y;
   double theta = (x == 0) ? M_PI/2.0 : atan(y/x);
   int k = (x<=0.0) ? 1 : ((y<0.0) ? 2 : 0.0);
   theta += k*M_PI;
   return theta * 180.0/M_PI;
}

void GetMeshAngleRange(Mesh * mesh, double & amin, double & amax)
{
   amin = infinity();
   amax = -infinity();
   int nbe = mesh->GetNBE();
   int dim = mesh->Dimension();

   for (int i = 0; i < nbe; ++i)
   {
      Vector center(dim);
      int geom = mesh->GetBdrElementBaseGeometry(i);
      ElementTransformation * T = mesh->GetBdrElementTransformation(i);
      T->Transform(Geometries.GetCenter(geom),center);
      double thetad = GetPointAngle(center);
      amin = min(amin,thetad);
      amax = max(amax,thetad);
   }
}

int get_angle_range(double angle, Array<double> angles)
{
   auto it = std::upper_bound(angles.begin(), angles.end(), angle);
   return std::distance(angles.begin(),it)-1;
}

void SetMeshAttributes(Mesh * mesh, int subdivisions, double ovlp)
{
   Array<double> angles(2*subdivisions);

   double amin, amax; 
   GetMeshAngleRange(mesh,amin,amax);
   angles[0] = amin;

   double length = (amax-amin)/subdivisions;
   double range;
   for (int i = 1; i<subdivisions; i++)
   {
      range = i*length;
      angles[2*i-1] = range-ovlp;
      angles[2*i] = range+ovlp;
   }
   angles[2* subdivisions-1] = amax;

   int ne = mesh->GetNE();
   int dim = mesh->Dimension();
   // set element attributes
   for (int i = 0; i < ne; ++i)
   {
      Element *el = mesh->GetElement(i);
      // roughly the element center
      Vector center(dim);
      mesh->GetElementCenter(i,center);
      double thetad = GetPointAngle(center);
      // Find the angle relative to (0,0,z)
      int attr = get_angle_range(thetad, angles) + 1;
      el->SetAttribute(attr);
   }
   mesh->SetAttributes();
   cout << "Max attributes " << mesh->attributes.Max() << endl;
   cout << "angles = " ; angles.Print(cout, 2*subdivisions);
   if (!angles.IsSorted()) 
      MFEM_WARNING("Check mesh partitioning angles ");
}

// remove/leave elements with attributes given by attr
Mesh * GetPartMesh(const Mesh * mesh0, const Array<int> & attr_, Array<int> & elem_map,
 bool complement)
{
   Array<int> bdr_attr;
   int max_attr     = mesh0->attributes.Max();
   int min_attr     = mesh0->attributes.Min();

   Array<int> attr;
   
   Array<int> all_attr(max_attr); all_attr = 0;
   for (int i = 0; i<attr_.Size(); i++)
   {
      all_attr[attr_[i]-1] = 1;
   }
   for (int i = min_attr; i<=max_attr; i++)
   {
      
      if (complement && all_attr[i-1]==0) attr.Append(i);
      if (!complement && all_attr[i-1]==1) attr.Append(i);
   }


   int max_bdr_attr = mesh0->bdr_attributes.Max();

   bdr_attr.SetSize(attr.Size());
   for (int i=0; i<attr.Size(); i++)
   {
      bdr_attr[i] = max_bdr_attr + attr[i];
   }

   Array<int> marker(max_attr);
   Array<int> attr_inv(max_attr);
   marker = 0;
   attr_inv = 0;
   for (int i=0; i<attr.Size(); i++)
   {
      marker[attr[i]-1] = 1;
      attr_inv[attr[i]-1] = i;
   }

   // Count the number of elements in the final mesh
   int num_elements = 0;
   for (int e=0; e<mesh0->GetNE(); e++)
   {
      int elem_attr = mesh0->GetElement(e)->GetAttribute();
      if (!marker[elem_attr-1]) { num_elements++; }
   }

   Mesh * mesh = new Mesh(mesh0->Dimension(), mesh0->GetNV(), num_elements);
   // Copy vertices
   for (int v=0; v<mesh0->GetNV(); v++)
   {
      mesh->AddVertex(mesh0->GetVertex(v));
   }

   // Copy elements
   elem_map.SetSize(num_elements);
   int k = 0;
   for (int e=0; e<mesh0->GetNE(); e++)
   {
      const Element * el = mesh0->GetElement(e);
      
      int elem_attr = el->GetAttribute();
      if (!marker[elem_attr-1])
      {
         Element * nel = mesh->NewElement(el->GetGeometryType());
         nel->SetAttribute(elem_attr);
         nel->SetVertices(el->GetVertices());
         mesh->AddElement(nel);
         elem_map[k++] = e;
      }
   }

   mesh->FinalizeTopology();
   mesh->RemoveUnusedVertices();

   const GridFunction * nodes0 = mesh0->GetNodes();

   int order = nodes0->FESpace()->GetOrder(0);
   if (order > 1)
   {
      mesh->SetCurvature(order, false, 3, Ordering::byVDIM);
   }

   GridFunction * nodes = mesh->GetNodes();
   int nel = mesh0->GetNE();
   // copy nodes
   int jel = 0;
   for (int iel = 0; iel< nel; iel++)
   {
      int elem_attr = mesh0->GetElement(iel)->GetAttribute();
      if (!marker[elem_attr-1]) 
      {
         Array<int> vdofs0,vdofs;
         nodes0->FESpace()->GetElementVDofs(iel,vdofs0);
         Vector x;
         nodes0->GetSubVector(vdofs0,x);
         nodes->FESpace()->GetElementVDofs(jel++,vdofs);
         nodes->SetSubVector(vdofs,x);
      }
   }
   return mesh;
}

// Partition mesh to nrsubmeshes (equally spaced in the azimuthal direction)
void PartitionMesh(Mesh * mesh, int nrsubmeshes, double ovlp, 
                   Array<Mesh*> & SubMeshes, Array<Array<int> *> & elems)
{
   cout << "Partitioning the global Mesh" << endl;

   SetMeshAttributes(mesh,nrsubmeshes,ovlp);
   int maxattr = mesh->attributes.Max();
   // Produce the subdomains
   char vishost[] = "localhost";
   int  visport   = 19916;
   SubMeshes.SetSize(nrsubmeshes);
   elems.SetSize(nrsubmeshes);
   for (int i = 0; i<nrsubmeshes; i++)
   {
      cout << "mesh " << i << endl;
      Array<int> attr;
      for (int j = 0; j<3; j++)
      {
         if (2*i+j >0 && 2*i+j <= maxattr) attr.Append(2*i+j);
      }
      Array<int> elem_map;
      // attr.Print();
      elems[i] = new Array<int>(0);
      SubMeshes[i] = GetPartMesh(mesh,attr,*elems[i],true);
      // socketstream mesh_sock(vishost, visport);
      // mesh_sock << "parallel " << nrsubmeshes <<  " " << i << "\n";
      // mesh_sock.precision(8);
      // mesh_sock << "mesh\n" << *SubMeshes[i] << flush;
      // cout << "nrelemes = " << mesh1->GetNE() << endl;
   }
}