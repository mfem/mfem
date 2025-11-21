#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


class SphereInBox {
public:
  SphereInBox();


private:
  void MakePatchTopology2D();
  void MakePatchTopology2D2();
  void MakeNURBS();
  void AddQuadPatch(const Array<int> &ind, int attr);

  NURBSPatch* GetLinearQuadPatch(const Vector &x0, const Vector &x1, const Vector &x2, const Vector &x3);

  int Dim;
  int NumOfVertices, NumOfElements, NumOfBdrElements;
  int NumOfEdges, NumOfFaces;

  Mesh *topo{nullptr};
  Mesh *mesh{nullptr};
  NURBSExtension *ne{nullptr};
};

SphereInBox::SphereInBox()
{
  // Create patch topology mesh
  MakePatchTopology2D2();

  MakeNURBS();
}

void SphereInBox::AddQuadPatch(const Array<int> &ind, int attr)
{
  //topo->AddQuad(ind, attr);

  Element* el = topo->NewElement(Element::QUADRILATERAL);
  el->SetVertices(ind);
  topo->AddElement(el);
}

void SphereInBox::MakePatchTopology2D()
{
   // TODO: input attributes?
   constexpr int sphereAttr = 1;
   constexpr int boxAttr = 2;
   constexpr int bottomBdryAttr = 1;
   constexpr int rightBdryAttr = 2;
   constexpr int topBdryAttr = 3;
   constexpr int leftBdryAttr = 4;
   constexpr int interiorBdryAttr = 5;
   
   Dim              = 2;
   NumOfVertices    = 12;
   NumOfElements    = 7;
   //NumOfBdrElements = 11;
   NumOfBdrElements = 0;

   topo = new Mesh(Dim, NumOfVertices, NumOfElements, NumOfBdrElements);

   // Create vertices
   double v[2] = {0.0, 0.0};

   // The first 6 vertices are along the bottom.
   for (int i=0; i<6; ++i)
     {
       v[0] = i * 0.2;
       topo->AddVertex(v); // Vertex i: (i * 0.2, 0)
     }

   v[0] = 0.4; v[1] = 1.0 / 3.0;
   topo->AddVertex(v); // Vertex 6: (0.4, 1/3)
   
   v[0] = 0.6; v[1] = 1.0 / 3.0;
   topo->AddVertex(v); // Vertex 7: (0.6, 1/3)

   v[0] = 0.2; v[1] = 2.0 / 3.0;
   topo->AddVertex(v); // Vertex 8: (0.2, 2/3)
   
   v[0] = 0.8; v[1] = 2.0 / 3.0;
   topo->AddVertex(v); // Vertex 9: (0.8, 2/3)
   
   v[0] = 0.0; v[1] = 1.0;
   topo->AddVertex(v); // Vertex 10: (0, 1)
   
   v[0] = 1.0; v[1] = 1.0;
   topo->AddVertex(v); // Vertex 11: (1, 1)
   
   // Create elements
   //int ind[4];
   Array<int> ind(4);

   // TODO: just use index values rather than ind?
   ind[0] = 2; ind[1] = 3; ind[2] = 7; ind[3] = 6;
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 1; ind[1] = 2; ind[2] = 6; ind[3] = 8;
   //ind[0] = 1; ind[1] = 2; ind[2] = 8; ind[3] = 6; // 1 2 6 8?
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 6; ind[1] = 7; ind[2] = 9; ind[3] = 8;
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 3; ind[1] = 4; ind[2] = 9; ind[3] = 7;
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 0; ind[1] = 1; ind[2] = 8; ind[3] = 10;
   //ind[0] = 0; ind[1] = 1; ind[2] = 10; ind[3] = 8; // 0 1 8 10?
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 8; ind[1] = 9; ind[2] = 11; ind[3] = 10;
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   //ind[0] = 4; ind[1] = 5; ind[2] = 11; ind[3] = 9;
   ind[0] = 4; ind[1] = 5; ind[2] = 9; ind[3] = 11; // 4 5 11 9?
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   // TODO: try AddElement as in MFEMTool::ConstructGlobalLinearFromCells

   // Create exterior boundary elements
   /*
   // The first 5 segments are along the bottom.
   for (int i=0; i<5; ++i)
     {
       topo->AddBdrSegment(i, i + 1, bottomBdryAttr);
     }

   topo->AddBdrSegment(11, 5, rightBdryAttr);
   topo->AddBdrSegment(11, 10, topBdryAttr);
   //topo->AddBdrSegment(0, 10, leftBdryAttr);
   //topo->AddBdrSegment(1, 8, interiorBdryAttr);
   //topo->AddBdrSegment(8, 9, interiorBdryAttr);
   //topo->AddBdrSegment(9, 4, interiorBdryAttr);
   */
   
   //topo->FinalizeQuadMesh(1, 1, true);

   //topo->FinalizeQuadMesh(1, 1, false);
   topo->FinalizeTopology();
   topo->Finalize(false, true);
   topo->CheckBdrElementOrientation(); // check and fix boundary element orientation

   ofstream mesh_ofs("patchTopology.mesh");
   mesh_ofs.precision(8);
   topo->Print(mesh_ofs);
}

void SphereInBox::MakePatchTopology2D2()
{
   // TODO: input attributes?
   constexpr int sphereAttr = 1;
   constexpr int boxAttr = 2;
   constexpr int bottomBdryAttr = 1;
   constexpr int rightBdryAttr = 2;
   constexpr int topBdryAttr = 3;
   constexpr int leftBdryAttr = 4;
   constexpr int interiorBdryAttr = 5;
   
   Dim              = 2;
   NumOfVertices    = 12;
   NumOfElements    = 7;
   //NumOfBdrElements = 11;
   NumOfBdrElements = 0;

   topo = new Mesh(Dim, NumOfVertices, NumOfElements, NumOfBdrElements);

   // Create vertices
   double v[2] = {0.0, 0.0};

   // The first 6 vertices are along the bottom.
   for (int i=0; i<6; ++i)
     {
       v[0] = i * 0.2;
       topo->AddVertex(v); // Vertex i: (i * 0.2, 0)
     }

   v[0] = 0.0; v[1] = 1.0;
   topo->AddVertex(v); // Vertex 6: (0, 1)

   v[0] = 0.2; v[1] = 2.0 / 3.0;
   topo->AddVertex(v); // Vertex 7: (0.2, 2/3)
   
   v[0] = 0.4; v[1] = 1.0 / 3.0;
   topo->AddVertex(v); // Vertex 8: (0.4, 1/3)
   
   v[0] = 0.6; v[1] = 1.0 / 3.0;
   topo->AddVertex(v); // Vertex 9: (0.6, 1/3)

   v[0] = 0.8; v[1] = 2.0 / 3.0;
   topo->AddVertex(v); // Vertex 10: (0.8, 2/3)
   
   v[0] = 1.0; v[1] = 1.0;
   topo->AddVertex(v); // Vertex 11: (1, 1)
   
   // Create elements
   //int ind[4];
   Array<int> ind(4);

   // TODO: just use index values rather than ind?
   ind[0] = 2; ind[1] = 3; ind[2] = 9; ind[3] = 8;
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 1; ind[1] = 2; ind[2] = 8; ind[3] = 7;
   //ind[0] = 1; ind[1] = 2; ind[2] = 8; ind[3] = 6; // 1 2 6 8?
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 8; ind[1] = 9; ind[2] = 10; ind[3] = 7;
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 3; ind[1] = 4; ind[2] = 10; ind[3] = 9;
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 0; ind[1] = 1; ind[2] = 7; ind[3] = 6;
   //ind[0] = 0; ind[1] = 1; ind[2] = 10; ind[3] = 8; // 0 1 8 10?
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 7; ind[1] = 10; ind[2] = 11; ind[3] = 6;
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   ind[0] = 4; ind[1] = 5; ind[2] = 11; ind[3] = 10;
   //topo->AddQuad(ind, boxAttr);
   AddQuadPatch(ind, boxAttr);

   // TODO: try AddElement as in MFEMTool::ConstructGlobalLinearFromCells

   // Create exterior boundary elements
   /*
   // The first 5 segments are along the bottom.
   for (int i=0; i<5; ++i)
     {
       topo->AddBdrSegment(i, i + 1, bottomBdryAttr);
     }

   topo->AddBdrSegment(11, 5, rightBdryAttr);
   topo->AddBdrSegment(11, 10, topBdryAttr);
   //topo->AddBdrSegment(0, 10, leftBdryAttr);
   //topo->AddBdrSegment(1, 8, interiorBdryAttr);
   //topo->AddBdrSegment(8, 9, interiorBdryAttr);
   //topo->AddBdrSegment(9, 4, interiorBdryAttr);
   */
   
   //topo->FinalizeQuadMesh(1, 1, true);

   //topo->FinalizeQuadMesh(1, 1, false);
   topo->FinalizeTopology();
   topo->Finalize(false, true);
   topo->CheckBdrElementOrientation(); // check and fix boundary element orientation

   ofstream mesh_ofs("patchTopology.mesh");
   mesh_ofs.precision(8);
   topo->Print(mesh_ofs);
}

// Counter-clockwise from bottom left of quad: x0, x1, x2, x3
NURBSPatch* SphereInBox::GetLinearQuadPatch(const Vector &x0, const Vector &x1, const Vector &x2, const Vector &x3)
{
  Vector cp(27); // TODO: store internally?

  Array<real_t> intervals_array({1});
  Vector intervals(intervals_array.GetData(), intervals_array.Size());
  const Array<int> continuity({-1, -1});
  const KnotVector kv(2, intervals, continuity);
  
  constexpr real_t h = 0.5;

  Vector bottom(2);
  Vector top(2);
  Vector p(2);
  for (int i=0; i<3; ++i)
    {
      const real_t xr = i * h;
      bottom.Set(xr, x1);
      bottom.Add(1.0 - xr, x0);

      top.Set(xr, x2);
      top.Add(1.0 - xr, x3);

      for (int j=0; j<3; ++j)
	{
	  const real_t yr = j * h;
	  p.Set(yr, top);
	  p.Add(1.0 - yr, bottom);

	  for (int l=0; l<2; ++l)
	    cp[((i + j * 3) * 3) + l] = p[l];
	  
	  //cp[((i + j * 3) * 3) + 0] = i * h; // x
	  //cp[((i + j * 3) * 3) + 1] = j * h; // y
	  cp[((i + j * 3) * 3) + 2] = 1.0; // Weight
	}
    }

  return new NURBSPatch(&kv, &kv, 3, cp.GetData());
}

void SphereInBox::MakeNURBS()
{
  Array<NURBSPatch*> patches(7);


  //paches[0] = GetPatch0(kv);
  //NURBSPatch::NURBSPatch(const KnotVector *kv0, const KnotVector *kv1, int dim_,
  //const real_t* control_points)

  // Ordered counter-clockwise from bottom left of quad
  Vector x0(2), x1(2), x2(2), x3(2); 

  x0[0] = 0.4; x0[1] = 0.0;
  x1[0] = 0.6; x1[1] = 0.0;
  x2[0] = 0.6; x2[1] = 1.0 / 3.0;
  x3[0] = 0.4; x3[1] = 1.0 / 3.0;
  patches[0] = GetLinearQuadPatch(x0, x1, x2, x3);
  
  x0[0] = 0.2; x0[1] = 0.0;
  x1[0] = 0.4; x1[1] = 0.0;
  x2[0] = 0.4; x2[1] = 1.0 / 3.0;
  x3[0] = 0.2; x3[1] = 2.0 / 3.0;
  patches[1] = GetLinearQuadPatch(x0, x1, x2, x3);
  
  x0[0] = 0.4; x0[1] = 1.0 / 3.0;
  x1[0] = 0.6; x1[1] = 1.0 / 3.0;
  x2[0] = 0.8; x2[1] = 2.0 / 3.0;
  x3[0] = 0.2; x3[1] = 2.0 / 3.0;
  patches[2] = GetLinearQuadPatch(x0, x1, x2, x3);
  
  x0[0] = 0.6; x0[1] = 0.0;
  x1[0] = 0.8; x1[1] = 0.0;
  x2[0] = 0.8; x2[1] = 2.0 / 3.0;
  x3[0] = 0.6; x3[1] = 1.0 / 3.0;
  patches[3] = GetLinearQuadPatch(x0, x1, x2, x3);
  
  x0[0] = 0.0; x0[1] = 0.0;
  x1[0] = 0.2; x1[1] = 0.0;
  x2[0] = 0.2; x2[1] = 2.0 / 3.0;
  x3[0] = 0.0; x3[1] = 1.0;
  patches[4] = GetLinearQuadPatch(x0, x1, x2, x3);
  
  x0[0] = 0.2; x0[1] = 2.0 / 3.0;
  x1[0] = 0.8; x1[1] = 2.0 / 3.0;
  x2[0] = 1.0; x2[1] = 1.0;
  x3[0] = 0.0; x3[1] = 1.0;
  patches[5] = GetLinearQuadPatch(x0, x1, x2, x3);
  
  x0[0] = 0.8; x0[1] = 0.0;
  x1[0] = 1.0; x1[1] = 0.0;
  x2[0] = 1.0; x2[1] = 1.0;
  x3[0] = 0.8; x3[1] = 2.0 / 3.0;
  patches[6] = GetLinearQuadPatch(x0, x1, x2, x3);
  
  ne = new NURBSExtension(topo, patches);
  mesh = new Mesh(*ne);

  ofstream mesh_ofs("sbox.mesh");
  mesh_ofs.precision(8);
  mesh->Print(mesh_ofs);

  //GridFunction *nodes = mesh->GetNodes();
}

int main(int argc, char *argv[])
{
  Mpi::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/nc3-nurbs.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   SphereInBox sb;

}

