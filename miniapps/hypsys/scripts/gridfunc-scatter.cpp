// This file is part of BLAST - a high-order finite element hydrocode
//
// MFEM has to be compiled in serial, then compile this with
// g++ -O3 -std=c++11 -I../../..  gridfunc-scatter.cpp -o gridfunc-scatter -L../../.. -lmfem -lrt

#include <fstream>

#include "../../../mfem.hpp"
using namespace mfem;

using namespace std;

int main(int argc, char *argv[])
{
  if (argc != 3)
    {
      cout << "usage: " << argv[0]
           << " in_mesh_file in_gf_file" << endl;
      return 1;
    }

  Mesh *mesh;

  {
    ifstream imesh (argv[1]);
    if (!imesh)
      {
	cout << "can not open mesh file: " << argv[1] << endl;
	return 2;
      }
    mesh = new Mesh (imesh, 1, 1);
  }

  int dim = mesh -> Dimension();
  cout << "mesh dimension: " << dim << endl;


  //////////////////////////////////
  ofstream         scatter_file("scatter.dat");
  int              sd = 4;
  RefinedGeometry *RefG;
  Vector           values;
  DenseMatrix      pointmat, vec_values;
  const IntegrationRule *ir;
  int points_percell=sd;

  cout << "Enter subdivision factor : " << flush;
  cin >> sd;
  cout << "Enter subdivisions per cell : " << flush;
  cin >> points_percell;
  points_percell--;

  scatter_file.precision(8);
  for (int lev = 0; lev < sd; lev++)
    mesh->UniformRefinement();

  GridFunction *gf;
  {
    ifstream igf(argv[2]);
    gf = new GridFunction(mesh, igf);
  }

  for (int i = 0; i < mesh->GetNE(); i++)
    {
      RefG = GlobGeometryRefiner.Refine(mesh->GetElementBaseGeometry(i), points_percell, 1);
      ir = &(RefG->RefPts);
      // GET VALUES
      gf->GetValues(i, *ir, values, pointmat);

      for (int j = 0; j < pointmat.Width(); j++) //num of points per element
	{
	  scatter_file << pointmat(0, j) << ' ';
	  //cout << pointmat(0,j) << ' ';
	  scatter_file << values(j) << '\n';
	  //cout << values(j) << endl;
	}
      //abort();
    }
  scatter_file.close();
  //////////////////////////////////

  delete gf;
  delete mesh;

  return 0;
}
