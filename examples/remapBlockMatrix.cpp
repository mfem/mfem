#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

class HypreMat
{
private:
  std::vector<int> I, J;
  std::vector<double> d;
  int size;
  
public:
  HypreMat() : size(0)
  {
    
  }

  int Size() const { return size; }
  
  void ReadMatrixParallel(std::string filename, const int np)
  {
    int lastRow = 0;
    int count = 0;
  
    std::vector<int> R;

    for (int p=0; p<np; ++p)
      {
	std::ifstream f(filename.c_str() + std::to_string(p), std::ifstream::in);

	int rowFirst, rowLast, colFirst, colLast, row, col;
	double v;
      
	f >> rowFirst >> rowLast >> colFirst >> colLast;

	I.resize(rowLast+2);

	for (row=rowFirst+1; row<=rowLast+1; ++row)
	  I[row] = 0;

	while (f.good())
	  {
	    f >> row >> col >> v;

	    if (row >= 2485)
	      abort();
	    
	    //if (row < lastRow)  // verify ascending order of rows
	    //  abort(); 

	    lastRow = row;
	  
	    I[row+1]++;
	    R.push_back(row);
	    J.push_back(col);
	    d.push_back(v);

	    count++;
	  }

	f.close();
      }

    // Partial sum of I
    
    size = I.size() - 1;

    I[0] = 0;
    for (int j=1; j<size; ++j)
      I[j+1] += I[j];

    //int *Idata = I.data();
    
    if (I[size] != J.size() || I[size] != d.size())
      abort();

    { // Reorder J and d
      const int nnz = J.size();
      if (nnz != d.size() || nnz != R.size())
	abort();
    
      std::vector<int> JJ(nnz);
      std::vector<double> dd(nnz);
      std::vector<int> cnt;

      JJ = J;
      dd = d;

      cnt.assign(I.size(), 0); // slightly larger than necessary

      for (int j=0; j<nnz; ++j)
	{
	  J[I[R[j]] + cnt[R[j]]] = JJ[j];
	  d[I[R[j]] + cnt[R[j]]] = dd[j];
	  cnt[R[j]]++;
	}
    }
  }

  void CopyReorder(HypreMat const& A, std::vector<int> const& permRow, std::vector<int> const& permCol)
  {
    if (permRow.size() != A.Size() || permCol.size() != A.Size())
      abort();

    size = A.Size();
    I.resize(size+1);

    I[0] = 0;

    for (int j=0; j<size; ++j)
      {
	const std::size_t nnz_j = A.I[j+1] - A.I[j];
	I[permRow[j]+1] = nnz_j;
      }

    for (int j=1; j<size; ++j)
      I[j+1] += I[j];

    const std::size_t nnz = I[size];
    J.resize(nnz);
    d.resize(nnz);
    
    std::vector<std::size_t> cnt;
    cnt.assign(size, 0);

    for (int j=0; j<size; ++j)
      {
	const int pj = permRow[j];
	const std::size_t nnz_j = A.I[j+1] - A.I[j];

	for (int k=0; k<nnz_j; ++k, cnt[pj]++)
	  {
	    J[I[pj] + cnt[pj]] = permCol[A.J[A.I[j] + k]];
	    d[I[pj] + cnt[pj]] = A.d[A.I[j] + k];
	  }
      }
  }

  void Print(std::string filename)
  {
    std::ofstream f(filename.c_str());
    
    for (int j=0; j<size; ++j)
      {
	for (int k=I[j]; k<I[j+1]; ++k)
	  f << j << " " << J[k] << " " << d[k] << endl;
      }

    f.close();
  }

  void Compare(HypreMat const& A)
  {
    if (size != A.Size())
      abort();

    for (int j=0; j<size; ++j)
      {
	for (int k=I[j]; k<I[j+1]; ++k)
	  {
	    int m = -1;

	    for (int l=A.I[j]; l<A.I[j+1]; ++l)
	      {
		if (A.J[l] == J[k])
		  m = l;
	      }

	    if (m < 0)
	      cout << "row " << j << " column " << J[k] << " not found" << endl;
	    //else if (fabs(A.d[m] - d[k]) > 1.0e-8)
	    else if (fabs(fabs(A.d[m]) - fabs(d[k])) > 0.0)
	      cout << "entry (" << j << ", " << J[k] << ") " << d[k] << " != " << A.d[m] << endl;
	  }
      }
  }
};

void ReadCrd(std::string filename, std::vector<double>& crd)
{
  std::ifstream f(filename.c_str(), std::ifstream::in);

  double c;
  
  while (f.good())
    {
      f >> c;
      crd.push_back(c);
    }

  // for some reason, the last value gets pushed twice
  crd.pop_back();
  
  f.close();
}

void ReadCrdParallel(std::string filenamebase, const int np, std::vector<double>& crd)
{
  for (int p=0; p<np; ++p)
    {
      ReadCrd(filenamebase + std::to_string(p), crd);  // results get concatenated in crd
    }
}

// crd is the DOF coordinates in serial, crdp in parallel.
void SerialToParallelDOFMap(std::vector<double> const& crd, std::vector<double> const& crdp,
			    std::vector<int>& sp)
{
  if (crdp.size() != crd.size())
    abort();

  const int dim = 3;
  const int n = crd.size() / dim;  // number of DOF's

  sp.resize(n);

  const double tol = 1.0e-8;
  
  // Simple O(n^2) search
  
  for (int i=0; i<n; ++i)
    {
      sp[i] = -1;
      
      for (int j=0; j<n; ++j)
	{
	  bool eq = true;
	  for (int l=0; l<dim; ++l)
	    {
	      if (fabs(crd[(dim*i)+l] - crdp[(dim*j)+l]) > tol)
		eq = false;
	    }

	  if (eq)
	    {
	      if (sp[i] != -1)
		abort();
	      
	      sp[i] = j;
	    }
	}

      if (sp[i] < 0)
	abort();
    }
}

int main()
{
  const int np = 2;
  
  std::vector<double> crd0, crdp0, crd1, crdp1;
  
  ReadCrd("dofcrd0Ser", crd0);
  ReadCrd("dofcrd1Ser", crd1);

  const int numCrd = crd0.size() / 3;

  if (3*numCrd != crd0.size())
    abort();

  //for (int i=7440; i<crd.size(); ++i)
  //cout << "crd[" << i << "] " << crd[i] << endl;

  ReadCrdParallel("dofcrd0Par", np, crdp0);
  ReadCrdParallel("dofcrd1Par", np, crdp1);

  if (3*numCrd != crdp0.size())
    abort();

  std::vector<int> sp0, sp1;  // serial to parallel DOF map for a subdomain
  SerialToParallelDOFMap(crd0, crdp0, sp0);
  SerialToParallelDOFMap(crd1, crdp1, sp1);

  /*
  for (int i=0; i<sp.size(); ++i)
    cout << i << " " << sp[i] << endl;
  */

  cout << "sp0" << endl;
  for (int i=0; i<sp0.size(); ++i)
    cout << i << " " << sp0[i] << endl;

  cout << "sp1" << endl;
  for (int i=0; i<sp1.size(); ++i)
    cout << i << " " << sp1[i] << endl;
  
  HypreMat Aser, Apar, AserToPar;

  /*
  Aser.ReadMatrixParallel("HypreAsdComplexIm1_Serial.0000", 1);
  Apar.ReadMatrixParallel("HypreAsdComplexIm1_Par5.0000", np);
  */
  
  Aser.ReadMatrixParallel("ifopSer.0000", 1);
  Apar.ReadMatrixParallel("ifopPar.0000", np);

  //AserToPar.CopyReorder(Aser, sp, sp);
  AserToPar.CopyReorder(Aser, sp0, sp1);

  /*
  Aser.Print("rbSer");
  AserToPar.Print("rbSerToPar");
  Apar.Print("rbPar");
  */

  cout << "Comparing " << endl;
  
  AserToPar.Compare(Apar);
  
  return 0;
}
