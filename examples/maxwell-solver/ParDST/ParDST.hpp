#pragma once
#include "../common/Utilities.hpp"
#include "../common/PML.hpp"
#include "../DST/DST.hpp"
using namespace std;
using namespace mfem;


class ParDST : public Solver//
{
private:
   // Constructor inputs
   ParSesquilinearForm *bf=nullptr;
   ParFiniteElementSpace * pfes = nullptr;
   ParMesh * pmesh = nullptr;
   const FiniteElementCollection * fec = nullptr;
   Array2D<double> Pmllength;
   int dim = 2;
   double omega = 0.5;
   Coefficient * ws;
   int nrlayers;

   int nrsubdomains = 0;
   int nx,ny,nz;


   void Getijk(int ip, int & i, int & j, int & k ) const;
   int GetPatchId(const Array<int> & ijk) const;
   void IdentifyCommonDofs(); // TODO (Use global Elem number for matching dofs)
   void TransferToNeighbors(int ip); // TODO
  
public:
   ParDST(ParSesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * ws_, int nrlayers_, int nx_=2, int ny_=2, int nz_=2);
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ParDST();
};


