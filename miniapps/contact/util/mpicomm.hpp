#include "mfem.hpp"

using namespace std;
using namespace mfem;


class MPICommunicator
{
private:
    MPI_Comm comm;
    int myid, num_procs;
    Array<unsigned int > origin_procs;
    Array<unsigned int > destination_procs;
    int offset, lsize;
    std::vector<int> offsets; 
    Array<int> send_count;
    Array<int> send_displ;
    Array<int> recv_count;
    Array<int> recv_displ;

public:
    MPICommunicator(MPI_Comm comm_, int offset_, int gsize);
    MPICommunicator(MPI_Comm comm_, Array<unsigned int> & destination_procs_);

    int get_rank(int dof);

    void Communicate(const Vector & x_s, Vector & x_r, int vdim, int ordering);
    Array<unsigned int> & GetOriginProcs() {return origin_procs;}
    void UpdateDestinationProcs() 
    {   
        destination_procs.SetSize(origin_procs.Size());
        destination_procs = origin_procs;
    }
    void Communicate(const Array<int> & x_s, Array<int> & x_r, int vdim, int ordering);
    void Communicate(const SparseMatrix & mat_s , SparseMatrix & mat_r);
    void Communicate(const Array<SparseMatrix*> & vmat_s, Array<SparseMatrix*> & vmat_r);
};
