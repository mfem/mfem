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
    void resetcounts()
    {
        send_count = 0;
        send_displ = 0;
        recv_count = 0;
        recv_displ = 0;
    }

public:
    MPICommunicator(MPI_Comm comm_, int offset_, int gsize);
    MPICommunicator(MPI_Comm comm_, Array<unsigned int> & destination_procs_);

    int get_rank(int dof);

    Array<unsigned int> & GetOriginProcs() {return origin_procs;}
    void UpdateDestinationProcs() 
    {   
        destination_procs.SetSize(origin_procs.Size());
        destination_procs = origin_procs;
        resetcounts();
    }
    void Communicate(const Vector & x_s, Vector & x_r, int vdim, int ordering);
    void Communicate(const Array<int> & x_s, Array<int> & x_r, int vdim, int ordering);
    void Communicate(const DenseMatrix & A_s, DenseMatrix & A_r, int vdim, int ordering);
    void Communicate(const Array<unsigned int> & x_s, Array<unsigned int> & x_r, int vdim, int ordering);
    void Communicate(const SparseMatrix & mat_s , SparseMatrix & mat_r);
    void Communicate(const Array<SparseMatrix*> & vmat_s, Array<SparseMatrix*> & vmat_r);
};
