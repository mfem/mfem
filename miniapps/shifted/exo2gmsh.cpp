#include "mfem.hpp"

#ifdef MFEM_USE_PETSC
#include "exodusII.h"
#endif

int main(int argc, char *argv[])
{
    // Parse command-line options.
    const char *mesh_file = "input.exo";
    const char *gmsh_file = "gmsh.out";


    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");

    args.AddOption(&gmsh_file, "-o", "--output",
                   "GMSH output file.");


    args.Parse();
    if (!args.Good())
    {
       args.PrintUsage(std::cout);
       return 1;
    }
    args.PrintOptions(std::cout);
#ifdef MFEM_USE_PETSC

    std::fstream gmsh;
    gmsh.open(gmsh_file,std::ios::out);
    gmsh<<"$MeshFormat\n2 0 8\n$EndMeshFormat"<<std::endl;

    int comp_ws=sizeof(double);
    int io_ws=0;
    float version;

    int exo_id=ex_open(mesh_file,EX_READ,&comp_ws,&io_ws,&version);

    //read the nodes and dump them to a file
    {
        int numn;
        ex_inquire(exo_id,EX_INQ_NODES,&numn ,NULL,NULL);
        std::cout<<"Num nodes ="<<numn<<std::endl;

        double* x=new double[numn];
        double* y=new double[numn];
        double* z=new double[numn];

        ex_get_coord(exo_id,x,y,z);
        gmsh<<"$Nodes"<<std::endl;
        gmsh<<numn<<std::endl;

        for(int i=0;i<numn;i++)
        {
            gmsh<<i+1<<" "<<x[i]<<" "<<y[i]<<" "<<z[i]<<std::endl;
        }

        gmsh<<"$EndNodes"<<std::endl;

        delete [] x;
        delete [] y;
        delete [] z;
    }

    //do the element processing
    {
        int numel_bl;
        ex_inquire(exo_id,EX_INQ_ELEM_BLK,&numel_bl,NULL,NULL);
        int* bl_id=new int[numel_bl];
        ex_get_elem_blk_ids(exo_id,bl_id);
        char el_type[MAX_STR_LENGTH+1];
        int numel;
        int numel_nod;
        int numel_attr;
        int el_id=0;


        int tot_el_num;
        ex_inquire(exo_id,EX_INQ_ELEM,&tot_el_num,NULL,NULL);
        gmsh<<"$Elements"<<std::endl;
        gmsh<<tot_el_num<<std::endl;


        for(int bl=0;bl<numel_bl;bl++)
        {

            ex_get_elem_block(exo_id,bl_id[bl],el_type,
                                  &numel,&numel_nod,&numel_attr);
            int* conn=new int[numel*numel_nod];
            ex_get_elem_conn(exo_id,bl_id[bl],conn);

            for(int i=0;i<numel;i++)
            {
                el_id++;
                gmsh<<el_id<<" ";
                //element type
                if(strcasecmp(el_type,"TETRA4")==0){
                    gmsh<<"4 ";
                }else if(strcasecmp(el_type,"TETRA10")==0){
                    gmsh<<"11 ";
                }
                else if(strcasecmp(el_type,"HEX8")==0){
                    gmsh<<"5 ";
                }else if(strcasecmp(el_type,"BAR2")==0){
                    gmsh<<"1 ";
                }else if(strcasecmp(el_type,"TRI3")==0){
                    gmsh<<"2 ";
                }else if(strcasecmp(el_type,"SHELL4")==0){
                    gmsh<<"3 ";
                }

                //dump the rest of the element data
                gmsh<<"3 "<<bl_id[bl];
                gmsh<<" 0 0"; //num tags, geom domain, volume

                for(int ii=0;ii<numel_nod;ii++)
                {
                    gmsh<<" "<<conn[i*numel_nod+ii];
                }
                gmsh<<std::endl;
              }
              delete [] conn;
        }
        delete [] bl_id;
        gmsh<<"$EndElements"<<std::endl;
    }

    gmsh.close();
    ex_close(exo_id);
#endif
}
