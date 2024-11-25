import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def read_data(filename, stringmatch):
    df = pd.read_csv(filename, delimiter=' ', header=None, names=['col1', 'col2', 'col3'])
    filtered_df = df[df['col3'] == stringmatch]
    np_array = filtered_df[['col2']].to_numpy()
    return np_array

# evaluate the ith Lagrange polynomial at xi
def lagrange_poly(gll, i, xi):
    num = 1
    den = 1
    for j in range(len(gll)):
        if j != i:
            num *= (xi - gll[j])
            den *= (gll[i] - gll[j])
    return num/den


def plot_lower_and_upper_bounds(nr, mr, npts, gll, intx, lbound, ubound, nodep, intp):
    rmin = 0.0
    rmax = 1.0
    sample_locs = np.zeros(npts)
    basis = np.zeros((npts))
    lower = np.zeros((npts))
    upper = np.zeros((npts))

    fname = "minmr_PL_"
    if (nodep == 0):
        fname += "GL" + str(nr)
    else:
        fname += "GLL" + str(nr)
    istring = ""
    if (intp == -1):
        istring = "_Cheb" + str(mr)
    elif (intp == 0):
        istring = "_GL" + str(mr)
    elif (intp == 1):
        istring = "_GLL" + str(mr)
    glstring = fname + istring
    pdf_pages = PdfPages(glstring+'.pdf')
    # print('minmr_PL_='+glstring+'_N='+str(nr)+'_M='+str(mr)+'.pdf')
    for i in range(nr):
        for j in range(npts):
            sample_locs[j] = rmin + (rmax-rmin)*j/npts
            basis[j] = lagrange_poly(gll, i, sample_locs[j])
            lower[j] = np.interp(sample_locs[j], intx, lbound[i, :])
            upper[j] = np.interp(sample_locs[j], intx, ubound[i, :])
        plt.figure(i)
        plt.plot(sample_locs, basis, 'k-', label='Solution')
        plt.plot(gll, gll*0, 'ko', markerfacecolor='none')
        plt.plot(intx, intx*0, 'kx--')
        plt.plot(sample_locs, lower, 'r-', label='Lower bound')
        plt.plot(sample_locs, upper, 'b-', label='Upper bound')
        # plt.show()
        # plt.savefig('minmr_PL_='+glstring+'_N='+str(nr)+'_M='+str(mr)+'.pdf',format='pdf',bbox_inches='tight')
        pdf_pages.savefig()
        plt.close()
    pdf_pages.close()


def plot_and_save_data(nr_in, nodep, intp):
    fname = "minmr_PL_"
    if (nodep == 0):
        fname += "GL_"
    else:
        fname += "GLL_"
    fname +=str(nr_in) + "_Int_" + str(intp) + ".txt"

    data = np.loadtxt(fname)
    nr = int(data[0])
    assert nr == nr_in, "NR mismatch"
    mr = int(data[1])
    nbrute = int(data[2])
    data = data[3:]
    gllx = data[0:nr]
    data = data[nr:]
    intx = data[0:mr]
    data = data[mr:]
    nmr = nr*mr
    lbound = data[0:nmr]
    data = data[nmr:]
    ubound = data[0:nmr]

    # print(nr, mr, nbrute)
    # print(gll)
    # print(intx)
    # print(lbound)
    lbound = np.reshape(lbound, (nr, mr))
    # print(nr, mr, len(ubound))
    ubound = np.reshape(ubound, (nr, mr))
    # print(lbound)
    # exit
    plot_lower_and_upper_bounds(nr, mr, nbrute, gllx, intx, lbound, ubound, nodep, intp)




def main():
    parser = argparse.ArgumentParser(description="A script that processes some arguments")

    # Add arguments
    parser.add_argument('--NMAX', type=int, help='Number of rows (N)', default=15)
    parser.add_argument('--NMIN', type=int, help='Number of rows (N)', default=5)
    parser.add_argument('--nodep', type=int, help='node type', default=1)
    parser.add_argument('--intp', type=int, help='intp type', default=-1)

    args = parser.parse_args()

    nmax = args.NMAX
    intp = args.intp
    nodep = args.nodep

    # loop from 3 to NMAX
    for nr in range(3, nmax+1):  # Add 1 to include nmax
        plot_and_save_data(nr, nodep, intp)





    # gll_nodes = read_data('glldata.out',"k10-gll-z")
    # nr = len(gll_nodes)

    # cheb_nodes = read_data('glldata.out',"k10-chebyshev-locs")
    # mr = len(cheb_nodes)

    # poly_bnds = read_data('glldata.out',"k10-bnds-r")
    # poly_bnds = np.reshape(poly_bnds[:], (nr,mr,2))

    # u_bnds = read_data('glldata.out',"k10-bnds-x")
    # u_bnds = np.reshape(u_bnds[:], (mr,2))

    # uvals = read_data('glldata.out',"k10-u")

    # # colors = ['#4169E1', '#FF4500', '#3CB371', '#FFD700', '#9400D3', '#FF6347', '#40E0D0']
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    # # print(gll_nodes)
    # # print(cheb_nodes)
    # # print(poly_bnds)
    # # print(npts,nr,mr)

    # rmin = gll_nodes[0]
    # rmax = gll_nodes[-1]
    # # print(rmin,rmax)
    # mylagpoly = np.zeros((npts,nr))
    # mylagpolyscaled = np.zeros((npts,nr))
    # upoly = np.zeros((npts))

    # fontsz = 16

    # lw = '1'

    # sample_locs = np.zeros(npts)
    # for j in range(npts):
    #     sample_locs[j] = rmin + (rmax-rmin)*j/npts

    # for i in range(nr+2):
    #     plt.figure(i+1)
    #     plt.plot(gll_nodes, np.zeros(nr), 'ko-', label='GLL')
    #     if (i != 0):
    #         plt.plot(cheb_nodes, np.zeros(mr), 'kx-', label='Chebyshev')

    # plt.figure(1)
    # for i in range(nr):
    #     for j in range(npts):
    #         mylagpoly[j,i] = lagrange_poly(gll_nodes, i, sample_locs[j])
    #         mylagpolyscaled[j,i] = mylagpoly[j,i]*uvals[i]
    #         upoly[j] += mylagpolyscaled[j,i]

    #     plt.figure(1)
    #     plt.plot(sample_locs, mylagpoly[:,i], color=colors[i % len(colors)], linestyle='-', label=f'Polynomial {i}')
    #     plt.figure(i+3)
    #     plt.plot(sample_locs, mylagpoly[:,i], color=colors[i % len(colors)], linestyle='-', label=f'Polynomial {i}')

    # plt.figure(2)
    # plt.plot(cheb_nodes, u_bnds, 'k-', linewidth=lw, label=f'Function bounds')
    # plt.plot(sample_locs, upoly[:], color=colors[nr % len(colors)], linestyle='-', label=f'Function')

    # for i in range(nr):
    #     plt.figure(i+3)
    #     plt.plot(cheb_nodes, poly_bnds[i,:,0], 'k-', linewidth=lw, label='Lower bound')
    #     plt.plot(cheb_nodes, poly_bnds[i,:,1], 'k-', linewidth=lw, label='Upper bound')

    # for i in range(nr+2):
    #     plt.figure(i+1)
    #     plt.xticks(fontsize=fontsz,rotation=45)
    #     plt.yticks(fontsize=fontsz)
    #     plt.savefig('lagrangepoly_nr='+str(nr)+'_mr='+str(mr)+'_'+str(i)+'.png',bbox_inches='tight')


    # plt.figure(nr+3)
    # plt.plot(sample_locs, upoly[:], color=colors[nr % len(colors)],  linestyle='-', label=f'Function')
    # plt.plot(gll_nodes, uvals, color=colors[nr % len(colors)], marker='o', linestyle='None', label=f'Function')
    # plt.xticks(fontsize=fontsz,rotation=45)
    # plt.yticks(fontsize=fontsz)
    # plt.savefig('lagrangepoly_nr='+str(nr)+'_mr='+str(mr)+'_'+str(nr+2)+'.png',bbox_inches='tight')


# Using the special variable
# __name__
if __name__=="__main__":
    main()
