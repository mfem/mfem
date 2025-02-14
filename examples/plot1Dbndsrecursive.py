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

def read_data_no_txt(filename):
    df = pd.read_csv(filename, delimiter=' ', header=None, names=['col1', 'col2', 'col3'])
    # filtered_df = df[df['col3'] == stringmatch]
    # np_array = filtered_df[['col2']].to_numpy()
    return df.to_numpy()

# evaluate the ith Lagrange polynomial at xi
def lagrange_poly(gll, i, xi):
    num = 1
    den = 1
    for j in range(len(gll)):
        if j != i:
            num *= (xi - gll[j])
            den *= (gll[i] - gll[j])
    return num/den
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_bases_bounds(bases_data_file):
    basesdata = np.loadtxt(bases_data_file)
    nr = int(basesdata[0])
    mr = int(basesdata[1])
    basesdata = basesdata[2:]
    gllx = basesdata[0:nr]
    basesdata = basesdata[nr:]
    intx = basesdata[0:mr]
    basesdata = basesdata[mr:]
    nmr = nr*mr
    lbound = basesdata[0:nmr]
    basesdata = basesdata[nmr:]
    ubound = basesdata[0:nmr]
    lbound = np.reshape(lbound, (nr, mr))
    ubound = np.reshape(ubound, (nr, mr))
    gllx = NormalizeData(gllx)
    intx = NormalizeData(intx)

    return nr, mr, gllx, intx, lbound, ubound

def get_bases_bounds_Tarik(bases_data_file):
    basesdata = np.loadtxt(bases_data_file)
    nr = int(basesdata[0])
    mr = int(basesdata[nr+1])
    basesdata = np.delete(basesdata, [0,nr+1])
    # print(basesdata)
    # basesdata = basesdata[2:]
    gllx = basesdata[0:nr]
    basesdata = basesdata[nr:]
    intx = basesdata[0:mr]
    basesdata = basesdata[mr:]
    lbound = np.zeros((nr,mr))
    ubound = np.zeros((nr,mr))
    # print(basesdata)
    for i in range(nr):
        lbound[i,:] = basesdata[0:mr]
        basesdata = basesdata[mr:]
        ubound[i,:] = basesdata[0:mr]
        basesdata = basesdata[mr:]
        # print(i,basesdata)
    # nmr = nr*mr
    # lbound = basesdata[0:nmr]
    # basesdata = basesdata[nmr:]
    # ubound = basesdata[0:nmr]
    # lbound = np.reshape(lbound, (nr, mr))
    # ubound = np.reshape(ubound, (nr, mr))
    gllx = NormalizeData(gllx)
    intx = NormalizeData(intx)

    return nr, mr, gllx, intx, lbound, ubound

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="A script that processes some arguments")

    # Add arguments
    parser.add_argument('--N', type=int, help='Number of rows (N)', default=5)
    parser.add_argument('--M', type=int, help='Number of columns (M)',default=6)
    parser.add_argument('--O', type=int, help='output suffix', default=0)

    args = parser.parse_args()


    N = args.N
    M = args.M
    O = args.O

    data_file = "recursive_bnd_" + str(N) + "_" + str(M) + "_out=" + str(O)+".txt"
    print(data_file)
    dataread = np.loadtxt(data_file)

    bases_data_file = "recursive_bnd_bases_GLL_" + str(N) + "_Cheb_" + str(M) + ".txt"
    nr, mr, gllx, intx, lbound, ubound = get_bases_bounds(bases_data_file)
    # bases_data_file = "recursive_bnd_bases_GLL_" + str(N) + "_Opt_" + str(M) + ".txt"
    bases_data_file = "../scripts/bounds/bnddata_spts_lobatto_" + str(N) + "_bpts_opt_" + str(M) + ".txt"
    print("Reading ",bases_data_file)
    nr, mr, gllT, intT, lboundT, uboundT = get_bases_bounds_Tarik(bases_data_file)
    assert nr == N, "NR mismatch in bases bounds file"
    assert mr == M, "MR mismatch in bases bounds file"

    bases_data_file = "../scripts/bounds/bnddata_spts_lobatto_" + str(N) + "_bpts_chebyshev_" + str(M) + ".txt"
    print("Reading ",bases_data_file)
    nr, mr, gllC, intC, lboundC, uboundC = get_bases_bounds_Tarik(bases_data_file)

    bases_data_file = "../scripts/bounds/bnddata_spts_lobatto_" + str(N) + "_bpts_chebyshev_" + str(M) + ".txt"
    print("Reading ",bases_data_file)
    nr, mr, gllTT, intTT, lboundTT, uboundTT = get_bases_bounds_Tarik(bases_data_file)


    pdf_pages = PdfPages('rec_bnd_N='+str(N)+'_M='+str(M)+ "_out=" + str(O)+'.pdf')

    if N != dataread[0]:
        print('N does not match the data file')
        return

    gllG = dataread[1:N+1]
    solG = dataread[N+1:N+1+N]
    remainingdata = dataread[N+1+N:] #four parts - intpts,min,max,depth
    print("GLL: ",gllG)
    print("Coefficients: ",solG)


    npdata = np.array(remainingdata, dtype=float)
    lennpdata = len(npdata)
    npdata = npdata.reshape((4,int(np.size(npdata)/4)))
    pts = npdata[0,:]
    minG = npdata[1,:]
    maxG = npdata[2,:]
    depth = npdata[3,:]
    depth_pos_indices = np.reshape((np.argwhere(depth > 0)), -1)
    # print(pts)

    npts=1000
    rmin = pts[0]
    rmax = np.max(pts)
    upoly = np.zeros((npts))
    sample_locs = np.zeros(npts)
    mylagpoly = np.zeros((npts,N))
    mylagpolyscaled = np.zeros((npts,N))
    for j in range(npts):
        sample_locs[j] = rmin + (rmax-rmin)*j/npts

    for i in range(N):
        for j in range(npts):
            mylagpoly[j,i] = lagrange_poly(gllG, i, sample_locs[j])
            mylagpolyscaled[j,i] = mylagpoly[j,i]*solG[i]
            upoly[j] += mylagpolyscaled[j,i]

    mindepth = int(np.min(np.abs(depth)))
    maxdepth = int(np.max(np.abs(depth)))


    # print("Modifying location of second-last GLL point for vis:")
    # modind = 880
    # gllG[-2] = sample_locs[modind]
    # solG[-2] = upoly[modind]

    ms = 1 #marker size

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plotindex = 0
    plotindex += 1
    # first plot the function and the bounds (using solid/dotted format)
    int_locs = None
    pos_indices = np.array([], dtype=int)
    for d in range(mindepth, maxdepth+1, 1):
        # print(d)
        new_indices = np.reshape((np.argwhere(depth == d)), -1)
        pos_indices = np.append(pos_indices, new_indices)
        pos_indices = np.reshape(pos_indices, -1)
        neg_indices = np.reshape((np.argwhere(depth == -d)), -1)
        nplots = int(len(pos_indices)/2)
        plt.figure(plotindex)
        plt.plot(sample_locs, upoly, 'r-', label='Solution', linewidth=3)
        plt.plot(sample_locs, upoly*0, 'k--', linewidth=1)
        plt.plot(gllG, solG, 'ko',markersize=ms*8)
        if (d == mindepth):
            pdf_pages.savefig()

        deb_indices = np.reshape((np.argwhere(np.abs(depth) == d)), -1)
        print(d, np.min(maxG[deb_indices]))
        if (d == mindepth):
            int_locs = np.unique(pts[deb_indices])
        for i in range(nplots):
            ids = pos_indices[2*i:2*i+2]
            plt.plot(pts[ids], minG[ids], color=colors[0],linestyle='-',marker='o', markersize=ms*8,linewidth=2,markerfacecolor='none')
            plt.plot(pts[ids], maxG[ids], color=colors[1],linestyle='-',marker='o', markersize=ms*8,linewidth=2,markerfacecolor='none')
        nplots = int(len(neg_indices)/2)
        for i in range(nplots):
            ids = neg_indices[2*i:2*i+2]
            plt.plot(pts[ids], minG[ids], color=colors[0],linestyle='dotted',marker='o', markersize=ms*8,linewidth=2,markerfacecolor='none')
            plt.plot(pts[ids], maxG[ids], color=colors[1],linestyle='dotted',marker='o', markersize=ms*8,linewidth=2,markerfacecolor='none')
        plt.savefig('rec_bnd_d='+str(d)+'_N='+str(N)+'_M='+str(M)+ "_out=" + str(O)+'.pdf',format='pdf',bbox_inches='tight')
        plotindex += 1
        plt.ylim([-0.3, 1.3])
        pdf_pages.savefig()


    # Plot bounds using solid line only
    pos_indices = np.array([], dtype=int)
    for d in range(mindepth, maxdepth+1, 1):
        # print(d)
        new_indices = np.reshape((np.argwhere(depth == d)), -1)
        pos_indices = np.append(pos_indices, new_indices)
        pos_indices = np.reshape(pos_indices, -1)
        neg_indices = np.reshape((np.argwhere(depth == -d)), -1)
        nplots = int(len(pos_indices)/2)
        plt.figure(plotindex)
        plt.plot(sample_locs, upoly, 'r-', label='Solution', linewidth=2)
        plt.plot(sample_locs, upoly*0, 'k--', linewidth=1)
        plt.plot(gllG, solG, 'ko',markersize=ms*3)

        deb_indices = np.reshape((np.argwhere(np.abs(depth) == d)), -1)
        if (d == mindepth):
            int_locs = np.unique(pts[deb_indices])
        for i in range(nplots):
            ids = pos_indices[2*i:2*i+2]
            # print(ids)
            plt.plot(pts[ids], minG[ids], color=colors[0],linestyle='-',marker='o', markersize=ms,linewidth=2)
            plt.plot(pts[ids], maxG[ids], color=colors[1],linestyle='-',marker='o', markersize=ms,linewidth=2)
        nplots = int(len(neg_indices)/2)
        for i in range(nplots):
            ids = neg_indices[2*i:2*i+2]
            plt.plot(pts[ids], minG[ids], color=colors[0],marker='o', markersize=ms,linewidth=2)
            plt.plot(pts[ids], maxG[ids], color=colors[1],marker='o', markersize=ms,linewidth=2)
        plt.savefig('rec_bnd_d='+str(d)+'_N='+str(N)+'_M='+str(M)+ "_out=" + str(O)+'.pdf',format='pdf',bbox_inches='tight')
        plotindex += 1
        plt.ylim([-0.3, 1.3])
        pdf_pages.savefig()
    pdf_pages.close()

    plt.rcParams['lines.linewidth'] = 2  # Sets the default linewidth to 2
    #plot all the bases now first and then individual bases with their bounds
    pdf_pages = PdfPages('rec_bnd_bases_N='+str(N)+'_M='+str(M)+ "_out=" + str(O)+'.pdf')
    plt.figure(plotindex)
    for i in range(N):
        plt.plot(sample_locs, mylagpoly[:,i],label=f'$\phi_{i}$')
    plt.ylim(-0.4,1.3)
    # plt.legend()
    plt.plot(gllG,0*solG,'ko-',markersize=10)
    # plt.plot(int_locs,0*int_locs,'rx')
    pdf_pages.savefig()

    plotindex += 1
    # plt.rcParams['xtick.major.size'] = 10
    # plt.rcParams['ytick.major.size'] = 10

    # Plot all the bases individually
    for i in range(N):
        plt.figure(plotindex)
        plt.plot(sample_locs, mylagpoly[:,i],label=f'$\phi_{i+1}$')
        # plt.plot(intx, lbound[i,:])
        # plt.plot(intx, ubound[i,:])
        plt.plot(gllG,0*solG,'ko-')
        # plt.plot(intx,0*intx,'rx')
        plt.ylim(-0.4,1.3)
        plt.legend(loc='upper left')
        # plt.title('$L_{\eta_{cheb},q},U_{\eta_{cheb},q}$')
        pdf_pages.savefig()
        plotindex += 1

    # Plot all the bases individually with GSLIB bounds
    for i in range(N):
        plt.figure(plotindex)
        plt.plot(sample_locs, mylagpoly[:,i],label=f'$\phi_{i+1}$')
        plt.plot(intx, lbound[i,:])
        plt.plot(intx, ubound[i,:])
        plt.plot(gllG,0*solG,'ko-')
        plt.plot(intx,0*intx,'rs',markerfacecolor='none')
        plt.ylim(-0.4,1.3)
        plt.legend(loc='upper left')
        plt.title('$L_{\eta_{cheb},q},U_{\eta_{cheb},q}$')
        pdf_pages.savefig()
        plotindex += 1

    # Plot all the bases individually with Chebyshev points and optimized bounds
    for i in range(N):
        plt.figure(plotindex)
        plt.plot(sample_locs, mylagpoly[:,i],label=f'$\phi_{i+1}$')
        plt.plot(intC, lboundC[i,:])
        plt.plot(intC, uboundC[i,:])
        plt.plot(gllC,0*solG,'ko-')
        plt.plot(intC,0*intC,'rs',markerfacecolor='none')
        plt.ylim(-0.4,1.3)
        plt.legend(loc='upper left')
        plt.title('$L_{\eta_{cheb},q_{opt}},U_{\eta_{cheb},q_{opt}}$')
        pdf_pages.savefig()
        plotindex += 1

    # Plot all the bases individually with optimized points and bounds
    for i in range(N):
        plt.figure(plotindex)
        plt.plot(sample_locs, mylagpoly[:,i],label=f'$\phi_{i+1}$')
        plt.plot(intT, lboundT[i,:])
        plt.plot(intT, uboundT[i,:])
        plt.plot(gllT,0*solG,'ko-')
        plt.plot(intT,0*intx,'rx')
        plt.ylim(-0.4,1.3)
        plt.legend(loc='upper left')
        plt.title('$L_{\eta_{opt},q_{opt}},U_{\eta_{opt},q_{opt}}$')
        pdf_pages.savefig()
        plotindex += 1

    # gslib bound and optimized points/bounds in same plot
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(N):
        plt.figure(plotindex)
        plt.plot(sample_locs, mylagpoly[:,i],label=f'$\phi_{i+1}$')
        # plt.plot(intx, lbound[i,:], color=cycle[1], linestyle='dashdot',linewidth=2)
        # plt.plot(intx, ubound[i,:],color=cycle[2], linestyle='dashdot',linewidth=2)
        # plt.plot(intx, lbound[i,:], color=cycle[1],linewidth=2)
        plt.plot(intx, ubound[i,:],color=cycle[2],linewidth=2)

        # plt.plot(intC, lboundC[i,:], label='$\underline{v}_{i,\eta_{cheb},q_{opt}}$',color=cycle[1], linestyle='--')
        # plt.plot(intC, uboundC[i,:],label='$\bar{v}_{i,\eta_{cheb},q_{opt}}$',color=cycle[2], linestyle='--')

        # plt.plot(intT, lboundT[i,:],color=cycle[3],linewidth=2)
        plt.plot(intT, uboundT[i,:],color=cycle[4],linewidth=2)

        plt.plot(gllG,0*solG,'ko-')
        plt.plot(intx,0*intx,'rs',markerfacecolor='none')
        plt.plot(intT,0*intT,'mx')
        plt.ylim(-0.4,1.3)
        plt.legend(loc='upper left')
        # plt.title('$L_{\eta_{cheb},q},U_{\eta_{cheb},q}$')
        pdf_pages.savefig()
        plotindex += 1

    # Plot all the bases individually with optimized points and bounds for M=N
    for i in range(N):
        plt.figure(plotindex)
        plt.plot(sample_locs, mylagpoly[:,i],label=f'$\phi_{i+1}$')
        plt.plot(intTT, lboundTT[i,:])
        plt.plot(intTT, uboundTT[i,:])
        plt.plot(gllTT,0*gllTT,'ko-')
        plt.plot(intTT,0*intTT,'rx')
        plt.ylim(-0.4,1.3)
        plt.legend(loc='upper left')
        # plt.title('$L_{\eta_{opt},q_{opt}},U_{\eta_{opt},q_{opt}}$')
        pdf_pages.savefig()
        plotindex += 1

    pdf_pages.close()


    # make animation of bounds with bases
    plt.rcParams['text.usetex'] = True
    plotindex += 1
    basis_cumul_val = np.zeros(npts)
    ubound_cumul_val = np.zeros(M)
    lbound_cumul_val = np.zeros(M)
    pdf_pages = PdfPages('rec_bnd_bases_animation_N='+str(N)+'_M='+str(M)+ "_out=" + str(O)+'.pdf')

    for i in range(N):
        plt.figure(plotindex)
        # plt.plot(sample_locs, upoly, 'r--', label='u(r)', linewidth=3)
        basis_cumul_val[:] += mylagpolyscaled[:,i]
        dummy = np.reshape(uboundTT[i,:],(-1))
        ubound_cumul_val[:] += solG[i]*dummy
        dummy = np.reshape(lboundTT[i,:],(-1))
        lbound_cumul_val[:] += solG[i]*dummy
        plt.plot(sample_locs, basis_cumul_val,color=colors[0],label=fr'$\sum_{{i=0}}^{{{i}}} u_i \phi_i(r)$')
        llabel = fr'$\sum_{{i=0}}^{{{i}}} u_i \underline{{\phi}}_i(r)$'
        plt.plot(intTT, ubound_cumul_val,color=colors[1],label=llabel)
        llabel = fr'$\sum_{{i=0}}^{{{i}}} u_i \bar{{\phi}}_i(r)$'
        plt.plot(intTT, lbound_cumul_val,color=colors[2],label=llabel)
        plt.plot(sample_locs, upoly*0, 'k--', linewidth=1)
        # if (i == 0)
        plt.legend(loc='upper right')
        pdf_pages.savefig()
        plotindex += 1

    pdf_pages.close()





# Using the special variable
# __name__
if __name__=="__main__":
    main()
