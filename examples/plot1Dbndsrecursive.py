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
    dataread = np.loadtxt(data_file)

    pdf_pages = PdfPages('rec_bnd_N='+str(N)+'_M='+str(M)+ "_out=" + str(O)+'.pdf')

    if N != dataread[0]:
        print('N does not match the data file')
        return

    gllG = dataread[1:N+1]
    solG = dataread[N+1:N+1+N]
    remainingdata = dataread[N+1+N:] #four parts - intpts,min,max,depth

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

    ms = 1 #marker size

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plotindex = 0
    plotindex += 1
    # first plot by depth
    pos_indices = np.array([], dtype=int)
    for d in range(mindepth, maxdepth+1, 1):
        # print(d)
        new_indices = np.reshape((np.argwhere(depth == d)), -1)
        pos_indices = np.append(pos_indices, new_indices)
        pos_indices = np.reshape(pos_indices, -1)
        neg_indices = np.reshape((np.argwhere(depth == -d)), -1)
        nplots = int(len(pos_indices)/2)
        plt.figure(plotindex)
        plt.plot(sample_locs, upoly, 'k-', label='Solution')

        plt.plot(sample_locs, upoly*0, 'k--', linewidth=1)
        deb_indices = np.reshape((np.argwhere(np.abs(depth) == d)), -1)
        print(d, np.min(maxG[deb_indices]))
        for i in range(nplots):
            ids = pos_indices[2*i:2*i+2]
            # print(ids)
            plt.plot(pts[ids], minG[ids], color=colors[0],linestyle='-',marker='o', markersize=ms,linewidth=1)
            plt.plot(pts[ids], maxG[ids], color=colors[1],linestyle='-',marker='o', markersize=ms,linewidth=1)
        nplots = int(len(neg_indices)/2)
        for i in range(nplots):
            ids = neg_indices[2*i:2*i+2]
            plt.plot(pts[ids], minG[ids], color=colors[0],linestyle='dotted',marker='o', markersize=ms,linewidth=1)
            plt.plot(pts[ids], maxG[ids], color=colors[1],linestyle='dotted',marker='o', markersize=ms,linewidth=1)
        plt.savefig('rec_bnd_d='+str(d)+'_N='+str(N)+'_M='+str(M)+ "_out=" + str(O)+'.pdf',format='pdf',bbox_inches='tight')
        plotindex += 1
        plt.ylim([-0.3, 1.3])
        pdf_pages.savefig()
    pdf_pages.close()



# Using the special variable
# __name__
if __name__=="__main__":
    main()
