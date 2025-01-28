import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc

def main():
    parser = argparse.ArgumentParser(description="A script that processes some arguments")

    # Add arguments
    parser.add_argument('--nodep', type=int, help='node type', default=1)
    parser.add_argument('--intp', type=int, help='intp type', default=-1)

    args = parser.parse_args()

    intp = args.intp
    nodep = args.nodep

    fname = "minmr_bnd_comp.txt"
    data = np.loadtxt(fname)
    data = np.reshape(data, (-1, 9))
    nmin = int(np.min(data[:, 0]))
    nmax = int(np.max(data[:, 0]))

    rc('text', usetex=True)

    pdf_pages = PdfPages('bndcompeff.pdf')
    for i in range(nmin, nmax+1):
        count = 1
        plt.figure(figsize=(12, 12))
        for xc in [1,6,7,8]:
            plt.subplot(2, 2, count)
            for et in range(2,3):  #error type
                dataf = data[data[:, 0] == i] #get data for that N
                # plt.figure()
                dataf2 = dataf[dataf[:, 2] == 0]
                plt.plot(dataf2[:, xc], dataf2[:, 3+et], 'ro-', label='GL+End')
                dataf2 = dataf[dataf[:, 2] == 1]
                plt.plot(dataf2[:, xc], dataf2[:, 3+et], 'go-', label='GLL')
                dataf2 = dataf[dataf[:, 2] == 2]
                plt.plot(dataf2[:, xc], dataf2[:, 3+et], 'bo-', label='Chebyshev')
                dataf2 = dataf[dataf[:, 2] == 3]
                plt.plot(dataf2[:, xc], dataf2[:, 3+et], 'ro--', label='Optimal GL+End')
                dataf2 = dataf[dataf[:, 2] == 4]
                plt.plot(dataf2[:, xc], dataf2[:, 3+et], 'go--', label='Optimal GLL')
                dataf2 = dataf[dataf[:, 2] == 5]
                plt.plot(dataf2[:, xc], dataf2[:, 3+et], 'bo--', label='Optimal Chebyshev')
                dataf2 = dataf[dataf[:, 2] == 6]
                plt.plot(dataf2[:, xc], dataf2[:, 3+et], 'co--', label='Optimal Uniform')
                # dataf2 = dataf[dataf[:, 2] == 7]
                # plt.plot(dataf2[:, xc], dataf2[:, 3+et], 'ko--', label='Optimal')

                dataf2 = dataf[dataf[:, 2] == 0]
                x1 = dataf2[1, xc]
                y1 = dataf2[1, 3+et]
                xrate = dataf2[:, xc]
                yrate = dataf2[:, 3+et]
                if xc == 1:
                    rate = 2.45
                    for j in range(0, len(yrate)):
                        yrate[j] = y1*(x1**rate)/(xrate[j]**rate)
                else:
                    if xc == 6:
                        rate = 1.0
                    elif xc == 7:
                        rate = 2.0
                    else:
                        rate = 2.25
                    for j in range(0, len(yrate)):
                        yrate[j] = y1*(xrate[j]**rate)/(x1**rate)
                # print(xrate)
                # print(yrate)
                # if (xc == 6):
                    # input(' ')
                plt.plot(xrate, yrate, 'c--', label=f'rate: {rate}')

                plt.legend()
                plt.yscale('log')
                if (xc == 6 or xc ==7 or xc ==8):
                    plt.xscale('log')
                plt.title('N='+str(i))
                if xc == 1:
                    plt.xlabel('M')
                elif xc == 6:
                    plt.xlabel(r"$\Delta x_{min}$")
                elif xc == 7:
                    plt.xlabel(r"$\Delta x_{max}$")
                elif xc == 8:
                    plt.xlabel(r"$\Delta x_{avg}$")

                if et == 0:
                    plt.ylabel('Error (Area)')
                elif et == 1:
                    plt.ylabel('Compactness (Linf)')
                elif et == 2:
                    plt.ylabel(r"$l_2\qquad\,\, \sum_{j=1}^{p+1}\sqrt{\frac{1}{N_{\tt samp}}\sum_{i=1}^{N_{\tt samp}}\left(\bar{\phi}_j(x_i)- \phi_j(x_i)\right)^2 + \left(\underline{\phi}_j(x_i) - \phi_j(x_i)\right)^2 }$")
                    # plt.ylabel(r"$l_2\,\, \sum_{j=1}^{p+1}\sqrt{\frac{1}{N_{\tt samp}}\sum_{i=1}^{N_{\tt samp}}(\bar{\phi}_j(x_i) - \phi_j(x_i))^2 + (\underline{\phi}_j(x_i) - \phi_j(x_i))^2}$")
                    # plt.ylabel(r${\tt l2} \sqrt{\frac{1}{N_{\tt samp}}\sum_{i=1}^{N_{\tt samp}}\big(LB_j(x_i) - L(x_i)\big)^2}$')
                # pdf_pages.savefig()
                # plt.close()
            count += 1
        pdf_pages.savefig()
        plt.close()
    pdf_pages.close()


# Using the special variable
# __name__
if __name__=="__main__":
    main()
