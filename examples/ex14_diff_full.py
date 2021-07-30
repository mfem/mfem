import os

errors = 0 
base_str = "./ex14 -pa "

# NURBS are not tensor

meshes = ["../data/inline-hex.mesh",
      #"../data/inline-quad.mesh",  
           # "../data/star.mesh"
           ]#,
#            "../data/escher.mesh",
 #           "../data/fichera.mesh",
  #          "../data/inline-hex.mesh",
   #         "../data/inline-quad.mesh"
    #        ]

maxrs = [3,3,3,2,2,2]

#print("b s k int     bdy    basis i r p err")
#print(" ")

def get_data(line):
   return repr(line).split('\\')[0].split("=")[1].strip()

for bas in ['--lob']:#,'--pos']:
   for thismesh in meshes:#list(range(0, len(meshes))):
      for o in [1,2]:#[1,2,3]:
         for r in [0,1]:#list(range(0, maxrs[o-1]+1)):
            for I in ['--int']: #['--noint','--int']:
               for B in ['--bdy']: #['--nobdy','--bdy']:
                  if( not ((I == '--noint') and (B == '--nobdy'))):
                     for sigma in [0, -1, 1]:
                        for kappa in [0, -1, 1]:
                           for beta in [0, 1]: 
                              if( sigma**2 + kappa**2 + beta**2 > 0 ):
                                 if( not(beta == 0 and sigma == 0 and kappa == 0 )):
                                    if((thismesh=="../data/inline-quad.mesh") or (thismesh=="../data/inline-hex.mesh")):
                                       inits = [0,2,3]
                                    else:
                                       inits = [0,2]
                                    for init in inits:
                                       mesh = " -m "+thismesh
                                       params = " -k "+str(kappa)+" -s "+str(sigma)+" -b "+str(beta)
                                       case = " "+str(I)+" "+str(B)+" "+str(bas)+" -i "+str(init)       
                                       hp = " -r "+str(r)+" -o "+str(o)
                                       runstr = base_str+mesh+params+case+hp+" &> ex14out.log"
                                       #print(runstr)
                                       os.system(runstr)
                                       with open('ex14out.log') as f:
                                          for line in f:
                                             if 'Timing pa' in line:
                                                patime = get_data(line)
                                             if 'Timing full' in line:
                                                fulltime = get_data(line)
                                             if '||ydiff||' in line:
                                                error = get_data(line)                                                
                                             if 'Segmentation Fault' in line:
                                                print("Segmentation Fault")
                                                print(runstr)
                                                quit()
                                             if 'Assertion failed' in line:
                                                print("Assertion failed")
                                                print(runstr)
                                                quit()
                                                

                                          if( float(error) > 1.0e-10 ):
                                             #s = [str(mesh),str(beta),str(sigma),str(kappa),str(I),str(B),str(bas),str(init),str(r),str(o),error,patime,fulltime]
                                             #print(" ".join(s))
                                             errors = errors + 1
                                          else:
                                             error = str(0)

                                          s = [str(mesh),str(beta),str(sigma),str(kappa),str(I),str(B),str(bas),str(init),str(r),str(o),error,fulltime,patime]
                                          print("error = ", error)

                                          s = ["./ex14 -pa ",str(mesh),
                                                " -b ",str(beta),
                                                " -s ",str(sigma),
                                                " -k ",str(kappa),
                                                str(I),str(B),str(bas),
                                                " -i ",str(init),
                                                " -r ",str(r),
                                                " -o ",str(o)]
                                          print(" ".join(s))

                                          if( errors >= 5 ):
                                             print("too many fails ")
                                             quit()
