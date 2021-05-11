import os

errors = 0 
base_str = "./ex14 -pa "

meshes = ["../data/inline-quad.mesh",
            "../data/star.mesh",
            "../data/escher.mesh",
            "../data/fichera.mesh"
            ]

print("b s k int     bdy    basis i r p err")
print(" ")

for meshi in list(range(0, len(meshes))):
   for r in [0,1,2]:
      for o in [1,2,3]:
         for init in [0,1,2,3,4]:
            for I in ['--noint','--int']:
               for B in ['--nobdy','--bdy']:
                  for sigma in [0, -1, 2]:
                     for kappa in [0, -1, 2]:
                        for beta in [0, 1, 2]: 
                           for bas in ['--lob']:#,'--pos']:
                              mesh = " -m "+meshes[meshi]
                              params = " -k "+str(kappa)+" -s "+str(sigma)#+" -b "+str(beta)
                              case = " "+str(I)+" "+str(B)+" "+str(bas)+" -i "+str(init)       
                              hp = " -r "+str(r)+" -o "+str(o)
                              runstr = base_str+mesh+params+case+hp+" &> ex14out.log"
                              print(runstr)
                              os.system(runstr)
                              with open('ex14out.log') as f:
                                 for line in f:
                                    pass
                                 last_line = line
                                 #print(last_line)
                                 error = str(last_line)
                                 #print(error)
                                 if( float(error) > 1.0e-10 ):
                                    s = [str(mesh),str(beta),str(sigma),str(kappa),str(I),str(B),str(bas),str(init),str(r),str(o),error]
                                    print(" ".join(s))
                                    errors = errors + 1
                                    if( errors >= 2 ):
                                       quit()
