import os

errors = 0 
base_str = "./ex14 -pa "

meshes = ["../data/inline-quad.mesh",
            "../data/star.mesh",
            "../data/square-disc-nurbs.mesh",
            "../data/disc-nurbs.mesh",
            #"../data/inline-hex.mesh",
            #"../data/pipe-nurbs.mesh",
            #"../data/amr-quad.mesh" # AMR!
            #"../data/fichera.mesh" #3D mesh, not yet implemented
            #"../data/amr-hex.mesh # AMR! 3D!
            #"../data/fichera-amr.mesh # 3D AMR!
            ]

print("b s k int     bdy    basis i r p err")
print(" ")

#sigmas = [-1,1,1]
#kappas = [-1,-1,0]

sigmas = [-1]
kappas = [-1]

I = '--int'
B = '--bdy'
beta = 1

def get_timing(line):
   return repr(line).split('\\')[0].split(":")[1].strip()

for meshi in list(range(0, len(meshes))):
   for r in [2,3]:
      for o in [1,2,3]:
         for init in [0]:
            for sigmakappa_case in range(0,len(sigmas)):
               sigma = sigmas[sigmakappa_case]
               kappa = kappas[sigmakappa_case]
               for bas in ['--lob']:#,'--pos']:
                  mesh = " -m "+meshes[meshi]
                  params = " -k "+str(kappa)+" -s "+str(sigma)#+" -b "+str(beta)
                  case = " "+str(I)+" "+str(B)+" "+str(bas)+" -i "+str(init)       
                  hp = " -r "+str(r)+" -o "+str(o)
                  runstr = base_str+mesh+params+case+hp+" &> ex14out.log"
                  #print(runstr)
                  os.system(runstr)
                  successes = 0
                  fails = 0
                  time_full = "-1"
                  time_pa = "-1"

                  with open('ex14out.log') as f:
                     for line in f:
                        if 'Number of PCG iterations' in line:
                           successes += 1
                        if 'PCG: No convergence!' in line:
                           fails += 1
                        if 'full solver time (ms):' in line:
                           time_full = get_timing(line)
                        if 'pa   solver time (ms):' in line:
                           time_pa = get_timing(line)

                  #print(" ".join([runstr,str(fails)]))
                  if(successes+fails != 2):
                     #print("Non-convergence error" )
                     fails = 9

                  s = [str(mesh),str(sigma),str(kappa),str(bas),str(init),str(r),str(o),str(fails),time_full,time_pa]
                  try:
                     print(" ".join(s))
                  except:
                     print("can't print")
                     print(s)
                  #if(successes == 0):
                  #   print("PA and full both failed ")
                  #   quit()
                  #if(successes == 1):
                  #   print("Only one failure to converge ")
                  #   quit()
