# This is the regression testing script for ex5-nguygen
# Currently, this uses cases 1 and 2
import subprocess

class bcolors:
	HEADER = '\033[95m'
	OKGREEN = '\033[92m'
	FAIL = '\033[91m'
	RESET = "\033[0m"

# Case 1:
print(bcolors.HEADER + "Running case 1:" + bcolors.RESET)
print("  HDG.........", end="", flush=True)

subprocess.call("./ex5-nguyen -nx 8 -ny 8 -p 1 -o 3 -dg -hb -reg", shell=True,stdout=subprocess.DEVNULL)
file = open("L2_t.txt", "r")
L2_t = float(file.read())
file.close()
subprocess.call("rm L2_t.txt", shell=True)

file = open("L2_q.txt", "r")
L2_q = float(file.read())
file.close()
subprocess.call("rm L2_q.txt", shell=True)

if L2_t and L2_q < 0.001:
	print(bcolors.OKGREEN + "SUCCESS" + bcolors.RESET)
else:
	print(bcolors.FAIL + "FAIL" + bcolors.RESET)

print("  RT..........", end="", flush=True)

subprocess.call("./ex5-nguyen -nx 8 -ny 8 -p 1 -o 3 -reg", shell=True,stdout=subprocess.DEVNULL)
file = open("L2_t.txt", "r")
L2_t = float(file.read())
file.close()
subprocess.call("rm L2_t.txt", shell=True)

file = open("L2_q.txt", "r")
L2_q = float(file.read())
file.close()
subprocess.call("rm L2_q.txt", shell=True)

if L2_t and L2_q < 0.001:
	print(bcolors.OKGREEN + "SUCCESS" + bcolors.RESET)
else:
	print(bcolors.FAIL + "FAIL" + bcolors.RESET)

# Case 2:
# Command line: ./ex5-nguyen -nx 20 -ny 20 -p 2 -c 25 -o 3 -dg -hb
print(bcolors.HEADER + "Running case 2:" + bcolors.RESET)
print("  HDG.........", end="", flush=True)

subprocess.call("./ex5-nguyen -nx 20 -ny 20 -p 2 -c 25 -o 3 -dg -hb -reg", shell=True,stdout=subprocess.DEVNULL)
file = open("L2_t.txt", "r")
L2_t = float(file.read())
file.close()
subprocess.call("rm L2_t.txt", shell=True)

file = open("L2_q.txt", "r")
L2_q = float(file.read())
file.close()
subprocess.call("rm L2_q.txt", shell=True)

if L2_t and L2_q < 0.001:
	print(bcolors.OKGREEN + "SUCCESS" + bcolors.RESET)
else:
	print(bcolors.FAIL + "FAIL" + bcolors.RESET)

print("  RT..........", end="", flush=True)

subprocess.call("./ex5-nguyen -nx 20 -ny 20 -p 2 -c 25 -o 3 -reg", shell=True,stdout=subprocess.DEVNULL)
file = open("L2_t.txt", "r")
L2_t = float(file.read())
file.close()
subprocess.call("rm L2_t.txt", shell=True)

file = open("L2_q.txt", "r")
L2_q = float(file.read())
file.close()
subprocess.call("rm L2_q.txt", shell=True)

if L2_t and L2_q < 0.001:
	print(bcolors.OKGREEN + "SUCCESS" + bcolors.RESET)
else:
	print(bcolors.FAIL + "FAIL" + bcolors.RESET)

