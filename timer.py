#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacobfaibussowitsch
"""

import os

class Runner(object):
  def __init__(self,*args):
    self.argList = list(args)
    self.setuptime = {}
    self.solvetime = {}
    self.itercount = {}
    self.jval = ' '.join(self.argList).partition('--jacobi-value')[2].split()[0]
    return

  def parseOutput(self,out,device,smoother):
    import re

    refelems = re.compile('^Number of finite element unknowns: ([0-9]+)')
    refsetup = re.compile('^Setup time = ([0-9]+\.[0-9]+)')
    refsolve = re.compile('^Solve time = ([0-9]+\.[0-9]+)')
    refiter  = re.compile('\s+Iteration :\s+([0-9]+)')

    if smoother not in self.setuptime.keys():
      self.setuptime[smoother] = {}
    if smoother not in self.solvetime.keys():
      self.solvetime[smoother] = {}
    if smoother not in self.itercount.keys():
      self.itercount[smoother] = {}

    for line in out.split('\n'):
      elems = refelems.search(line)
      if elems:
        self.elems = elems.group(1)
        continue
      setuptime = refsetup.search(line)
      if setuptime:
        if device not in self.setuptime[smoother]:
          self.setuptime[smoother][device] = [setuptime.group(1)]
        else:
          self.setuptime[smoother][device].append(setuptime.group(1))
        continue
      solvetime = refsolve.search(line)
      if solvetime:
        if device not in self.solvetime[smoother]:
          self.solvetime[smoother][device] = [solvetime.group(1)]
        else:
          self.solvetime[smoother][device].append(solvetime.group(1))
        continue
      itercnt = refiter.search(line)
      if itercnt:
        cnt = itercnt.group(1)

    self.itercount[smoother][device] = cnt
    return

  def run(self):
    import time
    import subprocess

    nit = 10
    for device in ['cpu','cuda']:
      argListBase = self.argList+['--device',device]
      for smoother in ['J','GS','DR']:
        argList = argListBase+['--smoother',smoother]
        argStr = ' '.join(argList)
        print('Running {}'.format(argStr),end='\t',flush=True)
        start = time.time()
        for _ in range(nit):
          with subprocess.Popen(argList,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as runner:
            (out,err) = runner.communicate()
            assert not runner.returncode,'{} raised returncode {}:\nstdout: {}\nstderr:{}'.format(argStr,runner.returncode,out.decode(),err.decode())
          self.parseOutput(out.decode(),device,smoother)
        print('success average time: {:.3f} seconds'.format((time.time()-start)/float(nit)))
    return

  def get(self):
    def averageTimes(timeDict):
      averaged = {}
      for smoother,devices in timeDict.items():
        averaged[smoother] = {}
        for device,timings in devices.items():
          averaged[smoother][device] = sum(float(t) for t in timings)/float(len(timings))
      return averaged

    return {
      'setuptime' : averageTimes(self.setuptime),
      'solvetime' : averageTimes(self.solvetime),
      'itercount' : self.itercount,
      'jval'      : self.jval,
      'elemcount' : self.elems,
      'arglist'   : self.argList,
    }


def main(exe,ordRange,refine,mesh,outFile,jacobiVal):
  assert os.path.exists(exe), 'Could not find {}'.format(exe)
  assert os.path.exists(mesh), 'Could not find {}'.format(mesh)

  results = {}
  try:
    for order in ordRange:
      runner = Runner(exe,'--order',str(order),'--refine',str(refine),'--mesh',mesh,'--jacobi-value',str(jacobiVal))
      runner.run()
      results[order] = runner.get()
  except Exception as e:
    print(e)
    res = runner.get()
    if res:
      results[order] = res
    print('\nsaving progress so far')

  with open(outFile,'w') as fd:
    import json
    print('Writing results to file:',outFile,end='\t')
    json.dump(results,fd)
    assert os.path.exists(outFile), '\nError writing results file {}'.format(outFile)
    print('success')
  return


if __name__ == '__main__':
  import argparse

  defaultOutput = 'output_o{ordmin}_{ordmax}_r{refine}_j{jval}'
  parser = argparse.ArgumentParser(description='Collect timing results for DRSmoothers',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('exec',help='path to example to time')
  parser.add_argument('-o','--order-range',metavar='int',default=[4,13],type=int,nargs=2,dest='ordrange')
  parser.add_argument('-r','--refine',metavar='int',default=0,type=int,dest='refine')
  parser.add_argument('-m','--mesh',default='../data/star.mesh')
  parser.add_argument('-t','--target',default=defaultOutput,help='path to output file')
  parser.add_argument('-jv','--jacobi-value',default=0.666,type=float,help='Jacobi smoother value',dest='jv')
  args = parser.parse_args()

  args.target = args.target.format(ordmin=args.ordrange[0],ordmax=args.ordrange[1],refine=args.refine,jval=str(args.jv).replace('.','_'))

  if not args.target.endswith('.json'):
    args.target += '.json'

  main(os.path.abspath(args.exec),range(*args.ordrange),args.refine,os.path.abspath(args.mesh),os.path.abspath(args.target),args.jv)
