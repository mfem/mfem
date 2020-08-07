function cmpRiemannSolvers()

ul = 0;
ur = 1;
n = 1/sqrt(2)*[1 1];
problem = 1;

fl = flux(ul, n, problem);
fr = flux(ur, n, problem);

sl = wavespeed(ul, n, problem);
sr = wavespeed(ur, n, problem);

LF = 0.5 * (dot(fl + fr, n) + max(abs(sl),abs(sr)) * (ul - ur))

if 0 < sl
  HLL = dot(fl, n);
elseif 0 > sr
  HLL = dot(fr, n);
else
  HLL = (dot(sr*fl - sl*fr, n) + sl*sr*(ur-ul)) / (sr-sl);
end

HLL
end

function f = flux(u, n, problem)

switch problem
  case 0
    f = velocity * u;
  case 1
    f = u*u/2 * [1; 1];
end
end

function s = wavespeed(u, n, problem)

switch problem
  case 0
    s = dot(velocity, n);
  case 1
    s = u * sum(n);
end
end

function v = velocity()
v=[1; -0.5];
end