M_gpu = load('gpu_mass.txt');
K_gpu = load('gpu_adv.txt');

M_cpu = load('cpu_mass.txt');
K_cpu = load('cpu_adv.txt');

Mass_error = norm(M_cpu-M_gpu,'inf')
Adv_error  = norm(K_cpu-K_gpu,'inf')