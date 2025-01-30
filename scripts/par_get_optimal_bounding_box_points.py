import concurrent.futures
import subprocess

def run_dum(N, M):
    command = f"python3 get_optimal_bounding_box_points.py --N {N} --M {M}"
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

def main():
    # Define the range of values for N and M
    N_values = range(8, 10)

    # Create a list of tasks
    tasks = []
    for N in N_values:
        for M in range(3 , 20):  # M ranges from N+1 to 30
            tasks.append((N, M))

    # Use ThreadPoolExecutor to run tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(run_dum, N, M): (N, M) for N, M in tasks}
        for future in concurrent.futures.as_completed(futures):
            N, M = futures[future]
            try:
                result = future.result()
                print(f"Result for N={N}, M={M}: {result}")
            except Exception as exc:
                print(f"Generated an exception for N={N}, M={M}: {exc}")

if __name__ == "__main__":
    main()
