# Parallel PSO-ND-I runs.
# Usage (from this directory):
#   julia --project=. PSO_ND_I_ParallelRuns.jl [number_of_runs]

using Distributed
using MAT

# Run settings.
const NWORKERS = 6
const N_RUNS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30

# Start workers.
if !isempty(workers())
    rmprocs(workers())
end
addprocs(NWORKERS)

# Load optimizer.
@everywhere cd(@__DIR__)
@everywhere begin
    include("PSO_ND_I_ADRC.jl")
    import .PSONDIADRC as RP
end

# Run with a unique seed.
@everywhere function run_pso_nd_i(run_id::Int)
    start_time = time()
    RP.optimize_adrc(run_id, 0; seed=12345 + 10000 * run_id)
    elapsed_seconds = time() - start_time
    println("Run $run_id completed in $(round(elapsed_seconds; digits=3)) s")
    return (i=run_id, tf=elapsed_seconds)
end

println("Logical CPU threads: ", Sys.CPU_THREADS)
println("Worker processes: $NWORKERS")
println("Independent PSO-ND-I runs: $N_RUNS")

# Execute parallel runs.
start_time = time()
results = pmap(run_pso_nd_i, 1:N_RUNS)
total_seconds = time() - start_time
println("Total elapsed time: $total_seconds s")

# Save runtime data.
output_directory = "pso-nd-i"
mkpath(output_directory)
output_file = joinpath(output_directory, "timing_$(N_RUNS)runs_$(NWORKERS)workers.mat")
matwrite(output_file, Dict(
    "tfProc" => total_seconds,
    "per_process" => results,
    "N_RUNS" => N_RUNS,
    "NWORKERS" => NWORKERS,
))
println("Timing results saved to: $output_file")
