# PSO-ND-I-for-Optimal-ADRC-Tuning-Code

Here are available the Julia codes used to implement the PSO-ND-I algorithm for optimal ADRC tuning in the associated research paper.

## 📁 Folder Structure

`Julia-Code/`

This folder contains the main Julia scripts required to run PSO-ND-I:

| File | Description |
|------|-------------|
| `ADRC_Simulation.jl` | Defines the **dynamic simulation** and **objective function** used to evaluate the controller parameters. |
| `PSO_ND_I_ADRC.jl` | Implements the **PSO-ND-I algorithm** with integral action, calling `simulate_adrc` to evaluate each particle's fitness. |
| `PSO_ND_I_ParallelRuns.jl` | Runs **multiple independent PSO-ND-I optimizations** in parallel and saves execution times. |

## 🚀 How to Run

1. Install Julia and the required packages:

   ```julia
   using Pkg
   Pkg.add(["MAT", "DifferentialEquations", "Symbolics", "RuntimeGeneratedFunctions", "Polynomials"])
   ```

2. Navigate to the `Julia-Code/` directory.
3. Run the main script:

   ```bash
   julia PSO_ND_I_ParallelRuns.jl 30
   ```

This script will:
- Execute 30 independent PSO-ND-I runs using 6 worker processes by default.
- Optimize the ADRC parameters using the simulation-based objective function.
- Save optimization results and execution times in `.mat` files.

The number of runs can be changed through the command-line argument, and the number of workers can be set using `NWORKERS` in the main script.
