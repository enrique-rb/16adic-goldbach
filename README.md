# A 16-adic Framework for Goldbach Partitions
**Version 1.0** **Author:** Enrique A. Ramirez Bochard  
**ORCID:** [0009-0005-9929-193X](https://orcid.org/0009-0005-9929-193X)

This repository contains the full paper, source code, and empirical data for the research project linking Goldbach partition densities to the dynamics of a simplified Collatz map.

## Abstract
This work presents a novel framework for investigating Goldbach's Conjecture by analyzing the distribution of prime pairs modulo 16. This approach reveals constrained additive structures and allows for the prediction of "hard" cases—even numbers with relatively few prime partitions. The core of this research is a new connection between the density of Goldbach partitions and the behavioral dynamics of a simplified, fixed-divisor variant of the Collatz map. We formalize this link with a Weighted Partition Count theorem, validate its predictions against empirical data, and propose a residue-aware sieving algorithm.

## Repository Contents

* `/paper`: Contains the manuscript "A 16-adic Framework for Goldbach Partitions" in both LaTeX source (`.tex`) and compiled PDF formats.
* `/code`: Contains all C++ and Python scripts used for verification, analysis, and data generation.
* `/data`: Contains the pre-computed empirical data (`goldbach_partitions_1M-2M.csv`) for Goldbach partition counts for all even numbers `n` in the interval `[1048576, 2097152]`. This data is used to validate the theoretical model in the paper.

## Reproducing the Results

### Dependencies
* **C++:** A modern C++ compiler (e.g., `g++`).
* **Python:** Python 3.x, `numpy`, `sympy`, `scikit-learn`, `matplotlib`.
    ```bash
    pip install numpy sympy scikit-learn matplotlib
    ```

### Step 1: Generate the Empirical Partition Data
The primary data file (`goldbach_partitions_1M-2M.csv`) can be regenerated using the C++ partition counter.

1.  **Compile the code:**
    ```bash
    g++ -std=c++17 -O3 code/goldbach_partition_counter.cpp -o partition_counter
    ```
2.  **Run the calculation:** (This may take some time)
    ```bash
    ./partition_counter --start=1048576 --end=2097152 --output=data/goldbach_partitions_1M-2M.csv
    ```

### Step 2: Run the ML Analysis and Exploration
The Python script `goldbach_activated_sums_v7.py` can be used to explore the Δₖ sets and perform the machine learning analysis described in the paper.

```bash
python code/goldbach_activated_sums_v7.py --num-primes 1000 --ml-analysis --show-plots
```
## Program Descriptions and Recommendations
I did also include a powerful and well-differentiated set of tools. The best program to use depends entirely on your specific goal, as each one is/can be optimized for a different task.
Here is a detailed breakdown and my recommendation for each use case.

### Summary of Programs

| Program | Primary Goal | Key Technology | Strengths | Limitations |
| :----------------------------------------- | :------------------------------- | :------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------ |
| `goldbach_partition_counter.cpp` | Empirical Data Generation | C++, Optimized Counting Algorithms | - High-performance for accurate partition count generation.<br>- Specifically designed for generating the empirical data used in the paper. | - Purely a data generator; no analysis or verification features. |
| `goldbach_activated_sums_v7.py` | Theoretical Exploration & Analysis | Python, SymPy, Scikit-learn, Matplotlib | - Directly models your Δₖ sets.<br>- Predicts "hard" evens using Machine Learning.<br>- Rich visualization and data export.<br>- Highly configurable for research. | - Not built for speed.<br>- Unsuitable for verifying extremely large numbers. |
| `goldbach_segmented_sieve.cpp` | High-Performance Verification | C++, Segmented Sieve of Eratosthenes | - Extremely memory efficient; can test very large k.<br>- Fast and deterministic.<br>- Includes checkpointing to resume long tests. | - Purely a verifier; no data analysis features. |
| `goldbach_probabilistic_parallelized.cpp` | Verification for Very Large Numbers | C++, Miller-Rabin Primality Test | - Miller-Rabin is fast for individual large numbers.<br>- Deterministic for all numbers up to $2^{64}$.<br>- Can test a wide range of k. | - Inefficiently generates its initial prime list.<br>- The provided code is not actually parallelized (it's missing OpenMP pragmas). |
| `goldbach_tuple_hypothesis.py` / `.cpp` | Hypothesis Prototyping | Python/C++, Simple Sieve | - Simple, easy-to-understand logic.<br>- Good for quickly testing the binary interval hypothesis on a small scale. | - The simple sieve consumes a lot of memory.<br>- Not feasible for k much larger than 25-30. |

### Recommendations
Based on this analysis, here is which program I would prefer for each task:

##### For Exploring Your Theory and Analyzing "Hard" Numbers:
Winner: **goldbach_activated_sums_v7.py**
This is the most unique and valuable tool, as it is the only program that directly implements the theoretical concepts from the research papers, such as the Δₖ sets.
Use this program when you want to:
* Generate data on which even numbers are resolved by adding new primes.
* Use the machine learning model to predict which even numbers are difficult to resolve. This directly supports your goal of identifying "hard" cases based on residue classes.
* Visualize the partition structure for specific even numbers.
* Export data for further analysis in papers or other tools. This program is the engine for your research framework.
##### For Brute-Force Verification up to the Highest Possible Number:
Winner: **goldbach_segmented_sieve.cpp**
This is the most robust and well-engineered verification tool. The segmented sieve is the industry-standard algorithm for this kind of task precisely because it balances speed and memory efficiency.
Use this program when you want to:
* Verify that the Goldbach conjecture holds for all even numbers in a large interval, like $[2^{33}, 2^{34})$.
* Run a multi-day or multi-week computation, relying on its checkpointing to save progress.
* Generate a definitive "no counterexamples found up to N" statement for your research. This is your workhorse for pushing the boundaries of computational verification.
##### For Generating Empirical Partition Data:
Winner: **goldbach_partition_counter.cpp**
This is the dedicated tool for creating the empirical data sets used to validate your theoretical models. It's optimized for this specific task and is crucial for reproducing the results presented in your paper.
Use this program when you want to:
* Re-generate the goldbach_partitions_1M-2M.csv file or similar datasets.
* Create new empirical data for different ranges to further test the theory.
##### For Prototyping or Simple Checks:
Winner: **goldbach_tuple_hypothesis.py** (or the **.cpp** version for more speed) 
These programs are great for quickly testing a hypothesis on a small scale without the complexity of the other tools. The Python version is particularly good for experimentation because it's easy to read and modify.
Use this program when you want to:
* Quickly check the "binary interval hypothesis" for k up to around 20.
* Have a simple, readable example of the core Goldbach check.
* Tweak the logic or test a new idea before implementing it in the more complex C++ programs.

### Conclusion
You don't have to choose just one. You have a complete toolkit for a computational and theoretical attack on the problem. I would recommend using them in a workflow:

* Generate Data: Start by using goldbach_partition_counter.cpp to create the foundational empirical data.
* Explore & Hypothesize: Use goldbach_activated_sums_v7.py to analyze the structure of the problem and generate hypotheses about which numbers are "hard", leveraging the generated data.
* Verify & Prove: Use goldbach_segmented_sieve.cpp to perform the heavy-duty computations that verify the conjecture holds true for massive ranges, providing confidence in your theoretical findings.
* Prototype & Test: Use goldbach_tuple_hypothesis.py for any quick, small-scale tests of new ideas that come up during your research.

### Citing this Work
If you use the materials in this repository, please cite the research paper:

```bibtex
@misc{ramirez_bochard_2025_goldbach,
  author       = {Enrique A. Ramirez Bochard},
  title        = {{A 16-adic Framework for Goldbach Partitions: Linking Collatz Dynamics to Additive Prime Structures}},
  month        = Jun,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15722143},
  url          = {[https://doi.org/10.5281/zenodo.15722143](https://doi.org/10.5281/zenodo.15722143)}
}
```
