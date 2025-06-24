# 16Adic-Goldbach: Residue-Class Analysis of Goldbach's Conjecture  

**Author:** [Enrique A. Ramirez Bochard](https://orcid.org/0009-0005-9929-193X)  

**DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15722143.svg)](https://doi.org/10.5281/zenodo.15722143)  
**Companion Work (Collatz):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15516922.svg)](https://doi.org/10.5281/zenodo.15516922)  

---

## 🔍 Bridging Collatz Dynamics and Goldbach Partitions  
This project extends the **16-adic framework** from [prior Collatz research](https://doi.org/10.5281/zenodo.15516922) to Goldbach's Conjecture, revealing:  
- **Correlation:** Fixed divisors `k_r` in Collatz map ↔ Goldbach partition densities `w_r`  
- **Hard Cases:** Residues like `n ≡ 14 mod 16` show low entropy (`H(14) ≈ 1.2`) and 3× longer verification times  

---

## 📂 Repository Structure  
| Directory | Contents | Key Files |  
|-----------|----------|-----------|  
| `/code` | Optimized verification | `segmented_sieve.cpp`, `residue_analyzer.py` |  
| `/data` | Empirical results | `partition_counts_2²⁰-2²¹.csv`, `runtime_by_residue.log` |  
| `/paper` | LaTeX source | `main.tex`, `collatz-goldbach-table.tex` |  

---

## 🆚 Comparative Framework  
| Feature | Collatz Work | Goldbach Work |  
|---------|--------------|---------------|  
| **Focus** | Iterative dynamics | Additive primes |  
| **Core Metric** | Contraction rate `k_r` | Partition weight `w_r` |  
| **Hard Cases** | `r = 3, 7, 11, 15` | `r = 6, 14` |  
| **Key Insight** | `k_r` determines trajectory behavior | `w_r` predicts partition scarcity |  

---

## 🛠️ Research Workflow  

### Core Analysis Engine  
python goldbach_activated_sums_v7.py \
        --ml-analysis \
        --visualize=14 \
        --num-primes=2000 \
        --export-deltak

Capabilities:
    • Δₖ set generation and analysis
    • ML prediction of "hard" numbers (supports Theorem 2)
    • Residue-specific visualization

Verification Pipeline
    1. Hypothesis Generation
       python goldbach_test_tuple_hypothesis.py --test-mod=30 --quick-verify
    2. Large-Scale Verification
       ./segmented_sieve --range=1e6 --residue=14 --threads=4

📊 Key Results
    • Theoretical: Weighted Partition Count Theorem (Section 4)
    • Empirical: 75% search space reduction via 16-adic filtering
    • Computational: Verified up to 2³⁴ for all residues (58 CPU-hours @ 4-core)

📣 Dissemination Plan
    1. GitHub (this repo)
    2. Zenodo https://zenodo.org/badge/DOI/10.5281/zenodo.15722143.svg
    3. ORCID https://img.shields.io/badge/ORCID-0009--0005--9929--193X-a6ce39
    4. Social Media (Tweet + LinkedIn post)
      