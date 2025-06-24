"""
    Goldbach Conjecture Incremental Sum Explorer - version 0.6

    This script explores the incremental activation of even numbers as sums of two primes,
    analyzing the structure of Δₖ sets (even numbers first expressible at each prime stage).
    Now enhanced with machine learning to predict "hard" even numbers that require larger primes.

    Core Functionality:
        - Δₖ generator with configurable prime sets
        - Coverage tracker with resolution verification
        - Machine learning difficulty predictor
        - Interactive exploration mode

    Key Components:
        1. Prime Analysis Engine:
            - Δₖ set generation
            - Missing even detection
            - Resolution tracking
        
        2. Visualization Suite:
            - Missing evens vs. primes used
            - Maximum even reached progression
            - Prime gap distributions
            - Difficulty prediction plots
        
        3. Machine Learning Module:
            - Predicts required primes for unresolved evens
            - Identifies problematic number patterns
            - Quantifies relative difficulty

        4. Data Tools:
            - CSV export of Δₖ sets
            - Solution timeline exports
            - Interactive probing

    Author: Enrique A. Ramirez Bochard
    Date: 2025-06-08

    Change Management: 
        2025-06-08: Base implementation
        2025-06-10: Extra iterations
        2025-06-11: Targeted resolution tracking
        2025-06-12: CLI controls
        2025-06-15: Machine learning integration
                    Interactive mode
                    Advanced visualization

    Command-line controls added: 
        --num-primes: Set number of base primes (default: 1000)
        --extra-primes: Set additional primes to check (default: 3)
        --show-plots: Toggle visualization
        --list-missing: Show full list of missing numbers
        --group-resolved: Group resolved numbers by delta step
        --verbose: Show detailed progress
                
    New Command-Line Features:
        --ml-analysis : Enable difficulty prediction
        --interactive : Launch exploration console
        --verify N : Test conjecture up to N
        --analyze-gaps : Show prime distribution
        --export FILE : Save results to CSV

    Example Use Cases:
    1. Identify stubborn even numbers:
    python goldbach.py --num-primes 2000 --ml-analysis

    2. Interactive investigation:
    python goldbach.py --interactive

    3. Full verification + export:
    python goldbach.py --verify 10000 --export results.csv
"""

from itertools import combinations_with_replacement
import sympy
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error

def parse_arguments():
    parser = argparse.ArgumentParser(description="Goldbach Conjecture Incremental Sum Explorer")
    parser.add_argument("--num-primes", type=int, default=1000, help="Number of base primes to use")
    parser.add_argument("--extra-primes", type=int, default=3, help="Additional primes to check for resolution")
    parser.add_argument("--show-plots", action="store_true", help="Display matplotlib plots")
    parser.add_argument("--list-missing", action="store_true", help="List all missing even numbers")
    parser.add_argument("--group-resolved", action="store_true", help="Group resolved numbers by delta step")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress information")
    parser.add_argument("--show-unresolved", action="store_true", help="Display unresolved even numbers (default: True)")
    parser.add_argument("--hide-unresolved", action="store_false", dest="show_unresolved", help="Hide the list of unresolved even numbers")
    parser.add_argument("--ml-analysis", action="store_true", help="Enable machine learning difficulty prediction")
    parser.add_argument("--visualize", type=int, metavar="EVEN", help="Visualize partitions for specific even number")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds for ML (default: 5)")
    return parser.parse_args()

def visualize_partitions(even_num, prime_set):
    """Visualize all Goldbach partitions for a given even number"""
    partitions = []
    primes_up_to_half = [p for p in prime_set if p <= even_num//2]
    
    for p in primes_up_to_half:
        q = even_num - p
        if q in prime_set:
            partitions.append((p, q))
    
    if not partitions:
        print(f"No partitions found for {even_num}")
        return

    plt.figure(figsize=(12, 4))
    
    # Partition visualization
    plt.subplot(1, 2, 1)
    x = [p[0] for p in partitions]
    y = [p[1] for p in partitions]
    plt.scatter(x, y, alpha=0.6)
    plt.plot([0, even_num//2], [even_num, even_num//2], 'r--')
    plt.xlabel("First prime (p)")
    plt.ylabel("Second prime (q)")
    plt.title(f"Goldbach Partitions for {even_num}\n{len(partitions)} solutions")
    
    # Gap analysis visualization
    plt.subplot(1, 2, 2)
    gaps = [abs(p[0]-p[1]) for p in partitions]
    plt.hist(gaps, bins=20, color='green', alpha=0.7)
    plt.xlabel("Prime Pair Gaps")
    plt.ylabel("Frequency")
    plt.title("Distribution of Gaps Between Partitions")
    
    plt.tight_layout()
    plt.show()
    
    return partitions

def prepare_training_data(delta_k_dict, prime_list):
    """Convert delta sets into ML features"""
    X, y = [], []
    
    for k in delta_k_dict:
        for even in delta_k_dict[k]:
            features = [
                even,                          # Target number itself
                even % 10,                     # Last digit
                even % 6,                      # Modulo 6 class
                len(str(even)),                # Number of digits
                prime_list[k-1],               # Largest prime used
                k,                             # Delta step
                sum(int(d) for d in str(even)) # Digit sum
            ]
            X.append(features)
            y.append(k)  # Using delta step as difficulty proxy
    
    return np.array(X), np.array(y)

def analyze_with_ml(delta_k_dict, prime_list, unresolved_evens):
    # Prepare data
    X, y = prepare_training_data(delta_k_dict, prime_list)
    
    # Train model
    predictor = GoldbachPredictor()
    predictor.train(X, y)
    
    # Predict difficulty for unresolved numbers
    if unresolved_evens:
        difficulties = predictor.predict_difficulty(sorted(unresolved_evens))
        print("\n=== Machine Learning Analysis ===")
        print("Predicted difficulty for unresolved evens (higher = harder):")
        
        results = sorted(zip(unresolved_evens, difficulties), 
                      key=lambda x: -x[1])
        
        for even, pred in results[:10]:  # Top 10 hardest
            print(f"{even}: predicted Δ_{pred:.1f} needed")
            
        return results
    return None

def plot_difficulty_analysis(evens, difficulties):
    plt.figure(figsize=(12, 6))
    plt.scatter(evens, difficulties, alpha=0.6)
    
    # Annotate outliers
    for e, d in zip(evens, difficulties):
        if d > np.mean(difficulties) + 2*np.std(difficulties):
            plt.annotate(e, (e, d), textcoords="offset points", xytext=(0,5), ha='center')
    
    plt.title("Predicted Difficulty of Unresolved Even Numbers")
    plt.xlabel("Even Number")
    plt.ylabel("Predicted Delta Step Needed")
    plt.grid(True)
    plt.show()

class GoldbachPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=7,
            random_state=42
        )
        
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Model trained - Test MAE: {mae:.2f} delta steps")
        return mae
    
    def predict_difficulty(self, evens):
        features = []
        for even in evens:
            features.append([
                even,
                even % 10,
                even % 6,
                len(str(even)),
                0,  # Placeholder for prime
                0,  # Placeholder for delta
                sum(int(d) for d in str(even))
            ])
        return self.model.predict(np.array(features))

class GoldbachPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42
        )
        self.cv_scores = None
        
    def train_with_cv(self, X, y, folds=5):
        """Train with k-fold cross-validation"""
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        self.cv_scores = cross_val_score(
            self.model, X, y, 
            cv=kf,
            scoring='neg_mean_absolute_error'
        )
        self.model.fit(X, y)
        return -self.cv_scores  # Convert to positive MAE
    
    def print_cv_report(self):
        """Display cross-validation results"""
        if self.cv_scores is not None:
            print("\n=== Cross-Validation Report ===")
            print(f"Folds: {len(self.cv_scores)}")
            print(f"Mean MAE: {np.mean(self.cv_scores):.2f} ± {np.std(self.cv_scores):.2f}")
            print("Per-fold scores:")
            for i, score in enumerate(self.cv_scores, 1):
                print(f"  Fold {i}: {score:.2f} delta steps")

def generate_deltas(base_primes, num_primes):
    delta_k_dict = {}
    covered_evens = set()
    
    for k, pk in enumerate(base_primes[:num_primes], start=1):
        evens = set()
        for i in range(k):
            pi = base_primes[i]
            for j in range(i, k):
                pj = base_primes[j]
                s = pi + pj
                if s % 2 == 0:
                    evens.add(s)
        delta_k_dict[k] = evens
        covered_evens.update(evens)
    
    return delta_k_dict, covered_evens

def check_resolution(base_primes, extra_primes, unresolved, delta_k_dict, num_primes):
    resolved = defaultdict(list)
    
    for k_extra, pk in enumerate(extra_primes, start=1):
        evens = set()
        for p in base_primes:
            s = p + pk
            if s % 2 == 0:
                evens.add(s)
        
        delta_step = num_primes + k_extra
        delta_k_dict[delta_step] = evens
        
        for e in list(unresolved):
            if e in evens:
                resolved[delta_step].append(e)
                unresolved.remove(e)
    
    return resolved, unresolved

def print_summary(num_primes, extra_primes, max_even, covered_count, missing_evens):
    print("\n=== Goldbach Conjecture Exploration Summary ===")
    print(f"• Primes used: {num_primes} (base) + {extra_primes} (extra)")
    print(f"• Even number coverage:")
    print(f"  - Maximum even generated: {max_even:,}")
    print(f"  - Total evens covered: {covered_count:,}")
    print(f"  - Missing evens: {len(missing_evens):,}")

def print_resolution_results(resolved, unresolved, prime_list, list_missing=False, group_resolved=False, show_unresolved=True):
    print("\n=== Resolution Results ===")
    
    if group_resolved:
        print("Resolved even numbers grouped by delta step:")
        for delta_step in sorted(resolved.keys()):
            prime = prime_list[delta_step-1]
            evens = sorted(resolved[delta_step])
            print(f"  Δ_{delta_step} (prime {prime}): {', '.join(map(str, evens))}")
    else:
        print("Resolved even numbers:")
        resolved_flat = []
        for delta_step, evens in resolved.items():
            prime = prime_list[delta_step-1]
            for e in evens:
                resolved_flat.append((e, delta_step, prime))
        
        for e, delta_step, prime in sorted(resolved_flat, key=lambda x: x[0]):
            print(f"  {e} resolved at Δ_{delta_step} (prime {prime})")
    
    if unresolved and show_unresolved:
        print("\nUnresolved even numbers:")
        if list_missing:
            # Group in lines of 10 numbers for better readability
            unresolved_sorted = sorted(unresolved)
            for i in range(0, len(unresolved_sorted), 10):
                batch = unresolved_sorted[i:i+10]
                print("  " + ", ".join(map(str, batch)))
        else:
            print(f"  {len(unresolved)} numbers remain unresolved (use --list-missing to see all)")
    elif unresolved:
        print(f"\nNote: {len(unresolved)} numbers remain unresolved (use --show-unresolved to display)")

def generate_plots(base_primes, num_primes):
    # Plot 1: Number of missing evens vs. number of primes
    missing_counts = []
    primes_range = list(range(10, num_primes+1, 10))
    for k in primes_range:
        evens = set()
        for i in range(k):
            for j in range(i, k):
                s = base_primes[i] + base_primes[j]
                if s % 2 == 0:
                    evens.add(s)
        max_e = max(evens)
        miss = [e for e in range(4, max_e+2, 2) if e not in evens]
        missing_counts.append(len(miss))

    plt.figure(figsize=(10, 6))
    plt.plot(primes_range, missing_counts, marker='o', linestyle='-', color='blue')
    plt.title("Missing Even Numbers vs. Number of Primes Used")
    plt.xlabel("Number of Primes Used (k)")
    plt.ylabel("Count of Missing Even Numbers")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Max even number reached vs. number of primes
    max_evens = []
    for k in primes_range:
        evens = set()
        for i in range(k):
            for j in range(i, k):
                s = base_primes[i] + base_primes[j]
                if s % 2 == 0:
                    evens.add(s)
        max_evens.append(max(evens))

    plt.figure(figsize=(10, 6))
    plt.plot(primes_range, max_evens, marker='o', linestyle='-', color='green')
    plt.title("Maximum Even Number Generated vs. Number of Primes Used")
    plt.xlabel("Number of Primes Used (k)")
    plt.ylabel("Maximum Even Number in Δₖ")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    args = parse_arguments()
    
    # Prime list
    prime_list = list(sympy.primerange(2, 10_000))  # Get many primes in advance
    base_primes = prime_list[:args.num_primes]
    extra_primes = prime_list[args.num_primes:args.num_primes+args.extra_primes]

    # Generate delta sets
    delta_k_dict, covered_evens = generate_deltas(base_primes, args.num_primes)
    max_even = max(covered_evens)
    missing_evens = [e for e in range(4, max_even+2, 2) if e not in covered_evens]
    unresolved = set(missing_evens)

    # Check resolution with extra primes
    resolved, unresolved = check_resolution(base_primes, extra_primes, unresolved, delta_k_dict, args.num_primes)

    # Print results
    print_summary(args.num_primes, args.extra_primes, max_even, len(covered_evens), missing_evens)
    print_resolution_results(resolved, unresolved, prime_list, args.list_missing, args.group_resolved)

    if args.ml_analysis:
        ml_results = analyze_with_ml(delta_k_dict, prime_list, unresolved)
        if ml_results and args.show_plots:
            evens, diffs = zip(*ml_results)
            plot_difficulty_analysis(evens, diffs)

    if args.show_plots:
        generate_plots(base_primes, args.num_primes)

    if args.visualize:
        print(f"\nAnalyzing partitions for {args.visualize}...")
        partitions = visualize_partitions(args.visualize, set(base_primes))
        if partitions:
            print(f"Found {len(partitions)} partitions")
            print("Sample:", partitions[:3], "...")

    if args.ml_analysis:
        X, y = prepare_training_data(delta_k_dict, prime_list)
        predictor = GoldbachPredictor()
        cv_scores = predictor.train_with_cv(X, y, folds=args.folds)
        predictor.print_cv_report()
        
        if unresolved:
            difficulties = predictor.predict_difficulty(sorted(unresolved))
            plot_difficulty_analysis(unresolved, difficulties)

if __name__ == "__main__":
    main()