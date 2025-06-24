#include <iostream>
#include <vector>
#include <bitset>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdint>

void test_hypothesis_original_behavior(int k_max) {
    for (int k = 2; k <= k_max; ++k) {
        const uint64_t lower = 1ULL << k;
        const uint64_t upper = (1ULL << (k + 1)) - 1;
        
        // Generate primes up to upper
        std::vector<bool> sieve(upper + 1, true);
        sieve[0] = sieve[1] = false;
        for (uint64_t p = 2; p * p <= upper; ++p) {
            if (sieve[p]) {
                for (uint64_t multiple = p * p; multiple <= upper; multiple += p) {
                    sieve[multiple] = false;
                }
            }
        }

        // Collect primes <= 2^(k-1)
        const uint64_t half_limit = 1ULL << (k - 1);
        std::vector<uint64_t> primes_half;
        for (uint64_t p = 2; p <= half_limit; ++p) {
            if (sieve[p]) primes_half.push_back(p);
        }

        // Test even numbers
        std::vector<uint64_t> failed_evens;
        for (uint64_t N = lower; N <= upper; N += 2) {
            bool found = false;
            for (uint64_t p : primes_half) {
                if (p > N) break;  // Safety check
                const uint64_t q = N - p;
                if (q >= 2 && q <= upper && sieve[q]) {
                    found = true;
                    break;
                }
            }
            if (!found) failed_evens.push_back(N);
        }

        // Original-style output
        std::cout << "\nInterval [2^" << k << ", 2^" << (k + 1) << "):\n";
        std::cout << "Primes <= 2^" << (k - 1) << ": " << primes_half.size() << " primes\n";
        if (failed_evens.empty()) {
            std::cout << "All evens satisfy the hypothesis!\n";
        } else {
            std::cout << "Counterexamples: ";
            for (auto N : failed_evens) std::cout << N << " ";
            std::cout << "\n";
        }
    }
}

int main() {
    int k_max;
    std::cout << "Enter maximum k (e.g., 20): ";
    std::cin >> k_max;

    auto start = std::chrono::high_resolution_clock::now();
    test_hypothesis_original_behavior(k_max);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nTime taken: " << elapsed.count() << " seconds\n";
    return 0;
}