#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <algorithm>

// Miller-Rabin primality test (deterministic for n < 2^64)
bool is_prime(uint64_t n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0) return false;

    // Write n-1 as d*2^s
    uint64_t d = n - 1;
    uint64_t s = 0;
    while (d % 2 == 0) {
        d /= 2;
        s++;
    }

    // Bases that cover all n < 2^64
    const uint64_t bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    
    for (uint64_t a : bases) {
        if (a >= n) continue;
        
        uint64_t x = 1;
        // Compute a^d mod n using modular exponentiation
        uint64_t p = d;
        uint64_t a_mod = a;
        while (p > 0) {
            if (p % 2 == 1) x = (__uint128_t(x) * a_mod) % n;
            a_mod = (__uint128_t(a_mod) * a_mod) % n;
            p /= 2;
        }
        
        if (x == 1 || x == n - 1) continue;
        
        bool composite = true;
        for (uint64_t r = 1; r < s; r++) {
            x = (__uint128_t(x) * x) % n;
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }
    return true;
}

void test_goldbach_probabilistic(int k_min, int k_max, bool save_primes = false) {
    for (int k = k_min; k <= k_max; ++k) {
        const uint64_t lower = 1ULL << k;
        const uint64_t upper = (1ULL << (k + 1)) - 1;
        const uint64_t half_limit = 1ULL << (k - 1);

        std::cout << "\n=== Testing k=" << k << " [" << lower << ", " << upper << "] ===\n";
        std::cout << "Generating primes <= " << half_limit << "...\n";

        // Generate primes up to half_limit
        std::vector<uint64_t> primes_half;
        for (uint64_t p = 2; p <= half_limit; ++p) {
            if (is_prime(p)) {
                primes_half.push_back(p);
            }
            if (p % 10000000 == 0) {
                std::cout << "Progress: " << p << "/" << half_limit 
                          << " (" << primes_half.size() << " primes)\r" << std::flush;
            }
        }

        // Save primes to file if requested
        if (save_primes) {
            std::ofstream prime_file("primes_" + std::to_string(k) + ".txt");
            for (uint64_t p : primes_half) prime_file << p << "\n";
        }

        std::cout << "\nTesting even numbers...\n";
        std::vector<uint64_t> failed_evens;
        uint64_t evens_tested = 0;
        uint64_t total_evens = (upper - lower) / 2 + 1;

        for (uint64_t N = lower; N <= upper; N += 2) {
            bool found = false;
            for (uint64_t p : primes_half) {
                if (p > N / 2) break;
                if (is_prime(N - p)) {
                    found = true;
                    break;
                }
            }

            if (!found) failed_evens.push_back(N);

            // Progress reporting
            if (++evens_tested % 1000 == 0) {
                std::cout << "Progress: " << evens_tested << "/" << total_evens 
                          << " (" << (evens_tested * 100 / total_evens) << "%)"
                          << " | Current N=" << N << "\r" << std::flush;
            }
        }

        // Save results
        std::ofstream checkpoint("goldbach_checkpoint_k" + std::to_string(k) + ".txt");
        checkpoint << "k=" << k << " tested up to N=" << upper << "\n";
        checkpoint << "Primes <= " << half_limit << ": " << primes_half.size() << "\n";
        checkpoint << "Failed evens: " << failed_evens.size() << "\n";
        for (uint64_t N : failed_evens) checkpoint << N << "\n";

        std::cout << "\nResults for k=" << k << ":\n";
        std::cout << "- Primes <= " << half_limit << ": " << primes_half.size() << "\n";
        if (failed_evens.empty()) {
            std::cout << "- All evens validated!\n";
        } else {
            std::cout << "- Counterexamples found: " << failed_evens.size() << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Goldbach Conjecture Probabilistic Tester - using OpenMP\n";
    std::cout << "---------------------------------------\n";

    int k_min = (argc > 1) ? std::stoi(argv[1]) : 34;
    int k_max = (argc > 2) ? std::stoi(argv[2]) : 40;
    test_goldbach_probabilistic(k_min, k_max, true); // Enable prime saving
 
    return 0;
}