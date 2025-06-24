#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <algorithm>

// Memory-safe segmented sieve
std::vector<uint64_t> get_primes_segment(uint64_t start, uint64_t end, 
                                        const std::vector<uint64_t>& base_primes) {
    std::vector<bool> segment(end - start + 1, true);
    
    for (uint64_t p : base_primes) {
        uint64_t first_multiple = std::max(p * p, ((start + p - 1) / p) * p);
        for (uint64_t m = first_multiple; m <= end; m += p) {
            segment[m - start] = false;
        }
    }
    
    std::vector<uint64_t> primes;
    for (uint64_t i = start; i <= end; ++i) {
        if (segment[i - start] && i >= 2) {
            primes.push_back(i);
        }
    }
    return primes;
}

void test_goldbach_with_checkpoints(int k_min, int k_max) {
    const uint64_t SEGMENT_SIZE = 1 << 26; // 67MB segments (safe for 16GB RAM)
    
    // Load checkpoint
    int current_k = k_min;
    std::ifstream checkpoint_in("goldbach_test_tuple_segmented_sieve_checkpoint.txt");
    if (checkpoint_in) {
        checkpoint_in >> current_k;
        std::cout << "Resuming from k=" << current_k << "\n";
    }
    
    for (int k = current_k; k <= k_max; ++k) {
        const uint64_t LOWER = 1ULL << k;
        const uint64_t UPPER = (1ULL << (k + 1)) - 1;
        const uint64_t HALF_LIMIT = 1ULL << (k - 1);
        const uint64_t SQRT_UPPER = static_cast<uint64_t>(std::sqrt(UPPER));
        
        std::cout << "\n=== Testing k=" << k << " [" << LOWER << ", " << UPPER << "] ===\n";
        
        // Generate small primes up to sqrt(UPPER)
        std::vector<uint64_t> base_primes = get_primes_segment(2, SQRT_UPPER, {});
        
        // Generate primes up to HALF_LIMIT in segments
        std::vector<uint64_t> primes_half;
        for (uint64_t low = 2; low <= HALF_LIMIT; low += SEGMENT_SIZE) {
            uint64_t high = std::min(low + SEGMENT_SIZE - 1, HALF_LIMIT);
            auto chunk = get_primes_segment(low, high, base_primes);
            primes_half.insert(primes_half.end(), chunk.begin(), chunk.end());
            std::cout << "Generated primes up to " << high << " (" 
                      << primes_half.size() << " primes so far)\r" << std::flush;
        }
        
        // Test even numbers in batches
        std::vector<uint64_t> failed_evens;
        uint64_t evens_tested = 0;
        uint64_t total_evens = (UPPER - LOWER) / 2 + 1;
        
        for (uint64_t N = LOWER; N <= UPPER; N += 2) {
            bool found_partition = false;
            
            for (uint64_t p : primes_half) {
                if (p > N / 2) break;
                
                uint64_t q = N - p;
                bool q_is_prime = true;
                
                if (q <= SQRT_UPPER) {
                    if (!std::binary_search(base_primes.begin(), base_primes.end(), q)) {
                        q_is_prime = false;
                    }
                } else {
                    for (uint64_t bp : base_primes) {
                        if (bp * bp > q) break;
                        if (q % bp == 0) {
                            q_is_prime = false;
                            break;
                        }
                    }
                }
                
                if (q_is_prime) {
                    found_partition = true;
                    break;
                }
            }
            
            if (!found_partition) {
                failed_evens.push_back(N);
            }
            
            // Progress reporting
            if (++evens_tested % 10000 == 0) {
                std::cout << "Progress: " << evens_tested << "/" << total_evens 
                          << " (" << (evens_tested * 100 / total_evens) << "%)"
                          << " | Current N=" << N << "\r" << std::flush;
            }
        }
        
        // Save checkpoint
        std::ofstream checkpoint_out("goldbach_test_tuple_segmented_sieve_checkpoint.txt");
        checkpoint_out << k;
        
        // Report results
        std::cout << "\nResults for k=" << k << ":\n";
        std::cout << "- Primes ≤ " << HALF_LIMIT << ": " << primes_half.size() << "\n";
        if (failed_evens.empty()) {
            std::cout << "- All evens validated!\n";
        } else {
            std::cout << "- Counterexamples found: " << failed_evens.size() << "\n";
            for (uint64_t N : failed_evens) std::cout << "  " << N << "\n";
            break; // Stop if counterexamples found
        }
    }
}

int main() {
    std::cout << "Goldbach Conjecture Tester (Segmented Sieve)\n";
    std::cout << "-------------------------------------------\n";
    
    // Safe testing range for 16GB RAM
    test_goldbach_with_checkpoints(33, 36);
    
    return 0;
}