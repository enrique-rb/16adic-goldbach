#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <cstring> // For strcmp()

// Segmented sieve function (unchanged)
std::vector<uint64_t> get_primes_segment(uint64_t start, uint64_t end, const std::vector<uint64_t>& base_primes) {
    std::vector<bool> segment(end - start + 1, true);
    for (uint64_t p : base_primes) {
        uint64_t first_multiple = std::max(p * p, ((start + p - 1) / p) * p);
        for (uint64_t m = first_multiple; m <= end; m += p) {
            segment[m - start] = false;
        }
    }
    std::vector<uint64_t> primes;
    for (uint64_t i = start; i <= end; ++i) {
        if (segment[i - start] && i >= 2) primes.push_back(i);
    }
    return primes;
}

// Modified to accept output_filename
void test_goldbach_range(uint64_t start, uint64_t end, const std::string& output_filename) {
    std::ofstream out(output_filename);
    out << "n,residue_mod_16,partition_count,example_partition\n";

    uint64_t sqrt_end = static_cast<uint64_t>(std::sqrt(end));
    std::vector<uint64_t> base_primes = get_primes_segment(2, sqrt_end, {});

    // Generate primes up to end/2 (unchanged)
    std::vector<uint64_t> primes_half = get_primes_segment(2, end / 2, base_primes);

    for (uint64_t n = start; n <= end; n += 2) {
        uint64_t residue = n % 16;
        uint64_t partition_count = 0;
        std::string example_partition;

        for (uint64_t p : primes_half) {
            if (p > n / 2) break;
            uint64_t q = n - p;
            if (std::binary_search(primes_half.begin(), primes_half.end(), q)) {
                partition_count++;
                if (example_partition.empty()) {
                    example_partition = std::to_string(n) + " = " + std::to_string(p) + " + " + std::to_string(q);
                }
            }
        }

        out << n << "," << residue << "," << partition_count << ",\"" << example_partition << "\"\n";
        
        // Progress report
        if (n % 10000 == 0) {
            std::cout << "Progress: " << n << "/" << end << " (" 
                      << (n - start) * 100 / (end - start) << "%)\r" << std::flush;
        }
    }
    std::cout << "\nDone! Results saved to " << output_filename << "\n";
}

int main(int argc, char* argv[]) {
    uint64_t start = 1048576;  // Default: 2^20
    uint64_t end = 2097152;    // Default: 2^21
    std::string output_file = "validation_subset.csv"; // Default

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--start") == 0 && i + 1 < argc) {
            start = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "--end") == 0 && i + 1 < argc) {
            end = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i]; // Critical: This must be used!
        }
    }

    std::cout << "Goldbach Conjecture Tester\n";
    std::cout << "Testing range [" << start << ", " << end << "]\n";
    std::cout << "Output file: " << output_file << "\n";  // Must match actual output
    std::cout << "-------------------------------------------\n";

    test_goldbach_range(start, end, output_file);  // Must pass output_file!
    return 0;
}
