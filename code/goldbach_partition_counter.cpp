// Filename: goldbach_partition_counter.cpp (Corrected Version)
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <cstring>

// This version calculates correct partition counts for a given range.
void calculate_partitions_in_range(uint64_t start, uint64_t end, const std::string& output_filename) {
    if (start % 2 != 0) start++; // Ensure start is even

    std::cout << "Step 1: Generating a full prime sieve up to " << end << "...\n";
    std::vector<bool> is_prime_sieve(end + 1, true);
    is_prime_sieve[0] = is_prime_sieve[1] = false;
    for (uint64_t p = 2; p * p <= end; ++p) {
        if (is_prime_sieve[p]) {
            for (uint64_t m = p * p; m <= end; m += p) {
                is_prime_sieve[m] = false;
            }
        }
    }
    std::cout << "Sieve generation complete.\n";

    std::cout << "Step 2: Calculating partitions for each even number...\n";
    std::ofstream out(output_filename);
    out << "n,residue_mod_16,partition_count,example_partition\n";

    for (uint64_t n = start; n <= end; n += 2) {
        uint64_t residue = n % 16;
        uint64_t partition_count = 0;
        std::string example_partition;

        for (uint64_t p = 2; p <= n / 2; ++p) {
            if (is_prime_sieve[p]) {
                uint64_t q = n - p;
                if (is_prime_sieve[q]) {
                    partition_count++;
                    if (example_partition.empty()) {
                        example_partition = std::to_string(n) + " = " + std::to_string(p) + " + " + std::to_string(q);
                    }
                }
            }
        }

        out << n << "," << residue << "," << partition_count << ",\"" << example_partition << "\"\n";
        
        // Progress report
        if ((n - start) % 10000 == 0) {
            std::cout << "Progress: " << n << "/" << end << " (" 
                      << (n - start) * 100 / (end - start) << "%)\r" << std::flush;
        }
    }
    std::cout << "\nDone! Results saved to " << output_filename << "\n";
}

int main(int argc, char* argv[]) {
    uint64_t start = 1048576;  // Default: 2^20
    uint64_t end = 2097152;    // Default: 2^21
    std::string output_file = "goldbach_partitions_1M-2M.csv"; // Default

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--start") == 0 && i + 1 < argc) {
            start = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "--end") == 0 && i + 1 < argc) {
            end = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    std::cout << "Goldbach Partition Counter\n";
    std::cout << "Calculating partitions for [" << start << ", " << end << "]\n";
    std::cout << "Output will be saved to: " << output_file << "\n";
    std::cout << "-------------------------------------------\n";

    calculate_partitions_in_range(start, end, output_file);
    return 0;
}
