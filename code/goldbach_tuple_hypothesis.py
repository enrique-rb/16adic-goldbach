import math

def sieve_of_eratosthenes(limit):
    """Generate primes up to 'limit' using Sieve of Eratosthenes."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for num in range(2, int(math.sqrt(limit)) + 1):
        if sieve[num]:
            sieve[num*num : limit+1 : num] = [False] * len(sieve[num*num : limit+1 : num])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def test_binary_interval_hypothesis(k_max):
    """Test the hypothesis for all binary intervals up to k=k_max."""
    for k in range(2, k_max + 1):
        lower = 2 ** k
        upper = 2 ** (k + 1)
        primes = sieve_of_eratosthenes(upper)
        primes_up_to_half = [p for p in primes if p <= 2 ** (k - 1)]
        
        failed_evens = []
        for N in range(lower, upper + 1, 2):  # Even numbers only
            found = False
            for p in primes_up_to_half:
                q = N - p
                if q in primes:  # Check if q is prime (using set for O(1) lookup)
                    found = True
                    break
            if not found:
                failed_evens.append(N)
        
        print(f"\nResults for [2^{k}, 2^{k+1}):")
        #print(f"Primes ≤ 2^{k-1}: {primes_up_to_half}")
        if not failed_evens:
            print(f"All evens in [2^{k}, 2^{k+1}] satisfy the hypothesis!")
        else:
            print(f"Counterexamples found: {failed_evens}")

# Test up to k=6 (i.e., N ∈ [2^6, 2^7] = [64, 128])
test_binary_interval_hypothesis(k_max=18)

print("End of calculations.")
print("")