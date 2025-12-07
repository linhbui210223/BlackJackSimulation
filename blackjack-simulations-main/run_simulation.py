import simulation
import sys

def main():
    print("=" * 60)
    print("BLACKJACK SIMULATION")
    print("=" * 60)
    
    # 1. Simulate dealer bust probabilities
    print("\n1. Dealer Bust Probabilities (Monte Carlo)")
    print("-" * 60)
    print("Running 1,000,000 simulations... ", end='', flush=True)
    dealer_bust_results = simulation.simulate_dealer_bust()
    print("DONE!")
    for total, probability in sorted(dealer_bust_results.items()):
        if total == -1:
            print(f"Bust: {probability:.4f}")
        elif total == 22:
            print(f"Blackjack: {probability:.4f}")
        else:
            print(f"Total {total}: {probability:.4f}")
    
    # 2. Simulate dealer results by upcard
    print("\n2. Dealer Results by Upcard (Monte Carlo)")
    print("-" * 60)
    print("Running 1,000,000 simulations... ", end='', flush=True)
    upcard_results = simulation.simulate_dealer_results_upcard()
    print("DONE!")
    for upcard in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]:
        print(f"\nDealer showing {upcard}:")
        for total, prob in sorted(upcard_results[upcard].items()):
            if total == -1:
                print(f"  Bust: {prob:.4f}")
            elif total == 22:
                print(f"  Blackjack: {prob:.4f}")
            else:
                print(f"  Total {total}: {prob:.4f}")
    
    # 3. Test basic strategy (stand on 17)
    print("\n3. Player Strategy 1 (Stand on 17) Results")
    print("-" * 60)
    print("Running 1,000,000 simulations... ", end='', flush=True)
    ps1_results, expectation = simulation.simulate_ps1()
    print("DONE!")
    for outcome, probability in ps1_results.items():
        print(f"{outcome.capitalize()}: {probability:.4f}")
    print(f"Expected value: {expectation:.4f}")
    
    # # 4. Generate dealer upcard heatmap (Markov Chain)
    # print("\n4. Generating Dealer Upcard Heatmap (Markov Chain)...")
    # print("-" * 60)
    # print("Computing transition matrix... ", end='', flush=True)
    # simulation.generate_dealer_upcard_heatmap()
    # print("Heatmap displayed (close window to continue)")
    
    # # 5. Generate player stay expectations heatmap
    # print("\n5. Generating Player Stay Expectations Heatmap...")
    # print("-" * 60)
    # print("Computing expectations... ", end='', flush=True)
    # simulation.generate_player_stay_expectations_heatmap()
    # print("Heatmap displayed")
    
    # print("\n" + "=" * 60)
    # print("SIMULATION COMPLETE")
    # print("=" * 60)

if __name__ == "__main__":
    main()