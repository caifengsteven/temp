from tradingrrl import LayeredRRL
import matplotlib.pyplot as plt
import numpy as np

# Create a simple test for the strategy
def test_strategy():
    # Initialize the RRL model with default parameters
    rrl = LayeredRRL(save_path='test_run')
    
    # Set some parameters
    rrl.stop_loss = 0.3
    rrl.rho = 0.1
    rrl.sigma = 0.2
    rrl.n_epoch = 100  # Reduced for faster testing
    
    # Train and test the model
    rrl.train()
    
    # Plot the results
    if len(rrl.all_W) > 0:
        print(f"Strategy performance:")
        print(f"Final profit: {rrl.all_W[-1]}")
        print(f"Max profit: {max(rrl.all_W)}")
        print(f"Min profit: {min(rrl.all_W)}")
        
        # Calculate some statistics
        returns = np.diff(rrl.all_W)
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        print(f"Win rate: {win_rate:.2%}")
        
        # Plot cumulative profit
        plt.figure(figsize=(12, 6))
        plt.plot(rrl.all_W)
        plt.title('Cumulative Profit')
        plt.xlabel('Time')
        plt.ylabel('Profit')
        plt.grid(True)
        plt.savefig('profit_curve.png')
        print("Saved profit curve to profit_curve.png")
    else:
        print("No trading results were generated.")

if __name__ == "__main__":
    test_strategy()
