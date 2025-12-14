"""
Supermarket Checkout System Performance Analysis
Performs queuing analysis and simulations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

class SupermarketCheckoutSimulation:
    """
    Simulates supermarket checkout queue system
    """
    
    def __init__(self, df, num_cashiers=3):
        """
        Initialize simulation with data and parameters
        
        Args:
            df (pd.DataFrame): Customer data
            num_cashiers (int): Number of checkout counters
        """
        self.df = df.copy()
        self.num_cashiers = num_cashiers
        self.results = {}
        
    def calculate_basic_metrics(self):
        """Calculate arrival and service rate metrics"""
        # Convert arrival times to datetime
        self.df['Arrival_Datetime'] = pd.to_datetime(self.df['Arrival_Time'])
        
        # Sort by arrival time
        self.df = self.df.sort_values('Arrival_Datetime')
        
        # Calculate inter-arrival times
        self.df['Inter_Arrival_Seconds'] = self.df['Arrival_Datetime'].diff().dt.total_seconds()
        
        # Calculate metrics
        avg_arrival_rate = 1 / self.df['Inter_Arrival_Seconds'].iloc[1:].mean()
        avg_service_rate = 1 / self.df['Service_Time_Seconds'].mean()
        
        return avg_arrival_rate, avg_service_rate
    
    def simulate_queue(self, method='analytical'):
        """
        Simulate queue using different methods
        
        Args:
            method (str): 'analytical' for queuing theory or 'discrete' for discrete event
        """
        if method == 'analytical':
            return self._analytical_simulation()
        else:
            return self._discrete_event_simulation()
    
    def _analytical_simulation(self):
        """Use M/M/c queuing theory formulas"""
        λ, μ = self.calculate_basic_metrics()  # Arrival and service rates
        
        ρ = λ / (self.num_cashiers * μ)  # Utilization factor
        
        if ρ >= 1:
            print("Warning: System is unstable (ρ >= 1)")
            return None
        
        # Probability of zero customers in system
        sum_term = 0
        for n in range(self.num_cashiers):
            sum_term += (self.num_cashiers * ρ) ** n / np.math.factorial(n)
        
        P0 = 1 / (sum_term + 
                  (self.num_cashiers * ρ) ** self.num_cashiers / 
                  (np.math.factorial(self.num_cashiers) * (1 - ρ)))
        
        # Average number of customers in queue
        Lq = (P0 * (λ/μ) ** self.num_cashiers * ρ) / \
             (np.math.factorial(self.num_cashiers) * (1 - ρ) ** 2)
        
        # Average wait time in queue
        Wq = Lq / λ
        
        # Average time in system
        W = Wq + 1/μ
        
        # Average number in system
        L = λ * W
        
        return {
            'Utilization (ρ)': ρ,
            'Avg Customers in Queue (Lq)': Lq,
            'Avg Wait Time in Queue (Wq mins)': Wq * 60,
            'Avg Time in System (W mins)': W * 60,
            'Avg Customers in System (L)': L
        }
    
    def _discrete_event_simulation(self):
        """Discrete event simulation of queue"""
        arrivals = self.df['Arrival_Datetime'].values
        service_times = self.df['Service_Time_Seconds'].values
        
        # Initialize - use pandas Timestamp for consistency
        cashier_free_times = [pd.Timestamp.min] * self.num_cashiers
        wait_times = []
        queue_lengths = []
        timeline = []
        
        for i, (arrival, service) in enumerate(zip(arrivals, service_times)):
            # Find earliest available cashier
            earliest_free = min(cashier_free_times)
            cashier_index = cashier_free_times.index(earliest_free)
            
            # Calculate wait time
            if earliest_free > arrival:
                wait_time = (earliest_free - arrival).total_seconds()
                start_service = earliest_free
            else:
                wait_time = 0
                start_service = arrival
            
            # Update cashier free time
            end_service = start_service + pd.Timedelta(seconds=service)
            cashier_free_times[cashier_index] = end_service
            
            # Record metrics
            wait_times.append(wait_time)
            
            # Calculate queue length at this arrival (simplified)
            queue_length = sum(1 for free in cashier_free_times 
                             if free > arrival)
            queue_lengths.append(queue_length)
            timeline.append(arrival)
        
        return {
            'Avg Wait Time (seconds)': np.mean(wait_times),
            'Max Wait Time (seconds)': np.max(wait_times),
            'Avg Queue Length': np.mean(queue_lengths),
            'Max Queue Length': np.max(queue_lengths),
            'Wait Times': wait_times,
            'Queue Lengths': queue_lengths,
            'Timeline': timeline
        }
    
    def analyze_scenarios(self, cashier_options=[2, 3, 4, 5]):
        """Analyze performance under different cashier configurations"""
        scenario_results = []
        
        for cashiers in cashier_options:
            self.num_cashiers = cashiers
            analytical = self._analytical_simulation()
            
            if analytical:
                scenario_results.append({
                    'Cashiers': cashiers,
                    'Utilization': analytical['Utilization (ρ)'],
                    'Avg Wait Time (mins)': analytical['Avg Wait Time in Queue (Wq mins)'],
                    'Avg Customers in Queue': analytical['Avg Customers in Queue (Lq)']
                })
        
        return pd.DataFrame(scenario_results)

def create_visualizations(df, simulation_results):
    """Create comprehensive visualizations"""
    
    # 1. Service Time Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['Service_Time_Seconds']/60, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Service Time (Minutes)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Customer Service Times')
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/service_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Items vs Service Time Scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Number_of_Items'], df['Service_Time_Seconds']/60, 
                alpha=0.6, s=50)
    plt.xlabel('Number of Items')
    plt.ylabel('Service Time (Minutes)')
    plt.title('Relationship Between Items and Service Time')
    
    # Add trend line
    z = np.polyfit(df['Number_of_Items'], df['Service_Time_Seconds']/60, 1)
    p = np.poly1d(z)
    plt.plot(df['Number_of_Items'], p(df['Number_of_Items']), 
             "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/items_vs_service.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Daily Customer Arrival Pattern
    plt.figure(figsize=(12, 6))
    daily_counts = df.groupby('Day').size()
    plt.bar(daily_counts.index, daily_counts.values, color='skyblue', edgecolor='black')
    plt.xlabel('Day')
    plt.ylabel('Number of Customers')
    plt.title('Daily Customer Volume During Peak Hours (5-6 PM)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(daily_counts.values):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('visualizations/daily_customer_volume.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    
    print("="*60)
    print("SUPERMARKET CHECKOUT PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Step 1: Load or generate data
    try:
        df = pd.read_csv('supermarket_checkout_data.csv')
        print("✓ Data loaded from file")
    except:
        print("Generating new dataset...")
        # Import generator function
        from supermarket_data_generator import generate_supermarket_data
        df = generate_supermarket_data()
        df.to_csv('supermarket_checkout_data.csv', index=False)
    
    # Step 2: Initial analysis
    print("\n" + "-"*60)
    print("DATASET SUMMARY")
    print("-"*60)
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Day'].min()} to {df['Day'].max()}")
    print(f"Average items per customer: {df['Number_of_Items'].mean():.1f}")
    print(f"Average service time: {df['Service_Time_Seconds'].mean()/60:.2f} minutes")
    print(f"Total service hours required: {df['Service_Time_Seconds'].sum()/3600:.2f} hours")
    
    # Step 3: Create visualizations directory
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Step 4: Run simulation
    print("\n" + "-"*60)
    print("QUEUE SIMULATION RESULTS")
    print("-"*60)
    
    sim = SupermarketCheckoutSimulation(df, num_cashiers=3)
    
    # Analytical results
    analytical_results = sim.simulate_queue(method='analytical')
    if analytical_results:
        print("\nAnalytical Results (M/M/c Queue Theory):")
        for key, value in analytical_results.items():
            print(f"  {key}: {value:.2f}")
    
    # Discrete event simulation
    discrete_results = sim.simulate_queue(method='discrete')
    if discrete_results:
        print(f"\nDiscrete Event Simulation Results:")
        print(f"  Average Wait Time: {discrete_results['Avg Wait Time (seconds)']/60:.2f} minutes")
        print(f"  Maximum Wait Time: {discrete_results['Max Wait Time (seconds)']/60:.2f} minutes")
        print(f"  Average Queue Length: {discrete_results['Avg Queue Length']:.2f} customers")
    
    # Step 5: Scenario analysis
    print("\n" + "-"*60)
    print("SCENARIO ANALYSIS - VARYING NUMBER OF CASHIERS")
    print("-"*60)
    
    scenario_df = sim.analyze_scenarios([2, 3, 4, 5])
    print(scenario_df.to_string(index=False))
    
    # Step 6: Create scenario comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Wait Time vs Cashiers
    plt.subplot(2, 2, 1)
    plt.plot(scenario_df['Cashiers'], scenario_df['Avg Wait Time (mins)'], 
             marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Cashiers')
    plt.ylabel('Average Wait Time (Minutes)')
    plt.title('Wait Time vs Number of Cashiers')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Utilization vs Cashiers
    plt.subplot(2, 2, 2)
    plt.plot(scenario_df['Cashiers'], scenario_df['Utilization']*100, 
             marker='s', linewidth=2, markersize=8, color='green')
    plt.xlabel('Number of Cashiers')
    plt.ylabel('Utilization (%)')
    plt.title('Cashier Utilization')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Queue Length vs Cashiers
    plt.subplot(2, 2, 3)
    plt.plot(scenario_df['Cashiers'], scenario_df['Avg Customers in Queue'], 
             marker='^', linewidth=2, markersize=8, color='red')
    plt.xlabel('Number of Cashiers')
    plt.ylabel('Average Queue Length')
    plt.title('Queue Length vs Number of Cashiers')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Cost-Benefit Analysis (simplified)
    plt.subplot(2, 2, 4)
    
    # Simple cost model: Cost = Labor cost - Customer satisfaction benefit
    labor_cost = scenario_df['Cashiers'] * 20  # $20 per cashier per hour
    wait_time_cost = scenario_df['Avg Wait Time (mins)'] * 10  # $10 per minute of wait time
    total_cost = labor_cost + wait_time_cost
    
    plt.plot(scenario_df['Cashiers'], total_cost, 
             marker='d', linewidth=2, markersize=8, color='purple')
    plt.xlabel('Number of Cashiers')
    plt.ylabel('Estimated Cost Index')
    plt.title('Cost-Benefit Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/scenario_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 7: Create visualizations
    create_visualizations(df, analytical_results)
    
    # Step 8: Save results
    scenario_df.to_csv('simulation_results.csv', index=False)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Outputs generated:")
    print("  1. supermarket_checkout_data.csv - Raw data")
    print("  2. simulation_results.csv - Scenario analysis")
    print("  3. visualizations/ - All graphs and charts")
    print("\nRecommendation based on analysis:")
    
    optimal_cashiers = scenario_df.loc[scenario_df['Avg Wait Time (mins)'] < 3, 'Cashiers'].min()
    if pd.isna(optimal_cashiers):
        optimal_cashiers = scenario_df['Cashiers'].iloc[-1]
    
    print(f"  → Optimal number of cashiers: {optimal_cashiers}")
    print(f"  → Expected wait time: {scenario_df.loc[scenario_df['Cashiers'] == optimal_cashiers, 'Avg Wait Time (mins)'].values[0]:.1f} minutes")
    print(f"  → Cashier utilization: {scenario_df.loc[scenario_df['Cashiers'] == optimal_cashiers, 'Utilization'].values[0]*100:.1f}%")

if __name__ == "__main__":
    main()