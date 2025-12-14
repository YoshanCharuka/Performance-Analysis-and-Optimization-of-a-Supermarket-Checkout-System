"""
Supermarket Checkout Data Generator
Generates realistic data for 4 days of peak hours (5-6 PM)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_supermarket_data():
    """
    Generates synthetic data for supermarket checkout system.
    
    Returns:
        pandas.DataFrame: Contains customer arrival and service data
    """
    customers_data = []
    customer_id = 1
    
    # Define 4 days of peak hours (5:00 PM - 6:00 PM)
    days = [
        '2025-12-01 17:00:00',
        '2025-12-02 17:00:00', 
        '2025-12-03 17:00:00',
        '2025-12-04 17:00:00'
    ]
    
    for day_start in days:
        base_time = datetime.strptime(day_start, '%Y-%m-%d %H:%M:%S')
        current_time = base_time
        hour_end = base_time + timedelta(hours=1)
        
        # Generate 30-40 customers per hour (realistic for peak time)
        num_customers = np.random.randint(30, 41)
        
        for _ in range(num_customers):
            # Arrival time follows exponential distribution
            inter_arrival = np.random.exponential(90)  # Avg 90 seconds between arrivals
            current_time += timedelta(seconds=inter_arrival)
            
            if current_time > hour_end:
                break
                
            # Number of items (normal distribution with min 1)
            num_items = max(1, int(np.random.normal(15, 8)))
            
            # Service time calculation: 3 sec/item + 30 sec base
            base_service = num_items * 3 + 30
            service_time = max(20, np.random.normal(base_service, 10))
            
            customers_data.append({
                'Customer_ID': customer_id,
                'Day': base_time.strftime('%Y-%m-%d'),
                'Arrival_Time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Number_of_Items': num_items,
                'Service_Time_Seconds': round(service_time, 2),
                'Arrival_Timestamp': current_time  # For simulation
            })
            
            customer_id += 1
    
    return pd.DataFrame(customers_data)

if __name__ == "__main__":
    # Generate dataset
    print("Generating supermarket checkout dataset...")
    df = generate_supermarket_data()
    
    # Save to CSV without timestamp column for cleaner data
    df_to_save = df.drop('Arrival_Timestamp', axis=1)
    df_to_save.to_csv('supermarket_checkout_data.csv', index=False)
    
    # Display summary statistics
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE")
    print("="*50)
    print(f"Total customers generated: {len(df)}")
    print(f"Date range: {df['Day'].min()} to {df['Day'].max()}")
    print(f"Average customers per day: {len(df)/4:.1f}")
    print(f"Average items per customer: {df['Number_of_Items'].mean():.1f}")
    print(f"Average service time: {df['Service_Time_Seconds'].mean():.1f} seconds")
    print(f"\nDataset saved as 'supermarket_checkout_data.csv'")