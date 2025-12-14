import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_supermarket_data(start_date_str='2025-07-01 17:00:00',
                              days_count=5,
                              num_counters=3,
                              min_customers_per_hour=30,
                              max_customers_per_hour=40,
                              inter_arrival_mean_seconds=90,
                              inter_arrival_scale=0.8,
                              service_time_scale=1.5,
                              extra_delay_probability=0.10,
                              extra_delay_seconds=30,
                              counter_speed_factors=None,
                              arrival_burst=None):
    """
    Generate simulated supermarket checkout data with multiple counters (servers),
    wait times, service start/end times, queue length at arrival, and times in minutes.

    Defaults are tuned to produce higher wait times.

    New / tuned parameters to increase wait:
    - inter_arrival_scale (default 0.8): <1 makes arrivals more frequent (more congestion).
    - service_time_scale (default 1.5): >1 increases average service times.
    - extra_delay_probability (default 0.10): chance of an unexpected delay (adds extra_delay_seconds).
    - counter_speed_factors: list of multipliers per counter (>1 = slower). If None, all 1.0.
      Example: [1.0, 1.3, 1.2] makes counter 2 slower by 30%, etc.
    - arrival_burst: optional dict to create a short high-arrival-rate burst within the hour:
        { 'start_min': 5, 'end_min': 20, 'burst_multiplier': 0.5, 'probability': 0.8 }
      burst_multiplier multiplies the inter-arrival mean (smaller -> more arrivals). probability
      is the chance that any given arrival during the burst window uses the burst multiplier.
    """
    customers_data = []
    customer_id = 1

    # Default counter speeds
    if counter_speed_factors is None:
        counter_speed_factors = [1.0] * num_counters
    else:
        # If provided but shorter, pad with 1.0; if longer, truncate
        if len(counter_speed_factors) < num_counters:
            counter_speed_factors = list(counter_speed_factors) + [1.0] * (num_counters - len(counter_speed_factors))
        else:
            counter_speed_factors = list(counter_speed_factors)[:num_counters]

    # Default arrival burst (None = no burst)
    # Example burst: more arrivals between 17:05 and 17:20 on each day
    if arrival_burst is None:
        arrival_burst = {
            'start_min': 5,
            'end_min': 20,
            'burst_multiplier': 0.5,   # multiply inter-arrival mean by 0.5 => twice as frequent
            'probability': 0.8         # probability that an arrival during window uses the burst multiplier
        }

    # parse start date and generate consecutive days_count days at 17:00
    start_base = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    day_bases = [(start_base + timedelta(days=i)) for i in range(days_count)]

    for base_time in day_bases:
        # We'll generate arrivals within the 17:00 - 18:00 window
        current_time = base_time
        hour_end = base_time + timedelta(hours=1)

        # Random number of customers for this peak hour
        num_customers = np.random.randint(min_customers_per_hour, max_customers_per_hour + 1)

        # Prepare server next-free times (initialized to base_time => all free at 17:00)
        servers_next_free = [base_time for _ in range(num_counters)]

        # Keep a list of service_start times of already-assigned customers to compute queue length
        assigned_service_starts = []

        for _ in range(num_customers):
            # Decide inter-arrival mean for this arrival (apply global scale)
            inter_mean = inter_arrival_mean_seconds * max(0.01, inter_arrival_scale)

            # If in burst window, occasionally shorten inter-arrival to create surge
            minutes_since_start = (current_time - base_time).total_seconds() / 60.0
            in_burst_window = (arrival_burst is not None and
                               arrival_burst.get('start_min', 0) <= minutes_since_start <= arrival_burst.get('end_min', 60))
            if in_burst_window and np.random.rand() < arrival_burst.get('probability', 0):
                inter_mean = inter_mean * max(0.01, arrival_burst.get('burst_multiplier', 0.5))

            inter_arrival = np.random.exponential(inter_mean)
            current_time += timedelta(seconds=inter_arrival)

            if current_time > hour_end:
                break

            # Number of items (normal distribution with min 1)
            num_items = max(1, int(np.random.normal(15, 8)))

            # Base service time = 3 sec/item + 30 sec base
            base_service = num_items * 3 + 30

            # Scale service time to increase/decrease average processing time
            service_time_seconds = max(20, np.random.normal(base_service * service_time_scale,
                                                            10 * max(0.5, service_time_scale)))

            arrival_time = current_time

            # Find the server that becomes free the earliest
            earliest_server_idx = int(np.argmin(servers_next_free))
            earliest_free_time = servers_next_free[earliest_server_idx]

            # Apply counter speed factor (slower counters multiply service time)
            speed_factor = counter_speed_factors[earliest_server_idx]
            service_time_seconds *= speed_factor

            # Occasionally add an extra delay (e.g., price check, system slowdown)
            if np.random.rand() < extra_delay_probability:
                service_time_seconds += extra_delay_seconds

            if earliest_free_time <= arrival_time:
                # Server free at arrival: no wait
                service_start = arrival_time
                wait_seconds = 0.0
            else:
                # Must wait until that server is free
                service_start = earliest_free_time
                wait_seconds = (service_start - arrival_time).total_seconds()

            service_end = service_start + timedelta(seconds=service_time_seconds)

            # Update server next free time
            servers_next_free[earliest_server_idx] = service_end

            # Queue length at arrival: count previously assigned customers whose service_start > arrival_time
            queue_length_at_arrival = sum(1 for ts in assigned_service_starts if ts > arrival_time)

            # Record this customer's service_start for future queue length calculations
            assigned_service_starts.append(service_start)

            customers_data.append({
                'Customer_ID': customer_id,
                'Day': base_time.strftime('%Y-%m-%d'),
                'Counter_ID': earliest_server_idx + 1,  # 1-index counters for readability
                'Arrival_Time': arrival_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Service_Start_Time': service_start.strftime('%Y-%m-%d %H:%M:%S'),
                'Service_End_Time': service_end.strftime('%Y-%m-%d %H:%M:%S'),
                'Number_of_Items': num_items,
                'Service_Time_Seconds': round(service_time_seconds, 2),
                'Service_Time_Minutes': round(service_time_seconds / 60.0, 3),
                'Wait_Time_Seconds': round(wait_seconds, 2),
                'Wait_Time_Minutes': round(wait_seconds / 60.0, 3),
                'Queue_Length_At_Arrival': queue_length_at_arrival
            })

            customer_id += 1

    df = pd.DataFrame(customers_data)

    # Sort by arrival time just in case
    if not df.empty:
        df['Arrival_dt'] = pd.to_datetime(df['Arrival_Time'])
        df = df.sort_values(['Arrival_dt', 'Day']).drop(columns=['Arrival_dt']).reset_index(drop=True)

    return df


if __name__ == '__main__':
    # Example: produce noticeably more wait by:
    # - reducing counters to 2
    # - increasing service_time_scale
    # - making one counter slower
    # - using a burst period early in the hour
    df = generate_supermarket_data(start_date_str='2025-07-01 17:00:00',
                                   days_count=5,
                                   num_counters=3,
                                   min_customers_per_hour=35,
                                   max_customers_per_hour=45,
                                   inter_arrival_mean_seconds=90,
                                   inter_arrival_scale=0.75,
                                   service_time_scale=1.6,
                                   extra_delay_probability=0.15,
                                   extra_delay_seconds=40,
                                   counter_speed_factors=[1.0, 1.4],
                                   arrival_burst={'start_min': 2, 'end_min': 20, 'burst_multiplier': 0.4, 'probability': 0.9})

    # Display basic info
    print("Dataset Overview:")
    print(f"Total customers: {len(df)}")
    print(f"Date range: {df['Day'].min()} to {df['Day'].max()}")
    print(f"Average customers per day: {len(df)/df['Day'].nunique():.1f}")
    print(f"Average items per customer: {df['Number_of_Items'].mean():.1f}")
    print(f"Average service time: {df['Service_Time_Seconds'].mean():.1f} seconds")
    print(f"Average wait time: {df['Wait_Time_Seconds'].mean():.1f} seconds")
    print(f"95th percentile wait time (s): {df['Wait_Time_Seconds'].quantile(0.95):.1f}")
    print(f"Average queue length at arrival: {df['Queue_Length_At_Arrival'].mean():.2f}")

    print("\nFirst 10 customers:")
    print(df.head(10).to_string(index=False))

    print("\nDaily summary:")
    daily_summary = df.groupby('Day').agg({
        'Customer_ID': 'count',
        'Number_of_Items': 'mean',
        'Service_Time_Seconds': 'mean',
        'Wait_Time_Seconds': 'mean',
        'Queue_Length_At_Arrival': 'mean'
    }).rename(columns={'Customer_ID': 'Customers'}).round(2)
    print(daily_summary)

    # Save to CSV
    df.to_csv('supermarket_checkout_data_with_more_wait_controls_5days.csv', index=False)
    print(f"\nDataset saved as 'supermarket_checkout_data_with_more_wait_controls_5days.csv'")