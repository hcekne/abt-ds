import pandas as pd
import random
import feather
from ds_abt.utilities.embeddingAnalysis import calculate_embeddings

class Experiment:
    def __init__(self, initial_data, pre_existing_customer_embeddings=None, pre_existing_product_embeddings=None):
        self.initial_data = initial_data

        self.agents = []
        self.daily_results = []
        
        # Load pre-existing embeddings if available, otherwise calculate them
        if pre_existing_customer_embeddings is not None:
            self.customer_embeddings = pre_existing_customer_embeddings
            self.product_embeddings = pre_existing_product_embeddings
        else:
            self.customer_embeddings, self.product_embeddings = calculate_embeddings(self.initial_data)

        # use initial data to calculate various metrics and 
        # data structures we need

        # calculateSeasonal 
        

        # Initialize agents
        for index, row in initial_data.iterrows():
            agent = Agent(
                customer_id=row['customer_id'],
                avg_duration=row['avg_duration'],
                is_one_time_shopper=row['is_one_time_shopper'],
                duration_until_censoring=row['duration_until_censoring'],
                churn_score=row['churn_score']
            )
            self.agents.append(agent)
            
    def run_simulation(self, num_days, seasonality_factor=1, churn_threshold=0.9):
        for day in range(1, num_days + 1):
            daily_shopping_data = []
            
            for agent in self.agents:
                agent.check_for_churn(churn_threshold)
                
                if agent.should_shop_today(seasonality_factor):
                    invoice_num = f'INV_{day}_{random.randint(1000, 9999)}'
                    product_bought = random.choice(self.product_list)
                    shopping_value = random.uniform(20, 100)  # Replace with actual logic if available
                    
                    daily_shopping_data.append({
                        'day': day,
                        'customer_id': agent.customer_id,
                        'invoice_num': invoice_num,
                        'product_bought': product_bought,
                        'shopping_value': shopping_value
                    })
                    
            self.daily_results.append(pd.DataFrame(daily_shopping_data))
            
    def write_to_feather(self, simulation_id):
        # Concatenate daily results into a single DataFrame
        full_results = pd.concat(self.daily_results, ignore_index=True)
        
        # Write to Feather file
        feather.write_dataframe(full_results, f'/mnt/data/simulation_{simulation_id}.feather')
        
    def aggregate_analysis(self):
        # Perform aggregate analysis, like shopping count, value, and number of shoppers per day
        # Assuming self.daily_results is a list of DataFrames
        aggregate_data = []
        
        for daily_df in self.daily_results:
            if daily_df.empty:
                aggregate_data.append({
                    'day': daily_df['day'].iloc[0] if not daily_df.empty else None,
                    'num_shoppers': 0,
                    'total_value': 0
                })
                continue
            
            num_shoppers = daily_df['customer_id'].nunique()
            total_value = daily_df['shopping_value'].sum()
            
            aggregate_data.append({
                'day': daily_df['day'].iloc[0],
                'num_shoppers': num_shoppers,
                'total_value': total_value
            })
        
        return pd.DataFrame(aggregate_data)
