# # agent.py

# class Agent:
#     def __init__(self, id, initial_state, rules):
#         self.id = id
#         self.state = initial_state
#         self.rules = rules

#     def update_state(self):
#         new_state = self.rules(self.state)
#         self.state = new_state

#     def __repr__(self):
#         return f"Agent {self.id} with state {self.state}"


# agent.py

class Agent:
    def __init__(self, customer_id, avg_duration, is_one_time_shopper, duration_until_censoring, churn_score):
        self.customer_id = customer_id
        self.avg_duration = avg_duration  # Average duration between shops
        self.is_one_time_shopper = is_one_time_shopper  # True if the customer is a one-time shopper
        self.duration_until_censoring = duration_until_censoring  # Duration until last shop event
        self.churn_score = churn_score  # Churn score
        self.has_churned = False  # Initialize as False, can be updated later

    def should_shop_today(self, seasonality_factor=1):
        # If the customer has churned, they won't shop
        if self.has_churned:
            return False

        # If one-time shopper and duration_until_censoring is high, they won't shop
        if self.is_one_time_shopper and self.duration_until_censoring > 30:
            return False

        # Determine the probability of shopping today
        prob_of_shopping = 1 / (self.avg_duration * seasonality_factor)
        
        # Generate a random number and check if the agent should shop today
        return random.random() < prob_of_shopping

    def update_churn_score(self, new_data):
        # Update churn score based on new data (this is a simplified example)
        # In a real-world scenario, you'd use more complex calculations here
        self.churn_score = new_data

    def check_for_churn(self, churn_threshold=0.9):
        # Update the has_churned status based on the churn_threshold
        self.has_churned = self.churn_score > churn_threshold