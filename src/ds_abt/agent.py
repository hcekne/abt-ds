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
    def __init__(self, id, initial_state, rules):
        self.id = id
        self.state = initial_state
        self.rules = rules

    def update_state(self):
        new_state = self.rules(self.state)
        self.state = new_state

    def __repr__(self):
        return f"Agent {self.id} with state {self.state}"