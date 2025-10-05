class GameEnvironment:
    def __init__(self):
        self.configuration = ['E', 'E', 'E', ' ', 'W', 'W', 'W']

    # Check if the goal state has been reached
    def is_goal_reached(self):
        return self.configuration == ['W', 'W', 'W', ' ', 'E', 'E', 'E']

    # Get all possible actions based on the current configuration
    def possible_actions(self):
        available_actions = []
        for index in range(len(self.configuration)):
            if self.configuration[index] == 'E':
                if index + 1 < len(self.configuration) and self.configuration[index + 1] == ' ':
                    available_actions.append((index, index + 1))
                elif index + 2 < len(self.configuration) and self.configuration[index + 2] == ' ':
                    available_actions.append((index, index + 2))
            elif self.configuration[index] == 'W':
                if index - 1 >= 0 and self.configuration[index - 1] == ' ':
                    available_actions.append((index, index - 1))
                elif index - 2 >= 0 and self.configuration[index - 2] == ' ':
                    available_actions.append((index, index - 2))
        return available_actions

    # Update the configuration based on the selected action
    def execute_action(self, action):
        self.configuration[action[0]], self.configuration[action[1]] = self.configuration[action[1]], self.configuration[action[0]]


class SearchAgent:
    def __init__(self, environment):
        self.environment = environment

    # Perform BFS to find a solution
    def breadth_first_search(self):
        queue = [([], self.environment)]
        while queue:
            current_path, current_env = queue.pop(0)
            if current_env.is_goal_reached():
                return current_path
            for action in current_env.possible_actions():
                new_env = GameEnvironment()
                new_env.configuration = current_env.configuration.copy()
                new_env.execute_action(action)
                queue.append((current_path + [action], new_env))

    # Perform DFS to find a solution
    def depth_first_search(self):
        stack = [([], self.environment)]
        while stack:
            current_path, current_env = stack.pop()
            if current_env.is_goal_reached():
                return current_path
            for action in current_env.possible_actions():
                new_env = GameEnvironment()
                new_env.configuration = current_env.configuration.copy()
                new_env.execute_action(action)
                stack.append((current_path + [action], new_env))


# Initialize the game environment and agent
game_env = GameEnvironment()
search_agent = SearchAgent(game_env)

# Perform searches
bfs_solution = search_agent.breadth_first_search()
dfs_solution = search_agent.depth_first_search()

# Display solutions
print("BFS Solution Path: ", bfs_solution)
print("DFS Solution Path: ", dfs_solution)