import random
from collections import defaultdict
import numpy as np


class TicTacToe:

    def __init__(self):
        self.board = [0] * 9  # 0: empty, 1: X (Player 1), 2: O (Player 2)
        self.current_player = 1

    def reset(self):
        self.board = [0] * 9
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        return "".join(map(str, self.board))

    def available_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def check_winner(self):
        win_conditions = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),  # Rows
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),  # Columns
            (0, 4, 8),
            (2, 4, 6),  # Diagonals
        ]
        for a, b, c in win_conditions:
            if (
                self.board[a] != 0
                and self.board[a] == self.board[b] == self.board[c]
            ):
                return self.board[a]  # 1 or 2
        if 0 not in self.board:
            return 0  # Draw
        return None  # Game ongoing

    def step(self, action):
        self.board[action] = self.current_player
        winner = self.check_winner()

        if winner is not None:
            done = True
            if winner == 0:
                reward = 0.2  # Draw
            else:
                reward = 1.0  # Current player won
        else:
            done = False
            reward = 0.0

        # Switch turns
        self.current_player = 2 if self.current_player == 1 else 1
        return self.get_state(), reward, done, winner

    def render(self):
        symbols = {0: ".", 1: "X", 2: "O"}
        for r in range(3):
            print(" ".join(symbols[self.board[r * 3 + c]] for c in range(3)))
        print()


class QLearningAgent:

    def __init__(
        self, player_id, alpha=0.2, gamma=0.95, epsilon=1.0, epsilon_decay=0.99995, min_epsilon=0.01
    ):
        self.player_id = player_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        # Map state string -> array of 9 Q-values
        self.q_table = defaultdict(lambda: np.zeros(9, dtype=np.float32))

    def choose_action(self, state, valid_actions, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploit: choose best Q-value among VALID moves only
        q_values = self.q_table[state]
        valid_q = {a: q_values[a] for a in valid_actions}
        max_q = max(valid_q.values())
        # Break ties randomly among best actions
        best_actions = [a for a, q in valid_q.items() if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_valid_actions, done):
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            max_future_q = (
                max([self.q_table[next_state][a] for a in next_valid_actions])
                if next_valid_actions
                else 0.0
            )
            target = reward + self.gamma * max_future_q

        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# ==========================================
# Training via Self-Play
# ==========================================
def train_agents(episodes=60000):
    env = TicTacToe()
    agent_x = QLearningAgent(player_id=1)
    agent_o = QLearningAgent(player_id=2)

    print(f"Training Q-learning agents across {episodes} self-play episodes...")

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False

        # Trajectory trackers for each player to perform TD updates
        last_state = {1: None, 2: None}
        last_action = {1: None, 2: None}

        while not done:
            current_player = env.current_player
            agent = agent_x if current_player == 1 else agent_o
            other_player = 2 if current_player == 1 else 1
            other_agent = agent_o if current_player == 1 else agent_x

            valid_moves = env.available_actions()
            action = agent.choose_action(state, valid_moves)

            # Record decision
            last_state[current_player] = state
            last_action[current_player] = action

            next_state, reward, done, winner = env.step(action)

            if done:
                if winner == 0:  # Draw
                    agent.update(
                        state, action, reward, next_state, [], done=True
                    )
                    other_agent.update(
                        last_state[other_player],
                        last_action[other_player],
                        reward,
                        next_state,
                        [],
                        done=True,
                    )
                else:  # Current player won, other player lost
                    agent.update(
                        state, action, 1.0, next_state, [], done=True
                    )
                    other_agent.update(
                        last_state[other_player],
                        last_action[other_player],
                        -1.0,
                        next_state,
                        [],
                        done=True,
                    )
            else:
                # If other player already has a past state, update their Q-value based on this new transition
                if last_state[other_player] is not None:
                    other_agent.update(
                        last_state[other_player],
                        last_action[other_player],
                        0.0,
                        next_state,
                        env.available_actions(),
                        done=False,
                    )

            state = next_state

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

        if ep % 15000 == 0:
            print(
                f"Episode {ep}/{episodes} | States explored (X): {len(agent_x.q_table)} | Epsilon: {agent_x.epsilon:.3f}"
            )

    print("Training complete.\n")
    return agent_x


# ==========================================
# Interactive Play: Human vs. Q-Agent
# ==========================================
def play_vs_human(trained_agent):
    env = TicTacToe()
    state = env.reset()
    done = False

    print("--- Game Started: You are 'O' (Player 2), Agent is 'X' (Player 1) ---")
    print("Cells are indexed 0 to 8:\n0 1 2\n3 4 5\n6 7 8\n")

    while not done:
        env.render()
        if env.current_player == 1:
            action = trained_agent.choose_action(
                state, env.available_actions(), greedy=True
            )
            print(f"Agent chose cell: {action}")
            state, _, done, winner = env.step(action)
        else:
            valid_moves = env.available_actions()
            user_input = -1
            while user_input not in valid_moves:
                try:
                    user_input = int(
                        input(f"Your turn! Choose an empty cell {valid_moves}: ")
                    )
                except ValueError:
                    continue
            state, _, done, winner = env.step(user_input)

    env.render()
    if winner == 1:
        print("Agent (X) wins!")
    elif winner == 2:
        print("You (O) win!")
    else:
        print("Game ended in a draw.")


if __name__ == "__main__":
    # Train agent with Player 1 (X) perspective
    agent = train_agents(episodes=60000)
    # Launch interactive match
    play_vs_human(agent)
