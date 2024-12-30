import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque

id2card = {
    1: "Attack",
    2: "Skip",
    3: "Shuffle",
    4: "See the Future",
    5: "Favor",
    6: "Rainbow-Ralphing Cat",
    7: "Tacocat",
    8: "Hairy Potato Cat",
    9: "Beard Cat",
    10: "Cattermelon",
    11: "Exploding Kitten",
    12: "Diffuse",
    13: "Nope",
}

card2id = {v: k for k, v in id2card.items()}

class ExplodingKittensEnv:
    def __init__(self):
        self.deck = []
        self.players = [[], []]
        self.players_known_future = [[], []]
        self.players_last_action = [-1, -1]
        self.current_player = 0
        self.game_over = False
        self.winner = None
        self.last_action_noped=[False, False]
        self.players_attacked=[False, False]

    def reset(self):
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.players = [self._draw_cards(8) for _ in range(2)]
        self.deck.append(card2id['Exploding Kitten'])
        random.shuffle(self.deck)
        self.current_player = 0
        self.game_over = False
        self.winner = None
        return self._get_state()

    def step(self, action, nope, favor=0):
        self.players_last_action[self.current_player] = action
        if self.game_over:
            raise ValueError("Game is already over.")

        player_hand = self.players[self.current_player]
        rewards = [0, 0]

        if action != 0:
            if action not in player_hand:
                self._end_game(winner=1 - self.current_player)
                rewards[self.current_player] = -1000
                return self._get_state(), rewards, True, {}

            player_hand.remove(action)
            if not nope:
                if action == card2id["Attack"]:
                    self.current_player = 1 - self.current_player
                    self.players_attacked[1-self.current_player]=True
                elif action == card2id["Skip"]:
                    self.current_player = 1 - self.current_player
                elif action == card2id["See the Future"]:
                    self.players_known_future[self.current_player] = list(self.deck)[:3]
                elif action == card2id["Shuffle"]:
                    random.shuffle(self.deck)
                elif action == card2id["Favor"]:
                    if favor==0:
                        print("Did not recieve favor number --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                        exit()
                    else:
                        if favor in self.players[1-self.current_player]:
                            self.players[1-self.current_player].remove(favor)
                            self.players[self.current_player].append(favor)
                        else:
                            print("Illegal Favor --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                            exit()
                elif card2id['Rainbow-Ralphing Cat'] <= action <= card2id['Cattermelon']:
                    player_hand.remove(action)
                    player_hand.append(random.choice(self.players[self.current_player]))
                elif action == card2id["Diffuse"]:
                    print('Played Diffuse ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    exit()
                    self._end_game(winner=1 - self.current_player)
                    rewards[self.current_player] = -1000
                    return self._get_state(), rewards, True, {}
                elif action == card2id["Nope"]:
                    print('Played Nope ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    exit()
                    self._end_game(winner=1 - self.current_player)
                    rewards[self.current_player] = -1000
                    return self._get_state(), rewards, True, {}
                else:
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    exit()
                    self._end_game(winner=1 - self.current_player)
                    rewards[self.current_player] = -1000
                    return self._get_state(), rewards, True, {}
                self.last_action_noped[self.current_player] = False
            else:
                if card2id['Nope'] in self.players[1-self.current_player]:
                    self.players[1-self.current_player].remove(card2id['Nope'])
                    self.last_action_noped[self.current_player] = True
                else:
                    print("Illegal Nope --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                    exit()
                    self._end_game(winner=self.current_player)
                    rewards[self.current_player] = -1000
                    return self._get_state(), rewards, True, {}
        else:
            for i in self.players_known_future:
                if i:
                    i.pop()
            drawn_card = self.deck.pop()

            if drawn_card == card2id["Exploding Kitten"]:
                if card2id["Diffuse"] in player_hand:
                    player_hand.remove(card2id["Diffuse"])
                    self.deck.append(card2id["Exploding Kitten"])
                    random.shuffle(self.deck)
                else:
                    self._end_game(winner=1 - self.current_player)
                    rewards[self.current_player] = -10
                    rewards[1 - self.current_player] = 1000
                    return self._get_state(), rewards, True, {}
            else:
                player_hand.append(drawn_card)
            if not self.players_attacked[self.current_player]:
                rewards = [1, 1]
                self.current_player = 1 - self.current_player
            else:
                self.players_attacked[self.current_player]=False

        return self._get_state(), rewards, self.game_over, {}

    def _end_game(self, winner):
        self.game_over = True
        self.winner = winner

    def _get_state(self):
        return {
            "deck_size": len(self.deck),
            "player_hands": self.players,
            "player_known_futures": self.players_known_future,
            "current_player": self.current_player,
            "game_over": self.game_over,
            "player_last_action": self.players_last_action,
            "last_action_noped": self.last_action_noped,
            "players_attacked": self.players_attacked
        }

    def _create_deck(self):
        return [card2id["Attack"]] * 4 + [card2id["Skip"]] * 4 + [card2id["See the Future"]] * 5 + [card2id["Shuffle"]] * 4 + [card2id["Nope"]] * 5 + [card2id["Favor"]] * 4 +[card2id['Diffuse']] * 4 + [card2id['Rainbow-Ralphing Cat']] * 4 + [card2id['Tacocat']] * 4 + [card2id['Hairy Potato Cat']] * 4 + [card2id['Beard Cat']] * 4 + [card2id['Cattermelon']] * 4

    def _draw_cards(self, count):
        return [self.deck.pop() for _ in range(count) if self.deck]

def preprocess_state(state):
    deck_size = state["deck_size"]
    player_hands = state["player_hands"]
    player_known_futures = state["player_known_futures"]
    current_player = state["current_player"]

    out_hand = [0] * len(card2id)
    for card in player_hands[current_player]:
        out_hand[card - 1] += 1

    out_future = player_known_futures[current_player]
    while len(out_future) < 3:
        out_future.append(0)

    return np.array([deck_size] + out_hand + out_future + [state["player_last_action"][1 - current_player]] + [state["last_action_noped"][1 - current_player]]+state['players_attacked'])

def play_build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(len(preprocess_state(ExplodingKittensEnv()._get_state())),)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(11, activation='linear'))
    return model

def nope_build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(len(preprocess_state(ExplodingKittensEnv()._get_state())),)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(2, activation='linear'))
    return model

def favor_build_model():
    model=Sequential()
    model.add(Flatten(input_shape=(len(preprocess_state(ExplodingKittensEnv()._get_state())),)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(len(card2id), activation='linear'))
    return model

def train(env, player1, player2, player1_noper, player2_noper, player1_favor, player2_favor, episodes=500, batch_size=32, gamma=0.99, epsilon=1.0,
          epsilon_decay=0.95, epsilon_min=0.01):
    replay_buffer1 = deque(maxlen=2000)
    replay_buffer2 = deque(maxlen=2000)
    replay_buffer1_noper = deque(maxlen=2000)
    replay_buffer2_noper = deque(maxlen=2000)

    for episode in range(episodes):
        print(f"Episode {episode}/{episodes}")
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            current_player_model = player1 if env.current_player == 0 else player2
            current_player_noper = player1_noper if env.current_player == 0 else player2_noper
            current_player_favor = player2_favor if env.current_player == 0 else player1_favor
            current_player_hand = env.players[env.current_player]
            state_input = preprocess_state(state).reshape(1, -1)
            if card2id["Nope"] in env.players[1-env.current_player]:
                nope = current_player_noper.predict(state_input)
                nope = np.argmax(nope[0]) == 1
            else:
                nope = False
            # Choose action
            if random.uniform(0, 1) < epsilon:
                action = random.choice(list(filter(lambda x: x<5, current_player_hand)) + [0])  # Allow the 0 action (not playing a card)
            else:
                action_probs = []
                for ind, i in enumerate(current_player_model.predict(state_input)[0]):
                    if not (6<=ind<=10):
                        print(ind)
                        if list(preprocess_state(env._get_state()))[int(ind-1)]>=2:
                            action_probs.append(float('-inf'))
                        else:
                            action_probs.append(i)
                    else:
                        if ind in current_player_hand:
                            action_probs.append(i)
                        else:
                            action_probs.append(float('-inf'))
                action = np.argmax(action_probs[0])
                # Predict if the current player will use Nope card
            favor=0
            if action==card2id['Favor']:
                if env.players[1-env.current_player]:
                    favor=np.argmax([i if ind in env.players[1-env.current_player] else float('-inf') for ind, i in enumerate(current_player_favor.predict(state_input)[0])])+1
            next_state, rewards, done, _ = env.step(action, nope, favor)
            print(rewards)
            total_reward += rewards[env.current_player]

            # Update the main player model's replay buffer
            if env.current_player == 0:
                replay_buffer1.append((state, action, rewards[0], next_state, done))
                # Also update the noper model's replay buffer
                replay_buffer2_noper.append((nope, state, action, rewards[0], next_state, done))
            else:
                replay_buffer2.append((state, action, rewards[1], next_state, done))
                # Also update the noper model's replay buffer
                replay_buffer1_noper.append((nope, state, action, rewards[1], next_state, done))
            state = next_state
        # Train the main player models
        if len(replay_buffer1) > batch_size:
            replay_batch1 = random.sample(replay_buffer1, batch_size)
            train_model(player1, replay_batch1, gamma)
            train_nope_model(player2_noper, replay_buffer2_noper, gamma)

        if len(replay_buffer2) > batch_size:
            replay_batch2 = random.sample(replay_buffer2, batch_size)
            train_model(player2, replay_batch2, gamma)
            train_nope_model(player1_noper, replay_buffer1_noper, gamma)
        # Decay epsilon for exploration-exploitation tradeoff
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        player1.save('./player1_model.keras')
        player2.save('./player2_model.keras')
        player1_noper.save('./player1_noper_model.keras')
        player2_noper.save('./player2_noper_model.keras')
        player1_favor.save('./player1_favor_model.keras')
        player2_favor.save('./player2_favor_model.keras')

        # Print progress every 50 episodes
        print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

def train_model(model, replay_batch, gamma):
    states = []
    targets = []

    for state, action, reward, next_state, done in replay_batch:
        target = reward
        if not done:
            next_state_input = preprocess_state(next_state).reshape(1, -1)
            target += gamma * np.max(model.predict(next_state_input))

        state_input = preprocess_state(state).reshape(1, -1)
        target_f = model.predict(state_input)
        target_f[0][action] = target

        states.append(state_input)
        targets.append(target_f)

    # Convert to numpy arrays for batch processing
    states = np.vstack(states)
    targets = np.vstack(targets)

    # Fit model on the whole batch
    print(len(replay_batch))
    model.fit(states, targets, epochs=32)

def train_nope_model(model, replay_batch, gamma):
    states = []
    targets = []

    for nope, state, action, reward, next_state, done in replay_batch:
        target = reward
        if not done:
            next_state_input = preprocess_state(next_state).reshape(1, -1)
            target += gamma * np.max(model.predict(next_state_input))

        state_input = preprocess_state(state).reshape(1, -1)
        target_f = model.predict(state_input)
        target_f[0][int(nope)] = target

        states.append(state_input)
        targets.append(target_f)

    # Convert to numpy arrays for batch processing
    states = np.vstack(states)
    targets = np.vstack(targets)

    # Fit model on the whole batch
    model.fit(states, targets, epochs=32)


env = ExplodingKittensEnv()
player1 = play_build_model()
player1.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
player2 = play_build_model()
player2.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

player1_noper = nope_build_model()
player1_noper.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
player2_noper = nope_build_model()
player2_noper.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

player1_favor=favor_build_model()
player1_favor.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
player2_favor=favor_build_model()
player2_favor.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

train(env, player1, player2, player1_noper, player2_noper, player1_favor, player2_favor, episodes=40)

