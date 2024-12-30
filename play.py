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
                    if self.current_player == 0:
                        print(f'You got: {id2card[player_hand[-1]]}')
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
                print('Exploding Kitten drawn ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
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
def play_game_against_ai(env, ai_player, ai_noper_model, ai_favor_model):
    state = env.reset()  # Reset the environment for a new game
    done = False
    total_reward = 0

    while not done:
        current_player_hand = env.players[env.current_player]

        if env.current_player == 0:  # Human player (Player 1)
            print("\nYour hand:", [id2card[i] for i in current_player_hand])
            print(f"Cards left in deck: {len(env.deck)}")
            print(f"Known future cards: {env.players_known_future[env.current_player]}")

            # Prompt user for their action
            valid_actions = current_player_hand + [0]  # Include the 'pass' action (0)
            print(f"Available actions: {', '.join([id2card.get(action, 'Pass') for action in valid_actions])}")

            action = input(f"Choose your action (enter card name or 'pass'): ")
            if action == 'pass':
                action = 0
            else:
                action = card2id[action]
            if action not in valid_actions:
                print("Invalid action! You lose the game!")
                env.step(0)  # Handle invalid action scenario (game ends)
                break
            nope = False
            # After human action, ask if AI wants to "Nope" the action
            if card2id["Nope"] in env.players[1] and action != 0:  # AI has 'Nope' in its hand
                print(f"\nAI is about to take an action: {id2card.get(action, 'Pass')}")
                ai_noper_input = ai_noper_model.predict(preprocess_state(state).reshape(1, -1))
                ai_noper_decision = np.argmax(ai_noper_input)  # AI decides whether to 'Nope' or not
                nope = ai_noper_decision == 1
                if ai_noper_decision == 1:  # AI decides to "Nope"
                    print("AI decided to use 'Nope'!")

            # If the player plays the "Favor" card
            favor = 0
            if action == card2id["Favor"] and env.players[1 - env.current_player]:
                state_input = preprocess_state(state).reshape(1, -1)  # Reshape to (1, -1)
                favor=np.argmax([i if ind in env.players[1-env.current_player] else float('-inf') for ind, i in enumerate(ai_favor_model.predict(state_input)[0])])+1
                print(f"AI gives you: {id2card[favor]}")
        else:  # AI player (Player 2)
            print("\nAI's turn...")
            # AI selects action based on its trained model
            state_input = preprocess_state(state).reshape(1, -1)  # Reshape to (1, -1)
            action_values = []
            for ind, i in enumerate(ai_player.predict(state_input)[0]):
                if (6 <= ind <= 10):
                    if list(preprocess_state(env._get_state()))[ind-1] >= 2:
                        action_values.append(i)
                    else:
                        action_values.append(float('-inf'))
                else:
                    if ind in current_player_hand:
                        action_values.append(i)
                    else:
                        action_values.append(float('-inf'))
            action = np.argmax(action_values)  # Select the action with the highest predicted value
            print(f"AI chose: {id2card.get(action, 'Pass')}")
            nope = False
            # Ask the human player if they want to "Nope" the action
            if card2id["Nope"] in current_player_hand and action != 0:
                print("\nAI is about to take an action.")
                nope_action = input(
                    f"Do you want to 'Nope' the action '{id2card.get(action, 'Pass')}'? (y/n): ").strip().lower()
                print(f"Cards left in deck: {len(env.deck)}")
                nope = nope_action == 'y'

            # If the AI plays the "Favor" card
            favor = 0
            if action == card2id["Favor"] and env.players[1 - env.current_player]:
                print("\nAI requests a favor! Your hand:", [id2card[i] for i in current_player_hand])
                while True:
                    favor_card = input("Choose a card to give to AI: ")
                    if favor_card in id2card.values() and card2id[favor_card] in current_player_hand:
                        favor = card2id[favor_card]
                        print(f"You give: {favor_card}")
                        break
                    else:
                        print("Invalid card. Please choose a valid card to give.")

        # Apply the chosen action in the environment
        print(f"\nGame status: Player {env.current_player + 1} took action {id2card.get(action, 'Pass')}")
        next_state, rewards, done, _ = env.step(action, bool(nope), favor)
        total_reward += rewards[env.current_player]

        # Display game status after action
        print(f"Current game state: {next_state}")
        print(f"Total reward: {total_reward}")

        # Update the state for the next iteration
        state = next_state

        # Check if the game is over
        if done:
            if env.winner == 0:
                print("\nCongratulations! You won the game!")
            else:
                print("\nAI wins the game!")
            break

player1 = play_build_model()
player1.load_weights('player1_model.keras')

player1_noper = nope_build_model()
player1_noper.load_weights('player1_noper_model.keras')

player1_favor = favor_build_model()
player1_favor.load_weights('player1_favor_model.keras')

env = ExplodingKittensEnv()

play_game_against_ai(env, player1, player1_noper, player1_favor)