import random
from collections import Counter
from itertools import combinations

# Define card ranks and suits
RANKS = '23456789TJQKA'
SUITS = ['♠', '♥', '♦', '♣']

# Map ranks to values
RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}

class Card:
    def __init__(self, rank, suit):
        self.rank = rank.upper()
        self.suit = suit
        self.value = RANK_VALUES[self.rank]

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return self.__str__()

class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for rank in RANKS for suit in SUITS]
        random.shuffle(self.cards)

    def deal(self, num=1):
        return [self.cards.pop() for _ in range(num)]

class HandEvaluator:
    @staticmethod
    def evaluate_hand(cards):
        # Evaluate hand strength and return a tuple (rank, high cards)
        ranks = [card.value for card in cards]
        suits = [card.suit for card in cards]
        rank_counts = Counter(ranks)
        is_flush = any(suits.count(suit) >= 5 for suit in SUITS)
        is_straight = HandEvaluator.is_straight(ranks)
        counts = rank_counts.values()
        counts_list = list(counts)

        if is_straight and is_flush:
            return (9, HandEvaluator.get_high_card(ranks))
        elif 4 in counts:
            return (8, HandEvaluator.get_rank_by_count(rank_counts, 4))
        elif 3 in counts and 2 in counts:
            return (7, HandEvaluator.get_rank_by_count(rank_counts, 3, 2))
        elif is_flush:
            return (6, HandEvaluator.get_high_card(ranks))
        elif is_straight:
            return (5, HandEvaluator.get_high_card(ranks))
        elif 3 in counts:
            return (4, HandEvaluator.get_rank_by_count(rank_counts, 3))
        elif counts_list.count(2) >= 2:
            return (3, HandEvaluator.get_rank_by_count(rank_counts, 2, 2))
        elif 2 in counts:
            return (2, HandEvaluator.get_rank_by_count(rank_counts, 2))
        else:
            return (1, HandEvaluator.get_high_card(ranks))

    @staticmethod
    def is_straight(ranks):
        ranks = list(set(ranks))
        ranks.sort()
        for i in range(len(ranks) - 4):
            if ranks[i + 4] - ranks[i] == 4:
                return True
        # Check for wheel straight (A-2-3-4-5)
        if set([2, 3, 4, 5, 14]).issubset(ranks):
            return True
        return False

    @staticmethod
    def get_rank_by_count(rank_counts, *counts_needed):
        result = []
        for count in counts_needed:
            ranks_with_count = [rank for rank, cnt in rank_counts.items() if cnt == count]
            result.extend(sorted(ranks_with_count, reverse=True))
        return result

    @staticmethod
    def get_high_card(ranks):
        return sorted(ranks, reverse=True)

    @staticmethod
    def compare_hands(hand1, hand2):
        score1 = HandEvaluator.evaluate_hand(hand1)
        score2 = HandEvaluator.evaluate_hand(hand2)
        if score1[0] != score2[0]:
            return score1[0] - score2[0]
        else:
            for a, b in zip(score1[1], score2[1]):
                if a != b:
                    return a - b
            return 0  # Tie

class Player:
    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy
        self.hole_cards = []
        self.chips = 1000  # Initial chips
        self.current_bet = 0
        self.is_folded = False
        self.needs_to_act = True  # For betting rounds
        self.is_all_in = False

    def reset(self):
        self.hole_cards = []
        self.current_bet = 0
        self.is_folded = False
        self.needs_to_act = True
        self.is_all_in = False

    def make_decision(self, game_state):
        return self.strategy.decide(self, game_state)

class KellyCriterionStrategy:
    def decide(self, player, game_state):
        # Calculate hand strength
        hand_strength = self.evaluate_hand_strength(player.hole_cards, game_state['community_cards'])
        # Estimate probability of winning
        win_probability = hand_strength  # Simplified assumption

        # Kelly criterion formula: f* = p - q, where p is win probability, q = 1 - p
        q = 1 - win_probability
        kelly_fraction = win_probability - q

        # Calculate bet size
        bet_size = int(kelly_fraction * player.chips)
        bet_size = max(bet_size, 0)  # Bet size cannot be negative
        bet_size = min(bet_size, player.chips)  # Bet size cannot exceed player's chips

        current_bet = game_state['current_bet']
        to_call = current_bet - player.current_bet

        if bet_size >= to_call + game_state['minimum_raise']:
            # Decide to raise
            return ('raise', bet_size)
        elif to_call <= player.chips:
            # Decide to call
            return ('call', to_call)
        else:
            # Cannot call, must fold or go all-in
            if player.chips > 0:
                return ('call', player.chips)  # Go all-in
            else:
                return ('fold', 0)

    def evaluate_hand_strength(self, hole_cards, community_cards):
        all_cards = hole_cards + community_cards
        num_cards = len(all_cards)
        if num_cards < 5:
            # Use a heuristic based on hole cards
            return self.evaluate_preflop_hand_strength(hole_cards)
        else:
            max_score = 0
            for combo in combinations(all_cards, 5):
                score = HandEvaluator.evaluate_hand(combo)[0]
                if score > max_score:
                    max_score = score
            return max_score / 9  # Normalize between 0 and 1

    def evaluate_preflop_hand_strength(self, hole_cards):
        # Simple heuristic for pre-flop hand strength
        ranks = sorted([card.value for card in hole_cards], reverse=True)
        high_card = ranks[0]
        low_card = ranks[1]
        if high_card == low_card:
            # Pair
            return 0.9 + (high_card - 2) / (14 - 2) * 0.1  # Between 0.9 and 1.0
        elif hole_cards[0].suit == hole_cards[1].suit:
            # Suited cards
            return 0.6 + (high_card + low_card - 4) / (28 - 4) * 0.3  # Between 0.6 and 0.9
        else:
            # Off-suit cards
            return 0.3 + (high_card + low_card - 4) / (28 - 4) * 0.3  # Between 0.3 and 0.6

def poker_advisor():
    print("Welcome to the Poker Advisor!")
    print("Select a strategy:")
    print("1. Kelly Criterion Strategy")
    # You can add more strategies here if desired
    strategy_choice = input("Enter the number of the strategy you want to use: ")
    if strategy_choice == '1':
        strategy = KellyCriterionStrategy()
    else:
        print("Invalid choice. Using Kelly Criterion Strategy by default.")
        strategy = KellyCriterionStrategy()

    # Get player's hole cards
    hole_cards = []
    for i in range(2):
        while True:
            card_input = input(f"Enter your hole card {i+1} (e.g., 'Ah' for Ace of hearts): ").strip()
            if validate_card_input(card_input):
                card = create_card_from_input(card_input)
                hole_cards.append(card)
                break
            else:
                print("Invalid card. Please try again.")

    # Get community cards
    community_cards = []
    num_community = int(input("Enter the number of community cards on the table (0-5): "))
    for i in range(num_community):
        while True:
            card_input = input(f"Enter community card {i+1}: ").strip()
            if validate_card_input(card_input):
                card = create_card_from_input(card_input)
                community_cards.append(card)
                break
            else:
                print("Invalid card. Please try again.")

    # Get game state information
    pot_size = int(input("Enter the current pot size: "))
    current_bet = int(input("Enter the current bet to call: "))
    minimum_raise = int(input("Enter the minimum raise amount: "))
    player_chips = int(input("Enter your current chip count: "))

    # Create a temporary player object
    player = Player("User", strategy)
    player.hole_cards = hole_cards
    player.chips = player_chips
    player.current_bet = 0

    # Build game state
    game_state = {
        'community_cards': community_cards,
        'pot': pot_size,
        'current_bet': current_bet,
        'minimum_raise': minimum_raise,
        'players': [],  # Other players can be added if needed
    }

    # Get decision from strategy
    action, amount = player.make_decision(game_state)

    # Output advice
    if action == 'fold':
        print("Advice: You should fold.")
    elif action == 'call':
        print(f"Advice: You should call with {amount} chips.")
    elif action == 'raise':
        print(f"Advice: You should raise to {player.current_bet + amount} chips.")
    else:
        print("Advice: No action recommended.")

def validate_card_input(card_str):
    if len(card_str) < 2 or len(card_str) > 3:
        return False
    rank = card_str[:-1].upper()
    suit = card_str[-1]
    if rank not in RANKS or suit not in SUITS:
        return False
    return True

def create_card_from_input(card_str):
    rank = card_str[:-1].upper()
    suit = card_str[-1]
    return Card(rank, suit)

if __name__ == "__main__":
    # Uncomment the line below to run the poker advisor
    poker_advisor()

    # If you want to run the full game simulation, uncomment the code below
    '''
    # Create players with different strategies
    player1 = Player("Alice", KellyCriterionStrategy())
    player2 = Player("Bob", KellyCriterionStrategy())
    player3 = Player("Charlie", KellyCriterionStrategy())
    player4 = Player("Diana", KellyCriterionStrategy())

    # Simulate multiple hands with more players
    game = TexasHoldEmGame([player1, player2, player3, player4])

    # Run the game until it's over
    while True:
        game_continues = game.play_hand()
        if not game_continues:
            break
    '''
