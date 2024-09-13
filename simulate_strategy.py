import random
from collections import Counter
from itertools import combinations

# Define card ranks and suits
RANKS = '23456789TJQKA'
SUITS = '♠♥♦♣'

# Map ranks to values
RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.value = RANK_VALUES[rank]

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

class TexasHoldEmGame:
    def __init__(self, players):
        self.players = players
        self.deck = None
        self.pot = 0
        self.side_pots = []  # For handling side pots
        self.community_cards = []
        self.betting_rounds = ['pre-flop', 'flop', 'turn', 'river']
        self.dealer_position = 0  # Start with the first player as the dealer
        self.hand_number = 0
        self.small_blind = 5
        self.big_blind = 10

    def play_hand(self):
        self.deck = Deck()
        self.pot = 0
        self.side_pots = []
        self.community_cards = []
        self.hand_number += 1

        # Increase blinds every 5 hands
        if self.hand_number % 5 == 0:
            self.small_blind *= 2
            self.big_blind *= 2
            print(f"Blinds increased to {self.small_blind}/{self.big_blind}")

        # Remove eliminated players
        self.players = [player for player in self.players if player.chips > 0]
        num_players = len(self.players)

        if num_players == 0:
            print("Game over! All players have been eliminated.")
            return False
        elif num_players == 1:
            print(f"Game over! {self.players[0].name} wins the game.")
            return False  # Indicate that the game is over

        # Adjust dealer position
        self.dealer_position = self.dealer_position % num_players

        print(f"--- Hand {self.hand_number} ---")
        print(f"Dealer: {self.players[self.dealer_position].name}")

        # Reset players and deal hole cards
        for player in self.players:
            player.reset()
            player.hole_cards = self.deck.deal(2)
            print(f"{player.name} has been dealt: {player.hole_cards}")

        # Determine positions
        small_blind_position = (self.dealer_position + 1) % num_players
        big_blind_position = (self.dealer_position + 2) % num_players

        # Post blinds
        self.post_blinds(small_blind_position, big_blind_position)

        # Set initial betting positions
        self.current_bettor_index = (big_blind_position + 1) % num_players

        for round_name in self.betting_rounds:
            if round_name == 'flop':
                self.community_cards.extend(self.deck.deal(3))
            elif round_name in ['turn', 'river']:
                self.community_cards.extend(self.deck.deal(1))

            print(f"\n--- {round_name.capitalize()} ---")
            if self.community_cards:
                print(f"Community Cards: {self.community_cards}")

            self.betting_round(round_name)
            if self.is_hand_over():
                break  # Exit the loop if the hand is over

        # Deal remaining community cards if necessary
        self.deal_remaining_community_cards()

        self.showdown()

        # Move dealer button
        self.dealer_position = (self.dealer_position + 1) % num_players
        print(f"\nEnd of Hand {self.hand_number}")
        for player in self.players:
            print(f"{player.name}'s chip count: {player.chips}")
        print("-" * 50)
        return True  # Indicate that the game continues

    def post_blinds(self, small_blind_position, big_blind_position):
        active_players = [player for player in self.players if player.chips > 0]
        num_active = len(active_players)
        if num_active < 2:
            return

        # Post small blind
        sb_player = self.players[small_blind_position]
        sb_amount = min(self.small_blind, sb_player.chips)
        sb_player.chips -= sb_amount
        sb_player.current_bet = sb_amount
        self.pot += sb_amount
        print(f"{sb_player.name} posts small blind of {sb_amount}")

        # Post big blind
        bb_player = self.players[big_blind_position]
        bb_amount = min(self.big_blind, bb_player.chips)
        bb_player.chips -= bb_amount
        bb_player.current_bet = bb_amount
        self.pot += bb_amount
        print(f"{bb_player.name} posts big blind of {bb_amount}")

    def betting_round(self, round_name):
        active_players = [player for player in self.players if not player.is_folded and player.chips > 0]
        if len(active_players) <= 1:
            return  # Hand is over if only one player remains

        # Initialize bets and needs_to_act flags
        current_bet = max(player.current_bet for player in self.players)
        minimum_raise = self.big_blind  # Set minimum raise amount

        for player in active_players:
            player.needs_to_act = True  # All players need to act at the start
        last_raiser = None

        # Determine betting order
        num_players = len(self.players)
        betting_order_indices = list(range(self.current_bettor_index, num_players)) + list(range(0, self.current_bettor_index))
        betting_order_players = [self.players[i] for i in betting_order_indices if self.players[i] in active_players]

        while True:
            all_players_done = True
            for player in betting_order_players:
                if player.is_folded or player.is_all_in or not player.needs_to_act:
                    continue
                all_players_done = False

                # Display current game state for the player
                print(f"\n{player.name}'s turn:")
                print(f"Current bet to call: {current_bet}")
                print(f"{player.name}'s current bet: {player.current_bet}")
                print(f"{player.name}'s chips: {player.chips}")
                action, amount = player.make_decision({
                    'community_cards': self.community_cards,
                    'pot': self.pot,
                    'current_bet': current_bet,
                    'minimum_raise': minimum_raise,
                    'players': self.players
                })

                if action == 'fold':
                    player.is_folded = True
                    active_players.remove(player)
                    print(f"{player.name} folds.")
                    if len(active_players) == 1:
                        return  # Only one player remains
                elif action == 'call':
                    to_call = min(amount, player.chips)
                    bet_amount = to_call
                    player.chips -= to_call
                    self.pot += to_call
                    player.current_bet += to_call
                    if player.chips == 0:
                        player.is_all_in = True
                        print(f"{player.name} goes all-in with {bet_amount}.")
                    else:
                        print(f"{player.name} calls with {bet_amount}.")
                    player.needs_to_act = False
                elif action == 'raise':
                    raise_amount = amount - (current_bet - player.current_bet)
                    if raise_amount < minimum_raise:
                        # Invalid raise, treat as call
                        to_call = min(current_bet - player.current_bet, player.chips)
                        bet_amount = to_call
                        player.chips -= to_call
                        self.pot += to_call
                        player.current_bet += to_call
                        if player.chips == 0:
                            player.is_all_in = True
                            print(f"{player.name} goes all-in with {bet_amount}.")
                        else:
                            print(f"{player.name} calls with {bet_amount}.")
                        player.needs_to_act = False
                    else:
                        bet = min(amount, player.chips)
                        bet_amount = bet
                        current_bet = player.current_bet + bet
                        player.chips -= bet
                        self.pot += bet
                        player.current_bet += bet
                        if player.chips == 0:
                            player.is_all_in = True
                            print(f"{player.name} goes all-in and raises to {current_bet}.")
                        else:
                            print(f"{player.name} raises to {current_bet}.")
                        last_raiser = player
                        # After a raise, all other players need to act again
                        for other_player in active_players:
                            if other_player != player and not other_player.is_all_in:
                                other_player.needs_to_act = True
                        player.needs_to_act = False
                # Break if hand is over
                if len(active_players) <= 1:
                    return
            if all_players_done:
                break  # Betting round is over

        # Handle side pots if necessary
        self.handle_side_pots()

        # Reset current bets for the next round
        for player in self.players:
            player.current_bet = 0

    def handle_side_pots(self):
        # Handle side pots when players are all-in
        # For simplicity, not implemented in detail
        pass

    def is_hand_over(self):
        active_players = [player for player in self.players if not player.is_folded and player.chips > 0]
        return len(active_players) <= 1

    def deal_remaining_community_cards(self):
        """Deals remaining community cards if there are fewer than 5."""
        total_community_cards_needed = 5
        cards_to_deal = total_community_cards_needed - len(self.community_cards)
        if cards_to_deal > 0:
            self.community_cards.extend(self.deck.deal(cards_to_deal))
            print(f"Dealing remaining {cards_to_deal} community card(s): {self.community_cards[-cards_to_deal:]}")
            print(f"Final Community Cards: {self.community_cards}")

    def showdown(self):
        active_players = [player for player in self.players if not player.is_folded]
        if len(active_players) == 1:
            winner = active_players[0]
            winner.chips += self.pot
            print(f"\n{winner.name} wins the pot of {self.pot} by default.")
        elif len(active_players) == 0:
            print("All players have folded. The pot remains unclaimed.")
        else:
            # Determine winners and distribute pots
            print("\n--- Showdown ---")
            self.distribute_pots(active_players)

    def distribute_pots(self, active_players):
        # In this simplified version, we'll assume no side pots for now
        best_score = None
        winners = []
        for player in active_players:
            all_cards = player.hole_cards + self.community_cards
            max_hand = max(
                combinations(all_cards, 5),
                key=lambda hand: HandEvaluator.evaluate_hand(hand)
            )
            score = HandEvaluator.evaluate_hand(max_hand)
            player.best_hand = max_hand
            player.hand_score = score
            print(f"{player.name}'s hand: {player.best_hand} - Rank: {score[0]}")
            if best_score is None or score > best_score:
                best_score = score
                winners = [player]
            elif score == best_score:
                winners.append(player)

        if len(winners) > 0:
            split_pot = self.pot // len(winners)
            for winner in winners:
                winner.chips += split_pot
                print(f"{winner.name} wins {split_pot} chips.")
        else:
            print("No winners could be determined.")

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
