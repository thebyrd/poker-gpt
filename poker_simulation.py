import random
import sys
import argparse
from collections import Counter
from itertools import combinations
from copy import deepcopy
import unittest

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
    def __init__(self, exclude_cards=None):
        self.cards = [Card(rank, suit) for rank in RANKS for suit in SUITS]
        if exclude_cards:
            self.cards = [card for card in self.cards if card not in exclude_cards]
        random.shuffle(self.cards)

    def deal(self, num=1):
        return [self.cards.pop() for _ in range(num)]

class HandEvaluator:
    @staticmethod
    def evaluate_hand(cards):
        ranks = [card.value for card in cards]
        suits = [card.suit for card in cards]
        rank_counts = Counter(ranks)
        counts = rank_counts.values()
        counts_list = list(counts)
        sorted_ranks = sorted(ranks, reverse=True)
        is_flush = False
        flush_suit = None

        # Check for flush
        for suit in SUITS:
            if suits.count(suit) >= 5:
                is_flush = True
                flush_suit = suit
                break

        # Check for straight
        unique_ranks = list(set(ranks))
        unique_ranks.sort()
        if 14 in unique_ranks:
            unique_ranks.insert(0, 1)  # Treat Ace as low for straight (A-2-3-4-5)
        is_straight = False
        high_straight_card = 0
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i + 4] - unique_ranks[i] == 4:
                is_straight = True
                high_straight_card = unique_ranks[i + 4]
        # Check for wheel straight (A-2-3-4-5)
        if not is_straight and set([14, 2, 3, 4, 5]).issubset(unique_ranks):
            is_straight = True
            high_straight_card = 5

        # Check for straight flush
        if is_flush and is_straight:
            flush_cards = [card for card in cards if card.suit == flush_suit]
            flush_ranks = [card.value for card in flush_cards]
            flush_ranks = list(set(flush_ranks))
            flush_ranks.sort()
            if 14 in flush_ranks:
                flush_ranks.insert(0, 1)
            for i in range(len(flush_ranks) - 4):
                if flush_ranks[i + 4] - flush_ranks[i] == 4:
                    return (9, [flush_ranks[i + 4]])
            # Check for wheel straight flush
            if set([14, 2, 3, 4, 5]).issubset(flush_ranks):
                return (9, [5])

        # Four of a Kind
        if 4 in counts_list:
            four_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
            kicker = max([rank for rank in ranks if rank != four_rank])
            return (8, [four_rank, kicker])

        # Full House
        if 3 in counts_list and 2 in counts_list:
            three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair_rank = max([rank for rank, count in rank_counts.items() if count == 2])
            return (7, [three_rank, pair_rank])

        # Flush
        if is_flush:
            flush_cards = [card.value for card in cards if card.suit == flush_suit]
            top_five = sorted(flush_cards, reverse=True)[:5]
            return (6, top_five)

        # Straight
        if is_straight:
            return (5, [high_straight_card])

        # Three of a Kind
        if 3 in counts_list:
            three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            kickers = sorted([rank for rank in ranks if rank != three_rank], reverse=True)[:2]
            return (4, [three_rank] + kickers)

        # Two Pair
        if counts_list.count(2) >= 2:
            pair_ranks = sorted([rank for rank, count in rank_counts.items() if count == 2], reverse=True)[:2]
            kickers = [rank for rank in ranks if rank not in pair_ranks]
            kicker = max(kickers) if kickers else 0
            return (3, pair_ranks + [kicker])

        # One Pair
        if 2 in counts_list:
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            kickers = sorted([rank for rank in ranks if rank != pair_rank], reverse=True)[:3]
            return (2, [pair_rank] + kickers)

        # High Card
        return (1, sorted_ranks[:5])

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
        self.total_bet = 0  # Total bet in the current hand
        self.is_folded = False
        self.needs_to_act = True  # For betting rounds
        self.is_all_in = False

    def reset(self):
        self.hole_cards = []
        self.current_bet = 0
        self.total_bet = 0
        self.is_folded = False
        self.needs_to_act = True
        self.is_all_in = False

    def make_decision(self, game_state):
        return self.strategy.decide(self, game_state)

class MonteCarloStrategy:
    def __init__(self, simulations=10000):
        self.simulations = simulations

    def decide(self, player, game_state):
        # Perform Monte Carlo simulations to estimate win probability
        win_probability = self.simulate(player, game_state)

        # Decide action based on win probability and pot odds
        pot = game_state['pot']
        current_bet = game_state['current_bet']
        to_call = current_bet - player.current_bet
        pot_odds = to_call / (pot + to_call) if pot + to_call > 0 else 0

        minimum_raise = game_state['minimum_raise']
        big_blind = game_state['big_blind']

        if to_call == 0:
            # No bet to call, consider betting
            if win_probability > 0.5:
                # Bet amount proportional to win probability
                bet_size = int(win_probability * player.chips)
                bet_size = max(bet_size, big_blind)
                bet_size = min(bet_size, player.chips)
                return ('raise', bet_size)
            else:
                # Check
                return ('call', 0)
        else:
            if win_probability > pot_odds:
                # Raise if win probability is significantly higher
                if win_probability - pot_odds > 0.1:
                    # Raise amount proportional to win probability
                    bet_size = int(win_probability * player.chips)
                    bet_size = max(bet_size, to_call + minimum_raise)
                    bet_size = min(bet_size, player.chips)
                    return ('raise', bet_size)
                else:
                    # Call
                    return ('call', to_call)
            else:
                # Fold
                return ('fold', 0)

    def simulate(self, player, game_state):
        wins = 0
        ties = 0
        total = 0

        hole_cards = player.hole_cards
        community_cards = game_state['community_cards']
        known_cards = hole_cards + community_cards
        unknown_cards = [card for card in Deck().cards if card not in known_cards]

        opponents = [p for p in game_state['players'] if p != player and not p.is_folded]
        num_opponents = len(opponents)

        for _ in range(self.simulations):
            deck_copy = unknown_cards.copy()
            random.shuffle(deck_copy)

            # Deal random hole cards to opponents
            opponents_hands = []
            for _ in range(num_opponents):
                opp_hole_cards = [deck_copy.pop(), deck_copy.pop()]
                opponents_hands.append(opp_hole_cards)

            # Complete the community cards
            total_community = community_cards.copy()
            cards_needed = 5 - len(total_community)
            total_community.extend(deck_copy[:cards_needed])

            # Evaluate player's best hand
            player_best_hand = self.get_best_hand(hole_cards, total_community)
            player_score = HandEvaluator.evaluate_hand(player_best_hand)

            # Evaluate opponents' best hands
            opponents_best = []
            for opp_hand in opponents_hands:
                opp_best_hand = self.get_best_hand(opp_hand, total_community)
                opp_score = HandEvaluator.evaluate_hand(opp_best_hand)
                opponents_best.append(opp_score)

            # Determine if player wins or ties
            player_wins = True
            tie = False
            for opp_score in opponents_best:
                result = self.compare_scores(player_score, opp_score)
                if result < 0:
                    player_wins = False
                    tie = False
                    break
                elif result == 0:
                    tie = True

            if player_wins and not tie:
                wins += 1
            elif tie:
                ties += 1
            total += 1

        win_probability = (wins + ties / 2) / total if total > 0 else 0
        return win_probability

    def compare_scores(self, score1, score2):
        if score1[0] != score2[0]:
            return score1[0] - score2[0]
        else:
            for a, b in zip(score1[1], score2[1]):
                if a != b:
                    return a - b
            return 0  # Tie

    def get_best_hand(self, hole_cards, community_cards):
        all_cards = hole_cards + community_cards
        max_hand = max(
            combinations(all_cards, 5),
            key=lambda hand: HandEvaluator.evaluate_hand(hand)
        )
        return max_hand

class KellyCriterionStrategy:
    def __init__(self, simulations=10000):
        self.monte_carlo = MonteCarloStrategy(simulations=simulations)

    def decide(self, player, game_state):
        # Use Monte Carlo simulations to estimate win probability
        win_probability = self.monte_carlo.simulate(player, game_state)
        print(f"{player.name}'s Win Probability: {win_probability}")
        pot = game_state['pot']
        current_bet = game_state['current_bet']
        to_call = current_bet - player.current_bet
        total_pot = pot + to_call
        q = 1 - win_probability

        minimum_raise = game_state['minimum_raise']
        big_blind = game_state['big_blind']

        if to_call == 0:
            # No bet to call, consider betting
            kelly_fraction = 2 * win_probability - 1
            if kelly_fraction <= 0:
                # Not favorable to bet
                return ('call', 0)  # Check
            bet_size = int(kelly_fraction * player.chips)
            bet_size = max(bet_size, big_blind)
            bet_size = min(bet_size, player.chips)
            # Decide to bet
            return ('raise', bet_size)
        else:
            b = to_call / total_pot if total_pot > 0 else 0  # Odds received
            if b > 0:
                kelly_fraction = (win_probability - q) / b
            else:
                kelly_fraction = 0

            if kelly_fraction <= 0:
                # Not favorable to call or raise
                return ('fold', 0)

            # Calculate bet size
            bet_size = int(kelly_fraction * player.chips)
            bet_size = max(bet_size, 0)
            bet_size = min(bet_size, player.chips)

            if bet_size >= to_call + minimum_raise:
                # Decide to raise
                return ('raise', bet_size)
            elif to_call <= player.chips:
                # Decide to call
                return ('call', to_call)
            else:
                # Cannot call, must fold
                return ('fold', 0)

class TexasHoldEmGame:
    def __init__(self, players):
        self.players = players
        self.deck = None
        self.pot = 0
        self.side_pots = []  # List of side pots
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

        self.handle_side_pots()

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
        sb_player.total_bet += sb_amount
        self.pot += sb_amount
        print(f"{sb_player.name} posts small blind of {sb_amount}")

        # Post big blind
        bb_player = self.players[big_blind_position]
        bb_amount = min(self.big_blind, bb_player.chips)
        bb_player.chips -= bb_amount
        bb_player.current_bet = bb_amount
        bb_player.total_bet += bb_amount
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

                to_call = current_bet - player.current_bet
                if to_call > player.chips:
                    to_call = player.chips

                # Update game_state for the player
                game_state = {
                    'community_cards': self.community_cards,
                    'pot': self.pot,
                    'current_bet': current_bet,
                    'minimum_raise': minimum_raise,
                    'big_blind': self.big_blind,
                    'players': self.players
                }

                action, amount = player.make_decision(game_state)

                if action == 'fold':
                    player.is_folded = True
                    active_players.remove(player)
                    print(f"{player.name} folds.")
                    if len(active_players) == 1:
                        return  # Only one player remains
                elif action == 'call':
                    if to_call == 0:
                        print(f"{player.name} checks.")
                    else:
                        if amount < to_call:
                            print(f"{player.name} cannot call less than the amount to call. Folding.")
                            player.is_folded = True
                            active_players.remove(player)
                            if len(active_players) == 1:
                                return  # Only one player remains
                            continue
                        bet_amount = min(to_call, player.chips)
                        player.chips -= bet_amount
                        self.pot += bet_amount
                        player.current_bet += bet_amount
                        player.total_bet += bet_amount
                        if player.chips == 0:
                            player.is_all_in = True
                            print(f"{player.name} goes all-in with {bet_amount}.")
                        else:
                            print(f"{player.name} calls with {bet_amount}.")
                    player.needs_to_act = False
                elif action == 'raise':
                    raise_amount = amount - (current_bet - player.current_bet)
                    if raise_amount < minimum_raise:
                        # Invalid raise, treat as fold
                        print(f"{player.name} attempted to raise less than the minimum. Folding.")
                        player.is_folded = True
                        active_players.remove(player)
                        if len(active_players) == 1:
                            return  # Only one player remains
                        player.needs_to_act = False
                    else:
                        total_bet = player.current_bet + raise_amount
                        if total_bet > player.chips:
                            total_bet = player.chips
                            raise_amount = total_bet - player.current_bet
                        bet_amount = total_bet - player.current_bet
                        player.chips -= bet_amount
                        player.current_bet += bet_amount
                        player.total_bet += bet_amount
                        self.pot += bet_amount
                        current_bet = player.current_bet
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
                else:
                    print(f"{player.name} takes no action.")
                    player.needs_to_act = False
                # Break if hand is over
                if len(active_players) <= 1:
                    return
            if all_players_done:
                break  # Betting round is over

        # Reset current bets for the next round
        for player in self.players:
            player.current_bet = 0

    def handle_side_pots(self):
        # Collect bets from players
        players_in_hand = [p for p in self.players if p.total_bet > 0]
        players_in_hand.sort(key=lambda p: p.total_bet)
        while players_in_hand:
            smallest_bet = players_in_hand[0].total_bet
            contributing_players = [p for p in players_in_hand if p.total_bet >= smallest_bet]
            pot = smallest_bet * len(contributing_players)
            self.side_pots.append({'amount': pot, 'players': contributing_players.copy()})
            for p in contributing_players:
                p.total_bet -= smallest_bet
                if p.total_bet == 0:
                    players_in_hand.remove(p)

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
        print("\n--- Showdown ---")
        active_players = [player for player in self.players if not player.is_folded]
        for i, side_pot in enumerate(self.side_pots):
            pot_amount = side_pot['amount']
            contributing_players = side_pot['players']
            eligible_players = [player for player in active_players if player in contributing_players]

            if not eligible_players:
                # Award the pot to the last active player
                last_player = active_players[0]
                last_player.chips += pot_amount
                print(f"No eligible winners for side pot {i+1}. {last_player.name} wins {pot_amount} chips by default.")
                continue

            best_score = None
            best_score_hand = None
            winners = []
            for player in eligible_players:
                all_cards = player.hole_cards + self.community_cards
                max_hand = max(
                    combinations(all_cards, 5),
                    key=lambda hand: HandEvaluator.evaluate_hand(hand)
                )
                score = HandEvaluator.evaluate_hand(max_hand)
                player.best_hand = max_hand
                player.hand_score = score
                print(f"{player.name}'s hand: {player.best_hand} - Rank: {score[0]}")
                if best_score is None or HandEvaluator.compare_hands(max_hand, best_score_hand) > 0:
                    best_score = score
                    best_score_hand = max_hand
                    winners = [player]
                elif HandEvaluator.compare_hands(max_hand, best_score_hand) == 0:
                    winners.append(player)

            if winners:
                split_pot = pot_amount // len(winners)
                remainder = pot_amount % len(winners)
                for idx, winner in enumerate(winners):
                    winner_share = split_pot + (1 if idx < remainder else 0)
                    winner.chips += winner_share
                    print(f"{winner.name} wins {winner_share} chips from side pot {i+1}.")
            else:
                # Award the pot to the last active player
                last_player = active_players[0]
                last_player.chips += pot_amount
                print(f"No eligible winners for side pot {i+1}. {last_player.name} wins {pot_amount} chips by default.")

def poker_advisor():
    print("Welcome to the Poker Advisor!")
    strategy = KellyCriterionStrategy(simulations=100000)
 
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
    big_blind = int(input("Enter the big blind amount: "))
    player_chips = int(input("Enter your current chip count: "))
    num_opponents = int(input("Enter the number of active opponents: "))

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
        'big_blind': big_blind,
        'players': [player] + [Player(f"Opponent_{i+1}", None) for i in range(num_opponents)],
    }

    # Get decision from strategy
    action, amount = player.make_decision(game_state)

    # Output advice
    if action == 'fold':
        print("Advice: You should fold.")
    elif action == 'call':
        if amount == 0:
            print("Advice: You should check.")
        else:
            print(f"Advice: You should call with {amount} chips.")
    elif action == 'raise':
        print(f"Advice: You should raise to {amount} chips.")
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

# Unit Tests
class TestPokerGame(unittest.TestCase):
    def setUp(self):
        self.player1 = Player("Alice", KellyCriterionStrategy())
        self.player2 = Player("Bob", KellyCriterionStrategy())
        self.player3 = Player("Charlie", KellyCriterionStrategy())
        self.player4 = Player("Diana", KellyCriterionStrategy())
        self.players = [self.player1, self.player2, self.player3, self.player4]
        self.game = TexasHoldEmGame(self.players)

    # Existing tests...

    def test_compare_hands_with_same_pair_different_kickers(self):
        # Player 1 has Pair of Aces with King kicker
        hand1 = [
            Card('A', '♠'), Card('A', '♦'),
            Card('K', '♣'), Card('Q', '♣'), Card('J', '♣')
        ]
        # Player 2 has Pair of Aces with Queen kicker
        hand2 = [
            Card('A', '♥'), Card('A', '♣'),
            Card('Q', '♦'), Card('J', '♦'), Card('9', '♣')
        ]
        result = HandEvaluator.compare_hands(hand1, hand2)
        self.assertTrue(result > 0)  # Player 1 should win

    def test_initial_chips(self):
        for player in self.players:
            self.assertEqual(player.chips, 1000)

    def test_blinds_posted_correctly(self):
        self.game.play_hand()
        small_blind = self.game.small_blind
        big_blind = self.game.big_blind
        sb_player = self.game.players[(self.game.dealer_position + 1) % len(self.players)]
        bb_player = self.game.players[(self.game.dealer_position + 2) % len(self.players)]
        self.assertEqual(sb_player.total_bet, small_blind)
        self.assertEqual(bb_player.total_bet, big_blind)

    def test_pot_distribution(self):
        total_chips_before = sum(player.chips for player in self.players)
        self.game.play_hand()
        total_chips_after = sum(player.chips for player in self.players)
        self.assertEqual(total_chips_before, total_chips_after)

    def test_no_unclaimed_pots(self):
        self.game.play_hand()
        # There should be no side pots left unclaimed
        self.assertEqual(len(self.game.side_pots), 0)

    def test_side_pot_calculations(self):
        # Force a scenario with side pots
        self.player1.chips = 100
        self.player2.chips = 200
        self.player3.chips = 300
        self.player4.chips = 400
        self.game.play_hand()
        total_chips_after = sum(player.chips for player in self.players)
        self.assertEqual(1000, total_chips_after)

    def test_strategy_decision_making(self):
        # Create a game state where folding is the best option
        self.player1.hole_cards = [Card('2', '♠'), Card('3', '♦')]
        game_state = {
            'community_cards': [],
            'pot': 100,
            'current_bet': 50,
            'minimum_raise': 10,
            'big_blind': 10,
            'players': self.players,
        }
        action, amount = self.player1.make_decision(game_state)
        self.assertEqual(action, 'fold')

    def test_player_cannot_call_with_less_than_to_call(self):
        self.player1.chips = 50
        self.player1.current_bet = 0
        game_state = {
            'community_cards': [],
            'pot': 100,
            'current_bet': 100,
            'minimum_raise': 10,
            'big_blind': 10,
            'players': self.players,
        }
        action, amount = self.player1.make_decision(game_state)
        self.assertIn(action, ['fold', 'call'])
        if action == 'call':
            self.assertEqual(amount, 50)

    def test_handle_all_in(self):
        self.player1.chips = 50
        self.player2.chips = 1000
        self.player3.chips = 1000
        self.player4.chips = 1000
        self.game.play_hand()
        total_chips_after = sum(player.chips for player in self.players)
        self.assertEqual(3050, total_chips_after)

    def test_winner_receives_correct_chips(self):
        # Set up a scenario where player1 wins
        self.player1.hole_cards = [Card('A', '♠'), Card('A', '♦')]
        self.player2.hole_cards = [Card('2', '♠'), Card('3', '♦')]
        self.player3.hole_cards = [Card('4', '♠'), Card('5', '♦')]
        self.player4.hole_cards = [Card('6', '♠'), Card('7', '♦')]
        self.game.community_cards = [Card('A', '♣'), Card('K', '♣'), Card('K', '♠'), Card('J', '♣'), Card('T', '♠')]
        self.player1.total_bet = 100
        self.player2.total_bet = 100
        self.player3.total_bet = 100
        self.player4.total_bet = 100
        self.game.pot = 400
        self.game.handle_side_pots()
        self.game.showdown()
        self.assertEqual(self.player1.chips, 1400)  # 1000 + 400
        self.assertEqual(self.player2.chips, 900) # doesn't work b/c need to call whatever method places the bet on the player
        self.assertEqual(self.player3.chips, 900)
        self.assertEqual(self.player4.chips, 900)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texas Hold'em Poker Simulation")
    parser.add_argument('--advisor', action='store_true', help='Run in poker advisor mode')
    parser.add_argument('--simulate', action='store_true', help='Run full game simulation mode')
    parser.add_argument('--test', action='store_true', help='Run unit tests')

    args = parser.parse_args()

    if args.test:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    elif args.advisor:
        poker_advisor()
    elif args.simulate:
        # Create players with KellyCriterionStrategy
        player1 = Player("Alice", KellyCriterionStrategy(simulations=1000000)) # Alice should win b/c she has more simulations
        player2 = Player("Bob", KellyCriterionStrategy(simulations=10000))
        player3 = Player("Charlie", KellyCriterionStrategy(simulations=10000))
        player4 = Player("Diana", KellyCriterionStrategy(simulations=10000))

        # Simulate multiple hands with more players
        game = TexasHoldEmGame([player1, player2, player3, player4])

        # Run the game until it's over
        while True:
            game_continues = game.play_hand()
            if not game_continues:
                break
    else:
        print("Please specify a mode to run:")
        print("Use '--advisor' for poker advisor mode.")
        print("Use '--simulate' for full game simulation mode.")
        print("Use '--test' to run unit tests.")
