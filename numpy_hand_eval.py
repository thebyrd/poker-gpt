import numpy as np

# Constants
SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
RANK_DICT = {r: i for i, r in enumerate(RANKS)}  # Map ranks to integers for fast comparisons
HAND_RANKING = {
    'high_card': 0,
    'one_pair': 1,
    'two_pair': 2,
    'three_of_a_kind': 3,
    'straight': 4,
    'flush': 5,
    'full_house': 6,
    'four_of_a_kind': 7,
    'straight_flush': 8,
    'royal_flush': 9
}

# Generate a full deck of cards
def generate_deck():
    return np.array([f'{rank}{suit}' for rank in RANKS for suit in SUITS])

# Helper function to extract rank values from cards
def extract_ranks(cards):
    return np.array([RANK_DICT[card[:-1]] for card in cards])

# Helper function to extract suit values from cards
def extract_suits(cards):
    return np.array([SUITS.index(card[-1]) for card in cards])

# Check for flush
def is_flush(suits):
    return len(np.unique(suits)) == 1

# Check for straight
def is_straight(ranks):
    ranks = np.sort(np.unique(ranks))
    if len(ranks) >= 5 and np.array_equal(ranks[-5:], np.arange(ranks[-1] - 4, ranks[-1] + 1)):
        return True
    if set(ranks) == {0, 1, 2, 3, 12}:  # A-2-3-4-5 straight
        return True
    return False

# Rank the hand
def best_hand_from_ranks_and_suits(ranks, suits):
    rank_counts = np.bincount(ranks, minlength=13)
    suit_counts = np.bincount(suits, minlength=4)

    # Check for flush
    if np.any(suit_counts >= 5):
        flush_suit = np.argmax(suit_counts)
        flush_cards = np.where(suits == flush_suit)[0]
        flush_ranks = np.sort(ranks[flush_cards])[-5:]  # Best 5 cards in the flush
        if is_straight(flush_ranks):
            if flush_ranks[-1] == 12:  # Royal Flush check
                return HAND_RANKING['royal_flush'], flush_ranks
            return HAND_RANKING['straight_flush'], flush_ranks
        return HAND_RANKING['flush'], flush_ranks

    # Check for straight
    if is_straight(ranks):
        sorted_ranks = np.sort(np.unique(ranks))[-5:]  # Highest straight
        return HAND_RANKING['straight'], sorted_ranks

    # Check for four of a kind
    if 4 in rank_counts:
        four_kind_rank = np.argmax(rank_counts == 4)
        kicker = np.max(ranks[ranks != four_kind_rank])
        return HAND_RANKING['four_of_a_kind'], [four_kind_rank, kicker]

    # Check for full house
    if 3 in rank_counts and 2 in rank_counts:
        three_kind_rank = np.argmax(rank_counts == 3)
        pair_rank = np.argmax(rank_counts == 2)
        return HAND_RANKING['full_house'], [three_kind_rank, pair_rank]

    # Check for three of a kind
    if 3 in rank_counts:
        three_kind_rank = np.argmax(rank_counts == 3)
        kickers = np.sort(ranks[ranks != three_kind_rank])[-2:]
        return HAND_RANKING['three_of_a_kind'], [three_kind_rank] + list(kickers)

    # Check for two pair
    if np.sum(rank_counts == 2) >= 2:
        pairs = np.argsort(np.where(rank_counts == 2)[0])[-2:]
        kicker = np.max(ranks[np.isin(ranks, pairs) == False])
        return HAND_RANKING['two_pair'], [pairs[-1], pairs[-2], kicker]

    # Check for one pair
    if 2 in rank_counts:
        pair_rank = np.argmax(rank_counts == 2)
        kickers = np.sort(ranks[ranks != pair_rank])[-3:]
        return HAND_RANKING['one_pair'], [pair_rank] + list(kickers)

    # High card
    return HAND_RANKING['high_card'], np.sort(ranks)[-5:]

# Vectorized Monte Carlo simulation using shuffling
def monte_carlo_simulation_vectorized(player_hand, community_cards, num_opponents=1, n_simulations=10000):
    deck = generate_deck()

    # Remove known cards (player's hand + community cards)
    known_cards = np.array(player_hand + community_cards)
    deck = deck[~np.isin(deck, known_cards)]  # Filter out known cards

    # Number of unknown cards: opponents' hole cards + remaining community cards
    total_unknown_cards_needed = (num_opponents * 2) + (5 - len(community_cards))

    # Repeat the deck for n_simulations and shuffle each deck independently
    shuffled_decks = np.array([np.random.permutation(deck) for _ in range(n_simulations)])

    # Extract unknown cards from shuffled decks
    sampled_cards = shuffled_decks[:, :total_unknown_cards_needed]

    # Separate opponent hands and remaining community cards
    opponent_hands = sampled_cards[:, :num_opponents * 2].reshape(n_simulations, num_opponents, 2)
    remaining_community_cards = sampled_cards[:, num_opponents * 2:]

    # Combine known community cards with remaining simulated community cards
    community_cards_matrix = np.tile(community_cards, (n_simulations, 1))
    final_community_cards = np.concatenate([community_cards_matrix, remaining_community_cards], axis=1)

    # Simulate all opponent hands and the player's hand
    player_hands = np.tile(np.array(player_hand), (n_simulations, 1))

    # Rank player's hand and opponent hands
    player_ranks = np.array([best_hand_from_ranks_and_suits(extract_ranks(np.concatenate([player_hands[i], final_community_cards[i]])),
                                                           extract_suits(np.concatenate([player_hands[i], final_community_cards[i]])))[0]
                             for i in range(n_simulations)])

    opponent_ranks = np.array([
        np.max([best_hand_from_ranks_and_suits(extract_ranks(np.concatenate([opponent_hands[i, j], final_community_cards[i]])),
                                               extract_suits(np.concatenate([opponent_hands[i, j], final_community_cards[i]])))[0]
                for j in range(num_opponents)]) for i in range(n_simulations)])

    # Calculate how many times the player wins
    player_wins = np.sum(player_ranks > opponent_ranks)
    opponent_wins = np.sum(player_ranks < opponent_ranks)

    # Calculate winning probabilities
    player_win_prob = player_wins / n_simulations
    opponent_win_prob = opponent_wins / n_simulations

    return player_win_prob, opponent_win_prob

# Example usage
player_hand = ['A♥', 'J♣']  # Player's hole cards
community_cards = [] # ['5♥', 'K♠', '6♦']  # Community cards on the board

# Run Monte Carlo simulation with 2 opponents and 100,000 simulations
player_win_prob, opponent_win_prob = monte_carlo_simulation_vectorized(player_hand, community_cards, num_opponents=3, n_simulations=10000)

# Display the results
print(f"Player win probability: {player_win_prob * 100:.2f}%")
print(f"Opponent win probability: {opponent_win_prob * 100:.2f}%")
