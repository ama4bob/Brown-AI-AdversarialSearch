from typing import Callable
from typing import Generic, Set, Tuple, TypeVar

from adversarialsearchproblem import (
    Action,
    AdversarialSearchProblem,
    State as GameState,
)

def simulate_state(asp: AdversarialSearchProblem[GameState, Action], state: GameState) -> Tuple[float, Action]:
    # Player 1 is +ve
    # Player 2 is -ve

    if asp.is_terminal_state(state):
        return (asp.evaluate_terminal(state)[0], None)

    player = state.player_to_move()
    if player == 0:
        best_action_so_far = (float('-inf'), None)
    else:
        best_action_so_far = (float('inf'), None)
        
    for action in asp.get_available_actions(state):
        child_state = asp.transition(state, action)
        child_score = simulate_state(asp,child_state)[0]
        if state == asp.get_start_state():
            print("min score: ", child_score, " action: ", action)
        if (player == 0):
            if(best_action_so_far[0] < child_score):
                best_action_so_far = (child_score, action)
        else:
            if (best_action_so_far[0] > child_score):
                best_action_so_far = (child_score, action)
    
    return best_action_so_far

def simulate_alpha_beta(asp: AdversarialSearchProblem[GameState, Action], state: GameState, alpha: float, beta: float) -> Tuple[float, Action]:
    # Player 1 is +ve
    # Player 2 is -ve

    if asp.is_terminal_state(state):
        return (asp.evaluate_terminal(state)[0], None)

    player = state.player_to_move()
    if player == 0:
        best_action_so_far = (float('-inf'), None)
    else:
        best_action_so_far = (float('inf'), None)
            
    for action in asp.get_available_actions(state):
        child_state = asp.transition(state, action)
        child_score = simulate_alpha_beta(asp, child_state, alpha, beta)[0]
        if state == asp.get_start_state():
            print("min score: ", child_score, " action: ", action)
        if (player == 0):
            if(best_action_so_far[0] < child_score):
                best_action_so_far = (child_score, action)
            if best_action_so_far[0] >= beta:
                return best_action_so_far
            alpha =  max(alpha, best_action_so_far[0])
        else:
            if (best_action_so_far[0] > child_score):
                best_action_so_far = (child_score, action)
            if best_action_so_far[0] <= alpha:
                return best_action_so_far
            beta =  min(beta, best_action_so_far[0])
    
    return best_action_so_far

def simulate_alpha_beta_cutoff(asp: AdversarialSearchProblem[GameState, Action], state: GameState, alpha: float, beta: float, cutoff: int, heuristic_func: Callable[[GameState], float]) -> Tuple[float, Action]:
    # Player 1 is +ve
    # Player 2 is -ve

    if asp.is_terminal_state(state):
        return (asp.evaluate_terminal(state)[0], None)

    if cutoff == 0:
        return (heuristic_func(state), None)

    player = state.player_to_move()
    if player == 0:
        best_action_so_far = (float('-inf'), None)
    else:
        best_action_so_far = (float('inf'), None)
            
    for action in asp.get_available_actions(state):
        child_state = asp.transition(state, action)
        child_score = simulate_alpha_beta_cutoff(asp, child_state, alpha, beta, cutoff - 1, heuristic_func)[0]
        if state == asp.get_start_state():
            print("min score: ", child_score, " action: ", action)
        if (player == 0):
            if(best_action_so_far[0] < child_score):
                best_action_so_far = (child_score, action)
            if best_action_so_far[0] >= beta:
                return best_action_so_far
            alpha =  max(alpha, best_action_so_far[0])
        else:
            if (best_action_so_far[0] > child_score):
                best_action_so_far = (child_score, action)
            if best_action_so_far[0] <= alpha:
                return best_action_so_far
            beta =  min(beta, best_action_so_far[0])
    
    return best_action_so_far

def max_value_cutoff (asp: AdversarialSearchProblem[GameState, Action], state: GameState, alpha: float, beta: float, cutoff: int, heuristic_func: Callable[[GameState], float]) -> Tuple[float, Action]:

    if asp.is_terminal_state(state):
        return (asp.evaluate_terminal(state)[0], None)

    if cutoff == 0:
        return (heuristic_func(state), None)

    score = (float('-inf'), None)
    for action in asp.get_available_actions(state):
        child_state = asp.transition(state, action)
        child_score = min_value_cutoff(asp, child_state, alpha, beta, cutoff - 1, heuristic_func)[0]
        if score[0] < child_score:
            score = (child_score, action)
        if score[0] >= beta:
            return score
        alpha = max(alpha, score[0])
        
    return score

def min_value_cutoff (asp: AdversarialSearchProblem[GameState, Action], state: GameState, alpha: float, beta: float, cutoff: int, heuristic_func: Callable[[GameState], float]) -> Tuple[float, Action]:

    if asp.is_terminal_state(state):
        return (asp.evaluate_terminal(state)[0], None)

    if cutoff == 0:
        return (heuristic_func(state), None)

    score = (float('inf'), None)
    for action in asp.get_available_actions(state):
        child_state = asp.transition(state, action)
        child_score = max_value_cutoff(asp, child_state, alpha, beta, cutoff - 1, heuristic_func)[0]
        if score[0] > child_score:
            score = (child_score, action)
        if score[0] <= alpha:
            return score
        beta = min(beta, score[0])

    return score 

def minimax(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
    """
    return simulate_state(asp, asp.get_start_state())[1]


def alpha_beta(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    #return max_value(asp, asp.get_start_state(), float('-inf'), float('inf'))[1]
    return simulate_alpha_beta(asp, asp.get_start_state(), float('-inf'), float('inf'))[1]

def alpha_beta_cutoff(
    asp: AdversarialSearchProblem[GameState, Action],
    cutoff_ply: int,
    heuristic_func: Callable[[GameState], float],
) -> Action:
    # See AdversarialSearchProblem:heuristic_func
    """
    This function should:
    - search through the asp using alpha-beta pruning
    - cut off the search after cutoff_ply moves have been made.

    Input:
        asp - an AdversarialSearchProblem
        cutoff_ply - an Integer that determines when to cutoff the search and
            use heuristic_func. For example, when cutoff_ply = 1, use
            heuristic_func to evaluate states that result from your first move.
            When cutoff_ply = 2, use heuristic_func to evaluate states that
            result from your opponent's first move. When cutoff_ply = 3 use
            heuristic_func to evaluate the states that result from your second
            move. You may assume that cutoff_ply > 0.
        heuristic_func - a function that takes in a GameState and outputs a
            real number indicating how good that state is for the player who is
            using alpha_beta_cutoff to choose their action. You do not need to
            implement this function, as it should be provided by whomever is
            calling alpha_beta_cutoff, however you are welcome to write
            evaluation functions to test your implemention. The heuristic_func
            we provide does not handle terminal states, so evaluate terminal
            states the same way you evaluated them in the previous algorithms.
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    return simulate_alpha_beta_cutoff(asp, asp.get_start_state(), float('-inf'), float('inf'), cutoff_ply, heuristic_func)[1]
