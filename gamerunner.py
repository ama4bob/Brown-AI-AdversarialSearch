import argparse
import time

from adversarialsearchproblem import AdversarialSearchProblem
import adversarialsearch as MyImplementation
from asps.tttproblem import TTTProblem, TTTUI
from asps.connect4problem import Connect4Problem, Connect4GUI


def get_custom_asp(args):
    """
    OPTIONAL: Include your own ASP through this getter!
    Will be called with the flag "--game=custom"

    Inputs:
            - args: args that have been passed from main(), for your convenience.
                    For example, you might want to use args.player1 and/or args.player2.
    Output:
            A tuple containing an AdversarialSearchProblem and a GameUI
    """
    pass
    # game = SomeProblem()
    # game_ui = SomeUI(game)
    # return game, game_ui


########################################################
# You should not have to modify the code below,        #
# but it may be useful to see how games are simulated. #
########################################################


def run_game(asp: AdversarialSearchProblem, bots, game_ui=None):
    """
    Inputs:
            - asp: a game to play, represented as an adversarial search problem
            - bots: a list in which the i'th element is adversarial search
                    algorithm that player i will use.
                    The algorithm must take in an ASP only and output an action.
            - game_ui (optional): a GameUI that visualizes ASPs and allows for
                    direct input in place of a bot that is None. If no argument is
                    passed, run_game() will not be interactive.
    Output:
            - the evaluation of the terminal state.
    """

    # Ensure game_ui is present if a bot is None:
    if not game_ui and any(bot is None for bot in bots):
        raise ValueError("A GameUI instance must be provided if any bot is None.")

    state = asp.get_start_state()
    if game_ui:
        game_ui.update_state(state)
        game_ui.render()

    while not (asp.is_terminal_state(state)):
        curr_bot = bots[state.player_to_move()]

        # Obtain decision from the bot itself, or from GameUI if bot is None:
        if curr_bot:
            decision = curr_bot(asp)

            # If the bot tries to make an invalid action,
            # returns any valid action:
            available_actions = asp.get_available_actions(state)
            if decision not in available_actions:
                decision = available_actions.pop()
        else:
            decision = game_ui.get_user_input_action()

        result_state = asp.transition(state, decision)
        asp.set_start_state(result_state)
        state = result_state

        if game_ui:
            game_ui.update_state(state)
            game_ui.render()

    return asp.evaluate_terminal(asp.get_start_state())


def main():
    # Setup parser; Default behavior is Tic-Tac-Toe, minimax, player vs. bot.
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=["ttt", "connect4", "custom"], default="ttt")
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument(
        "--player1", choices=["self", "minimax", "ab", "ab-cutoff"], default="self"
    )
    parser.add_argument(
        "--player2", choices=["self", "minimax", "ab", "ab-cutoff"], default="minimax"
    )
    parser.add_argument("--cutoff", type=int, default=None)
    args = parser.parse_args()
    player_args = [args.player1, args.player2]

    # Ensure cutoff is present, if required:
    if "ab-cutoff" in player_args and args.cutoff is None:
        parser.error(
            "Cannot run ab-cutoff without a cutoff set! Use the argument --cutoff=<your cutoff>."
        )

    # Assign players:
    players = [None, None]
    algorithm_dict = {
        "self": None,
        "minimax": MyImplementation.minimax,
        "ab": MyImplementation.alpha_beta,
    }  # (if not in dict, player is ab-cutoff)
    for i, player in enumerate(player_args):
        players[i] = algorithm_dict.get(
            player,
            lambda asp, i=i: MyImplementation.alpha_beta_cutoff(
                asp, args.cutoff, lambda s: asp.heuristic_func(s, i)
            ),
        )

    ### Game: Tic-Tac-Toe
    if args.game == "ttt":
        if args.dimension is not None:
            if args.dimension < 3:
                parser.error("--dimension must be at least 3 for Tic-Tac-Toe")
            game = TTTProblem(dim=args.dimension)
        else:
            game = TTTProblem()
        game_ui = TTTUI(game)

    ### Game: Connect Four
    if args.game == "connect4":
        if args.dimension is not None:
            if args.dimension < 4:
                parser.error("--dimension must be at least 4 for Connect Four")
            game = Connect4Problem(dims=(args.dimension, args.dimension))
        else:
            game = Connect4Problem()
        game_ui = Connect4GUI(game)

    ### Game: Custom
    if args.game == "custom":
        game, game_ui = get_custom_asp(args)

    ### Run the game and print the final scores:
    print(f"PLAYERS: {args.player1} (P1) vs. {args.player2} (P2)")
    p1_score, p2_score = run_game(game, players, game_ui)
    print(f"P1 score: {p1_score}, P2 score: {p2_score}")

    # time.sleep(10) #(uncomment to keep GUI visible after end of game)


if __name__ == "__main__":
    main()
