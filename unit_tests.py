import unittest

from adversarialsearch import alpha_beta, alpha_beta_cutoff, minimax
from asps.gamedag import DAGState, GameDAG


class IOTest(unittest.TestCase):
    """
    Tests IO for adversarial search implementations.
    Contains basic/trivial test cases.

    Each test function instantiates an adversarial search problem (DAG) and tests
    that the algorithm returns a valid action.

    It does NOT test whether the action is the "correct" action to take
    """

    def _get_test_dag(self):
        """
        An example of an implemented GameDAG from the gamedag class.

        Output: GameDAG to be used for testing
        """
        X = True
        _ = False
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: (-1, 1), 4: (-2, 2), 5: (-3, 3), 6: (-4, 4)}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        return dag

    def _check_result(self, result, dag):
        """
        Tests whether the result is one of the possible actions
        of the dag.
        Input:
            result- the return value of an adversarial search problem.
                This should be an action
            dag- the GameDAG that was used to test the algorithm
        """
        self.assertIsNotNone(result, "Output should not be None")
        start_state = dag.get_start_state()
        potentialActions = dag.get_available_actions(start_state)
        self.assertTrue(
            result in potentialActions, "Output should be an available action"
        )

    def _general_check_algorithm(self, algorithm):
        """
        instantiates an adversarial search problem (DAG), and
        checks that the result is an action
        Input:
            algorithm- a function that takes in an asp and returns an
            action
        """
        dag = self._get_test_dag()
        result = algorithm(dag)
        self._check_result(result, dag)

    def _dummy_heuristic_func(self, _):
        return 0

    def test_minimax(self):
        self._general_check_algorithm(minimax)
        print("minimax passes basic I/O specifications")

    def test_alpha_beta(self):
        self._general_check_algorithm(alpha_beta)
        print("alpha-beta passes basic I/O specifications")

    def test_alpha_beta_cutoff(self):
        dag = self._get_test_dag()
        cutoff = 1
        result = alpha_beta_cutoff(dag, cutoff, self._dummy_heuristic_func)
        self._check_result(result, dag)
        print("alpha-beta cutoff passes basic I/O specifications")


class CorrectActionTest(unittest.TestCase):
    """
    Tests "correct" action to take for adversarial search implementations.
    Contains simple test cases.

    Each test function instantiates an adversarial search problem (DAG) and tests
    that the algorithm returns the correct action for this simple DAG.
    """

    def _get_test_dag_2(self):
        """
        An example of an implemented GameDAG from the gamedag class.

        Output: GameDAG to be used for testing
        """
        X = True
        _ = False
        matrix = [
            [_, X, X, X, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, X, X, X, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, X, X, X, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, X, X, X],
            [_, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {
            4: (-1, 1),
            5: (-4, 4),
            6: (-5, 5),
            7: (2, -2),
            8: (3, -3),
            9: (8, -8),
            10: (-16, 16),
            11: (-3, 3),
            12: (-16, 16),
        }

        dag2 = GameDAG(
            matrix,
            start_state,
            terminal_evaluations,
        )
        return dag2

    def _output_check_result(self, result, dag):
        """
        Tests whether the result is the "correct" action
        of the dag.
        Input:
            result- the action of an adversarial search problem.
            dag- the GameDAG that was used to test the algorithm
        """
        self.assertIsNotNone(result, "Output should not be None")
        solution = 2
        self.assertTrue(result == solution, "Output producing incorrect action")

    def _output_check_algorithm(self, algorithm):
        """
        instantiates an adversarial search problem (DAG), and
        checks that the result is the "correct" action for a constant
        Input:
            algorithm- a function that takes in an asp and returns an
            action
        """
        dag2 = self._get_test_dag_2()
        result = algorithm(dag2)
        self._output_check_result(result, dag2)

    def _dummy_heuristic_func(self, _):
        return 0

    def test_minimax(self):
        self._output_check_algorithm(minimax)
        print("minimax produces correct action for simple DAG")

    def test_alpha_beta(self):
        self._output_check_algorithm(alpha_beta)
        print("alpha-beta produces correct action for simple DAG")

    def test_alpha_beta_cutoff(self):
        dag2 = self._get_test_dag_2()
        cutoff = 2
        result = alpha_beta_cutoff(dag2, cutoff, self._dummy_heuristic_func)
        self._output_check_result(result, dag2)
        print("alpha-beta cutoff produces correct action for simple DAG")


if __name__ == "__main__":
    unittest.main()
