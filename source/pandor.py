"""
IJCAI 2019 Submission:
A Correctness Result for Likelihood-based Synthesis of Finite-State Controllers

Pandor:
Probabilistic AND-OR search algorithm for a finite-state controller in a stochastic environment
"""

import argparse
import logging
import timeit
import time
import copy
import numpy as np
from typing import Tuple, Iterator

from controller import MealyController
import environments

NEW_ROWS = 5

S_WIN = "win"
S_FAIL = "fail"
A_STOP = "stop"

PRINT_WAIT_SECONDS = 1

# verbose flag
v = False


class PandorControllerNotFound(ValueError):
    pass


class HistoryItem:
    def __init__(self, q, s, p):
        self.q = q
        self.s = s
        self.p = p

    def __str__(self):
        return f"(q: {self.q}, s: {self.s}, p: {self.p:0.1f})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        """ self.p == None is a wildcard"""
        return self.q == other.q and \
               self.s == other.s and \
               ((self.p is None) or (other.p is None) or self.p == other.p)


class PAndOrPlanner:
    def __init__(self, env):
        self.env = env
        # Lower/upper bound for the LPC of the current controller
        self.lpc_desired = None
        self.num_steps = None

    def synth_plan(self, states_bound, lpc_desired):
        self.lpc_desired = lpc_desired

        # counters for stats
        self.num_steps = 0

        cont = MealyController(states_bound)

        alpha = {'win': [0.],
                 'fail': [0.],
                 'noter': [0.],
                 'loop': np.array([[0.]])}

        # For numerical stability, lpc_desired must be lower than 1.
        assert lpc_desired < 1.0

        empty_history = []

        try:
            good_cont, good_alpha = next(self.and_step(cont, cont.init_state, self.env.init_states_p, empty_history, alpha))
            print("Controller found with max ", states_bound, "states.")
            return good_cont, self.calc_lambda(good_alpha, empty_history)
        except StopIteration:
            print("No controller found with max ", states_bound, "states.")
            raise PandorControllerNotFound


    def and_step(self, c: MealyController, q, sl_next, history, alpha) \
            -> Iterator[Tuple[MealyController, dict]]:

        if sl_next == []:
            self.cumulate_alpha(alpha, history)
            yield (c, alpha)
        else:
            s_next, p_next = sl_next[0]

            logging.info("AND: Simulating s: %s, q: %s", self.env.str_state(s_next), q) if v else 0

            self.extend_alpha(alpha, history)

            for x in self.or_step(c, q, s_next, p_next, history, alpha):
                new_c, new_alpha = x

                # logging.debug("AND: (before calc) alpha['loop'] = %s", new_alpha['loop'])
                logging.debug("AND: (before calc) new_alpha['noter'] = %s", new_alpha['noter'])

                likelihoods = self.calc_lambda(new_alpha, history)

                # logging.debug("AND: (after  calc) alpha['loop'] = %s", new_alpha['loop'])
                logging.debug("AND: (after  calc) alpha['noter'] = %s", new_alpha['noter'])
                logging.debug("AND: likelihoods: %s", likelihoods)
                lpc_lower_bound = likelihoods['win']
                lpc_upper_bound = 1 - likelihoods['fail'] - likelihoods['noter']

                if lpc_lower_bound >= self.lpc_desired:
                    logging.info("AND: succeed at history %s", history)
                    yield (new_c, self.cumulate_alpha(new_alpha, history))

                elif lpc_upper_bound < self.lpc_desired:
                    logging.info("AND: fail at history %s", history)
                    logging.info("AND (do nothing)")
                    pass

                else:
                    yield from ((c_, self.cumulate_alpha(alpha_, history))
                                for c_, alpha_ in self.and_step(new_c, q, sl_next[1:], history, new_alpha))


    def or_step(self, c, q, s, p, history, alpha) \
            -> Iterator[Tuple[MealyController, dict]]:
        """
        :param p: probability of next state transition
        """

        # for debugging/stats only
        self.num_steps += 1

        if s is S_WIN:
            # len(history) is good because hist does not yet contain this step.
            alpha['win'][len(history)] += p
            logging.info("OR: terminated in goal state")
            yield c, alpha

        elif s is S_FAIL:
            alpha['fail'][len(history)] += p
            logging.info("OR: terminated in NOT goal state")
            yield (c, alpha)

        elif HistoryItem(q, s, None) in history:
            looping_timestep = history.index(HistoryItem(q, s, None))

            l_loop = p
            for h_item in history[looping_timestep + 1:]:
                l_loop *= h_item.p
            if l_loop == 1.:
                alpha['noter'][len(history)] += 1.
                logging.info("OR: repeated state")
            else:
                alpha['loop'][looping_timestep, len(history)-1] += p
                logging.info("OR: loop to level %d with prob %.1f", looping_timestep, p)

            yield (c, alpha)
        else:
            new_history = history + [HistoryItem(q, s, p)]
            obs = self.env.get_obs(s)

            if (q, obs) in c.transitions:
                q_next, action = c[q, obs]

                if (action not in self.env.legal_actions(s)) and not (action is A_STOP):
                    alpha['fail'][len(history) - 1] += p
                    logging.info("OR: illegal action {} in state {}".format(action, s))
                    yield (c, alpha)
                else:
                    sl_next = self.extended_next_states(action, s)
                    yield from self.and_step(c, q_next, sl_next, new_history, alpha)

            else:
                transition_list = self.get_mealy_qa_iterator(c, s, obs)

                # non-det branching of q',a
                for q_next, action in transition_list:
                    new_cont = copy.deepcopy(c)
                    new_cont[q, obs] = q_next, action

                    # new controller -> new alpha dict
                    new_alpha = copy.deepcopy(alpha)

                    logging.info("OR: Added:   (%s,%s) -> (%s,%s)",
                                 q, self.env.str_obs(obs),
                                 q_next, self.env.str_action(action))

                    sl_next = self.extended_next_states(action, s)

                    yield from self.and_step(new_cont, q_next, sl_next, new_history, new_alpha)

                logging.info("OR: all extensions failed")

    def extended_next_states(self, action, s):
        if action is A_STOP:
            if self.env.is_goal_state(s):
                sl_next = [(S_WIN, 1.0)]
            else:
                sl_next = [(S_FAIL, 1.0)]
        else:
            sl_next = sorted(self.env.next_states_p(s, action), key=lambda sp: sp[1], reverse=True)

        return sl_next

    @staticmethod
    def extend_alpha(alpha, history):
        """NB. This modifies alpha in place"""

        if len(alpha['win']) < len(history) + 1:
            # extend the size of alpha vectors
            for x in 'win', 'fail', 'noter':
                alpha[x] += [0.] * NEW_ROWS
            n_old, _ = alpha['loop'].shape
            alpha_loop_new = np.zeros((n_old + NEW_ROWS, n_old + NEW_ROWS))
            alpha_loop_new[:n_old, :n_old] = alpha['loop']
            alpha['loop'] = alpha_loop_new
        else:
            pass

    @staticmethod
    def cumulate_alpha(alpha, history):
        """NB modifies alpha in place

        Furthermore, this function is idempotent: calling it twice has the same effect as calling it once."""

        assert not (len(alpha['win']) < len(history) + 1)

        if history == []:
            # nothing to do.
            return alpha

        # cumulate alpha values from last layer
        n = len(history) - 1

        p_this = history[-1].p

        for x in 'win', 'fail', 'noter':
            alpha[x][n] += p_this * alpha[x][n+1] / (1 - alpha['loop'][n, n])
            alpha[x][n+1] = 0.

        for k in range(n):
            alpha['loop'][k, n-1] += p_this * alpha['loop'][k, n] / (1 - alpha['loop'][n, n])
            alpha['loop'][k, n] = 0.

        alpha['loop'][n, n] = 0.

        alpha['loop'][len(history), :] = 0.
        alpha['loop'][:, len(history)] = 0.

        return alpha

    @staticmethod
    def calc_lambda(alpha, history, epsilon=1e-6):
        """NB. Modifies alpha in place, but in a way that keeps lambda unchanged"""
        # history[0] .. history[n]
        n = len(history) - 1
        likelihoods_loop = np.empty(n+1)
        likelihoods_loop[:] = np.nan

        likelihoods = {}
        for x in 'win', 'fail', 'noter':
            likelihoods[x] = alpha[x][n+1]

        for k in range(n, -1, -1):
            p_k = history[k].p

            likelihoods_loop[k] = alpha['loop'][k, n]
            for m in range(n-1, k-1, -1):  # m = n-1 .. k
                likelihoods_loop[k] *= history[m + 1].p / (1 - likelihoods_loop[m + 1])
                likelihoods_loop[k] += alpha['loop'][k, m]

            if likelihoods_loop[k] > 1. - epsilon:
                # in this case, the whole tree below k loops back to history[k], so
                # for every key in 'win', 'fail', 'noter', likelihoods[key] == 0.
                # And as likelihoods_loop[k] ~= 1 here,
                #  avoid division by zero.

                likelihoods_loop[k] = 0.

                # fix it for future calls too:
                alpha['noter'][k] += p_k
                alpha['loop'][k:n + 1, k:n + 1] = 0.
                assert np.all(alpha['loop'][:k, k:n + 1] == 0.)

                for key in 'win', 'fail', 'noter':
                    assert likelihoods[key] == 0.
                    likelihoods[key] = alpha[key][k]

            else:
                for key in 'win', 'fail', 'noter':
                    likelihoods[key] = alpha[key][k] + \
                                        p_k * likelihoods[key] / (1. - likelihoods_loop[k])

            assert np.all(0. <= likelihoods_loop[k])
            assert np.all(likelihoods_loop[k] <= 1.)

        for key in 'win', 'noter':
            assert 0. <= likelihoods[key] <= 1.
        assert 0. <= likelihoods['fail']

        return likelihoods

    def get_mealy_qa_iterator(self, c, s, obs):
        if self.env.is_goal_state(s):
            return [(0, A_STOP)]

        legal_acts = self.env.legal_actions(s) + [A_STOP]
        q_list = range(min(c.num_states + 1, c.bound))

        return [(q_next, action) for q_next in q_list
                                 for action in legal_acts]


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env',
                           choices=['BridgeWalk',
                                    'WalkThroughFlapProb',
                                    'ProbHallAone',
                                    'ProbHallArect'],
                           help='Environment to run')
    argparser.add_argument('--max-states',
                           type=int,
                           help='Maximum number of controller states')
    argparser.add_argument('--lgt-desired',
                           type=float,
                           default=0.9999,
                           help='LGT*, the minimum goal termination likelihood')
    argparser.add_argument('-v', '--verbose',
                           action='store_true',
                           help='Print all logging messages')
    argparser.add_argument('--log-info',
                           action='store_true',
                           help='Print almost all logging messages (level INFO and above)')
    argparser.add_argument('--no-timeit',
                           action='store_true',
                           help="Don't time the execution")
    argparser.add_argument('--timeit-repeat',
                           type=int,
                           default=1,
                           help='Number of repeats for timing.')

    argparser.add_argument('env_args', type=int, nargs='*')

    args = argparser.parse_args()

    env_cls = getattr(environments, args.env)
    env = env_cls(*args.env_args)

    return args, env


def main(args, env):
    planner = PAndOrPlanner(env)

    try:
        good_cont, good_alpha = planner.synth_plan(args.max_states,
                                                   lpc_desired=args.lgt_desired)

        time.sleep(PRINT_WAIT_SECONDS)  # Wait for mesages of logging module
        for (q, o), (q_next, a) in good_cont.transitions.items():
            print("({},{}) → ({},{})".format(q, env.str_obs(o), q_next, env.str_action(a)))

    except PandorControllerNotFound:
        time.sleep(PRINT_WAIT_SECONDS)  # Wait for mesages of logging module
        print("No controller found")

    print("Number of steps taken: {}".format(planner.num_steps))


if __name__ == '__main__':
    args, env = parse_args()

    if args.verbose:
        v = True
        logging.basicConfig(level=logging.DEBUG)
    elif args.log_info:
        v = False
        logging.basicConfig(level=logging.INFO)
    else:
        v = False

    logging.info(f'Command-line options:\n{args}\n')

    if args.no_timeit:
        main(args, env)
    else:
        seconds_taken = (timeit.timeit(lambda: main(args, env),
                                       number=args.timeit_repeat)
                         / args.timeit_repeat) - PRINT_WAIT_SECONDS
        print('\n{:f} seconds'.format(seconds_taken))
