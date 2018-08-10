"""
AND-OR search algorithm for a finite-state controller in a probabilistic environment

Based on ./dandor.py.
(Originally based on Hu and De Giacomo: A Generic Technique for Synthesizing Bounded Finite-State Controllers (2013).)

"""

import environments

import logging
from collections import OrderedDict

import time
from itertools import dropwhile, product


AND_FAILURE = -1
AND_UNKNOWN = 0

# verbose flag
v = True


class PandorControllerFound(Exception):
    pass


class PandorControllerNotFound(Exception):
    pass


class MealyController:
    """ An N-bounded Mealy machine
    States: 0, 1, 2, ... k-1  where k <= N
    Observations
    """
    def __init__(self, bound):
        self.bound = bound
        self.transitions = OrderedDict()

    @property
    def num_states(self):
        """ Counts the number of states already defined
        (returns 1 for the empty controller)
        """

        if len(self.transitions) == 0:
            return 1
        else:
            return max(q for q, _ in self.transitions.values()) + 1

    @property
    def init_state(self):
        return 0

    def __getitem__(self, item):
        """
        :type item: tuple of (contr_state, observation)
        """
        return self.transitions[item]

    def __setitem__(self, key, value):
        q, obs = key
        q_next, act = value

        assert q < self.num_states and q_next <= self.num_states, \
            "Invalid controller state transition: {} → {}".format(key, value)

        if not q_next < self.bound:
            assert q_next < self.bound

        self.transitions[key] = value

    def __str__(self):
        n = self.num_states
        s = f"States: {n}\n"
        for i in sorted(self.transitions.items(), key=lambda x: x[0][0] * n + x[1][0]):
            s += f"{i}\n"
        return s


class HistoryItem:
    def __init__(self, q, s, l):
        self.q = q
        self.s = s
        self.l = l

    def __str__(self):
        return f"(q: {self.q}, s: {self.s}, l: {self.l:0.3f})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        """ self.l == 99 is a wildcard"""
        return self.q == other.q and \
               self.s == other.s and \
               (self.l == 99 or self.l == other.l)


class StackItem:
    def __init__(self, history, lpc_lower, lpc_upper):
        self.history = history
        self.lpc_lower = lpc_lower
        self.lpc_upper = lpc_upper

    def __eq__(self, other):
        return self.history == other.history and \
               self.lpc_lower == other.lpc_lower and \
               self.lpc_upper == other.lpc_upper


def set_test_controller(cont):
    env = environments.BridgeWalk()
    cont[0, False] = [1, env.A_LEFT]
    cont[1, False] = [1, env.A_FWD]
    cont[1, True] = [1, env.A_RIGHT]


class PAndOrPlanner:
    def __init__(self, env):
        self.env = env
        # attributes used by synth_plan() - def'd here only to suppress warnings
        self.contr, self.backtracking, self.backtrack_stack = None, None, None
        # Lower/upper bound for the LPC of the current controller
        self.lpc_lower_bound = None
        self.lpc_upper_bound = None
        self.lpc_desired = None

    def synth_plan(self, states_bound, lpc_desired):
        self.backtracking = False
        self.backtrack_stack = []
        self.contr = MealyController(states_bound)
        self.lpc_desired = lpc_desired
        self.lpc_lower_bound = 0.
        self.lpc_upper_bound = 1.

        # set_test_controller(self.contr)

        # Note: for numerical stability, lpc_desired must be lower than 1.
        assert lpc_desired < 1.0

        try:
            self.and_step(self.contr.init_state, self.env.init_states_p, [], first_and_step=True)
        except PandorControllerFound:
            print("Controller found with max ", states_bound, "states.")
            return True
        except PandorControllerNotFound:
            print("No controller found with max ", states_bound, "states.")
            return False

        assert False, "This should not happen."  # len(self.backtrack_stack) > 0 ?

    def and_step(self, q, action, history, first_and_step=False):
        """
        :return:
            - FAILURE if self.lpc_max < self.lpc_desired
            - AND_UNKNOWN otherwise
        :raises:
            - PandorControllerFound if a sufficiently correct controller is found
        """

        if first_and_step:
            sl_next = self.env.init_states_p
        else:
            sl_next = self.env.next_states_p(history[-1].s, action)

        sl_next.sort(key=lambda sp: sp[1], reverse=True)

        def get_backtracked_iterator():
            # don't care about or_steps that succeeded and to which we don't want to backtrack
            # history == self.backtrack_stack[-1].history[0:len(history)]
            # so the relevant element of sl_next is
            #  self.backtrack_stack[-1].history[len(history)]
            return dropwhile(lambda x: x[0] != self.backtrack_stack[-1].history[len(history)].s,
                             iter(sl_next))

        if self.backtracking:
            it = get_backtracked_iterator()
        else:
            it = iter(sl_next)

        while True:
            try:
                # Note: p = transition probability
                s_k, p_k = next(it)
            except StopIteration:
                # this cannot be the topmost AND step, because the early termination returns sooner
                assert len(history) > 0

                logging.info("AND: not fail at history %s", history) if v else 0

                return AND_UNKNOWN

            if self.backtracking:
                logging.info("AND: Redoing s: %s", self.env.str_state(s_k)) if v else 0
            else:
                logging.info("AND: Simulating s: %s, q: %s", self.env.str_state(s_k), q) if v else 0

            self.or_step(q, s_k, p_k, history[:])

            if self.lpc_lower_bound >= self.lpc_desired:
                logging.info("AND: succeed at history %s", history) if v else 0
                raise PandorControllerFound
            elif self.lpc_upper_bound < self.lpc_desired:
                logging.info("AND: fail at history %s", history) if v else 0
                self.backtracking = True

                if len(self.backtrack_stack) == 0:
                    logging.info("AND: Trying to backtrack but empty stack; fail.") if v else 0
                    raise PandorControllerNotFound

                # decide if we should backtrack left or up
                # (ignore the last element of self.backtrack_stack[-1].history)
                # Note: could make it iterative `is` instead of equality
                if history == self.backtrack_stack[-1].history[:min(len(history),
                                                                    len(self.backtrack_stack[-1].history)-1)]:
                    it = get_backtracked_iterator()
                    logging.info("AND: Backtracking left") if v else 0
                else:
                    logging.info("AND: Backtracking up") if v else 0
                    return AND_FAILURE
                # We set self.backtracking = False in or_step
                #   when we start doing business as usual:
                #   i.e. either when we arrive in an OR node from up
                #     or when we arrive in it from the left

    def or_step(self, q, s, p, history):
        """
        :param p: probability of next state transition
        :return: None
        """
        l = (history[-1].l * p) if len(history) > 0 else p

        if s in self.env.goal_states:
            self.lpc_lower_bound += l
            logging.info("OR: in goal state, new lower bound: %0.3f", self.lpc_lower_bound) if v else 0
            return

        if HistoryItem(q, s, 99) in history:
            self.lpc_upper_bound -= l
            logging.info("OR: repeated state, new upper bound: %0.3f", self.lpc_upper_bound) if v else 0
            return

        if not self.backtracking:
            history.append(HistoryItem(q, s, l))
            obs = self.env.get_obs(s)

            if (q, obs) in self.contr.transitions:
                q_next, action = self.contr[q, obs]
                if action not in self.env.legal_actions(s):
                    self.lpc_upper_bound -= l
                    return

                self.and_step(q_next, action, history)
                return

            # no (q_next,act) defined for (q,obs) ⇒ define new one with this iterator
            it = self.get_mealy_qa_iterator(s)

            # store a new checkpoint iff we're not backtracking currently
            self.backtrack_stack.append(StackItem(history[:],
                                                  self.lpc_lower_bound,
                                                  self.lpc_upper_bound))
            logging.info("OR: checkpoint at q: %s, s: %s\n    with history %s",
                         q, self.env.str_state(s), history) if v else 0

        else:  # backtracking
            history.append(HistoryItem(q, s, l))
            obs = self.env.get_obs(s)

            # this is the node of the last checkpoint
            # (enough to check the length because we're in the right branch now)
            if len(history) == len(self.backtrack_stack[-1].history):
                self.backtracking = False

                t = self.contr.transitions.popitem()
                assert (q, obs) == t[0]

                q_next_last, action_last = t[1]

                logging.info("OR: (redoing) Deleted: (%s,%s) -> (%s,%s)",
                             q, self.env.str_obs(obs),
                             q_next_last, self.env.str_action(action_last)) if v else 0

                it = self.get_mealy_qa_iterator(s,
                                                q_next_last,
                                                lambda x: x[1] != action_last)

                # burn the controller extension that caused the trouble earlier
                _ = next(it)
            else:
                # the controller transition and action should already be defined
                assert (q, obs) in self.contr.transitions
                q_next_last, action_last = self.contr[q, obs]
                assert action_last in self.env.legal_actions(s)

                logging.info("OR: redoing: (%s,%s) -> (%s,%s)",
                             q, self.env.str_obs(obs),
                             q_next_last, self.env.str_action(action_last)) if v else 0

                # same as the iterator above
                it = self.get_mealy_qa_iterator(s,
                                                q_next_last,
                                                lambda x: x[1] != action_last)

        # non-det branching of q',a
        # save function arguments for bracktracking (history is already in backtrack_stack)
        s_saved = s
        q_saved = q

        # important: q_next first, so we only add new states when necessary
        for q_next, action in it:
            # extend controller if not backtracking
            if not self.backtracking:
                self.contr[q, obs] = q_next, action
                logging.info("OR: Added:   (%s,%s) -> (%s,%s)",
                             q, self.env.str_obs(obs),
                             q_next, self.env.str_action(action)) if v else 0

            if self.and_step(q_next, action, history) != AND_FAILURE:
                # If we're here then self.backtracking is already False
                #   (if we came from up, then we cleared it;
                #    if we came from left, then the above call just succeeded
                assert not self.backtracking
                logging.info("OR: AND step didn't fail") if v else 0
                return
            else:
                # set backtracking to False: either there are more AND branches to try,
                #   or we'll set it to true in the and_step above.
                self.backtracking = False

                # restore saved values
                q, s = q_saved, s_saved
                len_history = len(self.backtrack_stack[-1].history)
                self.lpc_upper_bound = self.backtrack_stack[-1].lpc_upper
                self.lpc_lower_bound = self.backtrack_stack[-1].lpc_lower

                logging.info("OR: Backstep: %s",
                             [ (x.q, self.env.str_state(x.s), x.l) for x in history[len_history:] ]) if v else 0
                # it's enough to clip the history if we're backtracking in the or_step
                del history[len_history:]

                t = self.contr.transitions.popitem()
                logging.info("OR: Deleted: (%s,%s) -> (%s,%s)",
                             t[0][0], self.env.str_obs(t[0][1]),
                             t[1][0], self.env.str_action(t[1][1])) if v else 0

        self.backtrack_stack.pop()
        self.lpc_upper_bound -= history[-1].l
        logging.info("OR: all extensions failed, new upper bound: %0.3f", self.lpc_upper_bound) if v else 0
        return

    def get_mealy_qa_iterator(self, s, q_next_last=0, drop_func=lambda x: False):
        legal_acts = self.env.legal_actions(s)
        it = dropwhile(drop_func,
                       product(range(q_next_last, min(self.contr.bound, self.contr.num_states + 1)),
                               legal_acts))

        return it


if __name__ == '__main__':
    if v:
        logging.basicConfig(level=logging.INFO)

    env = environments.BridgeWalk()

    planner = PAndOrPlanner(env)
    success = planner.synth_plan(states_bound=2, lpc_desired=0.95)

    time.sleep(1)  # Wait for mesages of logging module
    if success:
        print(planner.contr)
