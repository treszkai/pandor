"""
AND-OR search algorithm for a finite-state controller in a probabilistic environment

Based on ./dandor.py.
(Originally based on Hu and De Giacomo: A Generic Technique for Synthesizing Bounded Finite-State Controllers (2013).)

   !! STILL IN DEVELOPMENT !!

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
        return len(set(k for k, _ in self.transitions.values())) or 1

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
            "Invalid controller state transition: %s → %s".format(key, value)

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


class PAndOrPlanner:
    def __init__(self, env):
        self.env = env
        # attributes used by synth_plan() - def'd here only to suppress warnings
        self.contr, self.backtracking, self.backtrack_stack = None, None, None
        # Lower/upper bound for the log LPC of the current controller
        self.lpc_lower_bound = None
        self.lpc_upper_bound = None
        self.lpc_desired = None

    def synth_plan(self, states_bound, lpc_desired):
        self.backtracking = False
        self.backtrack_stack = []
        self.contr = MealyController(states_bound)
        self.lpc_desired = lpc_desired
        self.lpc_lower_bound = -1000.
        self.lpc_upper_bound = 0.

        init_states = [(x, 0.0) for x in self.env.init_states]

        try:
            self.and_step(self.contr.init_state, init_states, [])
        except PandorControllerFound:
            return True

        return False

    def and_step(self, q, sl_next, history):
        """

        :param q:
        :param sl_next:
        :param history:
        :return:
            - FAILURE if self.lpc_max < self.lpc_desired
            - AND_UNKNOWN otherwise
        :raises:
            - PandorControllerFound if a controller is found
        """
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
                logging.info("AND: not fail at history %s", history) if v else 0
                return AND_UNKNOWN

            if self.backtracking:
                logging.info("AND: Redoing s: %s", self.env.str_state(s_k)) if v else 0
            else:
                logging.info("AND: Simulating s: %s, q: %s", self.env.str_state(s_k), q) if v else 0

            self.or_step(q, s_k, p_k, history[:])

            # TODO: epsilon
            if self.lpc_lower_bound >= self.lpc_desired:
                logging.info("AND: succeed at history %s", history) if v else 0
                raise PandorControllerFound
            elif self.lpc_upper_bound < self.lpc_desired:
                logging.info("AND: fail at history %s", history) if v else 0
                self.backtracking = True

                if len(self.backtrack_stack) == 0:
                    logging.info("AND: Backtracking up; empty stack") if v else 0
                    return AND_FAILURE

                # decide if we should backtrack left or up
                # (ignore the last element of self.backtrack_stack[-1].history)
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
        if not self.backtracking:
            l_old = 0. if len(history) == 0 else history[-1].l

            if s in self.env.goal_states:
                self.lpc_upper_bound += history[-1].l + p
                return

            if HistoryItem(q, s, 99) in history:
                self.lpc_lower_bound -= history[-1].l + p
                return

            l = l_old + p

            history.append(HistoryItem(q, s, l))
            obs = self.env.get_obs(s)

            if (q, obs) in self.contr.transitions:
                q_next, action = self.contr[q, obs]
                if action not in self.env.legal_actions(s):
                    self.lpc_lower_bound -= history[-1].l + p
                    return

                sl_next = self.env.next_states_p(s, action)
                self.and_step(q_next, sl_next, history)
                return

            # no (q_next,act) defined for (q,obs) ⇒ define new one with this iterator
            it = product(range(min(self.contr.bound, self.contr.num_states + 1)),
                         self.env.legal_actions(s))

            # store a new checkpoint iff we're not backtracking currently
            self.backtrack_stack.append(StackItem(history[:],
                                                  self.lpc_lower_bound,
                                                  self.lpc_upper_bound))
            logging.info("OR: checkpoint at q: %s, s: %s\n    with history %s",
                         q, self.env.str_state(s), history) if v else 0

        else:  # backtracking
            l = history[-1].l + p
            history.append(HistoryItem(q, s, l))
            obs = self.env.get_obs(s)

            t = self.contr.transitions.popitem()
            logging.info("OR: (redoing) Deleted: (%s,%s) -> (%s,%s)",
                         t[0][0], self.env.str_obs(t[0][1]),
                         t[1][0], self.env.str_action(t[1][1])) if v else 0

            q_next_last, action_last = t[1]

            it = dropwhile(lambda x: x[1] != action_last,
                           product(range(q_next_last, min(self.contr.bound, self.contr.num_states + 1)),
                                   self.env.legal_actions(s)))

            # this is the node of the last checkpoint
            # (enough to check the length because we're in the right branch now)
            if len(history) == len(self.backtrack_stack[-1].history):
                self.backtracking = False
                # burn the controller extension that caused the trouble earlier
                _ = next(it)

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

            sl_next = self.env.next_states_p(s, action)

            if self.and_step(q_next, sl_next, history) != AND_FAILURE:
                # If we're here then self.backtracking is already False
                #   (if we came from up, then we cleared it;
                #    if we came from left, then the above call just succeeded
                assert not self.backtracking
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
                             [(q, self.env.str_state(s), l) for q, s, l in history[-1:len_history-1:-1]]) if v else 0
                # it's enough to clip the history if we're backtracking in the or_step
                del history[len_history:]

                t = self.contr.transitions.popitem()
                logging.info("OR: Deleted: (%s,%s) -> (%s,%s)",
                             t[0][0], self.env.str_obs(t[0][1]),
                             t[1][0], self.env.str_action(t[1][1])) if v else 0

        self.backtrack_stack.pop()
        # TODO: log scale!
        self.lpc_lower_bound -= history[-1].l
        return


if __name__ == '__main__':
    if v:
        logging.basicConfig(level=logging.INFO)

    planner = PAndOrPlanner(env=environments.Climber())
    planner.synth_plan(states_bound=1, lpc_desired=0.0)

    time.sleep(1)
    print(planner.contr)
