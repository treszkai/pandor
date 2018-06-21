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
from itertools import dropwhile
from itertools import product


# verbose flag
v = True


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

        if q_next == self.bound:
            raise PandorControllerNotFound("Too many controller states")

        self.transitions[key] = value

    def __str__(self):
        n = self.num_states
        s = f"States: {n}\n"
        for i in sorted(self.transitions.items(), key=lambda x: x[0][0] * n + x[1][0]):
            s += f"{i}\n"
        return s


class PAndOrPlanner:
    def __init__(self, env):
        self.env = env
        # attributes used by synth_plan() - def'd here only to suppress warnings
        self.contr, self.backtracking, self.backtrack_stack = None, None, None

    def synth_plan(self, bound):
        self.backtracking = False
        self.backtrack_stack = []
        self.contr = MealyController(bound)

        self.and_step(self.contr.init_state, self.env.init_states, [])

    def and_step(self, q, sl_next, history):

        def get_backtracked_iterator():
            # don't care about or_steps that succeeded and to which we don't want to backtrack
            # history == self.backtrack_stack[-1][0:len(history)]
            # so the relevant element of sl_next is
            #  self.backtrack_stack[-1][len(history)]
            return dropwhile(lambda x: x[0] != self.backtrack_stack[-1][len(history)][1],
                             iter(sl_next))

        if self.backtracking:
            it = get_backtracked_iterator()
        else:
            it = iter(sl_next)

        while True:
            try:
                # Note: tp = transition probability
                s_k, p_k = next(it)
            except StopIteration:
                logging.info("AND: succeed at history %s", history) if v else 0
                return True

            if self.backtracking:
                logging.info("AND: Redoing s: %s", self.env.str_state(s_k)) if v else 0
            else:
                logging.info("AND: Simulating s: %s, q: %s", self.env.str_state(s_k), q) if v else 0

            if not self.or_step(q, s_k, p_k, history[:]):
                self.backtracking = True
                # decide if we should backtrack left or up
                # (ignore the last element of self.backtrack_stack[-1])
                if history == self.backtrack_stack[-1][:min(len(history), len(self.backtrack_stack[-1])-1)]:
                    it = get_backtracked_iterator()
                    logging.info("AND: Backtracking left") if v else 0
                else:
                    logging.info("AND: Backtracking up") if v else 0
                    return False
                # We set self.backtracking = False in or_step
                #   when we start doing business as usual:
                #   i.e. either when we arrive in an OR node from up
                #     or when we arrive in it from the left

    def or_step(self, q, s, p, history):
        if not self.backtracking:
            if s in self.env.goal_states:
                # TODO - increase self.max_lpc, return (note the log scale)
                return True

            if (q, s, _) in history:
                # TODO - decrease self.min_lpc, return
                return False

            l = history[-1][2] + p

            history.append((q, s, l))
            obs = self.env.get_obs(s)

            if (q, obs) in self.contr.transitions:
                q_next, action = self.contr[q, obs]
                if action not in self.env.legal_actions(s):
                    # TODO - decrease self.min_lpc, return
                    return False

                sl_next = self.env.next_states(s, action)
                # TODO - leave this as it is
                return self.and_step(q_next, sl_next, history)

            # no (q_next,act) defined for (q,obs) ⇒ define new one with this iterator
            it = product(range(self.contr.num_states+1),
                         self.env.legal_actions(s))

            # store a new checkpoint iff we're not backtracking currently
            self.backtrack_stack.append(history[:])
            logging.info("OR: checkpoint at q: %s, s: %s\n    with history %s",
                         q, self.env.str_state(s), history) if v else 0

        else:  # backtracking
            l = history[-1][2] + p
            history.append((q, s, l))
            obs = self.env.get_obs(s)

            t = self.contr.transitions.popitem()
            logging.info("OR: (redoing) Deleted: (%s,%s) -> (%s,%s)",
                         t[0][0], self.env.str_obs(t[0][1]),
                         t[1][0], self.env.str_action(t[1][1])) if v else 0

            q_next_last, action_last = t[1]

            it = dropwhile(lambda x: x[1] != action_last,
                           product(range(q_next_last, self.contr.num_states + 1),
                                   self.env.legal_actions(s)))

            # this is the node of the last checkpoint
            # (enough to check the length because we're in the right branch now)
            if len(history) == len(self.backtrack_stack[-1]):
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

            sl_next = self.env.next_states(s, action)

            if self.and_step(q_next, sl_next, history):
                # If we're here then self.backtracking is already False
                #   (if we came from up, then we cleared it;
                #    if we came from left, then the above call just succeeded
                assert not self.backtracking
                return True
            else:
                # set backtracking to False: either there are more AND branches to try,
                #   or we'll set it to true in the and_step above.
                self.backtracking = False

                # restore saved values
                q, s = q_saved, s_saved
                len_history = len(self.backtrack_stack[-1])

                logging.info("OR: Backstep: %s",
                             [(q, self.env.str_state(s)) for q, s in history[-1:len_history-1:-1]]) if v else 0
                # it's enough to clip the history if we're backtracking in the or_step
                del history[len_history:]

                t = self.contr.transitions.popitem()
                logging.info("OR: Deleted: (%s,%s) -> (%s,%s)",
                             t[0][0], self.env.str_obs(t[0][1]),
                             t[1][0], self.env.str_action(t[1][1])) if v else 0

        # TODO - decreas min lpc and return
        return False


if __name__ == '__main__':
    if v:
        logging.basicConfig(level=logging.INFO)

    planner = PAndOrPlanner(env=environments.WalkAB())
    planner.synth_plan(bound=2)

    time.sleep(1)
    print(planner.contr)
