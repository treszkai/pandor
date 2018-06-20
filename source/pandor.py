import logging
from collections import OrderedDict

import time
from itertools import dropwhile
from itertools import product


v = True


class PandorControllerNotFound(Exception):
    pass


class Environment:
    """ Class that describes the environment but does not simulate it """

    @property
    def init_states(self):
        raise NotImplementedError

    @property
    def goal_states(self):
        raise NotImplementedError

    def legal_actions(self, state):
        raise NotImplementedError

    def get_obs(self, state):
        raise NotImplementedError

    def next_states(self, state, action):
        """
        Returns a list of possible next environment states
        :rtype: list
        """
        raise NotImplementedError

    @staticmethod
    def str_state(s):
        return s

    @staticmethod
    def str_action(a):
        return a

    @staticmethod
    def str_obs(o):
        return o


class WalkAB(Environment):
    """ Environment of Fig. 1 of BPG2009
    States: {(n, visB): n ∈ {1,2,3,4,5}, visB ∈ {True, False} }
    Action set: {-1, +1} = {left, right}
    Action effects:
        - left( (n,visB) )  = ( max(n-1, 1), visB )
        - right( (n,visB) ) = ( min(n+1, 5), visB )
    Observables: (A, B) = (n == 1, n == 5)
    Init states: { (1, False), (2, False) }
    Goal states: { (1, True) }
    """

    @staticmethod
    def str_state(s):
        return "({}, {})".format(s[0], 'T' if s[1] else 'F')

    @staticmethod
    def str_action(a):
        if a == -1:
            return "Left"
        elif a == 1:
            return "Right"
        else:
            assert False, f"Nonsense action: {a}"

    @staticmethod
    def str_obs(o):
        return "({}, {})".format(" A" if o[0] else "¬A",
                                 " B" if o[1] else "¬B")

    @property
    def init_states(self):
        return [(1, False), (2, False)]

    @property
    def goal_states(self):
        return [(1, True)]

    def legal_actions(self, state):
        return [-1, +1]

    def get_obs(self, state):
        n = state[0]
        return n == 1, n == 5

    # Note: if there are many actions
    def next_states(self, state, action):
        n, vis_b = state
        vis_b |= action == +1 and n == 4
        n_ = n + action
        if n_ < 1:
            n = 1
        elif n_ > 5:
            n = 5
        else:
            n = n_
        return [(n, vis_b)]


class PrizeA:
    """ Prize-A environment, from Bonet2009
    States: {}
    Action set: {(0,1), (0,-1), (1,0), (-1,0)} = {right, left, up, down}
    """
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


class AndOrPlanner:
    def __init__(self, env):
        self.env = env
        # attributes used by synth_plan() - def'd here only to suppress warnings
        self.contr, self.backtracking, self.backtrack_stack = None, None, None

    def synth_plan(self, bound):
        self.backtracking = False
        self.backtrack_stack = []
        self.contr = MealyController(bound)
        # Note: this creates a controller first for one init state,
        #       and if it fails for another then backtracks it transition by transition.
        #       Is there a better method?
        return self.and_step(self.contr.init_state, self.env.init_states, [])

    # Note: history and controller could be moved from function arguments to class properties
    #       because both of them were just passed through and_step unmodified
    #       and there is a single instance of each
    #       (actually the whole and_step function could be removed)
    #       (maybe it's useful for the other two uses of the planner in Hu2013)
    #       (the and_step didn't even make the recursion step clearer)
    #       (and_step is needed for synth_plan though)
    def and_step(self, q, sl_next, history):

        def get_backtracked_iterator():
            # don't care about or_steps that succeeded and to which we don't want to backtrack
            # history == self.backtrack_stack[-1][0:len(history)]
            # so the relevant element of sl_next is
            #  self.backtrack_stack[-1][len(history)]
            return dropwhile(lambda x: x != self.backtrack_stack[-1][len(history)],
                             iter(sl_next))

        if self.backtracking:
            it = get_backtracked_iterator()
        else:
            it = iter(sl_next)

        while True:
            try:
                s_k = next(it)
            except StopIteration:
                logging.info("AND: succeed at history %s", history) if v else 0
                return True

            if self.backtracking:
                logging.info("AND: Redoing s: %s", self.env.str_state(s_k)) if v else 0
            else:
                logging.info("AND: Simulating s: %s, q: %s", self.env.str_state(s_k), q) if v else 0

            # Note: only need to copy the history if len(sl_next) > 1. I guess.
            #       Or alternatively, could save len(history) and clip it after each new or_step
            #       (like we do in or_step already after backtracking)
            if not self.or_step(q, s_k, history[:]):
                self.backtracking = True
                # decide if we should backtrack left or up
                # (ignore the last element of self.backtrack_stack[-1])
                # Note: would be enough to compare only the second entry of histories
                if history == self.backtrack_stack[-1][:min(len(history), len(self.backtrack_stack[-1])-1)]:
                    it = get_backtracked_iterator()
                    logging.info("AND: Backtracking left") if v else 0
                else:
                    logging.info("AND: Backtracking up") if v else 0
                    return False
                # Note: We set self.backtracking = True in and_step
                #   when an or_step failed and we start looking for the last checkpoint
                # We set self.backtracking = False in or_step
                #   when we start doing business as usual:
                #   i.e. either when we arrive in an OR node from up
                #     or when we arrive in it from the left

    def or_step(self, q, env_state, history):
        if not self.backtracking:
            if env_state in self.env.goal_states:
                return True

            if (q, env_state) in history:
                return False

            history.append((q, env_state))
            obs = self.env.get_obs(env_state)

            if (q, obs) in self.contr.transitions:
                q_next, action = self.contr[q, obs]
                if action not in self.env.legal_actions(env_state):
                    return False

                sl_next = self.env.next_states(env_state, action)
                return self.and_step(q_next, sl_next, history)

            # no (q_next,act) defined for (q,obs) ⇒ define new one with this iterator
            it = product(range(self.contr.num_states+1),
                         self.env.legal_actions(env_state))

            # store a new checkpoint iff we're not backtracking currently
            self.backtrack_stack.append(history[:])
            logging.info("OR: checkpoint at q: %s, s: %s\n    with history %s",
                         q, self.env.str_state(env_state), history) if v else 0

        else:  # backtracking
            history.append((q, env_state))
            obs = self.env.get_obs(env_state)

            t = self.contr.transitions.popitem()
            logging.info("OR: (redoing) Deleted: (%s,%s) -> (%s,%s)",
                         t[0][0], self.env.str_obs(t[0][1]),
                         t[1][0], self.env.str_action(t[1][1])) if v else 0

            q_next_last = t[1][0]
            action_last = t[1][1]

            it = dropwhile(lambda x: x[1] != action_last,
                           product(range(q_next_last, self.contr.num_states + 1),
                                   self.env.legal_actions(env_state)))

            # this is the node of the last checkpoint
            # (enough to check the length because we're in the right branch now)
            if len(history) == len(self.backtrack_stack[-1][history]):
                self.backtracking = False
                # burn the controller extension that caused the trouble earlier
                _ = next(it)

        # non-det branching of q',a

        # save function arguments for bracktracking (history is already in backtrack_stack)
        # Note: we have to make a shallow copy of the history b/c we modify it during recursion
        # Note: as the state transitions of the controller are ordered according to their
        #       being added, then we needn't store it in backtrack_stack.

        env_state_saved = env_state
        q_saved = q

        # important: q_next first, so we only add new states when necessary
        for q_next, action in it:
            # extend controller if not backtracking
            if not self.backtracking:
                self.contr[q, obs] = q_next, action
                logging.info("OR: Added:   (%s,%s) -> (%s,%s)",
                             q, self.env.str_obs(obs),
                             q_next, self.env.str_action(action)) if v else 0

            sl_next = self.env.next_states(env_state, action)

            if self.and_step(q_next, sl_next, history):
                # If we're here then self.backtracking is already False
                #   (if we came from up, then we cleared it;
                #    if we came from left, then the above call just succeeded
                assert not self.backtracking
                return True
            else:
                # Note: self.backtracking = True should be only when backtracking up and left, not when trying new branches of an OR node
                # set backtracking to False: either there are more AND branches to try, or we'll set it to true in the and_step above.
                self.backtracking = False

                # restore saved values
                q, env_state = q_saved, env_state_saved
                len_history = len(self.backtrack_stack[-1])

                logging.info("OR: Backstep: %s",
                             [(q, self.env.str_state(s)) for q, s in history[-1:len_history-1:-1]]) if v else 0
                # it's enough to clip the history if we're backtracking in the or_step
                del history[len_history:]

                t = self.contr.transitions.popitem()
                logging.info("OR: Deleted: (%s,%s) -> (%s,%s)",
                             t[0][0], self.env.str_obs(t[0][1]),
                             t[1][0], self.env.str_action(t[1][1])) if v else 0

        return False


if __name__ == '__main__':
    if v:
        logging.basicConfig(level=logging.INFO)

    planner = AndOrPlanner(env=WalkAB())
    planner.synth_plan(bound=2)

    time.sleep(1)
    print(planner.contr)
