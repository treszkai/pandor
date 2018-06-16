import logging
from collections import OrderedDict

import time


class PandorBacktrackException(Exception):
    pass


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
        if a == -1:  return "Left"
        elif a == 1: return "Right"
        else: assert False, f"Nonsense action: {a}"

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

    def synth_plan(self, bound):
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
        for s in sl_next:
            logging.info("Simulating s:%s, q:%s", self.env.str_state(s), q)
            self.or_step(q, s, history)

    def or_step(self, q, env_state, history):
        if env_state in self.env.goal_states:
            return

        if (q, env_state) in history:
            raise PandorBacktrackException

        history.append((q, env_state))
        obs = self.env.get_obs(env_state)

        if (q, obs) in self.contr.transitions:
            q_next, action = self.contr[q, obs]
            if action not in self.env.legal_actions(env_state):
                raise PandorBacktrackException
            sl_next = self.env.next_states(env_state, action)
            self.and_step(q_next, sl_next, history)
            return

        else:  # no (q_next,act) defined for (q,obs) ⇒ define new one
            # non-det branching of q',a

            # save function arguments for bracktracking
            # Note: we don't have to make a (full) checkpoint when either
            #       there's only one legal action or we're adding a new state - but who cares
            # Note: we have to make a shallow (?) copy of the arguments that we modify during recursion
            #       i.e. controller and history
            # Note: for memory efficiency, can store len(history) instead of history
            # Note: if the state transitions of the controller were ordered according to their
            #       being added, then we needn't make a copy of it either.
            # Note: memory - env_state need not be saved either, I guess
            self.backtrack_stack.append((q, env_state, len(history)))
            logging.debug("%s\n", self.backtrack_stack)

            # important: q_next first, so we only add new states when necessary
            for q_next in range(self.contr.num_states+1):
                for action in self.env.legal_actions(env_state):
                    # extend controller
                    self.contr[q, obs] = q_next, action
                    logging.info("Added:   (%s,%s) -> (%s,%s)",
                                 q, self.env.str_obs(obs),
                                 q_next, self.env.str_action(action))

                    sl_next = self.env.next_states(env_state, action)

                    try:
                        self.and_step(q_next, sl_next, history)
                        return
                    except PandorBacktrackException:
                        q, env_state, len_history = self.backtrack_stack[-1]
                        logging.info("Backstep: %s",
                                     [(q,self.env.str_state(s))
                                            for q,s in history[-1:len_history-1:-1]])
                        history = history[:len_history]

                        # TODO XXX As a result of the above line, some function calls will see a different list than others do
                        # (And in fact, this history clipping doesn't carry through AND steps.)
                        # Use a single history, common to the Planner object
                        # Is it correct if in the AND_step we save len_history and clip the history after the or_step succeeds?
                        # What if I use
                        #   del history[len_history:] ?
                        # Same result, just no new object.
                        # And do the history clipping in AND.
                        # Why did it work in Prolog? I guess it passes arguments by value. Why am I messing with this again?
                        # But even if I did that, this algo wouldn't backtrack over AND nodes – why?

                        t = self.contr.transitions.popitem()
                        logging.info("Deleted: (%s,%s) -> (%s,%s)",
                                     t[0][0], self.env.str_obs(t[0][1]),
                                     t[1][0], self.env.str_action(t[1][1]))

            assert False, "Are we ever here?"



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    planner = AndOrPlanner(env=WalkAB())
    planner.synth_plan(bound=2)

    time.sleep(1)
    print(planner.contr)