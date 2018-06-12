from collections import OrderedDict


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
        raise NotImplementedError


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

    def init_states(self):
        return [(1, False), (2, False)]

    def goal_states(self):
        return [(1, True)]

    def legal_actions(self, state):
        return [-1, +1]

    def get_obs(self, state):
        n = state[0]
        return n == 1, n == 5

    # Note: if there are many actions
    def next_states(self, state, action):
        """
        Returns a list of possible next environment states
        :rtype: list
        """
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


class Controller:
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


class AndOrPlanner:
    def __init__(self, env):
        self.env = env

    def synth_plan(self, bound):
        self.bound = bound
        self.backtrack_stack = []
        self.contr = Controller(bound)
        return self.and_step(self.contr.init_state, self.env.init_states, [])

    # Note: history and controller could be moved from function arguments to class properties
    #       because both of them were just passed through and_step unmodified
    #       and there is a single instance of each
    #       (actually the whole and_step function could be removed)
    #       (maybe it's useful for the other two uses of the planner in Hu2013)
    def and_step(self, q, sl_next, history):
        for s in sl_next:
            self.or_step(q, s, history)

    def or_step(self, q, env_state, history):
        if env_state in self.env.goal_states:
            return

        if (q, env_state) in history:
            raise PandorBacktrackException

        history.append((q, env_state))
        obs = self.env.get_obs(env_state)

        try:
            q_next, action = self.contr[q, obs]
            if action not in self.env.legal_actions(env_state):
                raise PandorBacktrackException
            sl_next = self.env.next_states(env_state, action)
            self.and_step(q_next, sl_next, history)

        except KeyError:  # no (q_next,act) defined for (q,obs) ⇒ define new one
            # non-det branching of q',a

            # save function arguments for bracktracking
            # Note: we don't have to make a (full) checkpoint when either
            #       there's only one legal action or we're adding a new state - but who cares
            # Note: we have to make a shallow (?) copy of the arguments that we modify during recursion
            #       i.e. controller and history
            # Note: for memory efficiency, can store len(history) instead of history
            # Note: if the state transitions of the controller were ordered according to their
            #       being added, then we needn't make a copy of it either.
            self.backtrack_stack.append((q, env_state, len(history)))

            # important: q_next first, so we only add new states when necessary
            for q_next in range(self.contr.num_states+1):
                for action in self.env.legal_actions(env_state):
                    # extend controller
                    self.contr[q, obs] = q_next, action

                    sl_next = self.env.next_states(env_state, action)

                    try:
                        self.and_step(q_next, sl_next, history)
                    except PandorBacktrackException:
                        q, env_state, len_history = self.backtrack_stack[-1]
                        history = history[:len_history]
                        self.contr.transitions.popitem()


if __name__ == '__main__':
    planner = AndOrPlanner(env=WalkAB())
    planner.synth_plan(bound=2)
