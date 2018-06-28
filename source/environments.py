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


class NoisyEnv(Environment):
    def next_states_p(self, state, action):
        """
        Returns a list of possible next environment states and their likelihoods

        :rtype: list(state, likelihood)
        """
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
        return [(5, True), (1, False), (2, False)]

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


class WalkThroughFlap(Environment):
    """ Environment to test AND backtracking
    States: {1(goal), 2(init0), 3(init1), 4(goal)}
    Actions: Left/Right, but Left in 3 leaves you in 3.
    Obs: Goal or not
    """

    @property
    def init_states(self):
        return [2, 3]

    @property
    def goal_states(self):
        return [1, 4]

    def legal_actions(self, state):
        return [-1, +1]

    def get_obs(self, state):
        return state in self.goal_states

    def next_states(self, state, action):
        if action == -1:
            if state == 3:
                return [3]
            else:
                return [max(state + action, 1)]
        if action == 1:
            return [min(state + action, 4)]


class TreeChop(Environment):
    """ TreeChop problem from Levesque 2005 (slightly modified: observation actions removed)
    fluents: axe_stored ∈ {T, F},
    states:
    actions: store, chop, TODO
    observables: up = (thickness == 0)

    """

    @staticmethod
    def str_state(s):
        pass

    @staticmethod
    def str_action(a):
        pass

    @staticmethod
    def str_obs(o):
        pass

    @property
    def init_states(self):
        pass

    @property
    def goal_states(self):
        pass

    def legal_actions(self, state):
        pass

    def get_obs(self, state):
        pass

    # Note: if there are many actions
    def next_states(self, state, action):
        pass

