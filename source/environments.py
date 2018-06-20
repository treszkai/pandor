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