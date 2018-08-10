class Environment:
    """ Class that describes the environment but does not simulate it """

    def __init__(self):
        assert type(self.init_states) is list
        s = self.init_states[0]
        assert type(self.goal_states) is list
        assert type(self.legal_actions(s)) is list
        a = self.legal_actions(s)[0]
        assert type(self.next_states(s, a)) is list

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
    def __init__(self):
        Environment.__init__(self)
        s = self.init_states[0]
        a = self.legal_actions(s)[0]
        assert type(self.next_states_p(s, a))is list
        assert all(type(p) is float for _,p in self.next_states_p(s, a))

    @property
    def init_states_p(self):
        """Initial belief distribution
        A list of states and their probabilities
        Either init_states_p() or init_states() must be overwritten.
        """
        sl_0 = self.init_states
        p_0 = 1. / len(sl_0)
        return [(s_0, p_0) for s_0 in self.init_states]

    @property
    def init_states(self):
        return [s_0 for s_0, p_0 in self.init_states_p]

    def next_states_p(self, state, action):
        """
        Returns a list of possible next environment states and their transition probabilities

        :rtype: list(state, probability)
        """
        raise NotImplementedError

    def next_states(self, state, action):
        return [s_next for s_next, _ in self.next_states_p(state, action)]


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
    States: {0(goal), 1(init0), 2(init1), 3(goal)}
    Actions: Left/Right, but Left in 2 leaves you in 2.
    Obs: Goal or not
    """

    @property
    def init_states(self):
        return [1,2]

    @property
    def goal_states(self):
        return [0,3]

    def legal_actions(self, state):
        return [-1, +1]

    def get_obs(self, state):
        return state in self.goal_states

    def next_states(self, state, action):
        if action == -1:
            if state == 2:
                return [2]
            else:
                return [max(state + action, 0)]
        if action == 1:
            return [min(state + action, 3)]


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


class Climber(NoisyEnv):
    """ Climber toy problem from Little 2007
    States: {up, down, dead, waiting}
    Actions: {try_jump, call_help, climb_down}
    Fully observable
    """

    S_UP = 0
    S_DOWN = 1
    S_UP_WAITING = 2
    S_DEAD = 3

    A_TRY_JUMP = 10
    A_CALL_HELP = 11
    A_CLIMB_DOWN = 12

    def get_obs(self, state):
        return state

    def legal_actions(self, state):
        return [Climber.A_TRY_JUMP, Climber.A_CALL_HELP, Climber.A_CLIMB_DOWN]

    @property
    def goal_states(self):
        return [Climber.S_DOWN]

    @property
    def init_states(self):
        return [Climber.S_UP]

    def next_states_p(self, state, action):
        C = Climber
        if state in [C.S_UP, C.S_UP_WAITING] and action == C.A_TRY_JUMP:
            return [(C.S_DOWN, 0.7), (C.S_DEAD, 0.3)]
        elif state == C.S_UP and action == C.A_CALL_HELP:
            return [(C.S_UP_WAITING, 1.)]
        elif state == C.S_UP_WAITING and action == C.A_CLIMB_DOWN:
            return [(C.S_DOWN, 1.)]
        else:
            return [(state, 1.)]

class BridgeWalk(NoisyEnv):
    A_LEFT = 0
    A_FWD = 1
    A_RIGHT = 2

    def __init__(self, init_N=4):
        self.init_N = init_N

        super().__init__()

    def get_obs(self, state):
        # return state[1]
        return state[0] == 0

    def legal_actions(self, state):
        # if state[1] == -1: # for debugging
        #     return []
        # else:
        return [self.A_FWD, self.A_LEFT, self.A_RIGHT]

    @property
    def goal_states(self):
        return [(0,0)]

    @property
    def init_states(self):
        return [(self.init_N,0)]

    def next_states_p(self, state, action):
        if action == self.A_FWD and state[1] == 0:
            s_next_1 = (max(state[0]-1, 0), 0)
            s_next_2 = (state[0], -1)
            return [(s_next_1, 0.9), (s_next_2, 0.1)]
        elif action == self.A_LEFT and state[1] == 0:
            return [((state[0], +1), 1.)]
        elif action == self.A_RIGHT and state[1] == 0:
            return [((state[0], -1), 1.)]
        elif state[1] == -1:  # dead
            return [(state, 1.)]
        elif action == self.A_FWD and state[1] == +1:
            return [((max(state[0]-1, 0), +1), 1.)]
        elif action == self.A_RIGHT and state[1] == +1:
            return [((state[0], 0), 1.)]
        elif action == self.A_LEFT and state[1] == +1:
            return [(state, 1.)]
        else:
            assert False, 'Illegal action'
