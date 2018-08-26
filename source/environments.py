class Environment:
    """ Class that describes the environment but does not simulate it """

    def __init__(self):
        assert type(self.init_states) is list
        s = self.init_states[0]
        assert type(self.is_goal_state(s)) is bool
        assert type(self.legal_actions(s)) is list
        a = self.legal_actions(s)[0]
        assert type(self.next_states(s, a)) is list

    @property
    def init_states(self):
        raise NotImplementedError

    @property
    def goal_states(self):
        raise NotImplementedError

    def is_goal_state(self, state):
        return state in self.goal_states

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
    """ Environment of Fig. 1 of BPG2009 (Hall-A one-dim)
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


class WalkThroughFlapProb(NoisyEnv):
    """ Environment to test AND backtracking
    States: {0(goal), 1(init0), 2(init1), 3(goal)}
    Actions: Left/Right, but Left in 2 leaves you in 2.
    Obs: Goal or not
    """

    @property
    def init_states(self):
        return [-1]
        # return [1,2]

    @property
    def goal_states(self):
        return [0,3]

    def legal_actions(self, state):
        if state == -1:
            return ["begin"]
        else:
            return [-1, +1]

    def get_obs(self, state):
        if state == -1: return "begin"
        return state in self.goal_states

    def next_states_p(self, state, action):
        if action == "begin":
            return [(1, 0.3), (2, .7)]
        if action == -1:
            if state == 2:
                return [(2,.9), (1,.1)]
            else:
                return [(max(state + action, 0), 1.0)]
        if action == 1:
            return [(min(state + action, 3), 1.0)]


class ProbHallAone(NoisyEnv):
    """ Noisy versioin of Fig. 1 of BPG2009 (Hall-A one-dim)
    States: {(n, visB): n ∈ {1,2,3,4}, visB ∈ {True, False} }
    Action set: {-1, +1} = {left, right}
    Action effects:
        - left( (n,visB) )  = ( max(n-1, 1), visB )
        - right( (n,visB) ) = ( min(n+1, 5), visB )
    Observables: (A, B) = (n == 1, n == 4)
    Init states: { (1, False), (2, False) }
    Goal states: { (1, True) }
    """

    def __init__(self, length=3, noisy=True):
        self.length = length
        self.noisy = noisy
        super().__init__()

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
            return a

    @staticmethod
    def str_obs(o):
        assert not(o[0] and o[1])
        if o[0]:
            return "A"
        elif o[1]:
            return "B"
        else:
            return "-"

    @property
    def init_states(self):
        return [(1, False)]

    @property
    def goal_states(self):
        return [(1, True)]

    def legal_actions(self, state):
        return [-1, +1]

    def get_obs(self, state):
        n, _ = state
        return n == 1, n == self.length

    def next_states_p(self, state, action):
        n, vis_b = state

        n += action
        if n < 1:
            n = 1
        elif n > self.length:
            n = self.length

        vis_b |= n == self.length
        next_state = (n, vis_b)

        if self.noisy:
            # if state == next_state or state[0] == 1 or state[0] == self.length:
            if state == next_state:
                return [(next_state, 1.0)]
            else:
                # return [(next_state, 1.0)]
                return [(state, 0.5), (next_state, 0.5)]
        else:
            return [(next_state, 1.0)]

    def next_states(self, state, action):
        sp_list = self.next_states_p(state, action)
        return [s for s,p in sp_list]


class ProbHallArect(NoisyEnv):
    """ Noisy version of (Hall-A n-by-n) by BPG2009
    States: (top,right,bot,left) x (1..n-1) x visA x ... x visD
    Action set: {left, right, up, down}
    Actions have 0.5 probability of succeeding
    Observables: A,B,C,D,–, depending on whether it's in a corner or not.
    Init states: top x 1 x true x false x false x false
    Goal states: top x 1 x true x true x true x true
    """

    A_LEFT = "←"
    A_RIGHT = "→"
    A_UP = "↑"
    A_DOWN = "↓"

    SIDE_TOP = 0
    SIDE_RIGHT = 10
    SIDE_BOTTOM = 20
    SIDE_LEFT = 30

    def __init__(self, length=3, noisy=True):
        self.length = length
        self.noisy = noisy
        super().__init__()

    @staticmethod
    def str_state(s):
        if type(s) is str:
            return s
        else:
            return "{}+{}{}{}{}".format(s[0] + s[1], 'A' if s[2] else 'a',
                'B' if s[3] else 'b', 'C' if s[4] else 'c', 'D' if s[5] else 'd')

    @staticmethod
    def str_action(a):
        return a

    @staticmethod
    def str_obs(o):
        return o

    @property
    def init_states(self):
        return [(self.SIDE_TOP, 1, False, False, False, False)]

    @property
    def goal_states(self):
        return [(self.SIDE_TOP, 1, True, True, True, True)]

    def legal_actions(self, state):
        return [self.A_LEFT, self.A_DOWN, self.A_RIGHT, self.A_UP]

    def get_obs(self, state):
        if state[1] == 1:
            if state[0] is self.SIDE_TOP:
                return "A"
            elif state[0] is self.SIDE_RIGHT:
                return "B"
            elif state[0] is self.SIDE_BOTTOM:
                return "C"
            elif state[0] is self.SIDE_LEFT:
                return "D"
        else:
            return "-"

    def next_states_p(self, state, action):
        side, n, vis_a, vis_b, vis_c, vis_d = state

        if side is self.SIDE_TOP:
            if action is self.A_RIGHT:
                n += 1
            elif action is self.A_LEFT:
                n -= 1
            elif action is self.A_DOWN and n == 1:
                n = self.length
                side = self.SIDE_LEFT
        elif side is self.SIDE_RIGHT:
            if action is self.A_DOWN:
                n += 1
            elif action is self.A_UP:
                n -= 1
            elif action is self.A_LEFT and n == 1:
                n = self.length
                side = self.SIDE_TOP
        elif side is self.SIDE_BOTTOM:
            if action is self.A_LEFT:
                n += 1
            elif action is self.A_RIGHT:
                n -= 1
            elif action is self.A_UP and n == 1:
                n = self.length
                side = self.SIDE_RIGHT
        elif side is self.SIDE_LEFT:
            if action is self.A_UP:
                n += 1
            elif action is self.A_DOWN:
                n -= 1
            elif action is self.A_RIGHT and n == 1:
                n = self.length
                side = self.SIDE_BOTTOM

        # don't change state if n == 0 (attempted move against the corner)
        if n == 0:
            n = 1

        # change sides if arrived at corner
        if n == self.length + 1:
            n = 1
            side += 10
            side %= 40

        if n == 1:
            vis_a |= side is self.SIDE_TOP
            vis_b |= side is self.SIDE_RIGHT
            vis_c |= side is self.SIDE_BOTTOM
            vis_d |= side is self.SIDE_LEFT

        next_state = (side, n, vis_a, vis_b, vis_c, vis_d)

        if self.noisy:
            if state == next_state or n == 1:  # TODO: corners not noisy
            # if state == next_state:
                return [(next_state, 1.0)]
            else:
                return [(state, 0.5), (next_state, 0.5)]
        else:
            return [(next_state, 1.0)]

    def next_states(self, state, action):
        sp_list = self.next_states_p(state, action)
        return [s for s,p in sp_list]


# class HallR(Environment):
#     """ Modification of the Hall-R problem (Bonet et al. 2009)
#     Goal: walk around a square-shaped hall.
#     States: {0,1,2,...11} x {true,false}^12 x {up,down,left,right}
#     Init: ⟨0,f,f,...,f⟩
#     Goals: ⟨*,t,t,...,t,*⟩
#     Actions: {fwd,turnright,turnleft}
#     Obs: {0,3,6,9,hall}x{facingwall,facinghall}
#     """
#
#     DIR_RIGHT='r'
#     g = 2**12-1
#
#     @property
#     def init_states(self):
#         return [(0,0,self.DIR_RIGHT)]
#
#     # @property
#     # def goal_states(self):
#     #     return self.g
#
#     def is_goal_state(self, state):
#         return state[1] == self.g
#
#     def legal_actions(self, state):
#         if state == -1:
#             return ["begin"]
#         else:
#             return [-1, +1]
#
#     def get_obs(self, state):
#         if state == -1: return "begin"
#         return state in self.goal_states
#
#     def next_states_p(self, state, action):
#         if action == "begin":
#             return [(1, 0.3), (2, .7)]
#         if action == -1:
#             if state == 2:
#                 return [(2,.9), (1,.1)]
#             else:
#                 return [(max(state + action, 0), 1.0)]
#         if action == 1:
#             return [(min(state + action, 3), 1.0)]


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
    A_LEFT = 'left'
    A_FWD = 'fwd'
    A_RIGHT = 'right'

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
        # return [self.A_LEFT, self.A_FWD, self.A_RIGHT]
        return [self.A_FWD, self.A_LEFT, self.A_RIGHT]
        # return [self.A_RIGHT, self.A_LEFT, self.A_FWD]

    @property
    def goal_states(self):
        return [(0,0)]

    @property
    def init_states(self):
        return [(self.init_N,0)]

    def next_states_p(self, state, action):
        if action is self.A_FWD and state[1] == 0:
            s_next_1 = (max(state[0]-1, 0), 0)
            s_next_2 = (state[0], -1)
            return [(s_next_1, 0.9), (s_next_2, 0.1)]
        elif action is self.A_LEFT and state[1] == 0:
            return [((state[0], +1), 1.)]
        elif action is self.A_RIGHT and state[1] == 0:
            return [((state[0], -1), 1.)]
        elif state[1] == -1:  # dead
            return [(state, 1.)]
        elif action is self.A_FWD and state[1] == +1:
            return [((max(state[0]-1, 0), +1), 1.)]
        elif action is self.A_RIGHT and state[1] == +1:
            return [((state[0], 0), 1.)]
        elif action is self.A_LEFT and state[1] == +1:
            return [(state, 1.)]
        else:
            assert False, 'Illegal action'


class LoopyTest(NoisyEnv):

    def get_obs(self, state):
        return 0

    def legal_actions(self, state):
        return [0]

    @property
    def goal_states(self):
        return [1,3]

    @property
    def init_states(self):
        return [0]

    def next_states_p(self, state, action):
        if state == 0:
            return [(1, 0.5), (2, .4), (7, .1)]
        elif state == 2:
            return [(3,.4), (2, .3), (5, .2), (6, .1)]
        elif state == 6:
            return [(6, 1.)]
        else:
            return [(state, 1.)]
        # if action is self.A_FWD and state[1] == 0:
        #     s_next_1 = (max(state[0]-1, 0), 0)
        #     s_next_2 = (state[0], -1)
        #     return [(s_next_1, 0.9), (s_next_2, 0.1)]
        # elif action is self.A_LEFT and state[1] == 0:
        #     return [((state[0], +1), 1.)]
        # elif action is self.A_RIGHT and state[1] == 0:
        #     return [((state[0], -1), 1.)]
        # elif state[1] == -1:  # dead
        #     return [(state, 1.)]
        # elif action is self.A_FWD and state[1] == +1:
        #     return [((max(state[0]-1, 0), +1), 1.)]
        # elif action is self.A_RIGHT and state[1] == +1:
        #     return [((state[0], 0), 1.)]
        # elif action is self.A_LEFT and state[1] == +1:
        #     return [(state, 1.)]
        # else:
        #     assert False, 'Illegal action'
