"""
AND-OR search algorithm for a finite-state controller

Based on the Prolog implementation in Hu and De Giacomo: A Generic Technique for Synthesizing Bounded Finite-State Controllers (2013).
"""


from controller import MealyController
import environments

import logging
import time
from itertools import dropwhile
from itertools import product


# verbose flag
# v = True
v = False


class PandorControllerNotFound(Exception):
    pass


class AndOrPlanner:
    def __init__(self, env):
        self.env = env
        # attributes used by synth_plan() - def'd here only to suppress warnings
        self.contr, self.backtracking, self.backtrack_stack = None, None, None

    def synth_plan(self, states_bound):
        self.backtracking = False
        self.backtrack_stack = []
        self.contr = MealyController(states_bound)
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
            return dropwhile(lambda x: x != self.backtrack_stack[-1][len(history)][1],
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
                if len(self.backtrack_stack) == 0:
                    logging.info("AND: Backtracking up; empty stack") if v else 0
                    return False
                elif history == self.backtrack_stack[-1][:min(len(history), len(self.backtrack_stack[-1])-1)]:
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
            if self.env.is_goal_state(env_state):
                return True

            if (q, env_state) in history:
                return False

            history.append((q, env_state))
            obs = self.env.get_obs(env_state)

            if (q, obs) in self.contr.transitions:
                # Note: for Moore machine, this can be defined as action being only q-dependent
                q_next, action = self.contr[q, obs]
                if action not in self.env.legal_actions(env_state):
                    return False

                sl_next = self.env.next_states(env_state, action)
                return self.and_step(q_next, sl_next, history)

            # for a Moore machine:
            # if δ(q,o) def'd then γ(q) also def'd.
            # So maybe γ(q) def'd but δ(q,o) undef.
            # if neither is defined:
            #   iterate over possible q' and a.
            # if γ def:
            #   iterate over possible q', but a = γ(q).
            #
            # (the code below is designed for Mealy only)

            # no (q_next,act) defined for (q,obs) ⇒ define new one with this iterator
            it = product(range(min(self.contr.num_states+1, self.contr.bound)),
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

            q_next_last, action_last = t[1]

            it = dropwhile(lambda x: x[1] != action_last,
                           product(range(q_next_last, min(self.contr.num_states + 1, self.contr.bound)),
                                   self.env.legal_actions(env_state)))

            # this is the node of the last checkpoint
            # (enough to check the length because we're in the right branch now)
            if len(history) == len(self.backtrack_stack[-1]):
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

                t = self.contr.transitions.popitem()
                logging.info("OR: Deleted: (%s,%s) -> (%s,%s)",
                             t[0][0], self.env.str_obs(t[0][1]),
                             t[1][0], self.env.str_action(t[1][1])) if v else 0

        self.backtrack_stack.pop()
        return False


def main():
    env = environments.ProbHallArect(noisy=False)

    planner = AndOrPlanner(env)
    success = planner.synth_plan(states_bound=3)

    if v:
        time.sleep(1)  # Wait for mesages of logging module

    if success:
        for (q,o),(q_next,a) in planner.contr.transitions.items():
            logging.warning("({},{}) → ({},{})".format(q, env.str_obs(o), q_next, env.str_action(a)))
    else:
        logging.warning("No controller found")


if __name__ == '__main__':
    if v:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    if 0:
        v = False
        print(timeit.timeit('main()', number=100, setup="from __main__ import main"))
    else:
        main()
