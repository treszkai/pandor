module Dandor
  (
    synthController
  ) where

import Config
import Environment
import Controller
import Data.Maybe
import Debug.Trace as Trace

data CombinedState = QS ContState EnvState deriving (Eq, Show)

synthController :: Controller
synthController = head $ andStep emptyController initialQ [initialState] emptyHist
  where
    initialQ = Q 0
    emptyHist = []

orStep :: Controller -> ContState -> EnvState -> [CombinedState] -> [Controller]
orStep c q s hl
  | traceShow ("orStep  c={" ++ show c ++ "} q={" ++ show q ++
               "}  s={" ++ show s ++ "} hl={" ++ show hl ++ "}") False = []
  | isGoalState s = [c]
  -- | isFail s = [] TODO?
  | (QS q s) `elem` hl = []
  | isJust maybeTransition = let (q', a) = fromJust maybeTransition
                                 s'l = [nextState s a]
                                 hl' = (QS q s):hl
                               in andStep c q' s'l hl'
  | otherwise = concat [let c' = ((q, o), (q', a)):c
                            s'l = [nextState s a]
                            hl' = (QS q s):hl
                          in andStep c' q' s'l hl' | q' <- map Q [0..k],
                                                     a <- legalActs]
  where 
    o = observe s
    maybeTransition = lookup (q, o) c
    k = min (numContStates c) (numStatesBound - 1)

andStep :: Controller -> ContState -> [EnvState] -> [CombinedState] -> [Controller]
andStep c q sl hl
  | traceShow ("andStep c={" ++ show c ++ "} q={" ++ show q ++
               "} sl={" ++ show sl ++ "} hl={" ++ show hl ++ "}") False = []
andStep c _ [] _ = [c]
andStep c q (s:sl) hl = concat [andStep c' q sl hl | c' <- orStep c q s hl]