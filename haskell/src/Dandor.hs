module Lib
  (
    printController,
    synthPlan
  ) where

import Config
import WalkAB

printController :: Maybe Controller -> IO ()
printController Nothing = putStrLn "No controller found"
printController (Just c) = putStrLn "Some controller found"

data ContState = Q Int
data CombinedState = (EnvState, ContState)
data Controller = [((ContState, Observation), (ContState, Action))]
data HistItem = [CombinedState]

synthPlan :: Int -> Controller
synthPlan = detAndStep emptyController initialQ [initialState] emptyHist
  where
    emptyController = []
    initialQ = Q 0
    emptyHist = []

orStep :: Controller -> ContState -> EnvState -> [DetHistItem] -> [Controller]
orStep c q s hl =
  | isGoal s = [c]
  | (q, s) `elem` hl = []
  | (lookup (q, o) c)@(Just (q', a)) = let s'l = nextStates a s
                                       in andStep c s'l hl
  | otherwise = let c' = ((q, o), (q',a)):c
                    s'l = nextStates a s
                    k = min (numStates c) (numStatesBound - 1)
                in concat [andStep c' q' s'l hl | q <- map Q [0..k],
                                                  a <- legalActs]
  where
    o = observe s

andStep :: Controller -> ContState -> [EnvState] -> [DetHistItem] -> [Controller]
andStep c _ [] _ = [c]
andStep c q s:sl hl = concat [andStep c' q sl hl | c' <- orStep c q s hl]


-- Pandor below

-- | synthPlan numStatesBound lpcDesired = (controller, lpc_min, lpc_max)
-- synthPlan :: Int -> Double -> (Maybe Controller, Double, Double)

-- | andStep controller q a = controller'
-- andStep :: Controller -> ContState -> Action -> Maybe Controller

-- | orStep controller q s p h = controller'
-- orStep :: Controller -> ContState -> EnvState -> Double -> [HistoryItem] -> Maybe Controller
