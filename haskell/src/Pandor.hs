{-
   TODO: Change notation in paper from A/→ to A:→
-}

module Pandor
  (
    printController,
    synthPlan,
    Controller
  ) where

import Config
import WalkAB

printController :: Maybe Controller -> IO ()
printController Nothing = putStrLn "No controller found"
printController (Just c) = putStrLn "Some controller found"

data ContState = Q Int
data CombinedState = (EnvState, ContState)
data Controller = [((ContState, Observation), (ContState, Action))]
data HistItem = [(CombinedState, Float)]
data Alphas = TODO

data ContGoodness = GoodCont | BadCont | MaybeGoodCont

synthPlan :: Controller
synthPlan = map fst $ head $ andStep emptyController initialQ [initialState] emptyHist
  where
    emptyController = []
    initialQ = Q 0
    emptyHist = []

orStep :: Controller -> ContState -> EnvState -> Float -> [HistItem] -> Alphas -> [Controller]
orStep c q s p hl alphas =
  | isGoal s = [(c, addGoal p alphas)]
  | isFail s = [(c, addFail p alphas)]
  | ((q, s), p) `elem` hl = [(c, addNoter p alphas)]
  | ((q, s) `elemIndex` (map fst hl))@(Just i) = [(c, addLoop p i alphas)]  # TODO divide by p_i?
  | (lookup (q, o) c)@(Just (q', a)) = let s'l = nextStates a s
                                           hl' = (s,q):hl
                                         in andStep c q' s'l hl' alphas
  | otherwise = let c' = ((q, o), (q',a)):c
                    s'l = nextStates a s
                    hl' = (s,q):hl
                    k = min (numStates c) (numStatesBound - 1)
                  in concat [andStep c' q' s'l hl' alphas | q' <- map Q [0..k],
                                                            a <- legalActs]
  where
    o = observe s

howGoodCont :: Alphas -> [HistItem] -> ContGoodness
howGoodCont alphas = TODO

andStep :: Controller -> ContState -> [EnvState] -> Float -> [HistItem] -> Alphas -> [Controller]
andStep c _ [] _ _ alphas = [(c, alphas)]
andStep c q s:sl p hl alphas = concat [case howGoodCont alphas' of
        GoodCont      -> [(c', alphas')]
        BadCont       -> []
        MaybeGoodCont -> [andStep c' q sl p hl alphas' | (c', alphas') <- orStep c q s hl alphas]
    ]
