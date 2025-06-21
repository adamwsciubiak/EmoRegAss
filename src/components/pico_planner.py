""" 

Implementation of Pico's planner agent accordingly to the paper

This file defines the data structures for the static planner derived from the paper (Section 4.2.).
Correlations of personality traits and emotion regulation techniques are yet to be refined with more precise based on Barańczuk (2019) or Hughes et al., (2020)

Glossery:

alpha               actions / actions catalog       (α),            a one of the five strategies defined in Section 2.1" (Section 4, para. 2)
state               emotional state                 (S),            a point in the 2D Arousal-Valence space (Section 2.2, Figure 3) -> tuple
delta_Sa            delta S alpha                   (ΔSα),          change in State in the effect of the Action (Section 4, para. 6; Table 2)
theta_a             theta alpha                     (θα),           "personality weights", correlation between the personality traits of the OCEAN model 
                                                                    and the regulation strategies (Table 1). In the current version: '+' = 1.0, '-' = -1.0.


perTrait            personality trait               (t)
epsilon_S           emotional equilibrium state     (ε)


"""





