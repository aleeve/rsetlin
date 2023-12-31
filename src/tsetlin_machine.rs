use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::iter::zip;

#[derive(Debug)]
enum FeedBack {
    Reward,
    Inaction,
    Penalty,
}

#[derive(Debug)]
enum FeedbackType {
    TypeI,
    TypeII,
}

#[derive(Clone, Default, Debug)]
struct Automaton {
    max_activation: i32,
    state: i32,
}
impl Automaton {
    fn new(max_activation: i32) -> Automaton {
        let state = thread_rng().gen_range(max_activation - 1..=max_activation);
        Automaton {
            max_activation,
            state,
        }
    }

    fn output(&self) -> bool {
        self.state >= self.max_activation
    }

    fn transition(&mut self, feedback: FeedBack) {
        match feedback {
            FeedBack::Reward => {
                if self.state >= self.max_activation {
                    if self.state < 2 * self.max_activation {
                        self.state += 1;
                    }
                } else {
                    if self.state > 0 {
                        self.state -= 1
                    }
                };
            }
            FeedBack::Penalty => {
                if self.state >= self.max_activation {
                    if self.state > 0 {
                        self.state -= 1;
                    }
                } else {
                    if self.state < 2 * self.max_activation {
                        self.state += 1
                    }
                };
            }
            FeedBack::Inaction => {}
        }
    }
}

#[derive(Clone, Default, Debug)]
struct Clause {
    max_activation: i32,
    feature_count: usize,
    features: Vec<Automaton>,
    state: Vec<bool>,
}

impl Clause {
    fn new(feature_count: usize, max_activation: i32) -> Clause {
        let mut c = Clause {
            max_activation,
            feature_count,
            features: vec![],
            state: vec![false], //hack to trigger reset
        };
        c.output();
        c
    }

    fn output(&mut self) {
        self.state = self.features.iter().map(|f| f.output()).collect();
        if self.state.iter().all(|x| !*x) {
            // Reseting dead clause, not sure if totally ok as the age of the clauses will
            // differ now.
            self.features = (0..self.feature_count)
                .map(|_| Automaton::new(self.max_activation))
                .collect();
            self.output();
        }
    }

    fn apply(&self, input: &Vec<bool>) -> bool {
        zip(input, &self.state).all(|(l, s)| l | !s)
    }
}

pub struct TsetlinMachine {
    s: f32,
    threshold: f32,
    positive_clauses: Vec<Clause>,
    negative_clauses: Vec<Clause>,
}

impl TsetlinMachine {
    pub fn new(
        num_clauses: usize,
        max_activation: i32,
        s: f32,
        threshold: f32,
        feature_count: usize,
    ) -> TsetlinMachine {
        let mut clauses = Vec::new();
        for _ in 0..=num_clauses {
            clauses.push(Clause::new(feature_count, max_activation));
        }
        let (positive_clauses, negative_clauses) = clauses.split_at(num_clauses / 2);
        TsetlinMachine {
            s,
            threshold,
            positive_clauses: positive_clauses.into(),
            negative_clauses: negative_clauses.into(),
        }
    }

    fn compute_clauses(&self, input: &Vec<bool>) -> (Vec<i32>, Vec<i32>, Vec<bool>) {
        // Get the negative features as well
        let input: Vec<bool> = input
            .par_iter()
            .flat_map(|l| vec![l.clone(), (!l).clone()])
            .collect();

        let positive: Vec<i32> = self
            .positive_clauses
            .par_iter()
            .map(|c| c.apply(&input) as i32)
            .collect();
        let negative: Vec<i32> = self
            .negative_clauses
            .par_iter()
            .map(|c| c.apply(&input) as i32)
            .collect();
        (positive, negative, input)
    }

    pub fn predict(&self, input: &Vec<bool>) -> bool {
        let (positive, negative, _) = self.compute_clauses(&input);
        positive.iter().sum::<i32>() > negative.iter().sum::<i32>()
    }

    /// Give feedback to multiple clauses in parallel
    fn feedback_clauses(
        &mut self,
        positive_clauses: bool,
        results: Vec<i32>,
        threshold: f32,
        input: &Vec<bool>,
        feedback_type: FeedbackType,
    ) {
        let clauses = match positive_clauses {
            true => &mut self.positive_clauses,
            false => &mut self.negative_clauses,
        };

        clauses
            .par_iter_mut()
            .zip(results.par_iter())
            .for_each(|(clause, outcome)| {
                if thread_rng().gen_range(0.0..1.0) < threshold {
                    match &feedback_type {
                        FeedbackType::TypeI => give_type_i_feedback(
                            self.s.clone(),
                            &input,
                            *outcome == 1,
                            &mut clause.features,
                        ),
                        FeedbackType::TypeII => {
                            give_type_ii_feedback(&input, *outcome == 1, &mut clause.features)
                        }
                    }
                    clause.output();
                };
            });
    }

    /// Fit model to single example
    pub fn fit(&mut self, input: Vec<bool>, target: bool) {
        let (positive, negative, input) = self.compute_clauses(&input);
        let v: f32 = (positive.iter().sum::<i32>() - negative.iter().sum::<i32>()) as f32;
        let threshold: f32 =
            (self.threshold - v.clamp(-self.threshold, self.threshold)) / (2f32 * self.threshold);
        if target {
            self.feedback_clauses(true, positive, threshold, &input, FeedbackType::TypeI);
            self.feedback_clauses(false, negative, threshold, &input, FeedbackType::TypeII);
        } else {
            self.feedback_clauses(true, positive, threshold, &input, FeedbackType::TypeII);
            self.feedback_clauses(false, negative, threshold, &input, FeedbackType::TypeI);
        };
    }
}

fn give_type_i_feedback(
    s: f32,
    input: &Vec<bool>,
    clause: bool,
    automata_family: &mut Vec<Automaton>,
) {
    for (literal, automaton) in zip(input, automata_family) {
        let action = automaton.output();
        let feedback = sample_type_i_feedback(s, action, literal, clause);
        automaton.transition(feedback);
    }
}

fn give_type_ii_feedback(input: &Vec<bool>, clause: bool, automata_family: &mut Vec<Automaton>) {
    for (literal, automaton) in zip(input, automata_family) {
        let action = automaton.output();
        let feedback = sample_type_ii_feedback(action, literal, clause);
        automaton.transition(feedback);
    }
}

fn sample_type_i_feedback(s: f32, include: bool, exists: &bool, clause: bool) -> FeedBack {
    let normal = thread_rng().gen_range(0.0..1.0) > 1f32 / s;
    if clause {
        if *exists {
            // The clause and the literal evaluates to true, normally we reward inclusion and
            // penalise exclusion. But stochastically sometimes we do nothing.
            if normal {
                if include {
                    FeedBack::Reward
                } else {
                    FeedBack::Penalty
                }
            } else {
                FeedBack::Inaction
            }
        } else {
            // The clause evaluate to true, but the literal isn't manifesting, normally we do
            // nothing but sometimes we want to try to add more features.
            if normal {
                FeedBack::Inaction
            } else {
                FeedBack::Reward
            }
        }
    } else {
        // The clause evaluates to false, normally
        // we do nothing but stochastically we push towards the other exclusion
        if normal {
            FeedBack::Inaction
        } else {
            if include {
                FeedBack::Penalty
            } else {
                FeedBack::Reward
            }
        }
    }
}

fn sample_type_ii_feedback(include: bool, exists: &bool, clause: bool) -> FeedBack {
    // Give penalty for excluding literal that would have falsified the clause
    if clause && !exists && !include {
        FeedBack::Penalty
    } else {
        FeedBack::Inaction
    }
}

#[test]
fn test_tm_clean_signal() {
    let mut tm = TsetlinMachine::new(50, 30, 4.0, 30.0, 3);
    let mut correct = 0;
    for _ in 0..200 {
        let data: [bool; 3] = thread_rng().gen();
        tm.fit(data.into(), data[1]);
    }
    let mut testset = vec![];
    for first in vec![true, false] {
        for second in vec![true, false] {
            for third in vec![true, false] {
                testset.push(vec![first, second, third])
            }
        }
    }

    for test in testset {
        let output = tm.predict(&test);
        if output == test[1] {
            correct += 1;
        }
    }
    assert!(correct == 8);
}

#[test]
fn test_tm_noisy_signal() {
    let mut tm = TsetlinMachine::new(100, 30, 4.0, 60.0, 3);
    let mut correct = 0;
    for _ in 0..500 {
        let data: [bool; 3] = thread_rng().gen();
        tm.fit(
            data.into(),
            if thread_rng().gen_range(0.0..1.0) < 0.1 {
                !data[1]
            } else {
                data[1]
            },
        );
    }
    let mut testset = vec![];
    for first in vec![true, false] {
        for second in vec![true, false] {
            for third in vec![true, false] {
                testset.push(vec![first, second, third])
            }
        }
    }

    for test in testset {
        let output = tm.predict(&test);
        if output == test[1] {
            correct += 1;
        }
    }
    assert!(correct == 8, "Correct: {}", correct);
}

#[test]
fn test_transition() {
    let mut a = Automaton {
        max_activation: 10,
        state: 12,
    };
    a.transition(FeedBack::Reward);
    assert!(a.state == 13);
    a.transition(FeedBack::Inaction);
    assert!(a.state == 13);
    a.transition(FeedBack::Penalty);
    assert!(a.state == 12);
    a.transition(FeedBack::Penalty);
    assert!(a.state == 11);
    a.transition(FeedBack::Penalty);
    assert!(a.state == 10);
    a.transition(FeedBack::Reward);
    assert!(a.state == 11);
}

#[test]
fn test_clause() {
    let mut c = Clause {
        max_activation: 12,
        feature_count: 1,
        features: vec![Automaton {
            state: 13,
            max_activation: 12,
        }],
        state: vec![],
    };
    c.output();
    assert!(c.apply(&vec![true]) == true);

    let mut c = Clause {
        max_activation: 12,
        feature_count: 2,
        features: vec![
            Automaton {
                state: 13,
                max_activation: 12,
            },
            Automaton {
                state: 12,
                max_activation: 12,
            },
        ],
        state: vec![],
    };
    c.output();
    assert!(c.apply(&vec![false, true]) == false);
}
