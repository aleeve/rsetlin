use crate::tselin_machine::TsetlinMachine as TM;
use pyo3::prelude::*;

#[pyclass]
struct TsetlinMachine {
    model: TM,
}

/// Interface fo tsetlin machine
#[pymethods]
impl TsetlinMachine {
    #[new]
    pub fn new(
        feature_count: usize,
        num_clauses: usize,
        max_activation: i32,
        s: f32,
        threshold: f32,
    ) -> Self {
        TsetlinMachine {
            model: TM::new(num_clauses, max_activation, s, threshold, feature_count),
        }
    }

    /// Take one step in the training process
    fn fit(&mut self, input: Vec<bool>, target: bool) {
        self.model.fit(input, target)
    }

    // Predict a single instance
    fn predict(&self, input: Vec<bool>) -> bool {
        self.model.predict(&input)
    }
}

/// rsetlin is a module
#[pymodule]
fn rsetlin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TsetlinMachine>()?;
    Ok(())
}
