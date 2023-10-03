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
    fn new(num_clauses: i32) -> Self {
        TsetlinMachine {
            model: TM::new(50, 30, 4.0, 30.0, 3),
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
