//! Deterministic bounded multi-track optimization using a native physics callback.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
/// One independently bounded physical parameter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Variable {
    /// Native evaluator's parameter path.
    pub path: String,
    /// Inclusive lower bound, in the parameter's SI unit.
    pub lower: f64,
    /// Inclusive upper bound, in the parameter's SI unit.
    pub upper: f64,
}
/// Track identifier resolved by the caller against its immutable inputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackCase {
    /// Stable track identity.
    pub id: String,
    /// Positive relative importance.
    pub weight: f64,
}
/// Squared residual objective with a fixed physical normalization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricTarget {
    /// Exact named metric key returned by the evaluator.
    pub path: String,
    /// Desired value in the metric's SI unit.
    pub desired: f64,
    /// Positive physical scale, never reestimated during search.
    pub scale: f64,
    /// Nonnegative importance (at least one target must have positive weight).
    pub weight: f64,
}
/// A metric must remain within an absolute tolerance of a fixed value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricConstraint {
    /// Exact evaluator metric key.
    pub path: String,
    /// Required physical value.
    pub value: f64,
    /// Nonnegative absolute tolerance in the metric's SI unit.
    pub tolerance: f64,
    /// Positive physical scale for ranking violations.
    pub scale: f64,
}
/// Independent uniform uncertainty applied by the physics callback.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Uncertainty {
    /// Physical parameter path to perturb.
    pub path: String,
    /// Nonnegative absolute half range, in the parameter's SI unit.
    pub half_range: f64,
}
/// Separate random streams; held-out samples never influence selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleSplit {
    /// Training scenarios shared by every candidate.
    Training,
    /// Independent final validation scenarios.
    Validation,
}
/// Replayable physical perturbations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UncertaintySample {
    /// Stream identity.
    pub split: SampleSplit,
    /// Stable zero-based index within the stream.
    pub index: usize,
    /// Additive SI deltas by parameter path.
    pub deltas: BTreeMap<String, f64>,
}
/// Bounded deterministic search settings and immutable objective definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationRequest {
    /// Ordered parameters passed to the callback.
    pub variables: Vec<Variable>,
    /// In-bounds starting candidate, also included in the population.
    pub initial_values: Vec<f64>,
    /// Weighted track set.
    pub tracks: Vec<TrackCase>,
    /// Desired metric residuals.
    pub metrics: Vec<MetricTarget>,
    /// Hard constraints checked on every track and sample.
    pub constraints: Vec<MetricConstraint>,
    /// Independent uniform perturbations.
    pub uncertainty: Vec<Uncertainty>,
    /// Positive number of training samples per track.
    pub training_samples: usize,
    /// Positive number of held-out samples per track.
    pub validation_samples: usize,
    /// Reproducibility seed.
    pub seed: u64,
    /// DE population size, at least four.
    pub population_size: usize,
    /// Maximum complete mutation rounds.
    pub generations: usize,
    /// Maximum training candidate evaluations, including initialization.
    /// A single held-out evaluation is reserved outside this budget.
    pub max_evaluations: usize,
    /// Fixed risk aggregation.
    pub risk: RiskWeights,
}
fn unique<'a>(items: impl Iterator<Item = &'a str>) -> bool {
    let mut seen = BTreeSet::new();
    items
        .into_iter()
        .all(|s| !s.trim().is_empty() && seen.insert(s))
}
impl OptimizationRequest {
    /// Validate domains and bound the maximum work and allocation sizes.
    pub fn validate(&self) -> Result<(), String> {
        self.risk.validate()?;
        if self.variables.is_empty()
            || self.variables.len() > 128
            || !unique(self.variables.iter().map(|v| v.path.as_str()))
            || self.variables.iter().any(|v| {
                !v.lower.is_finite()
                    || !v.upper.is_finite()
                    || v.lower >= v.upper
                    || !(v.upper - v.lower).is_finite()
            })
        {
            return Err("invalid variable bounds or paths".into());
        }
        if self.initial_values.len() != self.variables.len()
            || self
                .initial_values
                .iter()
                .zip(&self.variables)
                .any(|(x, v)| !x.is_finite() || *x < v.lower || *x > v.upper)
        {
            return Err("initial values must match bounds".into());
        }
        if self.tracks.is_empty()
            || self.tracks.len() > 1000
            || !unique(self.tracks.iter().map(|t| t.id.as_str()))
            || self
                .tracks
                .iter()
                .any(|t| !t.weight.is_finite() || t.weight <= 0.)
            || !self
                .tracks
                .iter()
                .map(|t| t.weight)
                .sum::<f64>()
                .is_finite()
        {
            return Err("invalid track weights or identities".into());
        }
        if self.metrics.is_empty()
            || !unique(self.metrics.iter().map(|m| m.path.as_str()))
            || self.metrics.iter().any(|m| {
                !m.desired.is_finite()
                    || !m.scale.is_finite()
                    || m.scale <= 0.
                    || !m.weight.is_finite()
                    || m.weight < 0.
            })
            || self.metrics.iter().map(|m| m.weight).sum::<f64>() <= 0.
        {
            return Err("invalid metric targets".into());
        }
        if !unique(self.constraints.iter().map(|c| c.path.as_str()))
            || self.constraints.iter().any(|c| {
                !c.value.is_finite()
                    || !c.tolerance.is_finite()
                    || c.tolerance < 0.
                    || !c.scale.is_finite()
                    || c.scale <= 0.
            })
        {
            return Err("invalid fixed-value constraints".into());
        }
        if !unique(self.uncertainty.iter().map(|u| u.path.as_str()))
            || self
                .uncertainty
                .iter()
                .any(|u| !u.half_range.is_finite() || u.half_range < 0.)
        {
            return Err("invalid uncertainty paths or ranges".into());
        }
        if !(1..=10_000).contains(&self.training_samples)
            || !(1..=10_000).contains(&self.validation_samples)
            || !(4..=1000).contains(&self.population_size)
            || self.generations > 100_000
            || self.max_evaluations < self.population_size
            || self.max_evaluations > 1_000_000
        {
            return Err("invalid optimization budget".into());
        }
        let cases = self.tracks.len() as u128
            * (self.training_samples as u128 * self.max_evaluations as u128
                + self.validation_samples as u128);
        if cases > 100_000_000 {
            return Err("optimization exceeds 100 million physics cases".into());
        }
        Ok(())
    }
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1. / 9007199254740992.)
    }
    fn index(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}
/// Generate the same train/validation scenarios independently of candidate order.
pub fn uncertainty_samples(
    request: &OptimizationRequest,
    split: SampleSplit,
) -> Result<Vec<UncertaintySample>, String> {
    request.validate()?;
    let (count, salt) = match split {
        SampleSplit::Training => (request.training_samples, 0x243f6a8885a308d3),
        SampleSplit::Validation => (request.validation_samples, 0x13198a2e03707344),
    };
    let mut rng = Rng(request.seed ^ salt);
    Ok((0..count)
        .map(|index| UncertaintySample {
            split,
            index,
            deltas: request
                .uncertainty
                .iter()
                .map(|u| (u.path.clone(), (2. * rng.unit() - 1.) * u.half_range))
                .collect(),
        })
        .collect())
}
/// A failed physics case or missing/nonfinite required metric.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationFailure {
    /// Sample index.
    pub sample: usize,
    /// Track identity.
    pub track: String,
    /// Diagnostic supplied by evaluator or metric validation.
    pub reason: String,
}
/// Complete objective evidence. A failure can never yield a valid score.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    /// Whether every requested case was attempted (false on cancellation).
    pub complete: bool,
    /// Complete, all physics valid, and all hard constraints satisfied.
    pub feasible: bool,
    /// Available only when every physics case and metric is valid.
    pub statistics: Option<LossStatistics>,
    /// Maximum normalized constraint excess across every case.
    pub violation: f64,
    /// Track-weighted losses, one per completed uncertainty sample.
    pub sample_losses: Vec<f64>,
    /// Failed case diagnostics.
    pub failures: Vec<EvaluationFailure>,
    /// Actual physics callback invocations.
    pub cases_evaluated: usize,
}
/// Evaluate a candidate with common random numbers and strict failure semantics.
pub fn evaluate_candidate<F, C>(
    r: &OptimizationRequest,
    values: &[f64],
    split: SampleSplit,
    evaluator: &mut F,
    cancelled: &mut C,
) -> Result<CandidateEvaluation, String>
where
    F: FnMut(&[f64], &TrackCase, &UncertaintySample) -> Result<BTreeMap<String, f64>, String>,
    C: FnMut() -> bool,
{
    let samples = uncertainty_samples(r, split)?;
    if values.len() != r.variables.len()
        || values
            .iter()
            .zip(&r.variables)
            .any(|(x, v)| !x.is_finite() || *x < v.lower || *x > v.upper)
    {
        return Err("candidate outside variable bounds".into());
    }
    let mut result = CandidateEvaluation {
        complete: true,
        feasible: false,
        statistics: None,
        violation: 0.,
        sample_losses: vec![],
        failures: vec![],
        cases_evaluated: 0,
    };
    let track_weight = r.tracks.iter().map(|t| t.weight).sum::<f64>();
    for sample in &samples {
        let mut sample_loss = 0.;
        let mut valid = true;
        for track in &r.tracks {
            if cancelled() {
                result.complete = false;
                return Ok(result);
            }
            result.cases_evaluated += 1;
            let evaluated = evaluator(values, track, sample).and_then(|m| {
                let mut loss = 0.;
                let mut violation: f64 = 0.;
                for target in &r.metrics {
                    let value = *m
                        .get(&target.path)
                        .ok_or_else(|| format!("missing metric {}", target.path))?;
                    if !value.is_finite() {
                        return Err(format!("nonfinite metric {}", target.path));
                    }
                    loss += target.weight * ((value - target.desired) / target.scale).powi(2);
                }
                for constraint in &r.constraints {
                    let value = *m
                        .get(&constraint.path)
                        .ok_or_else(|| format!("missing constraint metric {}", constraint.path))?;
                    if !value.is_finite() {
                        return Err(format!("nonfinite constraint metric {}", constraint.path));
                    }
                    violation = violation.max(
                        ((value - constraint.value).abs() - constraint.tolerance).max(0.)
                            / constraint.scale,
                    );
                }
                if !loss.is_finite() || !violation.is_finite() {
                    return Err("objective or constraint overflow".into());
                }
                Ok((loss, violation))
            });
            match evaluated {
                Ok((loss, violation)) => {
                    sample_loss += loss * (track.weight / track_weight);
                    result.violation = result.violation.max(violation);
                }
                Err(reason) => {
                    valid = false;
                    result.failures.push(EvaluationFailure {
                        sample: sample.index,
                        track: track.id.clone(),
                        reason,
                    });
                }
            }
        }
        if valid {
            result.sample_losses.push(sample_loss);
        }
    }
    if result.failures.is_empty() {
        match aggregate_losses(&result.sample_losses, &r.risk) {
            Ok(stats) => {
                result.statistics = Some(stats);
                result.feasible = result.violation == 0.;
            }
            Err(reason) => result.failures.push(EvaluationFailure {
                sample: 0,
                track: "aggregate".into(),
                reason,
            }),
        }
    }
    Ok(result)
}
/// Candidate vector together with its training evidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    /// Physical parameter values in request order.
    pub values: Vec<f64>,
    /// Training evaluation; validation is separately reported.
    pub evaluation: CandidateEvaluation,
}
/// Why the bounded search stopped. Completion does not guarantee feasibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Termination {
    /// Requested generations exhausted.
    GenerationsCompleted,
    /// Training evaluation budget exhausted.
    EvaluationBudget,
    /// Cancellation callback requested a stop.
    Cancelled,
}
/// Search result, fully serializable for immutable run snapshots.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Termination reason.
    pub termination: Termination,
    /// Best training-feasible candidate, if any.
    pub best_feasible: Option<Candidate>,
    /// Best complete but infeasible candidate, if any.
    pub best_infeasible: Option<Candidate>,
    /// Independent held-out evidence for the best training-feasible candidate.
    /// Validation failure does not retroactively change training evidence.
    pub validation: Option<CandidateEvaluation>,
    /// Training candidate attempts, including initialization.
    pub candidate_evaluations: usize,
    /// Fully completed DE rounds.
    pub generations_completed: usize,
}
fn better(a: &CandidateEvaluation, b: &CandidateEvaluation) -> bool {
    if a.feasible != b.feasible {
        return a.feasible;
    }
    let a_valid = a.complete && a.failures.is_empty() && a.statistics.is_some();
    let b_valid = b.complete && b.failures.is_empty() && b.statistics.is_some();
    if a_valid != b_valid {
        return a_valid;
    }
    if !a_valid {
        return a.failures.len() < b.failures.len();
    }
    if a.violation != b.violation {
        return a.violation < b.violation;
    }
    a.statistics.as_ref().unwrap().score < b.statistics.as_ref().unwrap().score
}
fn record(result: &mut OptimizationResult, candidate: &Candidate) {
    if !candidate.evaluation.complete {
        return;
    }
    let slot = if candidate.evaluation.feasible {
        &mut result.best_feasible
    } else {
        &mut result.best_infeasible
    };
    if slot
        .as_ref()
        .is_none_or(|current| better(&candidate.evaluation, &current.evaluation))
    {
        *slot = Some(candidate.clone());
    }
}
/// Native DE/rand/1/bin search (F=0.8, crossover=0.9), clamped to bounds.
/// Search uses only training scenarios; final validation is never selected on.
/// Cancellation is checked between physics cases; callbacks should separately
/// propagate cancellation inside expensive physics solves where needed.
pub fn optimize<F, C>(
    r: &OptimizationRequest,
    mut evaluator: F,
    mut cancelled: C,
) -> Result<OptimizationResult, String>
where
    F: FnMut(&[f64], &TrackCase, &UncertaintySample) -> Result<BTreeMap<String, f64>, String>,
    C: FnMut() -> bool,
{
    r.validate()?;
    let mut result = OptimizationResult {
        termination: Termination::GenerationsCompleted,
        best_feasible: None,
        best_infeasible: None,
        validation: None,
        candidate_evaluations: 0,
        generations_completed: 0,
    };
    let mut rng = Rng(r.seed ^ 0xa4093822299f31d0);
    let mut population = Vec::new();
    for i in 0..r.population_size {
        if cancelled() {
            result.termination = Termination::Cancelled;
            return Ok(result);
        }
        let values = if i == 0 {
            r.initial_values.clone()
        } else {
            r.variables
                .iter()
                .map(|v| v.lower + rng.unit() * (v.upper - v.lower))
                .collect()
        };
        let evaluation = evaluate_candidate(
            r,
            &values,
            SampleSplit::Training,
            &mut evaluator,
            &mut cancelled,
        )?;
        result.candidate_evaluations += 1;
        if !evaluation.complete {
            result.termination = Termination::Cancelled;
            return Ok(result);
        }
        let candidate = Candidate { values, evaluation };
        record(&mut result, &candidate);
        population.push(candidate);
    }
    'generations: for _ in 0..r.generations {
        for i in 0..population.len() {
            if cancelled() {
                result.termination = Termination::Cancelled;
                return Ok(result);
            }
            if result.candidate_evaluations >= r.max_evaluations {
                result.termination = Termination::EvaluationBudget;
                break 'generations;
            }
            let mut indices = Vec::new();
            while indices.len() < 3 {
                let j = rng.index(population.len());
                if j != i && !indices.contains(&j) {
                    indices.push(j);
                }
            }
            let forced = rng.index(r.variables.len());
            let values = r
                .variables
                .iter()
                .enumerate()
                .map(|(j, v)| {
                    if j == forced || rng.unit() < 0.9 {
                        (population[indices[0]].values[j]
                            + 0.8
                                * (population[indices[1]].values[j]
                                    - population[indices[2]].values[j]))
                            .clamp(v.lower, v.upper)
                    } else {
                        population[i].values[j]
                    }
                })
                .collect::<Vec<_>>();
            let evaluation = evaluate_candidate(
                r,
                &values,
                SampleSplit::Training,
                &mut evaluator,
                &mut cancelled,
            )?;
            result.candidate_evaluations += 1;
            if !evaluation.complete {
                result.termination = Termination::Cancelled;
                return Ok(result);
            }
            let candidate = Candidate { values, evaluation };
            record(&mut result, &candidate);
            if better(&candidate.evaluation, &population[i].evaluation) {
                population[i] = candidate;
            }
        }
        result.generations_completed += 1;
    }
    if let Some(best) = &result.best_feasible {
        let validation = evaluate_candidate(
            r,
            &best.values,
            SampleSplit::Validation,
            &mut evaluator,
            &mut cancelled,
        )?;
        if !validation.complete {
            result.termination = Termination::Cancelled;
        }
        result.validation = Some(validation);
    }
    Ok(result)
}
/// Fixed nonnegative weights on sample loss statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskWeights {
    /// Weight on arithmetic mean loss.
    pub mean: f64,
    /// Weight on population standard deviation.
    pub std_dev: f64,
    /// Weight on upper-tail conditional mean loss.
    pub cvar: f64,
    /// Worst fraction of probability mass, in (0,1].
    pub tail_fraction: f64,
}
impl RiskWeights {
    /// Validate probability and weight domains.
    pub fn validate(&self) -> Result<(), String> {
        let w = [self.mean, self.std_dev, self.cvar];
        if w.iter().any(|x| !x.is_finite() || *x < 0.)
            || w.iter().sum::<f64>() <= 0.
            || !self.tail_fraction.is_finite()
            || self.tail_fraction <= 0.
            || self.tail_fraction > 1.
        {
            return Err("invalid risk weights or tail fraction".into());
        }
        Ok(())
    }
}
/// Loss statistics across equally likely uncertainty samples.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LossStatistics {
    /// Arithmetic mean.
    pub mean: f64,
    /// Population standard deviation (not sample-estimator correction).
    pub std_dev: f64,
    /// Upper tail mean, including fractional boundary mass.
    pub cvar: f64,
    /// Weighted sum of mean, standard deviation and CVaR.
    pub score: f64,
}
/// Calculate exact empirical tail mass, without rounding tail size to an integer.
pub fn aggregate_losses(losses: &[f64], risk: &RiskWeights) -> Result<LossStatistics, String> {
    risk.validate()?;
    if losses.is_empty() || losses.iter().any(|x| !x.is_finite() || *x < 0.) {
        return Err("losses must be finite nonnegative and nonempty".into());
    }
    let n = losses.len() as f64;
    let mean = losses.iter().map(|x| x / n).sum::<f64>();
    let std_dev = (losses.iter().map(|x| (x - mean).powi(2) / n).sum::<f64>()).sqrt();
    let mut sorted = losses.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mass = (n * risk.tail_fraction).max(f64::MIN_POSITIVE);
    let mut remaining = mass;
    let mut cvar = 0.;
    for x in sorted {
        let take = remaining.min(1.);
        cvar += x * (take / mass);
        remaining -= take;
        if remaining <= 0. {
            break;
        }
    }
    let score = risk.mean * mean + risk.std_dev * std_dev + risk.cvar * cvar;
    if !score.is_finite() {
        return Err("objective overflow".into());
    }
    Ok(LossStatistics {
        mean,
        std_dev,
        cvar,
        score,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failures_missing_metrics_and_heldout_rejection_never_become_success() {
        let r = request();
        let failed = optimize(
            &r,
            |_, _, _| Err("solver did not converge".into()),
            || false,
        )
        .unwrap();
        assert!(failed.best_feasible.is_none());
        assert!(failed.validation.is_none());
        assert!(failed
            .best_infeasible
            .unwrap()
            .evaluation
            .statistics
            .is_none());
        let missing = evaluate_candidate(
            &r,
            &[1.],
            SampleSplit::Training,
            &mut |_, _, _| Ok(BTreeMap::new()),
            &mut || false,
        )
        .unwrap();
        assert!(!missing.feasible);
        assert!(missing.statistics.is_none());
        let result = optimize(
            &r,
            |x, _, s| {
                if s.split == SampleSplit::Validation {
                    Err("held-out instability".into())
                } else {
                    Ok(BTreeMap::from([("time".into(), x[0] - 1.)]))
                }
            },
            || false,
        )
        .unwrap();
        assert!(result.best_feasible.is_some());
        assert!(!result.validation.unwrap().feasible);
    }
    #[test]
    fn two_track_interior_optimum_matches_weighted_analytic_solution() {
        let mut r = request();
        r.generations = 80;
        r.max_evaluations = 1000;
        let result = optimize(
            &r,
            |x, t, _| {
                Ok(BTreeMap::from([(
                    "time".into(),
                    x[0] - if t.id == "a" { 0. } else { 2. },
                )]))
            },
            || false,
        )
        .unwrap();
        // (x² + 3(x-2)²)/16 has unique minimum at x=1.5.
        assert!((result.best_feasible.unwrap().values[0] - 1.5).abs() < 1e-6);
        assert!(result.validation.unwrap().feasible);
        let mut calls = 0;
        let e = evaluate_candidate(
            &r,
            &[1.],
            SampleSplit::Training,
            &mut |_, _, _| Ok(BTreeMap::from([("time".into(), 1.)])),
            &mut || {
                calls += 1;
                calls > 2
            },
        )
        .unwrap();
        assert!(!e.complete);
        assert!(!e.feasible);
        assert!(e.statistics.is_none());
        assert_eq!(e.cases_evaluated, 2);
    }
    #[test]
    fn bounded_search_repeats_and_heldout_is_separate() {
        let r = request();
        let eval = |x: &[f64], t: &TrackCase, _: &UncertaintySample| {
            Ok(BTreeMap::from([(
                "time".into(),
                x[0] - if t.id == "a" { 3. } else { 4. },
            )]))
        };
        let a = optimize(&r, eval, || false).unwrap();
        let b = optimize(&r, eval, || false).unwrap();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
        let best = a.best_feasible.unwrap();
        assert!((best.values[0] - 2.).abs() < 1e-10);
        assert!(a.validation.unwrap().feasible);
        assert!(a.candidate_evaluations <= r.max_evaluations);
        let train = uncertainty_samples(&r, SampleSplit::Training).unwrap();
        let validation = uncertainty_samples(&r, SampleSplit::Validation).unwrap();
        assert_ne!(train[0].deltas, validation[0].deltas);
        assert_eq!(
            train,
            uncertainty_samples(&r, SampleSplit::Training).unwrap()
        );
        let cancelled = optimize(&r, eval, || true).unwrap();
        assert_eq!(cancelled.termination, Termination::Cancelled);
        assert!(cancelled.best_feasible.is_none());
    }
    fn request() -> OptimizationRequest {
        OptimizationRequest {
            variables: vec![Variable {
                path: "x".into(),
                lower: 0.,
                upper: 2.,
            }],
            initial_values: vec![0.],
            tracks: vec![
                TrackCase {
                    id: "a".into(),
                    weight: 1.,
                },
                TrackCase {
                    id: "b".into(),
                    weight: 3.,
                },
            ],
            metrics: vec![MetricTarget {
                path: "time".into(),
                desired: 0.,
                scale: 2.,
                weight: 1.,
            }],
            constraints: vec![],
            uncertainty: vec![Uncertainty {
                path: "grip".into(),
                half_range: 0.1,
            }],
            training_samples: 3,
            validation_samples: 5,
            seed: 123,
            population_size: 12,
            generations: 40,
            max_evaluations: 500,
            risk: RiskWeights {
                mean: 1.,
                std_dev: 0.,
                cvar: 0.,
                tail_fraction: 0.2,
            },
        }
    }
    #[test]
    fn weighted_tracks_fixed_scales_and_hard_constraints() {
        let mut r = request();
        let mut eval = |_: &[f64], t: &TrackCase, _: &UncertaintySample| {
            Ok(BTreeMap::from([(
                "time".into(),
                if t.id == "a" { 2. } else { 4. },
            )]))
        };
        let e =
            evaluate_candidate(&r, &[1.], SampleSplit::Training, &mut eval, &mut || false).unwrap();
        assert!(e.feasible);
        assert_eq!(e.statistics.unwrap().score, 3.25);
        r.constraints.push(MetricConstraint {
            path: "time".into(),
            value: 3.,
            tolerance: 0.5,
            scale: 1.,
        });
        let e =
            evaluate_candidate(&r, &[1.], SampleSplit::Training, &mut eval, &mut || false).unwrap();
        assert!(!e.feasible);
        assert!((e.violation - 0.5).abs() < 1e-12);
        let e = evaluate_candidate(
            &r,
            &[1.],
            SampleSplit::Training,
            &mut |_, _, _| Err("physics failed".into()),
            &mut || false,
        )
        .unwrap();
        assert!(!e.feasible);
        assert!(e.statistics.is_none());
        assert_eq!(e.failures.len(), 6);
        r.constraints[0].tolerance = -1.;
        assert!(r.validate().is_err());
    }
    #[test]
    fn exact_mean_population_spread_and_fractional_tail() {
        let s = aggregate_losses(
            &[1., 2., 3., 4.],
            &RiskWeights {
                mean: 1.,
                std_dev: 2.,
                cvar: 3.,
                tail_fraction: 0.375,
            },
        )
        .unwrap();
        assert_eq!(s.mean, 2.5);
        assert!((s.std_dev - 1.25_f64.sqrt()).abs() < 1e-12);
        assert!((s.cvar - 11. / 3.).abs() < 1e-12);
        assert!((s.score - (2.5 + 2. * 1.25_f64.sqrt() + 11.)).abs() < 1e-12);
    }
}
