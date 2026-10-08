//! Bounded stochastic multi-track lap optimization.
//!
//! This module searches design variable values (each an opaque path meaningful
//! only to a caller-supplied physics callback) for the values that minimize a
//! weighted set of lap metric targets across a weighted set of tracks, subject to
//! hard metric constraints, while staying robust to bounded random uncertainty in
//! the underlying physical inputs. Every candidate is scored by actually
//! re-running the real lap physics through the evaluator callback passed to
//! [evaluate_candidate] and [optimize] — there is no metamodel or surrogate
//! standing in for the physics anywhere in this search.
//!
//! The search itself is a native, dependency-free bounded differential evolution
//! (DE/rand/1/bin). It is structurally parallel to the suspension-only optimizer
//! in `crate::optimize` — both use a feasibility-first candidate ordering and a
//! training/validation sample split, so a search can never be judged on the same
//! random samples it was allowed to select against. Unlike `crate::optimize`'s
//! session-based checkpoint/resume API, though, this module has none: [optimize]
//! is a single blocking call, and a run interrupted partway through (by the
//! cancellation callback or by the process stopping) cannot be resumed — it has
//! to be restarted from scratch.
//!
//! The JSON-pointer-addressed design variables, weighted tracks, metric
//! targets/constraints, training/validation uncertainty sampling, and
//! mean/standard-deviation/CVaR risk weighting (see [RiskWeights] and
//! [aggregate_losses]) implemented here follow the multi-track stochastic
//! optimization design in
//! <https://github.com/Ixiandesign/FKY-LAPSIM/blob/main/docs/fsae-platform-audit.md>.
//! This module itself never parses or dereferences a variable/metric `path`; it
//! is forwarded verbatim to the evaluator callback, which owns that
//! interpretation.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
/// One independently bounded design parameter searched over.
///
/// `path` is an opaque identifier forwarded verbatim to the evaluator callback
/// passed to [evaluate_candidate]/[optimize]; this module does not interpret it
/// itself (in the platform's usage it addresses a numeric leaf of the full-car
/// configuration — see the module documentation).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Variable {
    /// Evaluator-defined parameter identifier. Must be nonempty and unique among
    /// an [OptimizationRequest]'s `variables`.
    pub path: String,
    /// Inclusive lower bound, in the parameter's SI unit. Must be finite and
    /// strictly less than `upper`.
    pub lower: f64,
    /// Inclusive upper bound, in the parameter's SI unit. Must be finite, and
    /// `upper - lower` must also be finite.
    pub upper: f64,
}
/// One track in a weighted multi-track objective, identified by a caller-defined
/// key that the evaluator callback resolves against its own immutable track
/// inputs (this module holds no track data itself).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackCase {
    /// Stable track identity, unique among an [OptimizationRequest]'s `tracks`;
    /// passed to the evaluator callback unchanged.
    pub id: String,
    /// Positive relative importance of this track. Track losses are combined as
    /// a weighted average (see [evaluate_candidate]), so only the ratio between
    /// tracks' weights matters, not their absolute scale.
    pub weight: f64,
}
/// One objective term: drive a metric toward `desired`, contributing
/// `weight * ((value - desired) / scale)^2` to a candidate's loss on each
/// track/sample (see [evaluate_candidate]). `scale` is fixed at request time and
/// never reestimated from the data during the search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricTarget {
    /// Exact metric key the evaluator callback's returned map must contain for
    /// every case; missing or nonfinite values fail that case (see
    /// [EvaluationFailure]).
    pub path: String,
    /// Desired value, in the metric's own SI unit.
    pub desired: f64,
    /// Positive physical scale: the residual magnitude that counts as one unit of
    /// normalized squared error. Must be finite and strictly positive.
    pub scale: f64,
    /// Nonnegative importance relative to other targets. At least one target in
    /// an [OptimizationRequest] must have positive weight, or the request fails
    /// validation.
    pub weight: f64,
}
/// A hard feasibility limit: the named metric must stay within `tolerance` of
/// `value` on every evaluated track and uncertainty sample, or the candidate is
/// infeasible (see [CandidateEvaluation]'s `violation` field).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricConstraint {
    /// Exact metric key the evaluator callback's returned map must contain for
    /// every case; missing or nonfinite values fail that case.
    pub path: String,
    /// Required physical value, in the metric's own SI unit.
    pub value: f64,
    /// Nonnegative absolute tolerance around `value`, in the same unit. Zero
    /// means the metric must hit `value` exactly (subject to floating-point
    /// precision).
    pub tolerance: f64,
    /// Positive physical scale used to normalize how far outside tolerance a
    /// violation is, so violations across different constraints can be compared
    /// and maxed together.
    pub scale: f64,
}
/// One source of bounded random uncertainty. Each sample independently draws a
/// perturbation uniformly from `[-half_range, half_range]` for this `path` and
/// hands it to the evaluator callback (see [UncertaintySample]); this module does
/// not apply the perturbation itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Uncertainty {
    /// Parameter identifier to perturb, forwarded to the evaluator alongside the
    /// sampled delta; interpreted the same opaque way as [Variable]'s `path`.
    pub path: String,
    /// Nonnegative absolute half range, in the parameter's SI unit. Zero
    /// disables perturbation for this path while still recording it as a source.
    pub half_range: f64,
}
/// Which of two independent random streams an [UncertaintySample] belongs to.
/// [optimize] searches and selects candidates using only `Training` samples;
/// `Validation` samples come from a separate stream and are only ever used once,
/// after selection, to score the winning candidate — so held-out evidence can
/// never influence which candidate was chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleSplit {
    /// Training scenarios: the same fixed set (common random numbers) is reused
    /// for every candidate the search evaluates, so candidates are compared
    /// against identical noise rather than independently resampled noise.
    Training,
    /// Independent final validation scenarios, generated from a different salt
    /// than `Training` and never seen during the search itself.
    Validation,
}
/// One realized, replayable uncertainty draw: the additive perturbation applied
/// to every uncertainty path for one sample, so the same sample can be
/// regenerated deterministically by calling [uncertainty_samples] again with the
/// same request and split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UncertaintySample {
    /// Which random stream this sample was drawn from; see [SampleSplit].
    pub split: SampleSplit,
    /// Stable zero-based index within `split`'s stream (`0..training_samples` or
    /// `0..validation_samples`).
    pub index: usize,
    /// Additive SI-unit delta for each uncertainty path, keyed by each
    /// [Uncertainty]'s `path`.
    pub deltas: BTreeMap<String, f64>,
}
/// A complete, immutable multi-track lap optimization problem: design variables,
/// weighted tracks, the objective (targets/constraints), uncertainty sampling,
/// and a bounded search budget. Call [OptimizationRequest::validate] to check it,
/// or simply pass it to [optimize]/[evaluate_candidate]/[uncertainty_samples],
/// which each validate it themselves before doing any work.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationRequest {
    /// Design variables, in the order candidate value vectors use. At most 128,
    /// each with a unique nonempty path and finite, correctly ordered bounds.
    pub variables: Vec<Variable>,
    /// Starting candidate values, one per `variables` entry in the same order.
    /// Must already lie within each variable's bounds; this candidate is always
    /// evaluated first and seeds the initial population.
    pub initial_values: Vec<f64>,
    /// The weighted set of tracks every candidate is scored against. At most
    /// 1000, each with a unique id and a positive, finite weight.
    pub tracks: Vec<TrackCase>,
    /// The metrics to drive toward their desired values; see [MetricTarget]. At
    /// least one target is required, and the weights must sum to strictly
    /// positive.
    pub metrics: Vec<MetricTarget>,
    /// Hard feasibility limits checked on every track and every uncertainty
    /// sample; see [MetricConstraint]. May be empty (no hard constraints).
    pub constraints: Vec<MetricConstraint>,
    /// Independent bounded-uniform perturbations sampled once per uncertainty
    /// sample and forwarded to the evaluator; see [Uncertainty]. May be empty (no
    /// uncertainty — every sample is then identical).
    pub uncertainty: Vec<Uncertainty>,
    /// Number of training uncertainty samples per candidate, drawn once per
    /// request/split from [uncertainty_samples] and reused (common random
    /// numbers) across every candidate the search evaluates. Must be in
    /// `1..=10_000`.
    pub training_samples: usize,
    /// Number of held-out validation uncertainty samples, drawn from an
    /// independent stream and used only once, on the final selected candidate.
    /// Must be in `1..=10_000`.
    pub validation_samples: usize,
    /// Seed for both the DE search's mutation RNG and the training/validation
    /// sample RNGs; the same seed with the same request reproduces
    /// bit-identical results.
    pub seed: u64,
    /// Differential-evolution population size. Must be in `4..=1000`; the
    /// DE/rand/1/bin mutation needs at least three other population members
    /// besides the one being replaced.
    pub population_size: usize,
    /// Maximum number of complete DE mutation rounds to run. The search may stop
    /// earlier if `max_evaluations` is exhausted first or the cancellation
    /// callback fires. Must be at most 100,000.
    pub generations: usize,
    /// Hard cap on training candidate evaluations, including population
    /// initialization. Must be at least `population_size` and at most
    /// 1,000,000. The single final held-out validation evaluation, if any, is
    /// not counted against this budget.
    pub max_evaluations: usize,
    /// How this request's per-sample losses are aggregated into one score; see
    /// [RiskWeights].
    pub risk: RiskWeights,
}
fn unique<'a>(items: impl Iterator<Item = &'a str>) -> bool {
    let mut seen = BTreeSet::new();
    items
        .into_iter()
        .all(|s| !s.trim().is_empty() && seen.insert(s))
}
impl OptimizationRequest {
    /// Check every field's domain and cross-field invariant, and bound the total
    /// work the request could schedule.
    ///
    /// Called automatically by [uncertainty_samples], [evaluate_candidate], and
    /// [optimize] before they do any work, so callers do not need to call this
    /// directly unless they want to validate a request before use.
    ///
    /// # Errors
    ///
    /// Returns `Err` describing the first violated invariant, including: empty,
    /// duplicate-path, too many (>128), or out-of-order/nonfinite `variables`
    /// bounds; `initial_values` not matching `variables` in length or bounds;
    /// empty, duplicate-id, too many (>1000), or non-positive/nonfinite-weight
    /// `tracks`; empty or duplicate-path `metrics`, a nonfinite/non-positive
    /// `scale`, a nonfinite/negative `weight`, or a total target weight that is
    /// not strictly positive; duplicate-path `constraints` or a
    /// nonfinite/negative `tolerance` or nonfinite/non-positive `scale`;
    /// duplicate-path or nonfinite/negative `uncertainty` half-ranges;
    /// `training_samples`/`validation_samples` outside `1..=10_000`;
    /// `population_size` outside `4..=1000`; `generations` above `100_000`;
    /// `max_evaluations` below `population_size` or above 1,000,000; or a total
    /// scheduled physics-case count (`tracks.len() * (training_samples *
    /// max_evaluations + validation_samples)`) exceeding 100 million, which
    /// guards against a request that would take an unreasonable amount of
    /// compute regardless of whether each individual field looks legal on its
    /// own.
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
/// Deterministically generate one split's full set of uncertainty samples (see
/// [UncertaintySample]) from `request.seed`, independently of any candidate or
/// evaluation order — calling this twice with the same request and split always
/// returns identical samples, and training/validation samples are drawn from
/// distinct salted streams so they never coincide.
///
/// # Errors
///
/// Returns `Err` if `request` fails [OptimizationRequest::validate].
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
/// One failed case within a [CandidateEvaluation]: either the evaluator callback
/// itself returned `Err`, or it returned a metric map missing a required
/// [MetricTarget]/[MetricConstraint] path, or that metric's value was nonfinite.
/// Any single failure means the whole candidate evaluation cannot produce
/// statistics — see [evaluate_candidate].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationFailure {
    /// Zero-based index of the [UncertaintySample] this case belongs to.
    pub sample: usize,
    /// Identity of the [TrackCase] this case belongs to (or `"aggregate"` for a
    /// failure raised while aggregating losses across samples rather than within
    /// one track/sample case).
    pub track: String,
    /// Human-readable diagnostic: either the evaluator's own error string, or one
    /// of this module's own messages:
    /// `"missing metric <path>"`, `"nonfinite metric <path>"`,
    /// `"nonfinite constraint metric <path>"`, or
    /// `"objective or constraint overflow"`.
    pub reason: String,
}
/// The result of scoring one candidate's variable values against every
/// track/uncertainty-sample case in an [OptimizationRequest] (see
/// [evaluate_candidate]). A single failed case anywhere means `statistics` stays
/// `None` and `feasible` stays `false` — a failure can never yield a valid score,
/// regardless of how the other cases scored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    /// Whether every requested track/sample case was attempted. `false` only
    /// when the cancellation callback fired before all cases ran; the cases that
    /// had already run by then are still reflected in
    /// `cases_evaluated`/`failures`.
    pub complete: bool,
    /// `true` only when `complete` is `true`, every case succeeded with finite
    /// required metrics (`failures` is empty), and `violation` is exactly zero.
    pub feasible: bool,
    /// The risk-aggregated loss statistics (see [aggregate_losses]), present
    /// only when `complete` is `true` and `failures` is empty — i.e. every
    /// physics case and every required metric was valid. `None` whenever any
    /// case failed, even if the only remaining problem is a nonzero `violation`.
    pub statistics: Option<LossStatistics>,
    /// Maximum, across every evaluated case, of the normalized excess by which a
    /// [MetricConstraint] was violated (0 when no constraint was violated
    /// anywhere, including when `constraints` is empty).
    pub violation: f64,
    /// Track-weighted loss for each uncertainty sample that completed with no
    /// case failures, in sample order. A sample containing any failed case
    /// contributes no entry here, so this can be shorter than the requested
    /// sample count even when `complete` is `true`.
    pub sample_losses: Vec<f64>,
    /// Every failed case's diagnostic; see [EvaluationFailure]. Empty for a
    /// fully valid evaluation.
    pub failures: Vec<EvaluationFailure>,
    /// Number of physics callback invocations actually made (one per
    /// track/sample case attempted, whether it succeeded or failed).
    pub cases_evaluated: usize,
}
/// Score one candidate's `values` against every track in `r.tracks`, for every
/// uncertainty sample in `split` (see [uncertainty_samples]), by calling
/// `evaluator` once per track/sample case — always the real lap physics, never a
/// metamodel or surrogate.
///
/// `evaluator` receives the candidate's values, the [TrackCase] being scored, and
/// the [UncertaintySample] to apply, and must return a map of metric name to
/// value covering every [MetricTarget]/[MetricConstraint] path relevant to that
/// track. Using the same `split` and request always replays the same uncertainty
/// samples (common random numbers), so repeated candidates are compared against
/// identical noise rather than independently resampled noise.
///
/// `cancelled` is polled between cases; once it returns `true`, evaluation stops
/// immediately and the returned [CandidateEvaluation] has `complete: false`
/// reflecting only the cases already attempted. A failed or metric-incomplete
/// case is recorded as an [EvaluationFailure] rather than aborting the remaining
/// cases, so one bad case does not hide diagnostics for the others.
///
/// # Errors
///
/// Returns `Err` if `r` fails [OptimizationRequest::validate], or if `values`
/// does not have exactly `r.variables.len()` entries, or any entry is nonfinite
/// or outside its variable's bounds. These are request/candidate-shape errors;
/// physics or metric failures from `evaluator` itself are never surfaced as
/// `Err` here — they are recorded in the returned `failures` field instead,
/// leaving `statistics` as `None` and `feasible` as `false`.
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
/// One design point explored by the search: its variable values together with
/// the [CandidateEvaluation] scoring those values against the training split.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    /// Design variable values, in [OptimizationRequest]'s `variables` order.
    pub values: Vec<f64>,
    /// This candidate's evaluation against `SampleSplit::Training` samples.
    /// Held-out validation, when performed, is reported separately on
    /// [OptimizationResult]'s `validation` field, not here.
    pub evaluation: CandidateEvaluation,
}
/// Why [optimize] stopped searching. Any variant, including `Cancelled`, may or
/// may not have found a feasible candidate — check [OptimizationResult]'s
/// `best_feasible` field, not `termination`, to know whether the search
/// succeeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Termination {
    /// Every requested generation ran to completion without hitting the
    /// evaluation budget or cancellation.
    GenerationsCompleted,
    /// `max_evaluations` training candidate evaluations were spent before all
    /// requested generations completed.
    EvaluationBudget,
    /// The cancellation callback returned `true`, stopping the search (and
    /// skipping final validation) before it otherwise would have finished.
    Cancelled,
}
/// The complete outcome of one [optimize] call, fully serializable so a run's
/// result can be persisted as an immutable snapshot alongside the request that
/// produced it. There is no partial/checkpointed form of this result — it only
/// ever exists once a call to [optimize] returns (see the module documentation
/// on the absence of resume support).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Why the search stopped; see [Termination].
    pub termination: Termination,
    /// The best training-feasible candidate found across initialization and
    /// every generation, by the feasibility-first, then-lowest-score ordering
    /// used throughout this module. `None` if no candidate was ever feasible on
    /// training samples.
    pub best_feasible: Option<Candidate>,
    /// The best candidate that completed evaluation but was not feasible
    /// (constraint violation and/or no valid statistics), tracked only so a
    /// search that never found a feasible point still returns its closest
    /// attempt. `None` whenever `best_feasible` is `Some`, or no candidate
    /// completed at all.
    pub best_infeasible: Option<Candidate>,
    /// `best_feasible` re-evaluated once against independent held-out
    /// `SampleSplit::Validation` samples, or `None` if no feasible candidate was
    /// found (so there was nothing to validate). A validation evaluation that
    /// turns out infeasible does not retroactively change the training evidence
    /// recorded on `best_feasible` — the two are reported separately and must be
    /// read together.
    pub validation: Option<CandidateEvaluation>,
    /// Total training candidate evaluations spent, including population
    /// initialization; bounded by [OptimizationRequest]'s `max_evaluations`.
    pub candidate_evaluations: usize,
    /// Number of DE mutation rounds that ran to completion before the search
    /// stopped (may be fewer than [OptimizationRequest]'s `generations` if the
    /// evaluation budget or cancellation ended the search first).
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
/// Search for design variable values minimizing `r`'s multi-track, risk-weighted
/// lap objective, using a native DE/rand/1/bin differential evolution (mutation
/// factor F=0.8, crossover probability CR=0.9), clamped to each variable's
/// bounds, with every candidate scored by `evaluator` against real lap physics —
/// never a metamodel or surrogate.
///
/// The population is seeded with `r.initial_values` as candidate 0 and
/// `r.population_size - 1` further candidates drawn uniformly from each
/// variable's bounds, all evaluated against `SampleSplit::Training` samples.
/// Each subsequent generation replaces every population member in turn with a
/// DE/rand/1/bin trial vector (three other distinct population members combine
/// as `a + F * (b - c)`, per-dimension crossover against the current member with
/// one dimension always forced from the trial) if the trial evaluates better
/// under the feasibility-first ordering used throughout this module; the search
/// never evaluates against validation samples while selecting. Once the search
/// stops, if a training-feasible candidate was found, it is re-evaluated exactly
/// once against independent `SampleSplit::Validation` samples and the result
/// recorded on [OptimizationResult]'s `validation` field — this held-out
/// evidence never feeds back into which candidate was selected.
///
/// The search stops on whichever comes first: `r.generations` complete rounds,
/// `r.max_evaluations` training evaluations spent, or `cancelled` returning
/// `true` (checked between physics cases, i.e. between individual
/// track/uncertainty-sample calls to `evaluator`); see [Termination].
/// `evaluator` and `cancelled` run on the caller's thread with no internal
/// parallelism — callers with expensive physics solves should have `evaluator`
/// itself poll for cancellation internally if they need to abort mid-solve,
/// since `cancelled` is only checked between whole cases here.
///
/// This is a single, synchronous, non-resumable call: unlike `crate::optimize`'s
/// session-based checkpoint/resume API, there is no way to pause an in-progress
/// search and continue it later. If `cancelled` fires or the process is
/// interrupted, all search progress is lost and a fresh call starts over from a
/// new population.
///
/// # Errors
///
/// Returns `Err` if `r` fails [OptimizationRequest::validate]. Once validated,
/// every candidate generated internally is constructed to already lie within
/// bounds, so subsequent internal evaluations are not expected to error;
/// per-candidate physics or metric failures from `evaluator` are recorded in
/// that candidate's evaluation `failures` rather than propagated as an `Err`
/// from this function.
///
/// # Examples
///
/// ```text
/// // 1. Build an OptimizationRequest: one Variable per design parameter path,
/// //    a weighted TrackCase per track, one or more MetricTarget/MetricConstraint
/// //    entries naming the metrics `evaluator` will return, optional Uncertainty
/// //    entries, training/validation sample counts, a seed, and a search budget
/// //    (population_size/generations/max_evaluations) plus RiskWeights.
/// // 2. Call `optimize(&request, evaluator, cancelled)`, where `evaluator`
/// //    re-solves the real lap simulation for the given values/track/sample and
/// //    returns the named metrics, and `cancelled` polls for a stop request.
/// // 3. Inspect the OptimizationResult: `best_feasible` (and its `validation`
/// //    counterpart) for the winning design, `best_infeasible` if no feasible
/// //    candidate was found, and `termination` for why the search stopped.
/// ```
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
/// How a candidate's per-sample losses are combined into one scalar score (see
/// [LossStatistics]'s `score` field and [aggregate_losses]): a fixed
/// nonnegative-weighted sum of the mean, standard deviation, and CVaR
/// (conditional value at risk) of the loss distribution across uncertainty
/// samples. Weighting `std_dev`/`cvar` above zero makes the search prefer
/// designs that are consistent across the sampled uncertainty, not just good on
/// average.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskWeights {
    /// Nonnegative weight on the arithmetic mean loss across samples.
    pub mean: f64,
    /// Nonnegative weight on the population standard deviation of loss across
    /// samples (a variability/consistency penalty).
    pub std_dev: f64,
    /// Nonnegative weight on the upper-tail conditional mean loss (CVaR): the
    /// mean loss over the worst `tail_fraction` of samples.
    pub cvar: f64,
    /// Worst fraction of the sample loss distribution's probability mass that
    /// `cvar` averages over, in `(0, 1]`. `1.0` makes CVaR equal the plain mean.
    pub tail_fraction: f64,
}
impl RiskWeights {
    /// Check that `mean`/`std_dev`/`cvar` are finite, nonnegative, and sum to
    /// strictly positive, and that `tail_fraction` is finite and in `(0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any weight is nonfinite or negative, if all three
    /// weights sum to zero (or the sum is otherwise nonfinite), or if
    /// `tail_fraction` is nonfinite, non-positive, or greater than 1.
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
/// Risk statistics of a candidate's per-sample losses, treating every
/// uncertainty sample as equally likely; see [aggregate_losses] and
/// [RiskWeights].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LossStatistics {
    /// Arithmetic mean loss across samples.
    pub mean: f64,
    /// Population standard deviation of loss across samples (divides by `n`, not
    /// the Bessel-corrected `n - 1` sample-variance estimator).
    pub std_dev: f64,
    /// Mean loss over the worst `tail_fraction` of the sample mass (CVaR),
    /// including a proportional contribution from the sample straddling the
    /// tail boundary rather than rounding the tail to a whole number of
    /// samples.
    pub cvar: f64,
    /// `risk.mean * mean + risk.std_dev * std_dev + risk.cvar * cvar`: the
    /// single scalar the search minimizes, compared candidate-to-candidate using
    /// the feasibility-first ordering described on [optimize].
    pub score: f64,
}
/// Reduce per-sample `losses` (one per uncertainty sample, all equally likely) to
/// [LossStatistics] under `risk`'s weighting. The CVaR tail mass
/// (`losses.len() as f64 * risk.tail_fraction`) is computed exactly rather than
/// rounded to an integer sample count, so the sample straddling the tail
/// boundary contributes only its fractional share.
///
/// # Errors
///
/// Returns `Err` if `risk` fails [RiskWeights::validate], if `losses` is empty
/// or contains a nonfinite or negative value, or if the resulting weighted score
/// itself overflows to nonfinite.
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
