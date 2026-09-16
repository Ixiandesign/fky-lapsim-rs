//! Bounded, feasibility-first DE with common bounded uncertainty and resumable batches.
use crate::{CornerId, Error, Motion, Project, RideRequest, RideRun};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashSet,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
fn err(s: impl Into<String>) -> Error {
    Error { message: s.into() }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VariableKind {
    Continuous { lower: f64, upper: f64 },
    Discrete { choices: Vec<String> },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub path: String,
    #[serde(flatten)]
    pub kind: VariableKind,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub source: String,
    pub destination: String,
    pub factor: f64,
    pub offset: f64,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aggregation {
    MeanSquared,
    WorstSquared,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    pub corner: Option<CornerId>,
    pub metric: String,
    pub value: f64,
    pub values: Option<Vec<f64>>,
    pub scale: f64,
    pub weight: f64,
    pub aggregation: Aggregation,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub corner: Option<CornerId>,
    pub metric: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub scale: f64,
}
/// Same nonempty factor name shares one uniform latent; unnamed inputs are independent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perturbation {
    pub path: String,
    pub half_range: f64,
    pub factor: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizationRequest {
    pub variables: Vec<Variable>,
    pub relations: Vec<Relation>,
    pub targets: Vec<Target>,
    pub constraints: Vec<Constraint>,
    pub scenarios: Vec<Motion>,
    pub ride_request: Option<RideRequest>,
    pub uncertainty: Vec<Perturbation>,
    pub training_samples: usize,
    pub validation_samples: usize,
    pub seed: u64,
    pub population_size: usize,
    pub generations: usize,
    pub max_evaluations: usize,
    pub max_seconds: f64,
    pub workers: usize,
}
impl Default for OptimizationRequest {
    fn default() -> Self {
        Self {
            variables: vec![],
            relations: vec![],
            targets: vec![],
            constraints: vec![],
            scenarios: vec![],
            ride_request: None,
            uncertainty: vec![],
            training_samples: 0,
            validation_samples: 8,
            seed: 1,
            population_size: 12,
            generations: 20,
            max_evaluations: 256,
            max_seconds: 60.,
            workers: 1,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Failure {
    pub sample: usize,
    pub case: String,
    pub reason: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    pub complete: bool,
    pub feasible: bool,
    pub score: Option<f64>,
    pub contributions: Vec<f64>,
    /// Target-major values at nominal uncertainty, in scenario order (one for scalar domains).
    pub nominal_target_values: Vec<Vec<f64>>,
    pub sample_count: usize,
    pub failed_samples: usize,
    pub not_evaluated_cases: usize,
    pub violation: f64,
    pub requested_cases: usize,
    pub completed_cases: usize,
    pub physics_cases_completed: usize,
    pub physics_cases_failed: usize,
    pub failed_cases: usize,
    pub failures: Vec<Failure>,
    pub failures_truncated: bool,
    /// Compact exact failure mask, including failures whose textual detail was truncated.
    pub failed_case_bits: Vec<u64>,
    pub worst_squared_residual: f64,
    pub worst_sample: usize,
    pub model_fidelity: Option<String>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    Success,
    Failure,
    NotEvaluated,
}
impl Evaluation {
    /// case=0 is construction/property checks, 1..=S are Motions, S+1 is an optional ride.
    pub fn case_status(&self, sample: usize, case: usize) -> Option<CaseStatus> {
        let per = self.requested_cases.checked_div(self.sample_count)?;
        if sample >= self.sample_count || case >= per {
            return None;
        }
        let index = sample * per + case;
        Some(if index >= self.completed_cases {
            CaseStatus::NotEvaluated
        } else if self.failed_case_bits[index / 64] & (1u64 << (index % 64)) != 0 {
            CaseStatus::Failure
        } else {
            CaseStatus::Success
        })
    }
    pub fn better_than(&self, b: &Self) -> bool {
        if self.complete != b.complete {
            return self.complete;
        }
        if self.feasible != b.feasible {
            return self.feasible;
        }
        if self.feasible {
            return self.score < b.score;
        }
        (self.failed_cases, self.violation) < (b.failed_cases, b.violation)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub values: Vec<f64>,
    /// Absent when construction or project validation failed.
    pub project: Option<Project>,
    pub evaluation: Evaluation,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub status: String,
    pub baseline: Option<Candidate>,
    pub best_feasible: Option<Candidate>,
    pub best_infeasible: Option<Candidate>,
    pub validation: Option<Evaluation>,
    pub training_revalidation: Option<Evaluation>,
    pub candidate_attempts: usize,
    pub baseline_attempts: usize,
    pub search_attempts: usize,
    pub validation_attempts: usize,
    pub physics_cases_completed: usize,
    pub physics_cases_failed: usize,
    pub generation: usize,
    pub elapsed_seconds: f64,
}
/// Explicit numeric-leaf whitelist; optional numeric leaves must already exist.
pub fn parameter_registry(p: &Project) -> Vec<String> {
    let mut out = vec!["/chassis/sprung_mass".into()];
    for field in ["center_of_mass", "inertia"] {
        for k in 0..3 {
            out.push(format!("/chassis/{field}/{k}"));
        }
    }
    for (i, c) in p.corners.iter().enumerate() {
        let base = format!("/corners/{i}");
        for field in [
            "upper_front",
            "upper_rear",
            "lower_front",
            "lower_rear",
            "upper_ball",
            "lower_ball",
            "steering_inner",
            "steering_outer",
            "wheel_center",
            "pushrod_pickup",
            "rocker_pushrod",
            "rocker_shock",
            "shock_chassis",
        ] {
            for k in 0..3 {
                out.push(format!("{base}/{field}/{k}"));
            }
        }
        for field in ["spindle_axis", "rack_axis", "rocker_axis"] {
            for j in 0..2 {
                for k in 0..3 {
                    out.push(format!("{base}/{field}/{j}/{k}"));
                }
            }
        }
        for field in ["tire_radius", "tire_width"] {
            out.push(format!("{base}/{field}"));
        }
        for field in [
            "spring_rate",
            "preload",
            "compression_damping",
            "rebound_damping",
        ] {
            out.push(format!("{base}/spring_damper/{field}"));
        }
        for (field, v) in [
            ("min_length_m", c.spring_damper.min_length_m),
            ("max_length_m", c.spring_damper.max_length_m),
        ] {
            if v.is_some() {
                out.push(format!("{base}/spring_damper/{field}"));
            }
        }
        for (field, b) in [
            ("upper_arm", &c.component_masses.upper_arm),
            ("lower_arm", &c.component_masses.lower_arm),
            ("knuckle", &c.component_masses.knuckle),
            ("rocker", &c.component_masses.rocker),
        ] {
            if b.is_some() {
                let q = format!("{base}/component_masses/{field}");
                out.push(format!("{q}/mass_kg"));
                for k in 0..3 {
                    out.push(format!("{q}/center_of_mass/{k}"));
                    for j in 0..3 {
                        out.push(format!("{q}/inertia/{k}/{j}"));
                    }
                }
            }
        }
    }
    out
}
pub fn metric_registry() -> Vec<String> {
    let mut m: Vec<String> = serde_json::to_value(crate::Metrics::default())
        .unwrap()
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    m.extend(
        [
            "ride.rms_heave_acceleration_m_s2",
            "ride.peak_heave_m",
            "ride.peak_roll_rad",
            "ride.peak_pitch_rad",
            "ride.max_shock_speed_m_s",
            "ride.max_shock_force_n",
            "analysis.motion_ratio_gradient_per_m",
            "analysis.spring_wheel_rate_n_per_m",
            "analysis.camber_gain_deg_per_m",
            "analysis.toe_gain_deg_per_m",
            "vehicle.left_wheelbase_m",
            "vehicle.right_wheelbase_m",
            "vehicle.left_contact_wheelbase_m",
            "vehicle.right_contact_wheelbase_m",
            "vehicle.front.wheel_track_m",
            "vehicle.rear.wheel_track_m",
            "vehicle.front.contact_track_m",
            "vehicle.rear.contact_track_m",
        ]
        .map(str::to_string),
    );
    m
}
fn relation_order(r: &OptimizationRequest) -> Result<Vec<usize>, Error> {
    let mut pending: Vec<usize> = (0..r.relations.len()).collect();
    let mut done = vec![];
    while !pending.is_empty() {
        let Some(pos) = pending.iter().position(|&i| {
            !pending
                .iter()
                .any(|&j| r.relations[j].destination == r.relations[i].source)
        }) else {
            return Err(err("relation cycle"));
        };
        done.push(pending.remove(pos));
    }
    Ok(done)
}
fn validate(p: &Project, r: &OptimizationRequest) -> Result<(), Error> {
    p.validate()?;
    if r.variables.len() > 128
        || r.population_size > 256
        || r.population_size < 4
        || r.scenarios.len() > 128
        || r.training_samples > 256
        || r.validation_samples > 256
        || r.uncertainty.len() > 128
        || r.targets.len() > 128
        || r.constraints.len() > 128
        || r.workers == 0
        || r.workers > 8
        || !r.max_seconds.is_finite()
        || r.max_seconds < 0.
        || r.max_evaluations > 1_000_000
        || r.generations > 1_000_000
    {
        return Err(err("invalid optimization limits"));
    }
    let cases = r.scenarios.len() + usize::from(r.ride_request.is_some()) + 1;
    let scheduled = r
        .max_evaluations
        .checked_mul(r.training_samples + 1)
        .and_then(|n| n.checked_add(r.validation_samples + 1))
        .and_then(|n| n.checked_mul(cases));
    if scheduled.is_none_or(|n| n > 2_000_000)
        || r.relations.len() > 128
        || serde_json::to_vec(p).map_err(|e| err(e.to_string()))?.len() > 64_000
        || serde_json::to_vec(r).map_err(|e| err(e.to_string()))?.len() > 256_000
    {
        return Err(err("scheduled cases or input size exceeds configured cap"));
    }
    let reg: HashSet<_> = parameter_registry(p).into_iter().collect();
    let mut writes = HashSet::new();
    for v in &r.variables {
        if !writes.insert(v.path.clone()) {
            return Err(err("duplicate variable"));
        }
        match &v.kind {
            VariableKind::Continuous { lower, upper } => {
                if !reg.contains(&v.path)
                    || !lower.is_finite()
                    || !upper.is_finite()
                    || lower > upper
                    || !(upper - lower).is_finite()
                {
                    return Err(err("invalid continuous variable"));
                }
            }
            VariableKind::Discrete { choices } => {
                let allowed = if v.path.ends_with("/pushrod_body") {
                    vec!["upper_arm", "lower_arm", "knuckle"]
                } else if v.path.ends_with("/tire_profile") {
                    vec!["disk", "cylinder", "torus"]
                } else {
                    vec![]
                };
                if !(0..4).any(|i| {
                    v.path == format!("/corners/{i}/pushrod_body")
                        || v.path == format!("/corners/{i}/tire_profile")
                }) || choices.is_empty()
                    || choices.len() > 3
                    || choices.iter().any(|s| !allowed.contains(&s.as_str()))
                    || choices.iter().collect::<HashSet<_>>().len() != choices.len()
                {
                    return Err(err("invalid discrete variable"));
                }
            }
        }
    }
    for l in &r.relations {
        if !reg.contains(&l.source)
            || !reg.contains(&l.destination)
            || !writes.insert(l.destination.clone())
            || !l.factor.is_finite()
            || !l.offset.is_finite()
        {
            return Err(err("invalid relation"));
        }
    }
    relation_order(r)?;
    let mut inputs = HashSet::new();
    for u in &r.uncertainty {
        if !reg.contains(&u.path)
            || !inputs.insert(&u.path)
            || r.relations.iter().any(|l| l.destination == u.path)
            || !u.half_range.is_finite()
            || u.half_range < 0.
            || u.factor.as_ref().is_some_and(String::is_empty)
        {
            return Err(err("invalid uncertainty"));
        }
    }
    let check = |metric: &str, corner: Option<CornerId>| -> Result<(), Error> {
        if let Some(path) = metric.strip_prefix("project:") {
            if corner.is_some() || !reg.contains(path) {
                return Err(err("invalid project metric"));
            }
        } else if metric.starts_with("ride.") {
            if corner.is_some()
                || r.ride_request.is_none()
                || !metric_registry().contains(&metric.to_string())
            {
                return Err(err("invalid ride metric"));
            }
        } else if metric.starts_with("vehicle.") {
            if corner.is_some()
                || r.scenarios.is_empty()
                || !metric_registry().contains(&metric.to_string())
            {
                return Err(err("invalid vehicle metric"));
            }
        } else if corner.is_none()
            || r.scenarios.is_empty()
            || !metric_registry().contains(&metric.to_string())
        {
            return Err(err("invalid geometry metric"));
        }
        Ok(())
    };
    for t in &r.targets {
        check(&t.metric, t.corner)?;
        if !t.value.is_finite()
            || !t.scale.is_finite()
            || t.scale <= 0.
            || !t.weight.is_finite()
            || t.weight < 0.
            || t.values.as_ref().is_some_and(|v| {
                t.metric.starts_with("ride.")
                    || t.metric.starts_with("project:")
                    || v.len() != r.scenarios.len()
                    || v.iter().any(|v| !v.is_finite())
            })
        {
            return Err(err("invalid target"));
        }
    }
    if !r.targets.iter().any(|t| t.weight > 0.) {
        return Err(err("positive target weight required"));
    }
    for c in &r.constraints {
        check(&c.metric, c.corner)?;
        if !c.scale.is_finite()
            || c.scale <= 0.
            || c.min.iter().chain(c.max.iter()).any(|v| !v.is_finite())
            || matches!((c.min,c.max),(Some(a),Some(b)) if a>b)
        {
            return Err(err("invalid constraint"));
        }
    }
    Ok(())
}
fn set(v: &mut Value, path: &str, n: Value) -> Result<(), Error> {
    *v.pointer_mut(path).ok_or_else(|| err("missing path"))? = n;
    Ok(())
}
fn construct(p: &Project, r: &OptimizationRequest, x: &[f64], u: &[f64]) -> Result<Project, Error> {
    if x.len() != r.variables.len() {
        return Err(err("candidate dimension"));
    }
    let mut v = serde_json::to_value(p).map_err(|e| err(e.to_string()))?;
    for (var, &x) in r.variables.iter().zip(x) {
        let n = match &var.kind {
            VariableKind::Continuous { lower, upper } => {
                if !x.is_finite() || x < *lower || x > *upper {
                    return Err(err("candidate outside bounds"));
                }
                Value::from(x)
            }
            VariableKind::Discrete { choices } => {
                if !x.is_finite() || x.fract() != 0. || x < 0. || x >= choices.len() as f64 {
                    return Err(err("invalid choice index"));
                }
                Value::from(choices[x as usize].clone())
            }
        };
        set(&mut v, &var.path, n)?;
    }
    for (a, &z) in r.uncertainty.iter().zip(u) {
        let n = v
            .pointer(&a.path)
            .and_then(Value::as_f64)
            .ok_or_else(|| err("missing uncertainty input"))?
            + z;
        set(&mut v, &a.path, Value::from(n))?;
    }
    for i in relation_order(r)? {
        let l = &r.relations[i];
        let n = v
            .pointer(&l.source)
            .and_then(Value::as_f64)
            .ok_or_else(|| err("missing relation input"))?
            * l.factor
            + l.offset;
        set(&mut v, &l.destination, Value::from(n))?;
    }
    let p: Project = serde_json::from_value(v).map_err(|e| err(e.to_string()))?;
    p.validate()?;
    Ok(p)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}
fn samples(r: &OptimizationRequest, validation: bool) -> Vec<Vec<f64>> {
    let n = if validation {
        r.validation_samples
    } else {
        r.training_samples
    };
    let mut rng = Rng(r.seed ^ if validation { 0x1234abcd } else { 0xabcde123 });
    let mut out = vec![vec![0.; r.uncertainty.len()]];
    for _ in 0..n {
        let mut factors = std::collections::BTreeMap::new();
        out.push(
            r.uncertainty
                .iter()
                .map(|u| {
                    let a = if let Some(f) = &u.factor {
                        *factors
                            .entry(f.clone())
                            .or_insert_with(|| 2. * rng.unit() - 1.)
                    } else {
                        2. * rng.unit() - 1.
                    };
                    a * u.half_range
                })
                .collect(),
        );
    }
    out
}
pub fn ride_summary(
    run: &RideRun,
    duration: f64,
) -> Result<std::collections::BTreeMap<String, f64>, Error> {
    if run.termination.is_some()
        || duration <= 0.
        || run.samples.len() < 2
        || run.samples.first().unwrap().time_s != 0.
        || (run.samples.last().unwrap().time_s - duration).abs() > 1e-9 * duration.max(1.)
    {
        return Err(err("incomplete ride"));
    }
    let mut area = 0.;
    for w in run.samples.windows(2) {
        let dt = w[1].time_s - w[0].time_s;
        if !dt.is_finite() || dt <= 0. {
            return Err(err("invalid ride time"));
        }
        area += dt * (w[0].acceleration[0].powi(2) + w[1].acceleration[0].powi(2)) / 2.;
    }
    let mut m = std::collections::BTreeMap::new();
    m.insert(
        "ride.rms_heave_acceleration_m_s2".into(),
        (area / duration).sqrt(),
    );
    for (name, axis) in [
        ("ride.peak_heave_m", 0),
        ("ride.peak_roll_rad", 1),
        ("ride.peak_pitch_rad", 2),
    ] {
        m.insert(
            name.into(),
            run.samples
                .iter()
                .map(|s| s.displacement[axis].abs())
                .fold(0., f64::max),
        );
    }
    for (name, force) in [
        ("ride.max_shock_speed_m_s", false),
        ("ride.max_shock_force_n", true),
    ] {
        m.insert(
            name.into(),
            run.samples
                .iter()
                .flat_map(|s| {
                    if force {
                        s.shock_force_n
                    } else {
                        s.compression_velocity_m_s
                    }
                })
                .map(f64::abs)
                .fold(0., f64::max),
        );
    }
    if m.values().any(|x| !x.is_finite())
        || run.samples.iter().any(|s| {
            serde_json::to_value(s)
                .unwrap()
                .to_string()
                .contains("null")
        })
    {
        return Err(err("nonfinite ride"));
    }
    Ok(m)
}
fn stopped(token: Option<&AtomicBool>, start: Instant, seconds: f64) -> bool {
    token.is_some_and(|t| t.load(Ordering::Relaxed)) || start.elapsed().as_secs_f64() >= seconds
}
pub fn evaluate_candidate(
    p: &Project,
    r: &OptimizationRequest,
    x: &[f64],
) -> Result<Evaluation, Error> {
    evaluate_candidate_controlled(p, r, x, false, &|| false)
}
/// Independent held-out stream, including nominal. Costs one candidate bundle;
/// the caller owns cumulative external-algorithm accounting. No implicit training replay.
pub fn evaluate_validation_candidate(
    p: &Project,
    r: &OptimizationRequest,
    x: &[f64],
) -> Result<Evaluation, Error> {
    evaluate_candidate_controlled(p, r, x, true, &|| false)
}
/// Construct the nominal realized project (variables, then relations). No physics work.
pub fn candidate_project(
    p: &Project,
    r: &OptimizationRequest,
    x: &[f64],
) -> Result<Project, Error> {
    validate(p, r)?;
    construct(p, r, x, &vec![0.; r.uncertainty.len()])
}
/// One candidate attempt with per-call max_seconds and cooperative stop at each case.
/// Zero max_evaluations permits no work. Positive max_evaluations does not create a
/// shared external budget; adapters must count calls themselves. stop must be thread safe.
pub fn evaluate_candidate_controlled(
    p: &Project,
    r: &OptimizationRequest,
    x: &[f64],
    validation: bool,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<Evaluation, Error> {
    validate(p, r)?;
    let start = Instant::now();
    Ok(evaluate(p, r, x, validation, &|| {
        r.max_evaluations == 0 || start.elapsed().as_secs_f64() >= r.max_seconds || stop()
    }))
}
fn evaluate(
    p: &Project,
    r: &OptimizationRequest,
    x: &[f64],
    validation: bool,
    stop: &(dyn Fn() -> bool + Sync),
) -> Evaluation {
    let plan = samples(r, validation);
    let per = r.scenarios.len() + usize::from(r.ride_request.is_some()) + 1;
    let mut e = Evaluation {
        complete: true,
        feasible: false,
        score: None,
        contributions: vec![0.; r.targets.len()],
        nominal_target_values: vec![vec![]; r.targets.len()],
        sample_count: plan.len(),
        failed_samples: 0,
        not_evaluated_cases: 0,
        violation: 0.,
        requested_cases: plan.len() * per,
        completed_cases: 0,
        physics_cases_completed: 0,
        physics_cases_failed: 0,
        failed_cases: 0,
        failures: vec![],
        failures_truncated: false,
        failed_case_bits: vec![0; (plan.len() * per).div_ceil(64)],
        worst_squared_residual: 0.,
        worst_sample: 0,
        model_fidelity: None,
    };
    let fail = |e: &mut Evaluation, u, case: String, reason: String| {
        e.failed_cases += 1;
        if let Ok(ci) = case.parse::<usize>() {
            let index = u * per + ci;
            e.failed_case_bits[index / 64] |= 1u64 << (index % 64);
        }
        if e.failures.len() < 32 {
            e.failures.push(Failure {
                sample: u,
                case,
                reason,
            })
        } else {
            e.failures_truncated = true
        }
    };
    for (ui, u) in plan.iter().enumerate() {
        let failures_before = e.failed_cases;
        let mut vals: Vec<Vec<f64>> = vec![vec![]; r.targets.len()];
        let project = construct(p, r, x, u);
        for case in 0..per {
            if stop() {
                e.complete = false;
                break;
            }
            e.completed_cases += 1;
            let mut metrics = std::collections::BTreeMap::<String, f64>::new();
            let result = (|| -> Result<(), Error> {
                let p = project.as_ref().map_err(|e| err(e.to_string()))?;
                if case > 0 {
                    e.physics_cases_completed += 1;
                }
                if case == 0 {
                    let v = serde_json::to_value(p).unwrap();
                    for path in parameter_registry(p) {
                        if let Some(n) = v.pointer(&path).and_then(Value::as_f64) {
                            metrics.insert(format!("project:{path}"), n);
                        }
                    }
                } else if case <= r.scenarios.len() {
                    let m = &r.scenarios[case - 1];
                    let s = crate::simulate(p, m)?;
                    for c in &s.corners {
                        let v = serde_json::to_value(&c.metrics).unwrap();
                        for (k, v) in v.as_object().unwrap() {
                            if let Some(n) = v.as_f64() {
                                metrics.insert(format!("{:?}:{k}", c.id), n);
                            }
                        }
                    }
                    if r.targets
                        .iter()
                        .map(|t| &t.metric)
                        .chain(r.constraints.iter().map(|t| &t.metric))
                        .any(|s| s.starts_with("analysis.") || s.starts_with("vehicle."))
                    {
                        let a = crate::analyze(p, m)?;
                        for (name, value) in [
                            ("vehicle.left_wheelbase_m", a.left_wheelbase_m),
                            ("vehicle.right_wheelbase_m", a.right_wheelbase_m),
                            (
                                "vehicle.left_contact_wheelbase_m",
                                a.left_contact_wheelbase_m,
                            ),
                            (
                                "vehicle.right_contact_wheelbase_m",
                                a.right_contact_wheelbase_m,
                            ),
                            ("vehicle.front.wheel_track_m", a.front.wheel_track_m),
                            ("vehicle.rear.wheel_track_m", a.rear.wheel_track_m),
                            ("vehicle.front.contact_track_m", a.front.contact_track_m),
                            ("vehicle.rear.contact_track_m", a.rear.contact_track_m),
                        ] {
                            metrics.insert(name.into(), value);
                        }
                        for c in a.corners {
                            let id = c.id;
                            let v = serde_json::to_value(c).unwrap();
                            for (k, v) in v.as_object().unwrap() {
                                if let Some(n) = v.get("value").and_then(Value::as_f64) {
                                    metrics.insert(format!("{id:?}:analysis.{k}"), n);
                                }
                            }
                        }
                    }
                } else {
                    let rr = r.ride_request.as_ref().unwrap();
                    let run = crate::ride(p, rr)?;
                    e.model_fidelity = Some(run.model_fidelity.clone());
                    metrics = ride_summary(&run, rr.duration_s)?;
                }
                Ok(())
            })();
            if let Err(ex) = result {
                if case > 0 && project.is_ok() {
                    e.physics_cases_failed += 1;
                }
                fail(&mut e, ui, case.to_string(), ex.to_string());
                continue;
            }
            let relevant = |metric: &str| {
                if case == 0 {
                    metric.starts_with("project:")
                } else if case <= r.scenarios.len() {
                    !metric.starts_with("project:") && !metric.starts_with("ride.")
                } else {
                    metric.starts_with("ride.")
                }
            };
            let lookup = |metric: &str, corner: Option<CornerId>| {
                metrics
                    .get(&corner.map_or_else(|| metric.to_string(), |c| format!("{c:?}:{metric}")))
                    .copied()
                    .filter(|x| x.is_finite())
            };
            let mut missing = false;
            for (ti, t) in r.targets.iter().enumerate() {
                if relevant(&t.metric) {
                    if let Some(y) = lookup(&t.metric, t.corner) {
                        if ui == 0 {
                            e.nominal_target_values[ti].push(y);
                        }
                        let d = t.values.as_ref().map_or(t.value, |v| v[case - 1]);
                        let q = ((y - d) / t.scale).powi(2);
                        if !q.is_finite() {
                            missing = true
                        } else {
                            vals[ti].push(q);
                            if q > e.worst_squared_residual {
                                e.worst_squared_residual = q;
                                e.worst_sample = ui;
                            }
                        }
                    } else {
                        missing = true
                    }
                }
            }
            for c in &r.constraints {
                if relevant(&c.metric) {
                    if let Some(y) = lookup(&c.metric, c.corner) {
                        let v = c
                            .min
                            .map_or(0., |m| (m - y) / c.scale)
                            .max(c.max.map_or(0., |m| (y - m) / c.scale))
                            .max(0.);
                        if v.is_finite() {
                            let total = e.violation + v;
                            if total.is_finite() {
                                e.violation = total
                            } else {
                                missing = true;
                            }
                        } else {
                            missing = true
                        }
                    } else {
                        missing = true
                    }
                }
            }
            if missing {
                if case > 0 {
                    e.physics_cases_failed += 1;
                }
                fail(
                    &mut e,
                    ui,
                    case.to_string(),
                    "missing or nonfinite metric".into(),
                )
            }
        }
        for (i, t) in r.targets.iter().enumerate() {
            if !vals[i].is_empty() {
                let loss = match t.aggregation {
                    Aggregation::MeanSquared => vals[i].iter().sum::<f64>() / vals[i].len() as f64,
                    Aggregation::WorstSquared => vals[i].iter().copied().fold(0., f64::max),
                };
                e.contributions[i] += t.weight * loss / plan.len() as f64;
            }
        }
        if e.failed_cases > failures_before {
            e.failed_samples += 1;
        }
        if !e.complete {
            break;
        }
    }
    e.not_evaluated_cases = e.requested_cases - e.completed_cases;
    if e.contributions.iter().any(|c| !c.is_finite()) {
        for c in &mut e.contributions {
            if !c.is_finite() {
                *c = 0.;
            }
        }
        fail(
            &mut e,
            0,
            "0".into(),
            "nonfinite objective aggregate".into(),
        );
        e.failed_samples = e.failed_samples.max(1);
    }
    e.feasible = e.complete && e.failed_cases == 0 && e.violation == 0.;
    if e.complete && e.failed_cases == 0 {
        let s = e.contributions.iter().sum::<f64>();
        if s.is_finite() {
            e.score = Some(s)
        } else {
            e.feasible = false;
            for c in &mut e.contributions {
                if !c.is_finite() {
                    *c = 0.;
                }
            }
            fail(
                &mut e,
                0,
                "0".into(),
                "nonfinite objective aggregate".into(),
            );
            e.failed_samples = e.failed_samples.max(1);
        }
    }
    e
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionState {
    version: u32,
    identity: String,
    project: Project,
    request: OptimizationRequest,
    rng: Rng,
    population: Vec<Candidate>,
    pending: Vec<Vec<f64>>,
    pending_results: Vec<Candidate>,
    phase: String,
    result: OptimizationResult,
}
/// Checkpoint includes precomputed generation trials and completed results; pauses consume no time.
pub struct OptimizationSession {
    state: SessionState,
}
fn identity(p: &Project, r: &OptimizationRequest) -> String {
    serde_json::to_string(&(env!("CARGO_PKG_VERSION"), "optimization-v1", p, r)).unwrap()
}
fn checksum(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325, |a, b| {
        (a ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}
fn bounds(v: &Variable) -> (f64, f64) {
    match &v.kind {
        VariableKind::Continuous { lower, upper } => (*lower, *upper),
        VariableKind::Discrete { choices } => (0., (choices.len() - 1) as f64),
    }
}
fn mapped(v: &Variable, z: f64) -> f64 {
    let (l, h) = bounds(v);
    let x = l + z.clamp(0., 1.) * (h - l);
    match v.kind {
        VariableKind::Discrete { .. } => x.round(),
        _ => x,
    }
}
fn base_values(p: &Project, r: &OptimizationRequest) -> Vec<f64> {
    let v = serde_json::to_value(p).unwrap();
    r.variables
        .iter()
        .map(|a| match &a.kind {
            VariableKind::Continuous { lower, upper } => v
                .pointer(&a.path)
                .and_then(Value::as_f64)
                .unwrap()
                .clamp(*lower, *upper),
            VariableKind::Discrete { choices } => choices
                .iter()
                .position(|s| Some(s.as_str()) == v.pointer(&a.path).and_then(Value::as_str))
                .unwrap_or(0) as f64,
        })
        .collect()
}
impl OptimizationSession {
    pub fn start(p: &Project, r: &OptimizationRequest) -> Result<Self, Error> {
        validate(p, r)?;
        Ok(Self {
            state: SessionState {
                version: 1,
                identity: identity(p, r),
                project: p.clone(),
                request: r.clone(),
                rng: Rng(r.seed),
                population: vec![],
                pending: vec![],
                pending_results: vec![],
                phase: "baseline".into(),
                result: OptimizationResult {
                    status: "running".into(),
                    baseline: None,
                    best_feasible: None,
                    best_infeasible: None,
                    validation: None,
                    training_revalidation: None,
                    candidate_attempts: 0,
                    baseline_attempts: 0,
                    search_attempts: 0,
                    validation_attempts: 0,
                    physics_cases_completed: 0,
                    physics_cases_failed: 0,
                    generation: 0,
                    elapsed_seconds: 0.,
                },
            },
        })
    }
    pub fn result(&self) -> OptimizationResult {
        self.state.result.clone()
    }
    pub fn is_finished(&self) -> bool {
        self.state.phase == "done"
    }
    pub fn checkpoint(&self) -> Result<String, Error> {
        let payload = serde_json::to_string(&self.state).map_err(|e| err(e.to_string()))?;
        serde_json::to_string(&(checksum(&payload), payload)).map_err(|e| err(e.to_string()))
    }
    pub fn resume(p: &Project, r: &OptimizationRequest, json: &str) -> Result<Self, Error> {
        validate(p, r)?;
        if json.len() > 256_000_000 {
            return Err(err("checkpoint too large"));
        }
        let (sum, payload): (u64, String) =
            serde_json::from_str(json).map_err(|e| err(e.to_string()))?;
        if checksum(&payload) != sum {
            return Err(err("checkpoint checksum mismatch"));
        }
        let s: SessionState = serde_json::from_str(&payload).map_err(|e| err(e.to_string()))?;
        if s.version != 1
            || s.identity != identity(p, r)
            || identity(&s.project, &s.request) != s.identity
            || s.result.candidate_attempts > r.max_evaluations
            || s.result.generation > r.generations
            || !s.result.elapsed_seconds.is_finite()
            || s.result.elapsed_seconds < 0.
            || s.population.len() > r.population_size
            || s.pending.len() > r.population_size
            || s.pending_results.len() > s.pending.len()
            || !["baseline", "initial", "evolution", "validation", "done"]
                .contains(&s.phase.as_str())
        {
            return Err(err("checkpoint identity/state mismatch"));
        }
        if s.result.baseline_attempts > 1
            || s.result.validation_attempts > 1
            || s.result
                .baseline_attempts
                .checked_add(s.result.search_attempts)
                .and_then(|n| n.checked_add(s.result.validation_attempts))
                != Some(s.result.candidate_attempts)
            || (s.phase == "baseline"
                && (s.result.candidate_attempts != 0
                    || !s.pending.is_empty()
                    || !s.population.is_empty()))
            || (s.phase == "initial" && (!s.population.is_empty() || s.pending.is_empty()))
            || (s.phase == "evolution"
                && (s.population.len() != r.population_size
                    || s.pending.len() != r.population_size))
            || (s.phase != "done" && s.result.status != "running")
        {
            return Err(err("inconsistent checkpoint phase or accounting"));
        }
        for c in s
            .population
            .iter()
            .chain(s.pending_results.iter())
            .chain(s.result.best_feasible.iter())
            .chain(s.result.best_infeasible.iter())
        {
            let realized = construct(p, r, &c.values, &vec![0.; r.uncertainty.len()]).ok();
            if realized != c.project
                || c.values.len() != r.variables.len()
                || c.evaluation.score.is_some_and(|v| !v.is_finite() || v < 0.)
                || !c.evaluation.violation.is_finite()
                || c.evaluation.violation < 0.
                || c.evaluation.completed_cases > c.evaluation.requested_cases
                || c.evaluation.contributions.len() != r.targets.len()
                || c.evaluation
                    .contributions
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.)
                || c.evaluation.nominal_target_values.len() != r.targets.len()
                || c.evaluation.not_evaluated_cases
                    != c.evaluation.requested_cases - c.evaluation.completed_cases
                || c.evaluation.physics_cases_completed > c.evaluation.completed_cases
                || c.evaluation.physics_cases_failed > c.evaluation.physics_cases_completed
                || c.evaluation.failed_case_bits.len() != c.evaluation.requested_cases.div_ceil(64)
                || (c.evaluation.feasible
                    && (!c.evaluation.complete
                        || c.evaluation.failed_cases != 0
                        || c.evaluation.violation != 0.
                        || c.evaluation.score.is_none()))
                || c.evaluation
                    .score
                    .is_some_and(|v| v != c.evaluation.contributions.iter().sum::<f64>())
            {
                return Err(err("invalid checkpoint score"));
            }
        }
        for x in &s.pending {
            if x.len() != r.variables.len()
                || x.iter().zip(&r.variables).any(|(&x, v)| {
                    let (l, h) = bounds(v);
                    !x.is_finite()
                        || x < l
                        || x > h
                        || (matches!(v.kind, VariableKind::Discrete { .. }) && x.fract() != 0.)
                })
            {
                return Err(err("invalid checkpoint trial"));
            }
        }
        Ok(Self { state: s })
    }
    fn record(&mut self, c: &Candidate) {
        let r = &mut self.state.result;
        r.candidate_attempts += 1;
        r.search_attempts += 1;
        r.physics_cases_completed += c.evaluation.physics_cases_completed;
        r.physics_cases_failed += c.evaluation.physics_cases_failed;
        if !c.evaluation.complete {
            return;
        }
        let dest = if c.evaluation.feasible {
            &mut r.best_feasible
        } else {
            &mut r.best_infeasible
        };
        if dest
            .as_ref()
            .is_none_or(|a| c.evaluation.better_than(&a.evaluation))
        {
            *dest = Some(c.clone());
        }
    }
    fn end(&mut self, status: &str) {
        self.state.phase = "done".into();
        self.state.result.status = status.into();
    }
    fn prepare_initial(&mut self) {
        let s = &mut self.state;
        s.pending.push(base_values(&s.project, &s.request));
        let size = if s.request.variables.iter().all(|v| {
            let (l, h) = bounds(v);
            l == h
        }) {
            1
        } else {
            s.request.population_size
        };
        for _ in 1..size {
            s.pending.push(
                s.request
                    .variables
                    .iter()
                    .map(|v| mapped(v, s.rng.unit()))
                    .collect(),
            )
        }
        s.phase = "initial".into();
    }
    fn prepare_trials(&mut self) {
        let s = &mut self.state;
        let n = s.population.len();
        let d = s.request.variables.len();
        s.pending.clear();
        s.pending_results.clear();
        for i in 0..n {
            let mut donors = vec![];
            while donors.len() < 3 {
                let j = (s.rng.next() % n as u64) as usize;
                if j != i && !donors.contains(&j) {
                    donors.push(j)
                }
            }
            let active: Vec<usize> = s
                .request
                .variables
                .iter()
                .enumerate()
                .filter_map(|(k, v)| {
                    let (l, h) = bounds(v);
                    (l < h).then_some(k)
                })
                .collect();
            let forced = active[(s.rng.next() % active.len() as u64) as usize];
            let x = (0..d)
                .map(|k| {
                    let v = &s.request.variables[k];
                    let (l, h) = bounds(v);
                    if h == l {
                        return l;
                    }
                    let old = s.population[i].values[k];
                    if s.rng.unit() < 0.9 || k == forced {
                        let a = s.population[donors[0]].values[k];
                        let b = s.population[donors[1]].values[k];
                        let c = s.population[donors[2]].values[k];
                        mapped(v, ((a - l) + 0.7 * (b - c)) / (h - l))
                    } else {
                        old
                    }
                })
                .collect();
            s.pending.push(x)
        }
        s.phase = "evolution".into();
    }
    /// Executes at most max_work candidate attempts. Cancellation/deadline checked between cases;
    /// an individual native ride/geometry solve cannot be interrupted. Private pool capped at 8.
    pub fn advance(
        &mut self,
        max_work: usize,
        cancel: Option<&AtomicBool>,
    ) -> Result<OptimizationResult, Error> {
        let start = Instant::now();
        let remaining_seconds =
            (self.state.request.max_seconds - self.state.result.elapsed_seconds).max(0.);
        let mut work = 0;
        while !self.is_finished() && work < max_work {
            if stopped(cancel, start, remaining_seconds) {
                self.end(if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                    "cancelled_not_validated"
                } else {
                    "budget_not_validated"
                });
                break;
            }
            if self.state.result.candidate_attempts >= self.state.request.max_evaluations {
                self.end("budget_not_validated");
                break;
            }
            if self.state.phase == "baseline" {
                let mut br = self.state.request.clone();
                br.variables.clear();
                br.relations.clear();
                let evaluation = evaluate(&self.state.project, &br, &[], false, &|| {
                    stopped(cancel, start, remaining_seconds)
                });
                let c = Candidate {
                    values: vec![],
                    project: Some(self.state.project.clone()),
                    evaluation,
                };
                self.state.result.candidate_attempts += 1;
                self.state.result.baseline_attempts += 1;
                self.state.result.physics_cases_completed += c.evaluation.physics_cases_completed;
                self.state.result.physics_cases_failed += c.evaluation.physics_cases_failed;
                self.state.result.baseline = Some(c);
                work += 1;
                self.prepare_initial();
                continue;
            }
            let remaining =
                self.state.request.max_evaluations - self.state.result.candidate_attempts;
            if remaining <= 1 {
                self.state.phase = "validation".into();
            }
            if self.state.phase == "validation" {
                if let Some(best) = self.state.result.best_feasible.clone() {
                    // Reconstruct and repeat training, then independent held-out cases in one reserved attempt.
                    let report = evaluate(
                        &self.state.project,
                        &self.state.request,
                        &best.values,
                        false,
                        &|| stopped(cancel, start, remaining_seconds),
                    );
                    let held = evaluate(
                        &self.state.project,
                        &self.state.request,
                        &best.values,
                        true,
                        &|| stopped(cancel, start, remaining_seconds),
                    );
                    self.state.result.candidate_attempts += 1;
                    self.state.result.validation_attempts += 1;
                    self.state.result.physics_cases_completed +=
                        report.physics_cases_completed + held.physics_cases_completed;
                    self.state.result.physics_cases_failed +=
                        report.physics_cases_failed + held.physics_cases_failed;
                    let status = if !report.complete || !held.complete {
                        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                            "cancelled_not_validated"
                        } else {
                            "budget_not_validated"
                        }
                    } else if report.feasible && held.feasible {
                        "validated"
                    } else {
                        "validation_failed"
                    };
                    // Never return a cached score or project as freshly validated output.
                    if report.complete {
                        let fresh = Candidate {
                            project: construct(
                                &self.state.project,
                                &self.state.request,
                                &best.values,
                                &vec![0.; self.state.request.uncertainty.len()],
                            )
                            .ok(),
                            values: best.values,
                            evaluation: report.clone(),
                        };
                        if fresh.evaluation.feasible {
                            self.state.result.best_feasible = Some(fresh);
                        } else {
                            self.state.result.best_feasible = None;
                            self.state.result.best_infeasible = Some(fresh);
                        }
                    }
                    self.state.result.training_revalidation = Some(report);
                    self.state.result.validation = Some(held);
                    self.end(status);
                } else {
                    self.end("no_feasible_candidate");
                }
                break;
            }
            if self.state.pending_results.len() == self.state.pending.len() {
                if self.state.phase == "initial" {
                    self.state.population = std::mem::take(&mut self.state.pending_results);
                } else {
                    let trials = std::mem::take(&mut self.state.pending_results);
                    for (old, new) in self.state.population.iter_mut().zip(trials) {
                        if new.evaluation.better_than(&old.evaluation) {
                            *old = new
                        }
                    }
                    self.state.result.generation += 1;
                }
                if self.state.result.generation >= self.state.request.generations
                    || self.state.request.variables.iter().all(|v| {
                        let (l, h) = bounds(v);
                        l == h
                    })
                {
                    self.state.phase = "validation".into();
                } else {
                    self.prepare_trials();
                }
                continue;
            }
            let from = self.state.pending_results.len();
            let count = (max_work - work)
                .min(self.state.request.workers)
                .min(remaining - 1)
                .min(self.state.pending.len() - from);
            if count == 0 {
                self.state.phase = "validation".into();
                continue;
            }
            let xs = &self.state.pending[from..from + count];
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(
                    self.state
                        .request
                        .workers
                        .min(std::thread::available_parallelism().map_or(1, usize::from)),
                )
                .build()
                .map_err(|e| err(e.to_string()))?;
            use rayon::prelude::*;
            let candidates: Vec<Candidate> = pool.install(|| {
                xs.par_iter()
                    .map(|x| {
                        let evaluation =
                            evaluate(&self.state.project, &self.state.request, x, false, &|| {
                                stopped(cancel, start, remaining_seconds)
                            });
                        let project = construct(
                            &self.state.project,
                            &self.state.request,
                            x,
                            &vec![0.; self.state.request.uncertainty.len()],
                        )
                        .ok();
                        Candidate {
                            values: x.clone(),
                            project,
                            evaluation,
                        }
                    })
                    .collect()
            });
            for c in candidates {
                self.record(&c);
                self.state.pending_results.push(c);
            }
            work += count;
        }
        self.state.result.elapsed_seconds += start.elapsed().as_secs_f64();
        Ok(self.result())
    }
}
pub fn optimize(p: &Project, r: &OptimizationRequest) -> Result<OptimizationResult, Error> {
    let mut s = OptimizationSession::start(p, r)?;
    while !s.is_finished() {
        s.advance(r.workers.max(1), None)?;
    }
    Ok(s.result())
}
