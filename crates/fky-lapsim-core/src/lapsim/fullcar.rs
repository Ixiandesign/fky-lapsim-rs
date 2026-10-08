//! Full-car 7×7 matrix model (thesis §5.5, Eqs. 5-28…5-44) in quasi-static form.
//!
//! Coordinates `x = [z_s, φ, θ, z_u,FL, z_u,FR, z_u,RL, z_u,RR]` (sprung heave, roll, pitch, four
//! unsprung vertical motions). φ is positive when the left side rises, θ positive when the nose
//! lowers; corner `i` sits at `(x_i, y_i)` = `(a, +t_f/2), (a, −t_f/2), (−b, +t_r/2), (−b, −t_r/2)`
//! relative to the sprung CG; body motion there is `u_i = z_s + y_i φ − x_i θ` and spring
//! compression is `c_i = z_u,i − u_i`. `K x = F` with `K = [[BᵀKuB, −BᵀKu], [−KuB, Ku+Kt]]`.
use super::rates::{rates, ThesisSuspension};
use super::thesis::BikeParams;
use super::{err, G};
use crate::dynamics::{linearize_ride, spring_force, RideRequest};
use crate::study::simulate_on_road_mode;
use crate::{CornerId, Error, Motion, Project};
use nalgebra::{SMatrix, SVector, UnitQuaternion, Vector3};

/// 7×7 matrix type.
pub type M7 = SMatrix<f64, 7, 7>;
type V7 = SVector<f64, 7>;

/// Everything the 7×7 needs, independent of where it came from (thesis tables or geometry).
#[derive(Clone, Debug)]
pub struct FullCarParams {
    /// Sprung mass, kg.
    pub sprung_mass_kg: f64,
    /// Unsprung mass per corner FL, FR, RL, RR, kg.
    pub unsprung_mass_kg: [f64; 4],
    /// Roll inertia, kg·m².
    pub roll_inertia_kg_m2: f64,
    /// Pitch inertia, kg·m².
    pub pitch_inertia_kg_m2: f64,
    /// CG to front axle, m.
    pub a_m: f64,
    /// CG to rear axle, m.
    pub b_m: f64,
    /// Half track front, rear, m.
    pub half_track_m: [f64; 2],
    /// Wheel rate per corner (spring at the wheel), N/m.
    pub wheel_rate_n_m: [f64; 4],
    /// Anti-roll-bar rate at the wheel, front and rear, N/m.
    pub arb_wheel_rate_n_m: [f64; 2],
    /// Tyre vertical stiffness per corner, N/m.
    pub tyre_rate_n_m: [f64; 4],
    /// Damper rate at the wheel per corner, N·s/m (0 for the quasi-static solver).
    pub wheel_damping_n_s_m: [f64; 4],
    /// Static wheel loads FL, FR, RL, RR, N.
    pub static_wheel_load_n: [f64; 4],
    /// Static ride height front, rear, mm.
    pub static_rh_mm: [f64; 2],
    /// Wheelbase, m.
    pub wheelbase_m: f64,
    /// CG height used for longitudinal transfer, m.
    pub cg_height_m: f64,
    /// Roll lever arm (CG to roll axis), m.
    pub roll_lever_arm_m: f64,
    /// Mass used for the roll moment (thesis: total mass), kg.
    pub roll_mass_kg: f64,
    /// Mass used for longitudinal weight transfer (thesis: total mass), kg.
    pub pitch_mass_kg: f64,
    /// Roll-centre height front, rear, m.
    pub roll_centre_height_m: [f64; 2],
    /// Front anti-dive, percent.
    pub anti_dive_front_pct: f64,
    /// Rear anti-lift (braking), percent.
    pub anti_lift_rear_pct: f64,
    /// Front anti-lift (acceleration), percent.
    pub anti_lift_front_pct: f64,
    /// Rear anti-squat, percent.
    pub anti_squat_rear_pct: f64,
}

/// Build thesis-table parameters (rates mode) from the bike model and the thesis suspension.
///
/// # Errors
/// Propagates [`rates`] errors.
pub fn thesis_params(p: &BikeParams, s: &ThesisSuspension) -> Result<FullCarParams, Error> {
    let r = rates(p, s)?;
    let nsm = [
        s.nsm_mass_kg[0] / 2.0,
        s.nsm_mass_kg[0] / 2.0,
        s.nsm_mass_kg[1] / 2.0,
        s.nsm_mass_kg[1] / 2.0,
    ];
    let wd = p.wd_front_pct / 100.0;
    let w = p.mass_kg * G;
    Ok(FullCarParams {
        sprung_mass_kg: p.mass_kg - s.nsm_mass_kg[0] - s.nsm_mass_kg[1],
        unsprung_mass_kg: nsm,
        roll_inertia_kg_m2: s.roll_inertia_kg_m2,
        pitch_inertia_kg_m2: s.pitch_inertia_kg_m2,
        a_m: p.a_dist_m(),
        b_m: p.b_dist_m(),
        half_track_m: [p.front_track_m / 2.0, p.rear_track_m / 2.0],
        wheel_rate_n_m: [r.kw_n_m[0], r.kw_n_m[0], r.kw_n_m[1], r.kw_n_m[1]],
        arb_wheel_rate_n_m: r.kw_arb_n_m,
        tyre_rate_n_m: [
            s.tyre_stiffness_n_m[0],
            s.tyre_stiffness_n_m[0],
            s.tyre_stiffness_n_m[1],
            s.tyre_stiffness_n_m[1],
        ],
        wheel_damping_n_s_m: [0.0; 4],
        static_wheel_load_n: [
            w * wd / 2.0,
            w * wd / 2.0,
            w * (1.0 - wd) / 2.0,
            w * (1.0 - wd) / 2.0,
        ],
        static_rh_mm: s.static_rh_mm,
        wheelbase_m: p.wheelbase_m,
        cg_height_m: p.cg_height_m,
        roll_lever_arm_m: r.roll_lever_arm_m,
        roll_mass_kg: p.mass_kg,
        pitch_mass_kg: p.mass_kg,
        roll_centre_height_m: s.roll_centre_height_m,
        anti_dive_front_pct: s.anti_dive_front_pct,
        anti_lift_rear_pct: s.anti_lift_rear_pct,
        anti_lift_front_pct: s.anti_lift_front_pct,
        anti_squat_rear_pct: s.anti_squat_rear_pct,
    })
}

/// Loads for one quasi-static solve.
#[derive(Clone, Copy, Debug)]
pub struct LoadInputs {
    /// Longitudinal acceleration, m/s² (positive accelerating).
    pub ax: f64,
    /// Lateral acceleration, m/s² (positive = left turn).
    pub ay: f64,
    /// Front downforce, N.
    pub downforce_front_n: f64,
    /// Rear downforce, N.
    pub downforce_rear_n: f64,
    /// Aerodynamic drag, N.
    pub drag_n: f64,
    /// Tyre longitudinal force per wheel (vehicle frame, + forward), N.
    pub fx_wheel_n: [f64; 4],
    /// Tyre lateral force per wheel (vehicle frame, + left), N.
    pub fy_wheel_n: [f64; 4],
}

/// Result of a solve.
#[derive(Clone, Debug)]
pub struct Solution {
    /// Sprung heave relative to static, m.
    pub heave_m: f64,
    /// Roll, rad.
    pub roll_rad: f64,
    /// Pitch, rad.
    pub pitch_rad: f64,
    /// Unsprung vertical displacement per wheel, m.
    pub wheel_travel_m: [f64; 4],
    /// Total wheel normal load, N.
    pub wheel_load_n: [f64; 4],
    /// Camber per wheel, degrees (outward positive).
    pub camber_deg: [f64; 4],
    /// Toe per wheel, degrees.
    pub toe_deg: [f64; 4],
    /// Front ride height, mm.
    pub front_rh_mm: f64,
    /// Rear ride height, mm.
    pub rear_rh_mm: f64,
    /// Roll, degrees.
    pub roll_deg: f64,
    /// Pitch, degrees.
    pub pitch_deg: f64,
}

/// The 7×7 full-car model.
#[derive(Clone, Debug)]
pub struct FullCar {
    /// Mass matrix `M` (Eq. 5-36.1).
    pub mass: M7,
    /// Damping matrix `C`.
    pub damping: M7,
    /// Stiffness matrix `K`.
    pub stiffness: M7,
    /// Body-to-wheel-travel map `B` (rows `[1, y_i, −x_i]`).
    pub b: SMatrix<f64, 4, 3>,
    /// Parameters this model was built from.
    pub params: FullCarParams,
    /// Linearized geometry for the virtual-work load map; `None` in rates mode.
    pub geometry: Option<GeometryData>,
}

fn assemble(b: &SMatrix<f64, 4, 3>, ku: &SMatrix<f64, 4, 4>, kt: &SMatrix<f64, 4, 4>) -> M7 {
    let bt = b.transpose();
    let mut k = M7::zeros();
    k.fixed_view_mut::<3, 3>(0, 0).copy_from(&(bt * ku * b));
    k.fixed_view_mut::<3, 4>(0, 3).copy_from(&(-(bt * ku)));
    k.fixed_view_mut::<4, 3>(3, 0).copy_from(&(-(ku * b)));
    k.fixed_view_mut::<4, 4>(3, 3).copy_from(&(ku + kt));
    k
}

impl FullCar {
    /// Build from thesis-style (rates) parameters.
    ///
    /// # Errors
    /// Nonpositive/nonfinite masses, rates, geometry or tyre stiffness.
    pub fn from_rates(params: FullCarParams) -> Result<Self, Error> {
        let p = &params;
        let pos = [
            p.sprung_mass_kg,
            p.roll_inertia_kg_m2,
            p.pitch_inertia_kg_m2,
            p.a_m,
            p.b_m,
            p.half_track_m[0],
            p.half_track_m[1],
            p.wheelbase_m,
            p.cg_height_m,
            p.roll_mass_kg,
            p.pitch_mass_kg,
        ];
        if !pos.iter().all(|v| v.is_finite() && *v > 0.0)
            || !p
                .unsprung_mass_kg
                .iter()
                .chain(&p.wheel_rate_n_m)
                .chain(&p.tyre_rate_n_m)
                .all(|v| v.is_finite() && *v > 0.0)
            || !p
                .arb_wheel_rate_n_m
                .iter()
                .chain(&p.wheel_damping_n_s_m)
                .all(|v| v.is_finite() && *v >= 0.0)
        {
            return Err(err("invalid full-car parameters"));
        }
        let (a, bb) = (p.a_m, p.b_m);
        let y = [
            p.half_track_m[0],
            -p.half_track_m[0],
            p.half_track_m[1],
            -p.half_track_m[1],
        ];
        let x = [a, a, -bb, -bb];
        let mut b = SMatrix::<f64, 4, 3>::zeros();
        for i in 0..4 {
            b[(i, 0)] = 1.0;
            b[(i, 1)] = y[i];
            b[(i, 2)] = -x[i];
        }
        let mut ku = SMatrix::<f64, 4, 4>::zeros();
        let mut cu = SMatrix::<f64, 4, 4>::zeros();
        for i in 0..4 {
            ku[(i, i)] = p.wheel_rate_n_m[i];
            cu[(i, i)] = p.wheel_damping_n_s_m[i];
        }
        for (axle, &karb) in p.arb_wheel_rate_n_m.iter().enumerate() {
            let (l, r) = (2 * axle, 2 * axle + 1);
            ku[(l, l)] += karb / 2.0;
            ku[(r, r)] += karb / 2.0;
            ku[(l, r)] -= karb / 2.0;
            ku[(r, l)] -= karb / 2.0;
        }
        let mut kt = SMatrix::<f64, 4, 4>::zeros();
        for i in 0..4 {
            kt[(i, i)] = p.tyre_rate_n_m[i];
        }
        let mut mass = M7::zeros();
        mass[(0, 0)] = p.sprung_mass_kg;
        mass[(1, 1)] = p.roll_inertia_kg_m2;
        mass[(2, 2)] = p.pitch_inertia_kg_m2;
        for i in 0..4 {
            mass[(3 + i, 3 + i)] = p.unsprung_mass_kg[i];
        }
        Ok(Self {
            mass,
            damping: assemble(&b, &cu, &SMatrix::zeros()),
            stiffness: assemble(&b, &ku, &kt),
            b,
            params,
            geometry: None,
        })
    }

    /// Solve `K x = F` for the loads in `inp` (rates-mode load map: thesis §7.2, Eqs. 7-1, 7-8).
    ///
    /// # Errors
    /// Nonfinite input or a singular stiffness matrix.
    pub fn solve(&self, inp: &LoadInputs) -> Result<Solution, Error> {
        if let Some(g) = &self.geometry {
            return self.solve_geometry(g, inp);
        }
        let vals = [
            inp.ax,
            inp.ay,
            inp.downforce_front_n,
            inp.downforce_rear_n,
            inp.drag_n,
        ];
        if !vals.iter().all(|v| v.is_finite())
            || !inp
                .fx_wheel_n
                .iter()
                .chain(&inp.fy_wheel_n)
                .all(|v| v.is_finite())
        {
            return Err(err("nonfinite load inputs"));
        }
        let p = &self.params;
        let wt = p.pitch_mass_kg * inp.ax * p.cg_height_m / p.wheelbase_m; // Eq. 7-1
        let (anti_f, anti_r) = if inp.ax < 0.0 {
            (p.anti_dive_front_pct, p.anti_lift_rear_pct)
        } else {
            (p.anti_lift_front_pct, p.anti_squat_rear_pct)
        };
        let ds_f = inp.downforce_front_n - wt * (1.0 - anti_f / 100.0); // Eq. 7-8
        let ds_r = inp.downforce_rear_n + wt * (1.0 - anti_r / 100.0);
        let mut f = V7::zeros();
        f[0] = -(ds_f + ds_r);
        f[1] = p.roll_mass_kg * inp.ay * p.roll_lever_arm_m;
        f[2] = p.a_m * ds_f - p.b_m * ds_r;
        let x = self
            .stiffness
            .lu()
            .solve(&f)
            .ok_or_else(|| err("singular stiffness matrix"))?;
        let travel = [x[3], x[4], x[5], x[6]];
        let mut load = [0.0; 4];
        let t = [2.0 * p.half_track_m[0], 2.0 * p.half_track_m[1]];
        let fy_axle = [
            inp.fy_wheel_n[0] + inp.fy_wheel_n[1],
            inp.fy_wheel_n[2] + inp.fy_wheel_n[3],
        ];
        for i in 0..4 {
            let axle = i / 2;
            let side = if i % 2 == 0 { -1.0 } else { 1.0 }; // right (outside in a left turn) gains
            let lateral_geo = side * fy_axle[axle] * p.roll_centre_height_m[axle] / t[axle];
            let long_geo = if axle == 0 {
                -wt * anti_f / 100.0 / 2.0
            } else {
                wt * anti_r / 100.0 / 2.0
            };
            load[i] =
                p.static_wheel_load_n[i] - p.tyre_rate_n_m[i] * travel[i] + lateral_geo + long_geo;
        }
        Ok(Solution {
            heave_m: x[0],
            roll_rad: x[1],
            pitch_rad: x[2],
            wheel_travel_m: travel,
            wheel_load_n: load,
            camber_deg: [0.0; 4],
            toe_deg: [0.0; 4],
            front_rh_mm: p.static_rh_mm[0] + 1000.0 * (x[0] - p.a_m * x[2]),
            rear_rh_mm: p.static_rh_mm[1] + 1000.0 * (x[0] + p.b_m * x[2]),
            roll_deg: x[1].to_degrees(),
            pitch_deg: x[2].to_degrees(),
        })
    }
}

const IDS: [CornerId; 4] = [
    CornerId::FrontLeft,
    CornerId::FrontRight,
    CornerId::RearLeft,
    CornerId::RearRight,
];
fn corner_index(id: CornerId) -> usize {
    IDS.iter().position(|i| *i == id).unwrap()
}

/// Inputs for building the 7×7 from a suspension [`Project`] (corner order FL, FR, RL, RR).
#[derive(Clone, Debug)]
pub struct GeometryInputs<'a> {
    /// Suspension geometry and spring/damper/interconnect laws; `chassis.sprung_mass` is the
    /// whole lumped vehicle mass.
    pub project: &'a Project,
    /// Unsprung mass per corner, kg.
    pub unsprung_mass_kg: [f64; 4],
    /// Unsprung CG height per corner above ground, m.
    pub unsprung_cg_height_m: [f64; 4],
    /// Tyre vertical stiffness per corner, N/m.
    pub tyre_rate_n_m: [f64; 4],
    /// Static ride height front, rear, mm.
    pub static_rh_mm: [f64; 2],
    /// Aero centre of pressure relative to the CG (body frame), m.
    pub aero_cop_offset_m: [f64; 3],
}

// The kinematic quantities that depend on the 7 coordinates, as one flat vector so a single
// central difference yields every Jacobian row: shock compression (4), interconnect heave/roll
// channels (axle*2+channel), camber (4), toe (4), then x,y of the tyre patch (corner*2+axis) and
// of the wheel centre.
const SHOCK: usize = 0;
const ARM: usize = 4;
const CAMBER: usize = 8;
const TOE: usize = 12;
const PATCH: usize = 16;
const WHEEL: usize = 24;
const N_Q: usize = 32;

/// Linearized geometry data for the virtual-work load map (opaque to callers).
#[derive(Clone, Debug)]
pub struct GeometryData {
    /// Static value and Jacobian (per coordinate) of every kinematic quantity above.
    q0: [f64; N_Q],
    jac: [[f64; 7]; N_Q],
    /// Body-point Jacobians over (heave, roll, pitch): CG, front aero point, rear aero point.
    jcg: [[f64; 3]; 3],
    jaero_f: [[f64; 3]; 3],
    jaero_r: [[f64; 3]; 3],
    sprung_mass_kg: f64,
    /// Height of the aero application point above ground, m.
    aero_height_m: f64,
    unsprung_mass_kg: [f64; 4],
    /// Unsprung CG height minus wheel-centre height (the matrix already places the unsprung
    /// inertia at the wheel centre), m.
    unsprung_dh_m: [f64; 4],
    tyre_rate_n_m: [f64; 4],
}

/// The kinematic quantities at coordinates `x` (changes from the static pose; the unsprung
/// displacements are passed as road heights under each wheel).
fn kinematics(p: &Project, q0: [f64; 3], x: &[f64; 7]) -> Result<[f64; N_Q], Error> {
    let mut road = [0.0; 4];
    for (r, c) in road.iter_mut().zip(&p.corners) {
        *r = x[3 + corner_index(c.id)];
    }
    let m = Motion {
        heave: q0[0] + x[0],
        roll: q0[1] + x[1],
        pitch: q0[2] + x[2],
        rack_front: 0.0,
        rack_rear: 0.0,
    };
    let mut state = simulate_on_road_mode(p, &m, road, false)?;
    state.corners.sort_by_key(|c| corner_index(c.id));
    let mut q = [0.0; N_Q];
    for (i, c) in state.corners.iter().enumerate() {
        let (pts, met) = (&c.points, &c.metrics);
        // A longitudinal tyre force is reacted through the upright (outboard brakes), so the
        // patch behaves as a material point of the upright: its x motion adds the upright's
        // pitch rotation acting at the tyre radius.
        let [w, qx, qy, qz] = c.orientation;
        let pitch = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(w, qx, qy, qz))
            .euler_angles()
            .1;
        let radius = p
            .corners
            .iter()
            .find(|k| k.id == c.id)
            .map_or(0.0, |k| k.tire_radius);
        q[SHOCK + i] = met.shock_compression_m;
        q[CAMBER + i] = met.camber_deg;
        q[TOE + i] = met.toe_deg;
        q[PATCH + 2 * i] = pts.wheel_center[0] - radius * pitch;
        q[PATCH + 2 * i + 1] = pts.contact_point[1];
        q[WHEEL + 2 * i..WHEEL + 2 * i + 2].copy_from_slice(&pts.wheel_center[..2]);
    }
    for axle in 0..2 {
        let (l, r) = (
            &state.corners[2 * axle].metrics,
            &state.corners[2 * axle + 1].metrics,
        );
        q[ARM + 2 * axle] = (l.heave_arm_compression_m.unwrap_or(0.0)
            + r.heave_arm_compression_m.unwrap_or(0.0))
            / 2.0;
        q[ARM + 2 * axle + 1] = (l.roll_arm_compression_m.unwrap_or(0.0)
            - r.roll_arm_compression_m.unwrap_or(0.0))
            / 2.0;
    }
    Ok(q)
}

/// Quantities at `x` and their central-difference Jacobian (step `h`) over the 7 coordinates.
fn kinematic_jacobian(
    p: &Project,
    q0: [f64; 3],
    x: [f64; 7],
    h: f64,
) -> Result<([f64; N_Q], [[f64; 7]; N_Q]), Error> {
    let mut jac = [[0.0; 7]; N_Q];
    for k in 0..7 {
        let (mut xp, mut xm) = (x, x);
        xp[k] += h;
        xm[k] -= h;
        let (a, b) = (kinematics(p, q0, &xp)?, kinematics(p, q0, &xm)?);
        for (row, (a, b)) in jac.iter_mut().zip(a.iter().zip(&b)) {
            row[k] = (a - b) / (2.0 * h);
        }
    }
    Ok((kinematics(p, q0, &x)?, jac))
}

/// Generalized force of the springs and interconnects at coordinates `x` (rate-free).
fn internal_force(p: &Project, q0: [f64; 3], x: [f64; 7], h: f64) -> Result<[f64; 7], Error> {
    let (q, jac) = kinematic_jacobian(p, q0, x, h)?;
    let mut f = [0.0; 7];
    let mut add = |force: f64, row: usize| {
        for (fk, jk) in f.iter_mut().zip(&jac[row]) {
            *fk -= force * jk;
        }
    };
    for (i, id) in IDS.iter().enumerate() {
        let corner = p.corners.iter().find(|c| c.id == *id).unwrap();
        add(spring_force(&corner.spring_damper, q[SHOCK + i]), SHOCK + i);
    }
    for (axle, ic) in [&p.front_interconnect, &p.rear_interconnect]
        .into_iter()
        .enumerate()
    {
        if let Some(ic) = ic {
            for (ch, law) in [&ic.heave, &ic.roll].into_iter().enumerate() {
                add(
                    spring_force(law, q[ARM + 2 * axle + ch]),
                    ARM + 2 * axle + ch,
                );
            }
        }
    }
    Ok(f)
}

/// Jacobian over (heave, roll, pitch) of a point fixed to the chassis, which rotates about the CG
/// `c` by `Ry(pitch)·Rx(roll)` and heaves (how `simulate_on_road` places the chassis).
fn body_jacobian(q0: [f64; 3], p0: Vector3<f64>, c: Vector3<f64>) -> [[f64; 3]; 3] {
    let place = |dq: [f64; 3]| {
        let q: [f64; 3] = std::array::from_fn(|i| q0[i] + dq[i]);
        let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), q[2])
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), q[1]);
        c + Vector3::new(0.0, 0.0, q[0]) + r * (p0 - c)
    };
    let h = 1e-6;
    std::array::from_fn(|k| {
        let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
        a[k] = h;
        b[k] = -h;
        let d = (place(a) - place(b)) / (2.0 * h);
        [d.x, d.y, d.z]
    })
}

impl FullCar {
    /// Build the 7×7 from the real suspension geometry: tangent stiffness by differencing the
    /// virtual-work force of the exact kinematic model, loads through the contact-point Jacobians.
    ///
    /// # Errors
    /// Invalid tyre rates/masses (an unsprung mass total at or above the vehicle mass), a failed
    /// static equilibrium, or a kinematic solve failure.
    pub fn from_geometry(inp: &GeometryInputs) -> Result<Self, Error> {
        let p = inp.project;
        p.validate()?;
        let total = p.chassis.sprung_mass;
        let mu_sum: f64 = inp.unsprung_mass_kg.iter().sum();
        let positive = |v: &f64| v.is_finite() && *v > 0.0;
        let inputs = [
            &inp.tyre_rate_n_m[..],
            &inp.unsprung_mass_kg,
            &inp.unsprung_cg_height_m,
            &inp.static_rh_mm,
        ];
        if !inputs.iter().all(|vals| vals.iter().all(positive)) || mu_sum >= total {
            return Err(err("invalid geometry-mode inputs"));
        }
        let lin = linearize_ride(p, &RideRequest::default())?;
        let q0 = lin.equilibrium;
        let mut static_load = [0.0; 4];
        for (id, load) in lin.corner_ids.iter().zip(&lin.support_reaction_n) {
            static_load[corner_index(*id)] = *load;
        }
        // linearized stiffness K = -dF/dx by central differences of the internal force
        let (h, delta) = (5e-5, 2e-4);
        let mut k = M7::zeros();
        for l in 0..7 {
            let (mut xp, mut xm) = ([0.0; 7], [0.0; 7]);
            xp[l] = delta;
            xm[l] = -delta;
            let (fp, fm) = (internal_force(p, q0, xp, h)?, internal_force(p, q0, xm, h)?);
            for r in 0..7 {
                k[(r, l)] = -(fp[r] - fm[r]) / (2.0 * delta);
            }
        }
        let mut k = (k + k.transpose()) / 2.0;
        let mut mass = M7::zeros();
        mass[(0, 0)] = total - mu_sum;
        mass[(1, 1)] = p.chassis.inertia[0];
        mass[(2, 2)] = p.chassis.inertia[1];
        for i in 0..4 {
            k[(3 + i, 3 + i)] += inp.tyre_rate_n_m[i];
            mass[(3 + i, 3 + i)] = inp.unsprung_mass_kg[i];
        }
        let (q_static, jac) = kinematic_jacobian(p, q0, [0.0; 7], h)?;
        let c = Vector3::from(p.chassis.center_of_mass);
        let wheel_center = |i: usize| {
            p.corners
                .iter()
                .find(|c| c.id == IDS[i])
                .unwrap()
                .wheel_center
        };
        let x_front = (wheel_center(0)[0] + wheel_center(1)[0]) / 2.0;
        let x_rear = (wheel_center(2)[0] + wheel_center(3)[0]) / 2.0;
        let z_aero = c.z + inp.aero_cop_offset_m[2];
        let (a, b) = ((x_front - c.x).abs(), (c.x - x_rear).abs());
        let half = |l: usize, r: usize| (wheel_center(l)[1] - wheel_center(r)[1]).abs() / 2.0;
        let params = FullCarParams {
            sprung_mass_kg: total - mu_sum,
            unsprung_mass_kg: inp.unsprung_mass_kg,
            roll_inertia_kg_m2: p.chassis.inertia[0],
            pitch_inertia_kg_m2: p.chassis.inertia[1],
            a_m: a,
            b_m: b,
            half_track_m: [half(0, 1), half(2, 3)],
            wheel_rate_n_m: [0.0; 4],
            arb_wheel_rate_n_m: [0.0; 2],
            tyre_rate_n_m: inp.tyre_rate_n_m,
            wheel_damping_n_s_m: [0.0; 4],
            static_wheel_load_n: static_load,
            static_rh_mm: inp.static_rh_mm,
            wheelbase_m: a + b,
            cg_height_m: c.z + q0[0],
            roll_lever_arm_m: 0.0,
            roll_mass_kg: total,
            pitch_mass_kg: total,
            roll_centre_height_m: [0.0; 2],
            anti_dive_front_pct: 0.0,
            anti_lift_rear_pct: 0.0,
            anti_lift_front_pct: 0.0,
            anti_squat_rear_pct: 0.0,
        };
        let geometry = GeometryData {
            q0: q_static,
            jac,
            jcg: body_jacobian(q0, c, c),
            jaero_f: body_jacobian(q0, Vector3::new(x_front, c.y, z_aero), c),
            jaero_r: body_jacobian(q0, Vector3::new(x_rear, c.y, z_aero), c),
            sprung_mass_kg: params.sprung_mass_kg,
            aero_height_m: z_aero,
            unsprung_mass_kg: inp.unsprung_mass_kg,
            unsprung_dh_m: std::array::from_fn(|i| {
                inp.unsprung_cg_height_m[i] - wheel_center(i)[2]
            }),
            tyre_rate_n_m: inp.tyre_rate_n_m,
        };
        Ok(Self {
            mass,
            damping: M7::zeros(),
            stiffness: k,
            b: SMatrix::zeros(),
            params,
            geometry: Some(geometry),
        })
    }

    /// Front-to-rear pitch load transfer, N, from aerodynamic drag `drag_n` acting at the aero
    /// height against the tyre force at ground level (`D·h_aero/WB`). The thesis bike model
    /// neglects this term; only the geometry-mode matrix carries it (zero in rates mode).
    pub fn drag_transfer_n(&self, drag_n: f64) -> f64 {
        self.geometry
            .as_ref()
            .map_or(0.0, |g| drag_n * g.aero_height_m / self.params.wheelbase_m)
    }

    fn solve_geometry(&self, g: &GeometryData, inp: &LoadInputs) -> Result<Solution, Error> {
        let scalars = [
            inp.ax,
            inp.ay,
            inp.downforce_front_n,
            inp.downforce_rear_n,
            inp.drag_n,
        ];
        if ![&scalars[..], &inp.fx_wheel_n, &inp.fy_wheel_n]
            .iter()
            .all(|v| v.iter().all(|x| x.is_finite()))
        {
            return Err(err("nonfinite load inputs"));
        }
        // generalized force: tyre forces at the patches, unsprung inertia at the wheel centres,
        // sprung inertia at the CG, aero at the front and rear aero points
        let mut f = V7::zeros();
        for i in 0..4 {
            let (m_u, d) = (g.unsprung_mass_kg[i], 2 * i);
            for k in 0..7 {
                f[k] += g.jac[PATCH + d][k] * inp.fx_wheel_n[i]
                    + g.jac[PATCH + d + 1][k] * inp.fy_wheel_n[i]
                    - m_u * (g.jac[WHEEL + d][k] * inp.ax + g.jac[WHEEL + d + 1][k] * inp.ay);
            }
        }
        let df = inp.downforce_front_n + inp.downforce_rear_n;
        let share = if df > 1e-9 {
            inp.downforce_front_n / df
        } else {
            0.5
        };
        for k in 0..3 {
            f[k] -= g.sprung_mass_kg * (g.jcg[k][0] * inp.ax + g.jcg[k][1] * inp.ay);
            f[k] -= g.jaero_f[k][0] * inp.drag_n * share + g.jaero_f[k][2] * inp.downforce_front_n;
            f[k] -= g.jaero_r[k][0] * inp.drag_n * (1.0 - share)
                + g.jaero_r[k][2] * inp.downforce_rear_n;
        }
        let x = self
            .stiffness
            .lu()
            .solve(&f)
            .ok_or_else(|| err("singular stiffness matrix"))?;
        let p = &self.params;
        // The matrix puts each unsprung inertia at its wheel centre; the remaining couple
        // m·a·(h_cg − h_wheel-centre) is not carried by the rigid-z unsprung DOFs and transfers
        // load between the wheels of an axle (lateral) and between axles (longitudinal).
        let mh = |lo: usize| -> f64 {
            (lo..lo + 2)
                .map(|i| g.unsprung_mass_kg[i] * g.unsprung_dh_m[i])
                .sum()
        };
        let lateral = [
            mh(0) * inp.ay / (2.0 * p.half_track_m[0]),
            mh(2) * inp.ay / (2.0 * p.half_track_m[1]),
        ];
        let longitudinal = (mh(0) + mh(2)) * inp.ax / p.wheelbase_m / 2.0;
        let at = |row: usize| g.q0[row] + (0..7).map(|k| g.jac[row][k] * x[k]).sum::<f64>();
        let mut load = [0.0; 4];
        for (i, l) in load.iter_mut().enumerate() {
            let side = if i % 2 == 0 { -1.0 } else { 1.0 }; // right wheels gain in a left turn
            let axle_shift = if i < 2 { -longitudinal } else { longitudinal };
            *l = p.static_wheel_load_n[i] - g.tyre_rate_n_m[i] * x[3 + i]
                + side * lateral[i / 2]
                + axle_shift;
        }
        Ok(Solution {
            heave_m: x[0],
            roll_rad: x[1],
            pitch_rad: x[2],
            wheel_travel_m: [x[3], x[4], x[5], x[6]],
            wheel_load_n: load,
            camber_deg: std::array::from_fn(|i| at(CAMBER + i)),
            toe_deg: std::array::from_fn(|i| at(TOE + i)),
            front_rh_mm: p.static_rh_mm[0] + 1000.0 * (x[0] - p.a_m * x[2]),
            rear_rh_mm: p.static_rh_mm[1] + 1000.0 * (x[0] + p.b_m * x[2]),
            roll_deg: x[1].to_degrees(),
            pitch_deg: x[2].to_degrees(),
        })
    }
}
