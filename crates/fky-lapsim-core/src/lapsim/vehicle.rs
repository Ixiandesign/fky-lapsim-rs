//! Vehicle configuration for the QSS lap simulator: the extras the thesis model needs on top of a
//! [`LapVehicle`], μ derived from the Magic Formula tyres, and the assembled [`QssVehicle`].
use super::aeromap::AeroMap;
use super::coupled::{CoupleSettings, Coupled, TyreMode, TyreTable};
use super::err;
use super::forces::{StepModel, ThesisConst};
use super::fullcar::{FullCar, GeometryInputs};
use super::thesis::{AxleTyre, BikeParams, BrakeSystem, ConstAero, Correlation};
use super::tractive::TractiveTable;
use super::G;
use crate::dynamics::{linearize_ride, RideRequest};
use crate::lap::LapVehicle;
use crate::tire::TireInput;
use crate::tire_configuration::{tire_input, ConfiguredTire, Provenance};
use crate::{CornerId, Error};
use serde::{Deserialize, Serialize};

/// Unsprung mass properties of one corner (thesis Tables 2-1, 2-4).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnsprungCorner {
    /// Unsprung mass, kg.
    pub mass_kg: f64,
    /// Unsprung CG height above ground, m.
    pub cg_height_m: f64,
    /// Tyre vertical stiffness, N/m.
    pub tyre_vertical_stiffness_n_m: f64,
}

/// Per-wheel friction evaluation requested for the coupled model.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TyreModeKind {
    /// Thesis load-sensitivity formula per wheel.
    Thesis,
    /// Magic-Formula table in wheel load and camber.
    Table,
}

/// Lap-simulator inputs beyond the suspension/tyre/powertrain/aero already in a [`LapVehicle`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QssExtras {
    /// Unsprung properties per corner FL, FR, RL, RR (required).
    pub unsprung: [UnsprungCorner; 4],
    /// Static ride height front, rear, mm.
    pub static_ride_height_mm: [f64; 2],
    /// Steering-wheel to front-wheel ratio.
    pub steer_ratio: f64,
    /// Rolling-resistance coefficient magnitude.
    pub rolling_resistance_cr: f64,
    /// Brake hardware for the brake-pressure channel.
    pub brake: BrakeSystem,
    /// Optional AeroMap (needs the 7×7 matrix).
    #[serde(default)]
    pub aero_map: Option<AeroMap>,
    /// Per-wheel friction evaluation for the coupled model.
    pub tyre_mode: TyreModeKind,
    /// Correlation multipliers (default 1).
    #[serde(default)]
    pub correlation: Correlation,
    /// Explicit friction parameters overriding the Magic-Formula derivation (front, rear).
    #[serde(default)]
    pub tyre_override: Option<[AxleTyre; 2]>,
}
// (AeroMap needs `#[derive(Serialize, Deserialize)]`; add them to `aeromap.rs` structs as part of this task.)

impl QssExtras {
    /// Illustrative values for the synthetic demo vehicle; not a calibrated car.
    pub fn synthetic_demo() -> Self {
        let corner = |m: f64, h: f64| UnsprungCorner {
            mass_kg: m,
            cg_height_m: h,
            tyre_vertical_stiffness_n_m: 95.0e3,
        };
        Self {
            unsprung: [
                corner(8.0, 0.25),
                corner(8.0, 0.25),
                corner(9.0, 0.26),
                corner(9.0, 0.26),
            ],
            static_ride_height_mm: [35.0, 45.0],
            steer_ratio: 3.74,
            rolling_resistance_cr: 0.03,
            brake: super::thesis::p19().brake,
            aero_map: None,
            tyre_mode: TyreModeKind::Thesis,
            correlation: Correlation::default(),
            tyre_override: None,
        }
    }
}

/// Friction parameters fitted to a Magic Formula tyre, with their provenance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DerivedTyre {
    /// The thesis-form axle parameters.
    pub axle_tyre: AxleTyre,
    /// Static wheel load the line was anchored at, N.
    pub static_wheel_load_n: f64,
    /// Where the coefficients came from (tyre provenance text).
    pub source: String,
}

const SWEEP_STEP: f64 = 0.002;

/// Peak `(μx, μy)` of `tire` at wheel load `fz` and outward camber `camber_deg` (pure slip).
fn peak_mu(
    tire: &ConfiguredTire,
    id: CornerId,
    fz: f64,
    camber_deg: f64,
) -> Result<(f64, f64), Error> {
    let speed = 0.5 * (tire.domain.speed_m_s[0] + tire.domain.speed_m_s[1]).min(30.0);
    let base = tire_input(
        id,
        fz,
        [speed, 0.0],
        speed / tire.model.radius_m,
        tire.model.radius_m,
        camber_deg,
    )?;
    let (mut mux, mut muy) = (0.0_f64, 0.0_f64);
    let mut a = 0.0;
    while a <= tire.domain.max_abs_slip_angle_rad {
        let f = tire.evaluate(TireInput {
            slip_angle_rad: a,
            ..base
        })?;
        muy = muy.max(f.fy_n.abs() / fz);
        a += SWEEP_STEP;
    }
    let mut k = 0.0;
    while k <= tire.domain.max_abs_slip_ratio.min(0.5) {
        let f = tire.evaluate(TireInput {
            slip_ratio: k,
            ..base
        })?;
        mux = mux.max(f.fx_n.abs() / fz);
        k += SWEEP_STEP;
    }
    Ok((mux, muy))
}

/// Fit the thesis load-sensitivity line (Eqs. 2-12…2-15) to a Magic Formula tyre (spec §5.3).
///
/// # Errors
/// Returns an error for a nonpositive load or one outside the tyre's declared domain.
pub fn derive_axle_tyre(
    tire: &ConfiguredTire,
    static_wheel_load_n: f64,
) -> Result<DerivedTyre, Error> {
    let [lo, hi] = tire.domain.normal_load_n;
    if !static_wheel_load_n.is_finite()
        || static_wheel_load_n <= 0.0
        || static_wheel_load_n < lo
        || static_wheel_load_n > hi
    {
        return Err(err(
            "static wheel load is outside the tyre's calibrated load domain",
        ));
    }
    let f1 = (0.6 * static_wheel_load_n).max(lo.max(1.0));
    let f2 = (1.4 * static_wheel_load_n).min(hi);
    if f2 <= f1 {
        return Err(err(
            "tyre load domain too narrow to fit the load sensitivity",
        ));
    }
    let id = CornerId::FrontLeft;
    let (mx1, my1) = peak_mu(tire, id, f1, 0.0)?;
    let (mx2, my2) = peak_mu(tire, id, f2, 0.0)?;
    let (mxs, mys) = peak_mu(tire, id, static_wheel_load_n, 0.0)?;
    let speed = 0.5 * (tire.domain.speed_m_s[0] + tire.domain.speed_m_s[1]).min(30.0);
    let base = tire_input(
        id,
        static_wheel_load_n,
        [speed, 0.0],
        speed / tire.model.radius_m,
        tire.model.radius_m,
        0.0,
    )?;
    let half_deg = 0.5_f64.to_radians();
    let fy = tire
        .evaluate(TireInput {
            slip_angle_rad: half_deg,
            ..base
        })?
        .fy_n
        .abs();
    let norm_kg = static_wheel_load_n / G;
    Ok(DerivedTyre {
        axle_tyre: AxleTyre {
            mux: mxs,
            muy: mys,
            mux_norm_kg: norm_kg,
            muy_norm_kg: norm_kg,
            mux_sens_per_n: (mx1 - mx2) / (f2 - f1),
            muy_sens_per_n: (my1 - my2) / (f2 - f1),
            cornering_stiffness_n_per_deg: fy / 0.5,
        },
        static_wheel_load_n,
        source: match &tire.provenance {
            Provenance::Synthetic { description } => format!("synthetic: {description}"),
            Provenance::Measured { source, fit_notes } => {
                format!("measured: {source} ({fit_notes})")
            }
        },
    })
}

/// Build a peak-friction table over load and camber for one corner's tyre.
///
/// # Errors
/// Propagates tyre evaluation errors (loads outside the declared domain are clipped out of the
/// grid, never extrapolated).
pub fn build_tyre_table(
    tire: &ConfiguredTire,
    static_wheel_load_n: f64,
    id: CornerId,
) -> Result<TyreTable, Error> {
    let [lo, hi] = tire.domain.normal_load_n;
    if !static_wheel_load_n.is_finite() || static_wheel_load_n <= 0.0 {
        return Err(err("invalid static wheel load"));
    }
    let (l0, l1) = (
        (0.3 * static_wheel_load_n).max(lo.max(1.0)),
        (2.2 * static_wheel_load_n).min(hi),
    );
    if l1 <= l0 {
        return Err(err("tyre load domain too narrow for a friction table"));
    }
    let n = 8;
    let load_n: Vec<f64> = (0..n)
        .map(|i| l0 + (l1 - l0) * i as f64 / (n - 1) as f64)
        .collect();
    let cm = 4.0_f64.min(0.99 * tire.domain.max_abs_camber_rad.to_degrees());
    let camber_deg: Vec<f64> = (0..5).map(|j| -cm + 2.0 * cm * j as f64 / 4.0).collect();
    let (mut mux, mut muy) = (vec![], vec![]);
    for &fz in &load_n {
        let (mut rx, mut ry) = (vec![], vec![]);
        for &c in &camber_deg {
            let (x, y) = peak_mu(tire, id, fz, c)?;
            rx.push(x);
            ry.push(y);
        }
        mux.push(rx);
        muy.push(ry);
    }
    Ok(TyreTable {
        load_n,
        camber_deg,
        mux,
        muy,
    })
}

/// A vehicle assembled for the QSS lap simulator.
#[derive(Clone, Debug)]
pub struct QssVehicle {
    /// Thesis bike parameters.
    pub bike: BikeParams,
    /// 7×7 matrix from the suspension geometry.
    pub full_car: FullCar,
    /// Engine tractive force table.
    pub tractive: TractiveTable,
    /// Optional AeroMap.
    pub aero_map: Option<AeroMap>,
    /// Per-wheel friction evaluation.
    pub tyre_mode: TyreMode,
    /// Fitted friction parameters (front, rear) with provenance.
    pub derived_tyres: [DerivedTyre; 2],
    /// Correlation factors in force.
    pub correlation: Correlation,
}

impl QssVehicle {
    /// Assemble from a lap vehicle and the extras.
    ///
    /// # Errors
    /// Invalid vehicle or extras, failed equilibrium/kinematics, or tyre derivation errors.
    pub fn build(car: &LapVehicle, extras: &QssExtras) -> Result<Self, Error> {
        car.validate()?;
        let e = extras;
        let extras_ok = e.unsprung.iter().all(|u| {
            [u.mass_kg, u.cg_height_m, u.tyre_vertical_stiffness_n_m]
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
        }) && e.steer_ratio.is_finite()
            && e.steer_ratio > 0.0
            && e.rolling_resistance_cr.is_finite()
            && e.rolling_resistance_cr >= 0.0
            && e.static_ride_height_mm
                .iter()
                .all(|v| v.is_finite() && *v > 0.0);
        if !extras_ok {
            return Err(err("invalid QSS extras"));
        }
        if let Some(map) = &e.aero_map {
            map.validate()?;
        }
        let p = &car.suspension;
        let ids = [
            CornerId::FrontLeft,
            CornerId::FrontRight,
            CornerId::RearLeft,
            CornerId::RearRight,
        ];
        let lin = linearize_ride(p, &RideRequest::default())?;
        let mut load = [0.0; 4];
        for (k, id) in lin.corner_ids.iter().enumerate() {
            load[ids.iter().position(|i| i == id).unwrap()] = lin.support_reaction_n[k];
        }
        let total_w: f64 = load.iter().sum();
        let wd = 100.0 * (load[0] + load[1]) / total_w;
        let wc = |id: CornerId| p.corners.iter().find(|c| c.id == id).unwrap().wheel_center;
        let x_front = (wc(ids[0])[0] + wc(ids[1])[0]) / 2.0;
        let x_rear = (wc(ids[2])[0] + wc(ids[3])[0]) / 2.0;
        let wheelbase = (x_front - x_rear).abs();
        let front_track = (wc(ids[0])[1] - wc(ids[1])[1]).abs();
        let rear_track = (wc(ids[2])[1] - wc(ids[3])[1]).abs();
        let cg_h = p.chassis.center_of_mass[2] + lin.equilibrium[0];
        let wheel = |id: CornerId| car.wheels.iter().find(|w| w.id == id).unwrap();
        let radius = wheel(ids[0]).tire.model.radius_m;
        let axle_static = [(load[0] + load[1]) / 2.0, (load[2] + load[3]) / 2.0];
        let derived: [DerivedTyre; 2] = match &e.tyre_override {
            Some(o) => [
                DerivedTyre {
                    axle_tyre: o[0].clone(),
                    static_wheel_load_n: axle_static[0],
                    source: "explicit override".into(),
                },
                DerivedTyre {
                    axle_tyre: o[1].clone(),
                    static_wheel_load_n: axle_static[1],
                    source: "explicit override".into(),
                },
            ],
            None => [
                derive_axle_tyre(&wheel(ids[0]).tire, axle_static[0])?,
                derive_axle_tyre(&wheel(ids[2]).tire, axle_static[1])?,
            ],
        };
        let tyre_mode = match e.tyre_mode {
            TyreModeKind::Thesis => TyreMode::Thesis,
            TyreModeKind::Table => TyreMode::Table([
                build_tyre_table(&wheel(ids[0]).tire, axle_static[0], ids[0])?,
                build_tyre_table(&wheel(ids[2]).tire, axle_static[1], ids[2])?,
            ]),
        };
        let a = &car.aero;
        let a_dist = (1.0 - wd / 100.0) * wheelbase;
        let b_dist = wheelbase - a_dist;
        let ab = (100.0 * (b_dist + a.center_of_pressure_m[0]) / wheelbase).clamp(0.0, 100.0);
        let (area, cz, cx) = if a.reference_area_m2 > 0.0 {
            (
                a.reference_area_m2,
                a.downforce_coefficient,
                a.drag_coefficient,
            )
        } else {
            (1.0, 0.0, 0.0)
        };
        let bike = BikeParams {
            mass_kg: p.chassis.sprung_mass,
            wd_front_pct: wd,
            wheelbase_m: wheelbase,
            front_track_m: front_track,
            rear_track_m: rear_track,
            cg_height_m: cg_h,
            steer_ratio: e.steer_ratio,
            rolling_radius_m: radius,
            rolling_resistance_cr: e.rolling_resistance_cr,
            drive: car.powertrain.driven_axle,
            front: derived[0].axle_tyre.clone(),
            rear: derived[1].axle_tyre.clone(),
            aero: ConstAero {
                rho_kg_m3: a.air_density_kg_m3,
                cz_total: cz,
                aero_balance_front_pct: ab,
                cx_total: cx,
                area_m2: area,
            },
            brake: e.brake.clone(),
            correlation: e.correlation,
        };
        bike.validate()?;
        let tractive =
            TractiveTable::build(&car.powertrain, radius, e.correlation.engine_power, 0.05)?;
        let full_car = FullCar::from_geometry(&GeometryInputs {
            project: p,
            unsprung_mass_kg: std::array::from_fn(|i| e.unsprung[i].mass_kg),
            unsprung_cg_height_m: std::array::from_fn(|i| e.unsprung[i].cg_height_m),
            tyre_rate_n_m: std::array::from_fn(|i| e.unsprung[i].tyre_vertical_stiffness_n_m),
            static_rh_mm: e.static_ride_height_mm,
            aero_cop_offset_m: a.center_of_pressure_m,
        })?;
        Ok(Self {
            bike,
            full_car,
            tractive,
            aero_map: e.aero_map.clone(),
            tyre_mode,
            derived_tyres: derived,
            correlation: e.correlation,
        })
    }

    /// The step model for a run: `matrix == false` is the thesis-exact model (no AeroMap),
    /// `matrix == true` the coupled bike ↔ 7×7 model.
    ///
    /// # Errors
    /// An AeroMap with `matrix == false` (the map needs the matrix's ride height, roll and yaw).
    pub fn step_model(
        &self,
        matrix: bool,
        weight_transfer: bool,
    ) -> Result<Box<dyn StepModel>, Error> {
        if !matrix {
            if self.aero_map.is_some() {
                return Err(err("an AeroMap needs the 7x7 matrix (set matrix = true)"));
            }
            return Ok(Box::new(ThesisConst {
                params: self.bike.clone(),
                weight_transfer,
            }));
        }
        Ok(Box::new(Coupled {
            params: self.bike.clone(),
            car: self.full_car.clone(),
            aero_map: self.aero_map.clone(),
            tyres: self.tyre_mode.clone(),
            settings: CoupleSettings::default(),
        }))
    }
}
