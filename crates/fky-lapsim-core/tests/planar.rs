use dw_core::planar::{balance, contact_velocity, PlanarState, WheelForce};

#[test]
fn four_wheel_balance_retains_left_right_traction_yaw_and_aligning_moments() {
    let state = PlanarState {
        u_m_s: 10.,
        v_m_s: 2.,
        yaw_rate_rad_s: 0.5,
        ..Default::default()
    };
    let forces = [
        WheelForce {
            position_m: [1., 0.5, 0.],
            steer_rad: 0.,
            fx_n: 100.,
            fy_n: 20.,
            mz_nm: 3.,
        },
        WheelForce {
            position_m: [1., -0.5, 0.],
            steer_rad: 0.,
            fx_n: 200.,
            fy_n: 20.,
            mz_nm: 3.,
        },
        WheelForce {
            position_m: [-1., 0.5, 0.],
            steer_rad: 0.,
            fx_n: 0.,
            fy_n: 20.,
            mz_nm: 3.,
        },
        WheelForce {
            position_m: [-1., -0.5, 0.],
            steer_rad: 0.,
            fx_n: 0.,
            fy_n: 20.,
            mz_nm: 3.,
        },
    ];
    let d = balance(100., 50., state, &forces, [0., 0., 0.]).unwrap();
    assert_eq!(d.u_m_s2, 4.); // 300/100 + r*v
    assert_eq!(d.v_m_s2, -4.2); // 80/100 - r*u
    assert_eq!(d.yaw_acceleration_rad_s2, 1.24); // (50 Nm traction + 12 aligning)/50
    let vel = contact_velocity(state, [1., 0.5, 0.], 0.).unwrap();
    assert_eq!(vel, [9.75, 2.5]);
}

#[test]
fn steer_rotation_and_global_kinematics_are_consistent() {
    let state = PlanarState {
        u_m_s: 10.,
        heading_rad: std::f64::consts::FRAC_PI_2,
        ..Default::default()
    };
    let wheel = WheelForce {
        position_m: [1., 0., 0.],
        steer_rad: std::f64::consts::FRAC_PI_2,
        fx_n: 100.,
        fy_n: 0.,
        mz_nm: 0.,
    };
    let d = balance(100., 50., state, &[wheel; 4], [0., 0., 0.]).unwrap();
    assert!(d.x_m_s.abs() < 1e-12);
    assert_eq!(d.y_m_s, 10.);
    assert!((d.v_m_s2 - 4.).abs() < 1e-12);
    assert!((d.yaw_acceleration_rad_s2 - 8.).abs() < 1e-12);
    assert!(balance(0., 50., state, &[wheel; 4], [0., 0., 0.]).is_err());
}
