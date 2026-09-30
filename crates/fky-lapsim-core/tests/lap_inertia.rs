use fky_lapsim_core::lap::{angular_acceleration, angular_energy};

#[test]
fn euler_inertia_reduces_to_principal_axes_and_conserves_energy() {
    let i=[2.,3.,5.];
    let a=angular_acceleration(i,[0.,0.],[0.,0.,0.],[10.,6.,12.]).unwrap();
    for (actual,expected) in a.into_iter().zip([2.,3.,4.]) {
        assert!((actual-expected).abs()<1e-12); // yaw, roll, pitch
    }
    let q=[0.2,-0.3];
    let v=[0.4,0.7,-0.2];
    let acc=angular_acceleration(i,q,v,[0.;3]).unwrap();
    let h=1e-6;
    let ep=angular_energy(i,[q[0]+h*v[1],q[1]+h*v[2]],std::array::from_fn(|j|v[j]+h*acc[j]));
    let em=angular_energy(i,[q[0]-h*v[1],q[1]-h*v[2]],std::array::from_fn(|j|v[j]-h*acc[j]));
    assert!(((ep-em)/(2.*h)).abs()<1e-8);
    let force=[3.,-2.,4.];
    let acc=angular_acceleration(i,q,v,force).unwrap();
    let ep=angular_energy(i,[q[0]+h*v[1],q[1]+h*v[2]],std::array::from_fn(|j|v[j]+h*acc[j]));
    let em=angular_energy(i,[q[0]-h*v[1],q[1]-h*v[2]],std::array::from_fn(|j|v[j]-h*acc[j]));
    assert!(((ep-em)/(2.*h)-(0..3).map(|j|force[j]*v[j]).sum::<f64>()).abs()<1e-8);
}

#[test]
fn inertia_rejects_euler_singularity_and_nonphysical_inputs() {
    assert!(angular_acceleration([1.,2.,3.],[0.,std::f64::consts::FRAC_PI_2],[0.;3],[0.;3]).is_err());
    assert!(angular_acceleration([0.,2.,3.],[0.;2],[0.;3],[0.;3]).is_err());
}
