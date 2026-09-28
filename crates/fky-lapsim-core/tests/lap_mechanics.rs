use dw_core::{lap::suspension_forces, dynamics::{formula_car_demo, evaluate_ride}, RideRequest};

#[test]
fn suspension_virtual_work_matches_ride_and_corner_order_is_irrelevant() {
    let (p,_)=formula_car_demo().unwrap();
    let q=[-0.002,0.004,-0.003]; let v=[0.01,-0.02,0.03];
    let r=RideRequest::default();
    let ride=evaluate_ride(&p,&r,0.,q,v).unwrap();
    let f=suspension_forces(&p,q,v,0.,0.).unwrap();
    for i in 0..4 {assert!((f.normal_load_n[i]-ride.support_reaction_n[i]).abs()<0.05);}
    assert!((f.generalized_force[0]-p.chassis.sprung_mass*ride.acceleration[0]).abs()<0.05);
    let mut shuffled=p.clone(); shuffled.corners.swap(0,3);
    let b=suspension_forces(&shuffled,q,v,0.,0.).unwrap();
    for i in 0..4 {assert!((b.normal_load_n[i]-f.normal_load_n[i]).abs()<1e-9);}
}

#[test]
fn suspension_force_is_negative_potential_gradient_with_interconnects() {
    let p=dw_core::Project::example_with_interconnect();
    let q=[-0.02,0.003,0.002];
    let s=suspension_forces(&p,q,[0.;3],0.,0.).unwrap();
    for j in 0..3 {
        let h=1e-5; let mut qp=q; let mut qm=q; qp[j]+=h; qm[j]-=h;
        let ep=suspension_forces(&p,qp,[0.;3],0.,0.).unwrap().potential_energy_j;
        let em=suspension_forces(&p,qm,[0.;3],0.,0.).unwrap().potential_energy_j;
        assert!((s.generalized_force[j]+(ep-em)/(2.*h)).abs()<0.05);
    }
}
