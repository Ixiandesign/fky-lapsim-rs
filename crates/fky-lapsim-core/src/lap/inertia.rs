//! Exact Euler inertia for Rz(yaw) Ry(pitch) Rx(roll), about the chassis CG.
use nalgebra::{Matrix3,Vector3};
use crate::Error;
fn axes(q:[f64;2])->Matrix3<f64> {
    let (sr,cr)=q[0].sin_cos();
    let (sp,cp)=q[1].sin_cos();
    Matrix3::from_columns(&[Vector3::new(-sp,sr*cp,cr*cp),Vector3::x(),Vector3::new(0.,cr,-sr)])
}
/// Chassis rotational kinetic energy, J. Rates ordered [yaw,roll,pitch], rad/s.
pub fn angular_energy(inertia:[f64;3],roll_pitch:[f64;2],rates:[f64;3])->f64 {
    let w=axes(roll_pitch)*Vector3::from(rates);
    0.5*(0..3).map(|j|inertia[j]*w[j]*w[j]).sum::<f64>()
}
/// Angular accelerations from conjugate generalized moments, retaining Euler
/// gyroscopic terms and yaw/roll/pitch inertial coupling. Rates/output order is
/// [yaw,roll,pitch]; pose is [roll,pitch]. Near pitch ±90° is rejected.
pub fn angular_acceleration(inertia:[f64;3],q:[f64;2],rates:[f64;3],moments:[f64;3])->Result<[f64;3],Error> {
    if inertia.iter().chain(q.iter()).chain(rates.iter()).chain(moments.iter()).any(|x|!x.is_finite())
        || inertia.iter().any(|i|*i<=0.) || q[1].cos().abs()<1e-6 {
        return Err(Error {message:"invalid or singular rotational inertia state".into()});
    }
    let (sr,cr)=q[0].sin_cos(); let (sp,cp)=q[1].sin_cos();
    let [_,rd,pd]=rates;
    let g=axes(q);
    let gd=Matrix3::from_columns(&[Vector3::new(-cp*pd,cr*rd*cp-sr*sp*pd,-sr*rd*cp-cr*sp*pd),
        Vector3::zeros(),Vector3::new(0.,-sr*rd,-cr*rd)]);
    let v=Vector3::from(rates); let w=g*v;
    let im=Matrix3::from_diagonal(&Vector3::from(inertia));
    let mass=g.transpose()*im*g;
    let rhs=Vector3::from(moments)-g.transpose()*(im*gd*v+w.cross(&(im*w)));
    let a=mass.cholesky().ok_or_else(||Error{message:"singular rotational inertia".into()})?.solve(&rhs);
    if !a.iter().all(|x|x.is_finite()) {return Err(Error{message:"nonfinite angular acceleration".into()});}
    Ok(a.into())
}
