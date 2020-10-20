#[macro_use]
pub mod approx;
pub mod vec;
pub mod ray;
pub mod aabb;

pub use self::approx::*;
pub use self::vec::*;
pub use self::ray::*;
pub use self::aabb::*;

pub type Point3 = Vec3;

pub fn lerp(x: f32, y: f32, t: f32) -> f32 {
    x * (1. - t) + t * y
}

pub fn quadratic(a: f32, b: f32, c: f32) -> Option<(f32, f32)> {
    let (a, b, c) = (a as f64, b as f64, c as f64);
    let delta = b * b - 4. * a * c;

    if delta < 0. {
        return None;
    }

    let delta_sqrt = delta.sqrt();
    let q = if b < 0. {
        -0.5 * (b - delta_sqrt)
    } else {
        -0.5 * (b + delta_sqrt)
    };

    let (t0, t1) = (q / a, c / q);

    if t0 < t1 {
        Some((t0 as f32, t1 as f32))
    } else {
        Some((t1 as f32, t0 as f32))
    }
}

pub fn schlick(cos: f32, ri: f32) -> f32 {
    let r0 = (1.0 - ri) / (1.0 + ri);
    let r0 = r0 * r0;

    r0 + (1. - r0) * (1. - cos).powf(5.)
}
