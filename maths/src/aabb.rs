use crate::Point3;
use crate::ray::{Ray, RayConstraint};

#[derive(Debug, Copy, Clone)]
pub struct Aabb3 {
    pub min: Point3,
    pub max: Point3,
}

impl Aabb3 {
    pub fn intersect(&self, ray: &Ray, constraints: &RayConstraint) -> bool {
        use std::mem;

        for dim_idx in 0..3 {
            let inv_d = 1. / ray.d.dim(dim_idx);
            let mut t0 = inv_d * (self.min.dim(dim_idx) - ray.o.dim(dim_idx));
            let mut t1 = inv_d * (self.max.dim(dim_idx) - ray.o.dim(dim_idx));

            if inv_d < 0. {
                mem::swap(&mut t0, &mut t1);
            }

            let t_min = if t0 > constraints.start { t0 } else { constraints.start };
            let t_max = if t1 < constraints.end { t1 } else { constraints.end };

            if t_max <= t_min {
                return false;
            }
        }
        true
    }

    pub fn center(self) -> Point3 {
        self.min + (self.max - self.min) / 2.
    }

    pub fn union(self, rhs: Aabb3) -> Self {
        Aabb3 {
            min: self.min.min(rhs.min),
            max: self.max.max(rhs.max)
        }
    }


}

pub use self::aabb3::*;

pub mod aabb3 {
    use super::Aabb3;
    use crate::p3;

    pub fn empty() -> Aabb3 {
        let n_inf = std::f32::NEG_INFINITY;
        let inf = std::f32::INFINITY;

        Aabb3 {
            min: p3(inf, inf, inf),
            max: p3(n_inf, n_inf, n_inf)
        }
    }
}
