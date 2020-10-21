use crate::{Point3, Vec3};

#[derive(Debug, Copy, Clone)]
pub struct Ray {
    pub o: Point3,
    pub d: Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> Point3 {
        self.o + t * self.d
    }
}

pub fn ray(o: Point3, d: Vec3) -> Ray {
    Ray { o, d }
}

#[derive(Debug, Copy, Clone)]
pub struct RayConstraint {
    pub start: f32,
    pub end: f32,
}

impl RayConstraint {
    pub fn none() -> Self {
        Self {
            start: 0.,
            end: f32::INFINITY,
        }
    }
}
