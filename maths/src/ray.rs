use crate::{Point3, Vec3};

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
