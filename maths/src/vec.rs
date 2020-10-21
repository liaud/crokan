use std::ops::{Add, Mul, Sub, Neg, Div};
use crate::approx::ApproxEq;

#[derive(Debug, Copy, Clone)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

macro_rules! impl_vec {
    ($t:ident { $($field:ident),+ }) => {
        impl Add<$t> for $t {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                $t {
                    $(
                        $field: self.$field + rhs.$field,
                    )+
                }
            }
        }

        impl Sub<$t> for $t {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                $t {
                    $(
                        $field: self.$field - rhs.$field,
                    )+
                }
            }
        }

        impl Div<f32> for $t {
            type Output = Self;

            fn div(self, rhs: f32) -> Self {
                let inv = 1. / rhs;
                $t {
                    $(
                        $field: inv * self.$field,
                    )+
                }
            }
        }

        impl Mul<$t> for $t {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                $t {
                    $(
                        $field: self.$field * rhs.$field,
                    )+
                }
            }
        }

        impl Mul<f32> for $t {
            type Output = Self;

            fn mul(self, rhs: f32) -> Self {
                $t {
                    $(
                        $field: self.$field * rhs,
                    )+
                }
            }
        }

        impl Mul<$t> for f32 {
            type Output = $t;

            fn mul(self, rhs: $t) -> $t {
                $t {
                    $(
                        $field: self* rhs.$field,
                    )+
                }
            }
        }

        impl Neg for $t {
            type Output = Self;

            fn neg(self) -> Self {
                $t {
                    $(
                        $field: -self.$field,
                    )+
                }
            }
        }

        impl ApproxEq for $t {
            #[inline]
            fn approx_eq_ulps(self, rhs: Self, max_ulps: u32) -> bool {
                $(
                    self.$field.approx_eq_ulps(rhs.$field, max_ulps)
                ) && +
            }

            #[inline]
            fn approx_eq(self, rhs: Self, max_ulps: u32, max_diff: f32) -> bool {
                $(
                    self.$field.approx_eq(rhs.$field, max_ulps, max_diff)
                ) && +
            }

        }

        impl $t {
            pub fn lerp(self, rhs: Self, t: f32) -> Self {
                $t {
                    $(
                        $field: crate::lerp(self.$field, rhs.$field, t),
                    )+
                }
            }

            pub fn saturate(self) -> Self {
                $t {
                    $(
                        $field: self.$field.max(0.).min(1.),
                    )+
                }
            }

            pub fn min(self, rhs: Self) -> Self {
                $t {
                    $(
                        $field: self.$field.min(rhs.$field),
                    )*
                }
            }

            pub fn max(self, rhs: Self) -> Self {
                $t {
                    $(
                        $field: self.$field.max(rhs.$field),
                    )*
                }
            }
        }
    };
}

impl_vec!(Vec2 { x, y });
impl_vec!(Vec3 { x, y, z });
impl_vec!(Vec4 { x, y, z, w });

pub use self::v3::*;
pub use self::p3::*;

pub mod v2 {
    use super::Vec2;

    pub fn v2(x: f32, y: f32) -> Vec2 {
        Vec2 {
            x, y
        }
    }
}

pub mod v3 {
    use super::{Vec3, Vec2};

    pub const fn v3(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 {
            x, y, z
        }
    }

    pub const fn from_v2(xy: Vec2, z: f32) -> Vec3 {
        Vec3 {
            x: xy.x, y: xy.y, z
        }
    }

    pub const fn up() -> Vec3 {
        Vec3 {
            x: 0.,
            y: 1.,
            z: 0.,
        }
    }
}

pub mod p3 {
    use super::Vec3;

    pub const fn p3(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 {
            x, y, z
        }
    }
}

pub mod v4 {
    use super::Vec4;

    pub const fn v4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4 {
            x, y, z, w
        }
    }
}

impl Vec3 {
    pub fn unit(self) -> Self {
        let inv = 1. / self.length();
        inv * self
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Vec3 {
        Vec3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    pub fn reflect(self, n: Vec3) -> Vec3 {
        self - 2.*self.dot(n)*n
    }

    pub fn refract(self, n: Vec3, etai_over_etat: f32) -> Vec3 {
        let cos_theta = (-self).dot(n);
        let r_perp = etai_over_etat * (self + cos_theta*n);
        let r_parallel = -(1.0 - r_perp.dot(r_perp)).abs().sqrt() * n;

        r_perp + r_parallel
    }

    pub fn xy(self) -> Vec2 {
        Vec2 {
            x: self.x,
            y: self.y
        }
    }

    #[inline]
    pub fn dim(self, d: usize) -> f32 {
        if d == 0 {
            self.x
        } else if d == 1 {
            self.y
        } else if d == 2 {
            self.z
        } else {
            panic!(r#"invalid dimension"#);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    #[test]
    fn basic_ops() {
        let a = v3(0., 1., 2.);
        let b = v3(1., 1., 2.);

        assert_approx_eq!(a + b, v3(1., 2., 4.));
        assert_approx_eq!(a - b, v3(-1., 0., 0.));
        assert_approx_eq!(a * b, v3(0., 1., 4.));
        assert_approx_eq!(a / 2., v3(0., 0.5, 1.));
        assert_approx_eq!(3. * b, v3(3., 3., 6.));
        assert_approx_eq!(-b, v3(-1., -1., -2.));
    }

    #[test]
    fn unit_and_length() {
        let a = v3(0., 0., 0.);
        let b = v3(2., 3., 2.);
        let c = v3(1., 0., 0.);

        assert_approx_eq!(a.length(), 0.);
        assert_approx_eq!(b.length(), (17.0_f32).sqrt());
        assert_approx_eq!(b.length().powf(2.), b.dot(b));

        assert_approx_eq!(b.unit().length(), 1.0);
        assert_approx_eq!(c.length(), 1.0);
        assert_approx_eq!(c.unit(), c);
    }


    #[test]
    fn cross() {
        let a = v3(0., 0., 1.);
        let c = v3(1., 0., 0.);

        let b = a.cross(c);
        assert_approx_eq!(b, v3(0., 1., 0.));
        assert_approx_eq!(b.dot(a), 0.);
        assert_approx_eq!(b.dot(c), 0.);
        assert_approx_eq!(c.cross(a), -b);

    }
}
