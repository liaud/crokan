pub trait ApproxEq {
    fn approx_eq_ulps(self, rhs: Self, max_ulps: u32) -> bool;
    fn approx_eq(self, rhs: Self, max_ulps: u32, max_diff: f32) -> bool;
}

impl ApproxEq for f32 {
    #[inline]
    fn approx_eq_ulps(self, rhs: Self, max_ulps: u32) -> bool {
        let lhs = i32::from_le_bytes(self.to_le_bytes());
        let rhs = i32::from_le_bytes(rhs.to_le_bytes());

        if lhs.signum() != rhs.signum() {
            return lhs == rhs;
        }

        let ulps = (lhs - rhs).abs() as u32;
        ulps <= max_ulps
    }

    #[inline]
    fn approx_eq(self, rhs: Self, max_ulps: u32, epsilon: f32) -> bool {
        let abs_diff = (rhs - self).abs();
        if abs_diff <= epsilon {
            return true;
        }

        self.approx_eq_ulps(rhs, max_ulps)
    }
}

#[macro_export]
macro_rules! approx_eq {
    ($lhs:expr, $rhs:expr, ulps = $ulps:expr, epsilon = $epsilon:expr) => {
        $crate::approx::ApproxEq::approx_eq($rhs, $lhs, $ulps, $epsilon)
    };
    ($lhs:expr, $rhs:expr, epsilon = $epsilon:expr) => {
        $crate::approx::ApproxEq::approx_eq($rhs, $lhs, 4, $epsilon)
    };
    ($lhs:expr, $rhs:expr, ulps = $ulps:expr) => {
        $crate::approx::ApproxEq::approx_eq($rhs, $lhs, $ulps, std::f32::EPSILON)
    };
    ($lhs:expr, $rhs:expr) => {
        $crate::approx::ApproxEq::approx_eq($rhs, $lhs, 4, std::f32::EPSILON)
    };
}

#[macro_export]
macro_rules! assert_approx_eq {
    ($lhs:expr, $rhs:expr, ulps = $ulps:expr, epsilon = $epsilon:expr) => {
        if !approx_eq!($rhs, $lhs, ulps = $ulps, epsilon = $epsilon) {
            panic!(r#"assertion failed `(left == right)`
  left: `{:?}`,
 right: `{:?}: max_ulps: {}, epsilon: {}`"#, $lhs, $rhs, $ulps, $epsilon);
        }
    };

    ($lhs:expr, $rhs:expr, epsilon = $epsilon:expr) => {
        if !approx_eq!($rhs, $lhs, ulps = 4, epsilon = $epsilon) {
            panic!(r#"assertion failed `(left == right)`
  left: `{:?}`,
 right: `{:?}: max_ulps: {}, epsilon: {}`"#, $lhs, $rhs, 4, $epsilon);
        }
    };
    ($lhs:expr, $rhs:expr, ulps = $ulps:expr) => {
        if !approx_eq!($rhs, $lhs, ulps = $ulps, epsilon = std::f32::EPSILON) {
            panic!(r#"assertion failed `(left == right)`
  left: `{:?}`,
 right: `{:?}: max_ulps: {}, epsilon: {}`"#, $lhs, $rhs, $ulps, std::f32::EPSILON);
        }
    };

    ($lhs:expr, $rhs:expr) => {
        if !approx_eq!($rhs, $lhs, ulps = 4, epsilon = std::f32::EPSILON) {
            panic!(r#"assertion failed `(left == right)`
  left: `{:?}`,
 right: `{:?}: max_ulps: {}, epsilon: {}`"#, $lhs, $rhs, 4, std::f32::EPSILON);
        }
    };
}
