use std::ops::Range;

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new() -> Rng {
        Self {
            state: 0
        }
    }

    pub fn next(&mut self) -> u32 {
        const A: u64 = 2_862_933_555_777_941_757;

        // The only requirement for the additive constant is to be odd, prime with the modulus,
        // here 2^64.
        let c = (&self.state as *const _ as u64) + 1;
        self.state = A.wrapping_mul(self.state).wrapping_add(c);

        let st = self.state;
        let output = (((st ^ (st >> 18)) >> 27) as u32).rotate_right((st >> 59) as u32);

        output
    }


    pub fn next_zero_one(&mut self) -> f32
    {
        const ONE_MINUS_EPSILON: f32 = 1.0 - f32::EPSILON;
        let output = self.next();

        ONE_MINUS_EPSILON.min((output as f32) * f32::exp2(-32.))
    }


    pub fn next_in_range(&mut self, range: Range<u32>) -> u32
    {
        let span = range.end - range.start;
        let x: u32 = self.next();
        let mut l = (x as u64).wrapping_mul(span as u64) as u32;

        // This condition on range.end serve as an estimate of a threshold for values
        // that should be discard to debias the final modulo operation.
        if l < span {
            // Otherwise, we have to compute the exact threshold and perform
            // as many draw as necessary.
            let t = span.wrapping_neg() % span;
            while l < t {
                let x = self.next();
                l = (x as u64).wrapping_mul(span as u64) as u32;
            }
        }

        range.start + (l % span)
    }

    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        assert!(x.len() < u32::max_value() as usize);

        for lhs in 0..x.len() {
            let rhs = self.next_in_range(lhs as u32..(x.len() as u32));
            assert!(rhs < x.len() as u32);
            x.swap(lhs, rhs as usize);
        }
    }
}
