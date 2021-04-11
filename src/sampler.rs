use maths::{Point2, Vec2};

use crate::rand::Rng;

pub struct StratifiedSampler {
    sample_count: u32,
    dimension_count: u32,
    rng: Rng,
    samples: Vec<f32>,
}

impl StratifiedSampler {
    pub fn new(sample_count: u32, dimensions: u32) -> Self {
        Self {
            sample_count,
            dimension_count: dimensions,
            rng: Rng::new(),
            samples: Vec::with_capacity((sample_count * dimensions) as usize),
        }
    }

    pub fn generate_sample_vector(&mut self) -> SampleVec<'_> {
        const ONE_MINUS_EPSILON: f32 = 1.0 - f32::EPSILON;
        self.samples.clear();

        let dimension_count = self.samples.capacity() / self.sample_count as usize;
        let delta = 1.0 / self.sample_count as f32;

        /* Generate the jittered samples */
        for _ in 0..dimension_count {
           for sample_idx in 0..self.sample_count {
                let jitter = self.rng.next_zero_one();
                let sample = ONE_MINUS_EPSILON.min((sample_idx as f32 + jitter) * delta);
                self.samples.push(sample);
            }
        }

        /* At this point samples have been generate as an interpolation + jittering, dimensions
           are too correlated, shuffle each dimension. */
        for dimension_idx in 0..dimension_count {
            let offset = dimension_idx*self.sample_count as usize;
            let samples = &mut self.samples[offset..(offset + self.sample_count as usize) as usize];
            self.rng.shuffle(samples);
        }

        SampleVec {
            dimension_count: self.dimension_count,
            sample_count: self.sample_count,
            samples: &self.samples,
            current_sample: 0,
            current_dimension: 0
        }
    }
}

pub struct SampleVec<'a> {
    dimension_count: u32,
    sample_count: u32,
    samples: &'a [f32],
    current_sample: u32,
    current_dimension: u32,
}


impl SampleVec<'_> {
    pub fn advance_to_next_sample(&mut self) {
        assert!(self.current_sample < self.sample_count, "{} < {}", self.current_sample, self.sample_count);

        self.current_dimension = 0;
        self.current_sample += 1;
    }

    pub fn next_zero_one(&mut self) -> f32 {
        assert!(self.current_dimension < self.dimension_count, "{} < {}", self.current_dimension, self.dimension_count);

        let index = self.current_dimension * self.sample_count + self.current_sample;
        let sample = self.samples[index as usize];
        self.current_dimension += 1;
        sample
    }
}

#[derive(Copy, Clone)]
pub struct MitchellFilter {
    inv_radius: Vec2,
}

impl MitchellFilter {
    pub fn new(radius: Vec2) -> Self {
        Self {
            inv_radius: 1. / radius,
        }
    }

    pub fn evalp(self, p: Point2) -> f32 {
        self.eval(p.x * self.inv_radius.x) * self.eval(p.y)
    }

    fn eval(self, x: f32) -> f32 {
        const B: f32 = 1. / 3.;
        const C: f32 = 1. / 3.;

        let x = 2.*x; /* Mitchell-Netravali function is an even over the range [-2, 2],
                         x is expected to be the range of [-1, 1] */

        let x = x.abs();
        if x > 1. {
            ((-B - 6.*C)       * x*x*x
            + (6.*B + 30.*C)   * x*x
            + (-12.*B - 48.*C) * x
            + (8.*B + 24.*C))  * (1./6.)
        } else {
            ((12. - 9.*B - 6.*C)     * x*x*x
             + (-18. + 12.*B + 6.*C) * x*x
             + (6. - 2.*B))          * (1./6.)
        }
    }
}
