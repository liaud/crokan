use std::io::Error;
use std::path::Path;

#[derive(Debug, Copy, Clone, Default)]
pub struct LinearRgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Srgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug)]
pub struct Texture<P: Copy> {
    pixels: Vec<P>,
    pub width: u32,
    pub height: u32,
}

impl<P: Copy + Default> Texture<P> {
    pub fn new(width: u32, height: u32) -> Self {
        let mut pixels = Vec::new();
        pixels.resize((width * height) as usize, P::default());

        Self {
            pixels,
            width,
            height,
        }
    }

    pub fn pixel_mut(&mut self, x: u32, y: u32) -> &mut P {
        let index = (y * self.width + x) as usize;
        &mut self.pixels[index]
    }

    pub fn pixel(&self, x: u32, y: u32) -> P {
        let index = (y * self.width + x) as usize;
        self.pixels[index]
    }

    pub fn copy_to<DestP: Copy, M>(&self, dest: &mut Texture<DestP>, map: M)
        where M: Fn(P) -> DestP
    {
        assert_eq!(self.width, dest.width);
        assert_eq!(self.height, dest.height);

        for (dst, src) in dest.pixels.iter_mut().zip(self.pixels.iter()) {
            *dst = map(*src);
        }
    }
}

pub fn save_as_ppm(output: &Path, frame: &Texture<Srgb>) -> Result<(), Error> {
    use std::fs::File;
    use std::io::prelude::*;
    use std::io::BufWriter;

    let mut output = BufWriter::new(File::create(output)?);

    writeln!(output, "P3")?;
    writeln!(output, "{} {}", frame.width, frame.height)?;
    writeln!(output, "255")?;

    for y in 0..frame.height {
        for x in 0..frame.width {
            let p = frame.pixel(x, y);
            write!(output, "{} {} {}\n", p.r, p.g, p.b)?;

            if x != frame.height - 1 {
                write!(output, " ")?;
            }
        }
    }

    output.flush()?;

    Ok(())
}
