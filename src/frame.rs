use std::io::Error;
use std::path::Path;

#[derive(Debug, Copy, Clone)]
pub struct Srgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

pub struct FrameBuffer {
    pub pixels: Vec<Srgb>,
    pub width: u32,
    pub height: u32,
}

impl FrameBuffer {
    pub fn new(width: u32, height: u32) -> FrameBuffer {
        let mut pixels = Vec::new();
        pixels.resize((width * height) as usize, Srgb { r: 0, g: 0, b: 0 });

        FrameBuffer {
            pixels,
            width,
            height,
        }
    }

    pub fn pixel_mut(&mut self, x: u32, y: u32) -> &mut Srgb {
        let index = (y * self.width + x) as usize;
        &mut self.pixels[index]
    }

    pub fn pixel(&self, x: u32, y: u32) -> Srgb {
        let index = (y * self.width + x) as usize;
        self.pixels[index]
    }
}

pub fn save_as_ppm(output: &Path, frame: &FrameBuffer) -> Result<(), Error> {
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
