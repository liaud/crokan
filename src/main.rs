mod frame;

use crate::frame::{Texture, Srgb, LinearRgb};
use maths::*;
use std::io;
use std::path::Path;

const ASPECT: f32 = 3. / 2.;
const MAX_DEPTH: u32 = 50;
const SPP: u32 = 50;

const FRAME_WIDHT: u32 = 200;
const FRAME_HEIGHT: u32 = (FRAME_WIDHT as f32 / ASPECT) as u32;

#[derive(Debug, Copy, Clone)]
pub enum Material {
    Lambertian { albedo: Vec3 },
    Metallic { albedo: Vec3, fuzz: f32 },
    Dialectric { ri: f32 },
}

pub struct Sphere {
    center: Point3,
    radius: f32,
    material: Material,
}

fn random_scene(rng: &mut oorandom::Rand32) -> Vec<Sphere> {
    let mut spheres = Vec::new();

    let ground = Sphere {
        center: p3(0., -1000.0, 0.),
        radius: 1000.,
        material: Material::Lambertian {
            albedo: v3(0.5, 0.5, 0.5),
        },
    };
    spheres.push(ground);

    let rand_color = |rng: &mut oorandom::Rand32| {
        v3(
            rand_min_max(rng, (0.)..1.),
            rand_min_max(rng, (0.)..1.),
            rand_min_max(rng, (0.)..1.),
        )
    };

    for a in -11..11 {
        for b in -11..11 {
            let material_choice = rng.rand_float();
            let center = v3(a as f32 + 0.9 * rng.rand_float(), 0.2, b as f32 + 0.9 * rng.rand_float());

            if (center - p3(4., 0.2, 0.)).length() > 0.9 {
                let material = if material_choice < 0.8 {
                    Material::Lambertian {
                        albedo: rand_color(rng),
                    }
                } else if material_choice < 0.95 {
                    Material::Metallic {
                        albedo: rand_color(rng),
                        fuzz: rand_min_max(rng, (0.)..0.5),
                    }
                } else {
                    Material::Dialectric { ri: 1.5 }
                };

                spheres.push(Sphere {
                    center,
                    material,
                    radius: 0.2,
                });
            }
        }
    }

    spheres.push(Sphere {
        center: p3(0., 1., 0.),
        radius: 1.0,
        material: Material::Dialectric {
            ri: 1.5
        }
    });
    spheres.push(Sphere {
        center: p3(-4., 1.0, 0.),
        radius: 1.0,
        material: Material::Lambertian { albedo: v3(0.4, 0.2, 0.1) },
    });
    spheres.push(Sphere {
        center: p3(4., 1., 0.),
        radius: 1.0,
        material: Material::Metallic {
            albedo: v3(0.7, 0.6, 0.5),
            fuzz: 0.0
        }
    });

    spheres
}

fn main() -> Result<(), io::Error> {
    let mut rng = oorandom::Rand32::new(188557);

    let spheres = random_scene(&mut rng);
    let mut render_target: Texture<LinearRgb> = Texture::new(FRAME_WIDHT, FRAME_HEIGHT);
    render(&mut rng, &spheres[..], &mut render_target);

    let mut frame: Texture<Srgb> = Texture::new(FRAME_WIDHT, FRAME_HEIGHT);
    render_target.copy_to(&mut frame, |p| {
        let color = v3(p.r.sqrt(), p.g.sqrt(), p.b.sqrt());
        let rgb = v3(255., 255., 255.) * color.saturate();

        Srgb {
            r: rgb.x as u8,
            g: rgb.y as u8,
            b: rgb.z as u8,
        }
    });
    frame::save_as_ppm(Path::new("render.ppm"), &frame)
}

fn render(rng: &mut oorandom::Rand32, spheres: &[Sphere], render_target: &mut Texture<LinearRgb>) {
    let lookat_from = p3(13., 2., -3.);
    let lookat_to = p3(0., 0., 0.);

    let aperture = 0.1;
    let focus_dist = 10.;
    let vup = v3::up();

    let w = (lookat_to - lookat_from).unit();
    let u = vup.cross(w).unit();
    let v = w.cross(u);
    dbg!((u, v, w));

    let vfov = 20.0_f32.to_radians();
    let viewport_height = 2. * (vfov / 2.0).tan();

    let viewport_u = focus_dist * ASPECT * viewport_height * u;
    let viewport_v = focus_dist * viewport_height * v;

    let origin = lookat_from;
    let lower_left = origin - viewport_u / 2. - viewport_v / 2. + focus_dist * w;
    let lens_radius = aperture / 2.0;

    for y in 0..render_target.height {
        for x in 0..render_target.width {
            let weight = 1. / SPP as f32;
            let mut color = v3(0., 0., 0.);
            for _ in 0..SPP {
                let s = (x as f32 + rng.rand_float()) / (render_target.width - 1) as f32;
                let t = ((render_target.height - y) as f32 - rng.rand_float()) as f32
                    / (render_target.height - 1) as f32;

                let rd = lens_radius * random_in_unit_disk(rng);
                let offset = u * rd.x + v * rd.y;
                let ray = ray(
                    origin + offset,
                    (lower_left + viewport_u * s + viewport_v * t) - origin - offset,
                );

                color = color + weight * trace(rng, &spheres[..], &ray, MAX_DEPTH);
            }


            *render_target.pixel_mut(x, y) = LinearRgb {
                r: color.x,
                g: color.y,
                b: color.z,
            };
        }

        let per = y as f32 / render_target.height as f32 * 100.;
        println!("{}%", per);
    }
}

fn trace(rng: &mut oorandom::Rand32, spheres: &[Sphere], trace_ray: &Ray, depth: u32) -> Vec3 {
    if depth == 0 {
        return v3(0., 0., 0.);
    }

    if let Some(Intersection { t, entity }) =
        intersect_any(&spheres[..], &trace_ray, RayConstraint::none())
    {
        let sphere = &spheres[entity];
        let surface = surface_at(sphere, &trace_ray, t);

        match sphere.material {
            Material::Lambertian { albedo } => {
                let target = surface.p + surface.n + random_unit_vector(rng);
                let diffuse_ray = ray(surface.p + 0.001 * surface.n, target - surface.p);
                return albedo * trace(rng, spheres, &diffuse_ray, depth - 1);
            }
            Material::Metallic { albedo, fuzz } => {
                let reflected = trace_ray.d.reflect(surface.n);
                let scattered_ray = ray(surface.p + 0.001 * surface.n, reflected + fuzz * random_unit_vector(rng)) ;
                return albedo * trace(rng, spheres, &scattered_ray, depth - 1);
            }
            Material::Dialectric { ri } => {
                let etai_over_etat = if surface.face == Face::Front {
                    1.0 / ri
                } else {
                    ri
                };

                let ray_dir_unit = trace_ray.d.unit();

                let cos_theta = (-ray_dir_unit).dot(surface.n).min(1.0);
                assert!(cos_theta >= 0.);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                if etai_over_etat * sin_theta > 1.0 {
                    let reflected = ray_dir_unit.reflect(surface.n);
                    let reflected_ray = ray(surface.p + 0.001 * surface.n, reflected);

                    return trace(rng, spheres, &reflected_ray, depth - 1);
                }

                let reflect_probability = maths::schlick(cos_theta, etai_over_etat);
                if rng.rand_float() < reflect_probability {
                    let reflected = ray_dir_unit.reflect(surface.n);
                    let reflected_ray = ray(surface.p + 0.001 * surface.n, reflected);
                    return trace(rng, spheres, &reflected_ray, depth - 1);
                }

                let refracted_ray = ray(
                    surface.p - 0.001 * surface.n,
                    ray_dir_unit.refract(surface.n, etai_over_etat),
                );
                return trace(rng, spheres, &refracted_ray, depth - 1);
            }
        }
    }

    let end_bg_color = v3(255., 255., 255.) / 255.;
    let start_bg_color = v3(0.5, 0.7, 1.0);
    start_bg_color.lerp(end_bg_color, trace_ray.d.unit().y)
}

#[derive(Debug, Copy, Clone)]
pub struct RayConstraint {
    start: f32,
    end: f32,
}

impl RayConstraint {
    pub fn none() -> Self {
        Self {
            start: 0.,
            end: f32::INFINITY,
        }
    }
}
#[derive(Debug, Copy, Clone)]
pub struct Intersection {
    entity: usize,
    t: f32,
}

fn intersect_any(spheres: &[Sphere], ray: &Ray, constraint: RayConstraint) -> Option<Intersection> {
    let mut intersection: Option<Intersection> = None;
    for (i, sphere) in spheres.iter().enumerate() {
        let found = match intersect(sphere, ray) {
            Some(found) => found,
            None => continue,
        };

        let min_valid = if found.0 >= constraint.start && found.0 < constraint.end {
            found.0
        } else if found.1 >= constraint.start && found.1 < constraint.end {
            found.1
        } else {
            continue;
        };

        let candidate = Intersection {
            t: min_valid,
            entity: i,
        };

        match intersection {
            Some(current) if current.t > candidate.t => intersection = Some(candidate),
            None => intersection = Some(candidate),
            _ => {}
        }
    }

    intersection
}

fn intersect(sphere: &Sphere, ray: &Ray) -> Option<(f32, f32)> {
    let center_to_ray = ray.o - sphere.center;

    let a = ray.d.dot(ray.d);
    let b = 2.0 * ray.d.dot(center_to_ray);
    let c = center_to_ray.dot(center_to_ray) - sphere.radius * sphere.radius;

    maths::quadratic(a, b, c)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Face {
    Front,
    Back,
}

pub struct Surface {
    p: Point3,
    n: Vec3,
    face: Face,
}

fn surface_at(sphere: &Sphere, ray: &Ray, t: f32) -> Surface {
    let p = ray.at(t);
    let n = (p - sphere.center) / sphere.radius;

    let (n, face) = if ray.d.dot(n) < 0. {
        (n, Face::Front)
    } else {
        (-n, Face::Back)
    };

    Surface { p, face, n }
}

fn random_unit_vector(state: &mut oorandom::Rand32) -> Vec3 {
    let a = rand_min_max(state, (0.)..2. * std::f32::consts::PI);
    let z = rand_min_max(state, (-1.)..1.);
    let r = (1. - z * z).sqrt();

    let (sin, cos) = a.sin_cos();
    v3(r * cos, r * sin, z)
}

fn rand_min_max(state: &mut oorandom::Rand32, range: std::ops::Range<f32>) -> f32 {
    range.start + (range.end - range.start) * state.rand_float()
}

fn random_in_unit_disk(state: &mut oorandom::Rand32) -> Vec3 {
    loop {
        let p = v3(
            rand_min_max(state, (-1.)..1.),
            rand_min_max(state, (-1.)..1.),
            0.,
        );
        if p.dot(p) <= 1. {
            break p;
        }
    }
}
