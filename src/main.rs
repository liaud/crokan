mod bvh;
mod frame;
mod rand;
mod sampler;

use crate::frame::{LinearRgb, Srgb, Texture};
use crate::sampler::{SampleVec, StratifiedSampler};
use maths::*;
use sampler::MitchellFilter;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

const ASPECT: f32 = 16. / 9.;
const MAX_DEPTH: u32 = 50;
const SPP: u32 = 200;
const THREAD_COUNT: u32 = 16;
const THREAD_SPP: u32 = SPP / THREAD_COUNT;

const FRAME_WIDHT: u32 = 1920;
const FRAME_HEIGHT: u32 = (FRAME_WIDHT as f32 / ASPECT) as u32;

#[derive(Debug, Copy, Clone)]
pub enum Material {
    Lambertian { albedo: Vec3 },
    Metallic { albedo: Vec3, fuzz: f32 },
    Dialectric { ri: f32 },
    EmissiveDiffuse { intensity: Vec3 },
}

pub struct Sphere {
    center: Point3,
    radius: f32,
    material: Material,
}

fn random_scene(rng: &mut rand::Rng) -> Vec<Sphere> {
    let mut spheres = Vec::new();

    let ground = Sphere {
        center: p3(0., -1000.0, 0.),
        radius: 1000.,
        material: Material::Lambertian {
            albedo: v3(0.5, 0.5, 0.5),
        },
    };
    spheres.push(ground);

    let rand_color = |rng: &mut rand::Rng| {
        v3(
            rng.next_zero_one(),
            rng.next_zero_one(),
            rng.next_zero_one(),
        )
    };

    for a in -11..11 {
        for b in -11..11 {
            let material_choice = rng.next_zero_one();
            let center = v3(
                a as f32 + 0.9 * rng.next_zero_one(),
                0.2,
                b as f32 + 0.9 * rng.next_zero_one(),
            );

            if (center - p3(4., 0.2, 0.)).length() > 0.9 {
                let material = if material_choice < 0.8 {
                    Material::Lambertian {
                        albedo: rand_color(rng),
                    }
                } else if material_choice < 0.95 {
                    Material::Metallic {
                        albedo: rand_color(rng),
                        fuzz: 0.5 * rng.next_zero_one(),
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
        material: Material::Dialectric { ri: 1.5 },
    });
    spheres.push(Sphere {
        center: p3(-4., 1.0, 0.),
        radius: 1.0,
        material: Material::Lambertian {
            albedo: v3(0.4, 0.2, 0.1),
        },
    });
    spheres.push(Sphere {
        center: p3(4., 1., 0.),
        radius: 1.0,
        material: Material::Metallic {
            albedo: v3(0.7, 0.6, 0.5),
            fuzz: 0.0,
        },
    });

    spheres
}

fn debug_scene(_rng: &mut rand::Rng) -> Vec<Sphere> {
    vec![
        Sphere {
            center: p3(0., -1000.0, 0.),
            radius: 1000.,
            material: Material::Lambertian {
                albedo: v3(0.5, 0.5, 0.5),
            },
        },
        Sphere {
            center: p3(-2.5, 1., 0.),
            radius: 1.0,
            material: Material::Lambertian {
                albedo: v3(0.4, 0.2, 0.1),
            },
        },
        Sphere {
            center: p3(-1., 1.5, 1.5),
            radius: 0.7,
            material: Material::EmissiveDiffuse {
                intensity: v3(4.0, 3.8, 4.0),
            },
        },
        Sphere {
            center: p3(2.5, 1., 0.),
            radius: 1.0,
            material: Material::Metallic {
                albedo: v3(0.7, 0.6, 0.5),
                fuzz: 0.4,
            },
        },
    ]
}

#[derive(Copy, Clone, Default)]
pub struct Pixel {
    radiance_sum: LinearRgb,
    filter_weight_sum: f32,
}

fn main() -> Result<(), io::Error> {
    let mut scene_rng = rand::Rng::new();

    let spheres = Arc::new(random_scene(&mut scene_rng));
    //let spheres = Arc::new(debug_scene(&mut scene_rng));
    let bvh = Arc::new(bvh::StaticBvh::with_entities(&spheres[..], &mut scene_rng));
    dbg!(&bvh);

    let thread_targets = Arc::new(Mutex::new(Vec::with_capacity(THREAD_COUNT as usize)));
    let mut render_threads = Vec::with_capacity(THREAD_COUNT as usize);

    for thread_idx in 0..THREAD_COUNT {
        let spheres = spheres.clone();
        let bvh = bvh.clone();
        let thread_targets = thread_targets.clone();

        let mut render_target: Texture<Pixel> = Texture::new(FRAME_WIDHT, FRAME_HEIGHT);

        render_threads.push(thread::spawn(move || {
            render(thread_idx, &bvh, &spheres[..], &mut render_target);

            let mut targets = thread_targets.lock().unwrap();
            targets.push(render_target);
        }));
    }

    for render_thread in render_threads {
        render_thread
            .join()
            .expect("render thread failed unexpectedly");
    }

    let mut merged_targets: Texture<Pixel> = Texture::new(FRAME_WIDHT, FRAME_HEIGHT);
    let mut targets = thread_targets.lock().unwrap();

    for target in targets.drain(..) {
        for y in 0..target.height {
            for x in 0..target.width {
                let lhs = merged_targets.pixel(x, y);
                let rhs = target.pixel(x, y);

                let mut merged_pixel = merged_targets.pixel_mut(x, y);

                merged_pixel.radiance_sum = LinearRgb {
                    r: lhs.radiance_sum.r + rhs.radiance_sum.r,
                    g: lhs.radiance_sum.g + rhs.radiance_sum.g,
                    b: lhs.radiance_sum.b + rhs.radiance_sum.b,
                };
                merged_pixel.filter_weight_sum = lhs.filter_weight_sum + rhs.filter_weight_sum;
            }
        }
    }

    let mut frame: Texture<Srgb> = Texture::new(FRAME_WIDHT, FRAME_HEIGHT);
    merged_targets.copy_to(&mut frame, |p| {
        let color = LinearRgb {
            r: p.radiance_sum.r / p.filter_weight_sum,
            g: p.radiance_sum.g / p.filter_weight_sum,
            b: p.radiance_sum.b / p.filter_weight_sum,
        };

        let color = v3(color.r.sqrt(), color.g.sqrt(), color.b.sqrt());
        let rgb = v3(255., 255., 255.) * color.saturate();

        Srgb {
            r: rgb.x as u8,
            g: rgb.y as u8,
            b: rgb.z as u8,
        }
    });
    frame::save_as_ppm(Path::new("render.ppm"), &frame)
}

fn render(
    thread_idx: u32,
    bvh: &bvh::StaticBvh,
    spheres: &[Sphere],
    render_target: &mut Texture<Pixel>,
) {
    let filter_radius = v2(2., 2.);
    let filter = MitchellFilter::new(filter_radius);
    let mut sampler = StratifiedSampler::new(THREAD_SPP, MAX_DEPTH * 15);

    let lookat_from = p3(0., 4., -20.);
    let lookat_to = p3(0., 1., 0.);

    let aperture = 0.1;
    let focus_dist = (lookat_from - lookat_to).length();
    let vup = v3::up();

    let w = (lookat_to - lookat_from).unit();
    let u = vup.cross(w).unit();
    let v = w.cross(u);

    let vfov = 20.0_f32.to_radians();
    let viewport_height = 2. * (vfov / 2.0).tan();

    let viewport_u = focus_dist * ASPECT * viewport_height * u;
    let viewport_v = focus_dist * viewport_height * v;

    let origin = lookat_from;
    let lower_left = origin - viewport_u / 2. - viewport_v / 2. + focus_dist * w;
    let lens_radius = aperture / 2.0;

    let min_sampled_pixel = (
        (0.5 - filter_radius.x).floor() as i32,
        (0.5 - filter_radius.y).floor() as i32,
    );
    let max_sampled_pixel = (
        (render_target.width as f32 + 0.5 + filter_radius.x).ceil() as i32,
        (render_target.height as f32 + filter_radius.y + 0.5).ceil() as i32,
    );
    dbg!((min_sampled_pixel, max_sampled_pixel));

    let y_span = min_sampled_pixel.1 + max_sampled_pixel.1;
    dbg!(y_span);
    for y in min_sampled_pixel.1..max_sampled_pixel.1 {
        for x in min_sampled_pixel.0..max_sampled_pixel.0 {
            let mut samples = sampler.generate_sample_vector();

            for _ in 0..THREAD_SPP {
                let offset_in_sampled_pixel = v2(samples.next_zero_one(), -samples.next_zero_one());

                let pixel_sample =
                    v2(x as f32, y_span as f32 - y as f32) + offset_in_sampled_pixel;

                let pixel_sample_in_viewport = pixel_sample
                    / v2(
                        (render_target.width - 1) as f32,
                        (render_target.height - 1) as f32,
                    );

                let rd = lens_radius * random_in_unit_disk(&mut samples);
                let offset = u * rd.x + v * rd.y;

                let ray = ray(
                    origin + offset,
                    (lower_left
                        + viewport_u * pixel_sample_in_viewport.x
                        + viewport_v * pixel_sample_in_viewport.y)
                        - origin
                        - offset,
                );

                let color = trace(&mut samples, bvh, &spheres[..], &ray, MAX_DEPTH);

                let min_contributed_pixel =
                    (v2(pixel_sample.x, y_span as f32 - pixel_sample.y) - filter_radius)
                        .ceil()
                        .max(v2::ZERO);

                let max_contributed_pixel = v2(
                    pixel_sample.x + filter_radius.x,
                    y_span as f32 - pixel_sample.y + filter_radius.y,
                )
                .floor()
                .min(v2(render_target.width as f32, render_target.height as f32));

                for contributed_x in
                    (min_contributed_pixel.x as u32)..(max_contributed_pixel.x as u32)
                {
                    for contributed_y in
                        (min_contributed_pixel.y as u32)..(max_contributed_pixel.y as u32)
                    {
                        let pixel_contributed =
                            v2(contributed_x as f32, contributed_y as f32) + v2(0.5, 0.5);
                        let pix_sample_relative_to_contrib = v2(
                            pixel_sample.x - pixel_contributed.x,
                            (y_span as f32 - pixel_sample.y) - pixel_contributed.y);

                        let filter_weight = filter.evalp(pix_sample_relative_to_contrib);

                        let mut pixel = render_target.pixel_mut(contributed_x, contributed_y);
                        pixel.filter_weight_sum += filter_weight;
                        pixel.radiance_sum = LinearRgb {
                            r: pixel.radiance_sum.r + filter_weight * color.x,
                            g: pixel.radiance_sum.g + filter_weight * color.y,
                            b: pixel.radiance_sum.b + filter_weight * color.z,
                        };
                    }
                }

                samples.advance_to_next_sample();
            }
        }

        let per = y as f32 / render_target.height as f32 * 100.;
        println!("thread {}: {}%", thread_idx, per);
    }
}

struct MaterialSamples {
    reflect_unit_dir: Vec3,
    dialectric_reflect_choice: f32,
}

impl MaterialSamples {
    fn with_samples(samples: &mut SampleVec<'_>) -> Self {
        Self {
            reflect_unit_dir: random_unit_vector(samples),
            dialectric_reflect_choice: samples.next_zero_one(),
        }
    }
}

fn trace(
    samples: &mut SampleVec<'_>,
    bvh: &bvh::StaticBvh,
    spheres: &[Sphere],
    trace_ray: &Ray,
    depth: u32,
) -> Vec3 {
    if depth == 0 {
        return v3(0., 0., 0.);
    }

    let material_samples = MaterialSamples::with_samples(samples);

    if let Some(Intersection { t, entity }) =
        bvh.intersect(&spheres[..], &trace_ray, &RayConstraint::none())
    {
        let sphere = &spheres[entity];
        let surface = surface_at(sphere, &trace_ray, t);

        match sphere.material {
            Material::Lambertian { albedo } => {
                let target = surface.p + surface.n + material_samples.reflect_unit_dir;
                let diffuse_ray = ray(surface.p + 0.001 * surface.n, target - surface.p);
                return albedo * trace(samples, bvh, spheres, &diffuse_ray, depth - 1);
            }
            Material::Metallic { albedo, fuzz } => {
                let reflected = trace_ray.d.reflect(surface.n);
                let scattered_ray = ray(
                    surface.p + 0.001 * surface.n,
                    reflected + fuzz * material_samples.reflect_unit_dir,
                );
                return albedo * trace(samples, bvh, spheres, &scattered_ray, depth - 1);
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

                    return trace(samples, bvh, spheres, &reflected_ray, depth - 1);
                }

                let reflect_probability = maths::schlick(cos_theta, etai_over_etat);
                if material_samples.dialectric_reflect_choice < reflect_probability {
                    let reflected = ray_dir_unit.reflect(surface.n);
                    let reflected_ray = ray(surface.p + 0.001 * surface.n, reflected);
                    return trace(samples, bvh, spheres, &reflected_ray, depth - 1);
                }

                let refracted_ray = ray(
                    surface.p - 0.001 * surface.n,
                    ray_dir_unit.refract(surface.n, etai_over_etat),
                );
                return trace(samples, bvh, spheres, &refracted_ray, depth - 1);
            }
            Material::EmissiveDiffuse { intensity } => {
                return intensity;
            }
        }
    }

    let end_bg_color = v3(255., 255., 255.) / 255.;
    let start_bg_color = v3(0.5, 0.7, 1.0);
    start_bg_color.lerp(end_bg_color, trace_ray.d.unit().y)
}

#[derive(Debug, Copy, Clone)]
pub struct Intersection {
    entity: usize,
    t: f32,
}

pub(crate) fn intersect_sphere(
    sphere: &Sphere,
    ray: &Ray,
    constraints: &RayConstraint,
) -> Option<f32> {
    let center_to_ray = ray.o - sphere.center;

    let a = ray.d.dot(ray.d);
    let b = 2.0 * ray.d.dot(center_to_ray);
    let c = center_to_ray.dot(center_to_ray) - sphere.radius * sphere.radius;

    match maths::quadratic(a, b, c) {
        Some((t0, _)) if t0 >= constraints.start && t0 < constraints.end => Some(t0),
        Some((_, t1)) if t1 >= constraints.start && t1 < constraints.end => Some(t1),
        _ => None,
    }
}

fn bounding_box(sphere: &Sphere) -> Aabb3 {
    let offset = v3(sphere.radius, sphere.radius, sphere.radius);

    Aabb3 {
        min: sphere.center - offset,
        max: sphere.center + offset,
    }
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

fn random_unit_vector(samples: &mut SampleVec<'_>) -> Vec3 {
    let a = rand_min_max(samples, (0.)..2. * std::f32::consts::PI);
    let z = rand_min_max(samples, (-1.)..1.);
    let r = (1. - z * z).sqrt();

    let (sin, cos) = a.sin_cos();
    v3(r * cos, r * sin, z)
}

fn rand_min_max(samples: &mut SampleVec<'_>, range: std::ops::Range<f32>) -> f32 {
    range.start + (range.end - range.start) * samples.next_zero_one()
}

fn random_in_unit_disk(samples: &mut SampleVec<'_>) -> Vec3 {
    loop {
        let p = v3(
            rand_min_max(samples, (-1.)..1.),
            rand_min_max(samples, (-1.)..1.),
            0.,
        );
        if p.dot(p) <= 1. {
            break p;
        }
    }
}
