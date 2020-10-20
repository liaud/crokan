use crate::Sphere;
use maths::*;

enum BvhNode {
    Leaf {
        entity: usize,
    },
    Internal {
        left: usize,
        right: usize,
        aabb: Aabb3,
    },
}

pub struct StaticBvh {
    nodes: Vec<BvhNode>,
}

impl StaticBvh {
    pub fn intersect(
        &self,
        entities: &[Sphere],
        ray: &Ray,
        constraint: &RayConstraint,
    ) -> Option<f32> {
        if self.nodes.is_empty() {
            return None;
        }
        self.intersect_node(&self.nodes[0], ray, constraint)
    }

    fn intersect_node(
        &self,
        entities: &[Sphere],
        node: &BvhNode,
        ray: &Ray,
        constraints: &RayConstraint,
    ) -> Option<f32> {
        match node {
            Leaf { entity } => {
                crate::intersect_sphere(&entities[entity], ray, constraints)
            }
            Internal { left, right, aabb } => {
                let left_intersection = self.intersect_node(&self.nodes[node.left], ray, constraints);
                let constraints = match left_intersection {
                    Some(t) => RayConstraint {
                        max: t,
                        ..constraints
                    },
                    None => *constraints,
                };

                let right_intersection = self.intersect_node(&self.nodes[node.right], ray, &constraints);
                right_intersection.or(left_intersection)
            }
        }
    }
}
