use crate::Sphere;
use crate::rand;
use maths::*;

#[derive(Debug)]
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

#[derive(Debug)]
pub struct StaticBvh {
    nodes: Vec<BvhNode>,
    root: Option<usize>,
}

impl StaticBvh {
    pub fn with_entities(entities: &[Sphere], rng: &mut rand::Rng) -> Self {
        let expected_node_count = ((entities.len() as f32).log2() as usize * entities.len()) / 2;

        let bounding_boxes: Vec<(usize, Aabb3)> = entities
            .into_iter()
            .enumerate()
            .map(|(idx, sphere)| (idx, crate::bounding_box(sphere)))
            .collect();

        let mut bvh = StaticBvh {
            nodes: Vec::with_capacity(expected_node_count),
            root: None,
        };

        let root = bvh.build_with_entities(&bounding_boxes[..], rng);
        bvh.root = root;
        bvh
    }

    fn build_with_entities(
        &mut self,
        bounding_boxes: &[(usize, Aabb3)],
        rng: &mut rand::Rng,
    ) -> Option<usize> {
        match bounding_boxes.len() {
            0 => None,
            1 => {
                let (entity, aabb) = bounding_boxes[0];
                let offset = self.nodes.len();

                self.nodes.push(BvhNode::Internal {
                    left: offset + 1,
                    right: offset + 2,
                    aabb,
                });
                self.nodes.push(BvhNode::Leaf { entity });
                self.nodes.push(BvhNode::Leaf { entity });

                Some(offset)
            }
            len => {
                let split_axis = (3. * rng.next_zero_one()).min(2.0) as usize;
                let mut splitted: Vec<_> = bounding_boxes.into_iter().copied().collect();
                splitted.sort_by(|(_, lhs), (_, rhs)| {
                    lhs.center()
                        .dim(split_axis)
                        .partial_cmp(&rhs.center().dim(split_axis))
                        .expect("invalid float value")
                });

                let (lhs, rhs) = splitted.split_at(len / 2);
                let left = self.build_with_entities(&lhs[..], rng);
                let right = self.build_with_entities(&rhs[..], rng);

                let mut aabb = aabb3::empty();
                for (_, bb) in &splitted[..] {
                    aabb = aabb.union(*bb);
                }
                self.nodes.push(BvhNode::Internal {
                    left: left.unwrap(),
                    right: right.unwrap(),
                    aabb,
                });

                Some(self.nodes.len() - 1)
            }
        }
    }

    pub fn intersect(
        &self,
        entities: &[Sphere],
        ray: &Ray,
        constraint: &RayConstraint,
    ) -> Option<crate::Intersection> {
        if let Some(root) = self.root {
            return self.intersect_node(entities, &self.nodes[root], ray, constraint);
        }

        None
    }

    fn intersect_node(
        &self,
        entities: &[Sphere],
        node: &BvhNode,
        ray: &Ray,
        constraints: &RayConstraint,
    ) -> Option<crate::Intersection> {
        use self::BvhNode::*;

        match node {
            Leaf { entity } => crate::intersect_sphere(&entities[*entity], ray, constraints)
                .map(|t| crate::Intersection { entity: *entity, t }),
            Internal { left, right, aabb } => {
                if !aabb.intersect(ray, constraints) {
                    return None;
                }

                let left_intersection =
                    self.intersect_node(entities, &self.nodes[*left], ray, constraints);
                let constraints = match left_intersection {
                    Some(intersection) => RayConstraint {
                        end: intersection.t,
                        ..*constraints
                    },
                    None => *constraints,
                };

                let right_intersection =
                    self.intersect_node(entities, &self.nodes[*right], ray, &constraints);
                right_intersection.or(left_intersection)
            }
        }
    }
}
