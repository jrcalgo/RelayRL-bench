//! LunarLander-v3 faithful Rust port.
//!
//! Physics bug fix: legs now start at correct positions (joint at rest),
//! and constraints use an impulse-based pin joint (sequential impulse method)
//! instead of a spring, which is stable for low-mass leg bodies.

pub mod vec;

use std::any::Any;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use glam::Vec2;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use relayrl_env_trait::{EnvironmentError, TrainingPerformanceReturnFn};
use relayrl_types::prelude::tensor::burn::backend::Backend;
use relayrl_types::prelude::tensor::burn::{Float, Tensor, TensorData};

// ── Physics constants (exact match to gymnasium) ────────────────────────────

const FPS: f32 = 50.0;
const DT: f32 = 1.0 / FPS;
const SCALE: f32 = 30.0;
const GRAVITY: f32 = -10.0;

const MAIN_ENGINE_POWER: f32 = 13.0;
const SIDE_ENGINE_POWER: f32 = 0.6;
const INITIAL_RANDOM: f32 = 1000.0;

const VIEWPORT_W: f32 = 600.0;
const VIEWPORT_H: f32 = 400.0;
const W: f32 = VIEWPORT_W / SCALE; // 20.0
const H: f32 = VIEWPORT_H / SCALE; // 13.333…

const LEG_AWAY: f32 = 20.0 / SCALE;
const LEG_DOWN: f32 = 18.0 / SCALE;
const LEG_W_HALF: f32 = (2.0 / SCALE) / 2.0;
const LEG_H_HALF: f32 = (8.0 / SCALE) / 2.0;
const LEG_SPRING_TORQUE: f32 = 40.0;

const SIDE_ENGINE_HEIGHT: f32 = 14.0 / SCALE;
const SIDE_ENGINE_AWAY_S: f32 = 12.0 / SCALE;
const MAIN_ENGINE_Y_LOCATION: f32 = 4.0 / SCALE;

/// Lander polygon vertices in local space (scaled).
const LANDER_POLY: [(f32, f32); 6] = [
    (-14.0 / SCALE, 17.0 / SCALE),
    (-17.0 / SCALE, 0.0),
    (-17.0 / SCALE, -10.0 / SCALE),
    (17.0 / SCALE, -10.0 / SCALE),
    (17.0 / SCALE, 0.0),
    (14.0 / SCALE, 17.0 / SCALE),
];

/// Leg box half-extents in local space.
const LEG_POLY: [(f32, f32); 4] = [
    (-LEG_W_HALF, -LEG_H_HALF),
    (LEG_W_HALF, -LEG_H_HALF),
    (LEG_W_HALF, LEG_H_HALF),
    (-LEG_W_HALF, LEG_H_HALF),
];

const CHUNKS: usize = 11;

/// Baumgarte position-error correction factor (fraction corrected per step).
const BAUMGARTE_BETA: f32 = 0.3;

/// Constraint-solver iterations per physics step (improves convergence).
const CONSTRAINT_ITERS: usize = 4;

// Ground response.
const GROUND_RESTITUTION: f32 = 0.0;
const GROUND_FRICTION: f32 = 0.1;

// Sleep detection.
const SLEEP_VEL_THRESHOLD: f32 = 0.1;
const SLEEP_ANG_THRESHOLD: f32 = 0.05;
const SLEEP_FRAMES: u32 = 20;

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn rotate(v: Vec2, angle: f32) -> Vec2 {
    let (s, c) = angle.sin_cos();
    Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
}

#[inline]
fn cross2(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

#[inline]
fn perp(v: Vec2) -> Vec2 {
    Vec2::new(-v.y, v.x)
}

// ── Terrain ──────────────────────────────────────────────────────────────────

struct Terrain {
    chunk_x: [f32; CHUNKS],
    smooth_y: [f32; CHUNKS],
    helipad_y: f32,
}

impl Terrain {
    fn generate(rng: &mut SmallRng) -> Self {
        let mut height = [0.0f32; CHUNKS + 1];
        for h in height.iter_mut() {
            *h = rng.random::<f32>() * H / 2.0;
        }
        let helipad_y = H / 4.0;
        let mid = CHUNKS / 2;
        for offset in [mid - 2, mid - 1, mid, mid + 1, mid + 2] {
            height[offset] = helipad_y;
        }

        let mut chunk_x = [0.0f32; CHUNKS];
        for i in 0..CHUNKS {
            chunk_x[i] = W / (CHUNKS as f32 - 1.0) * i as f32;
        }

        let mut smooth_y = [0.0f32; CHUNKS];
        for i in 0..CHUNKS {
            let h_prev = if i == 0 { height[CHUNKS] } else { height[i - 1] };
            smooth_y[i] = 0.33 * (h_prev + height[i] + height[i + 1]);
        }

        Terrain { chunk_x, smooth_y, helipad_y }
    }

    fn height_at(&self, x: f32) -> f32 {
        if x <= self.chunk_x[0] {
            return self.smooth_y[0];
        }
        if x >= self.chunk_x[CHUNKS - 1] {
            return self.smooth_y[CHUNKS - 1];
        }
        for i in 0..(CHUNKS - 1) {
            if x < self.chunk_x[i + 1] {
                let t = (x - self.chunk_x[i]) / (self.chunk_x[i + 1] - self.chunk_x[i]);
                return self.smooth_y[i] * (1.0 - t) + self.smooth_y[i + 1] * t;
            }
        }
        self.smooth_y[CHUNKS - 1]
    }
}

// ── Rigid body ───────────────────────────────────────────────────────────────

struct Body {
    pos: Vec2,
    vel: Vec2,
    angle: f32,
    angvel: f32,
    inv_mass: f32,
    inv_inertia: f32,
    force: Vec2,
    torque: f32,
    sleep_timer: u32,
    sleeping: bool,
}

impl Body {
    fn new(pos: Vec2, angle: f32, mass: f32, inertia: f32) -> Self {
        Body {
            pos,
            vel: Vec2::ZERO,
            angle,
            angvel: 0.0,
            inv_mass: 1.0 / mass,
            inv_inertia: 1.0 / inertia,
            force: Vec2::ZERO,
            torque: 0.0,
            sleep_timer: 0,
            sleeping: false,
        }
    }

    fn world_point(&self, local: Vec2) -> Vec2 {
        self.pos + rotate(local, self.angle)
    }

    fn velocity_at_world(&self, world_pt: Vec2) -> Vec2 {
        let r = world_pt - self.pos;
        self.vel + perp(r) * self.angvel
    }

    fn apply_impulse_at(&mut self, impulse: Vec2, world_pt: Vec2) {
        let r = world_pt - self.pos;
        self.vel += impulse * self.inv_mass;
        self.angvel += cross2(r, impulse) * self.inv_inertia;
    }

    /// Phase 1: update velocities from accumulated forces + gravity.
    fn integrate_velocity(&mut self, gravity: f32) {
        if self.sleeping {
            return;
        }
        self.vel += (self.force * self.inv_mass + Vec2::new(0.0, gravity)) * DT;
        self.angvel += self.torque * self.inv_inertia * DT;
        self.force = Vec2::ZERO;
        self.torque = 0.0;
    }

    /// Phase 2: update positions from current velocities (after constraints applied).
    fn integrate_position(&mut self) {
        if self.sleeping {
            return;
        }
        self.pos += self.vel * DT;
        self.angle += self.angvel * DT;
    }

    fn check_sleep(&mut self) {
        if self.vel.length() < SLEEP_VEL_THRESHOLD && self.angvel.abs() < SLEEP_ANG_THRESHOLD {
            self.sleep_timer += 1;
            if self.sleep_timer >= SLEEP_FRAMES {
                self.sleeping = true;
            }
        } else {
            self.sleep_timer = 0;
            self.sleeping = false;
        }
    }
}

/// Mass and rotational inertia for the lander polygon (density 5.0).
fn lander_mass_inertia() -> (f32, f32) {
    let density = 5.0f32;
    let verts: Vec<Vec2> = LANDER_POLY.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
    let n = verts.len();
    let mut area = 0.0f32;
    let mut inertia = 0.0f32;
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        let c = cross2(a, b);
        area += c;
        inertia += (a.dot(a) + a.dot(b) + b.dot(b)) * c;
    }
    area = area.abs() / 2.0;
    inertia = inertia.abs() / 12.0;
    (density * area, density * inertia)
}

/// Mass and rotational inertia for a leg box (density 1.0).
fn leg_mass_inertia() -> (f32, f32) {
    let density = 1.0f32;
    let w = LEG_W_HALF * 2.0;
    let h = LEG_H_HALF * 2.0;
    let mass = density * w * h;
    let inertia = density * w * h * (w * w + h * h) / 12.0;
    (mass, inertia)
}

// ── Physics world ─────────────────────────────────────────────────────────────

struct PhysicsState {
    terrain: Terrain,
    lander: Body,
    legs: [Body; 2],
    anchor_lander: [Vec2; 2],
    anchor_leg: [Vec2; 2],
    motor_speed: [f32; 2],
    angle_lower: [f32; 2],
    angle_upper: [f32; 2],
    leg_contact: [bool; 2],
    game_over: bool,
    prev_shaping: Option<f32>,
    last_reward: f32,
    done: bool,
    last_obs: [f32; 8],
    rng: SmallRng,
}

impl PhysicsState {
    fn build(seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let terrain = Terrain::generate(&mut rng);

        let initial_x = W / 2.0;
        let initial_y = H;

        let (lm, li) = lander_mass_inertia();
        let mut lander = Body::new(Vec2::new(initial_x, initial_y), 0.0, lm, li);

        // Initial random force for one timestep (matches Python's ApplyForceToCenter).
        let fx: f32 = rng.gen_range(-INITIAL_RANDOM..INITIAL_RANDOM);
        let fy: f32 = rng.gen_range(-INITIAL_RANDOM..INITIAL_RANDOM);
        lander.vel += Vec2::new(fx, fy) * lander.inv_mass * DT;

        // Joint configuration:
        //   Left  leg (idx=0, Python i=-1): anchor_leg=(-LEG_AWAY, LEG_DOWN), motor=-0.3, limits=[0.4, 0.9]
        //   Right leg (idx=1, Python i=+1): anchor_leg=(+LEG_AWAY, LEG_DOWN), motor=+0.3, limits=[-0.9,-0.4]
        let anchor_lander = [Vec2::ZERO, Vec2::ZERO];
        let anchor_leg = [
            Vec2::new(-LEG_AWAY, LEG_DOWN),
            Vec2::new(LEG_AWAY, LEG_DOWN),
        ];
        let motor_speed = [-0.3f32, 0.3f32];
        let angle_lower = [0.4f32, -0.9f32];
        let angle_upper = [0.9f32, -0.4f32];

        // Initial leg angles (Python: angle = i * 0.05 where i=-1 left, i=+1 right).
        let leg_angles = [-0.05f32, 0.05f32];

        let (legm, legi) = leg_mass_inertia();

        // CRITICAL: position each leg so that the revolute joint starts at rest.
        // Constraint: leg.world_point(anchor_leg) == lander.world_point(anchor_lander)
        // With anchor_lander=0 and lander.angle=0:  lander.pos = leg.pos + rotate(anchor_leg, leg_angle)
        // => leg.pos = lander.pos - rotate(anchor_leg, leg_angle)
        let legs = std::array::from_fn::<Body, 2, _>(|idx| {
            let leg_angle = leg_angles[idx];
            let leg_pos = lander.pos - rotate(anchor_leg[idx], leg_angle);
            Body::new(leg_pos, leg_angle, legm, legi)
        });

        let mut state = PhysicsState {
            terrain,
            lander,
            legs,
            anchor_lander,
            anchor_leg,
            motor_speed,
            angle_lower,
            angle_upper,
            leg_contact: [false; 2],
            game_over: false,
            prev_shaping: None,
            last_reward: 0.0,
            done: false,
            last_obs: [0.0; 8],
            rng,
        };

        // Run one settle step (mirrors Python reset() calling step(0)).
        state.step_physics(0);
        // Clear settle-step side-effects so episode starts fresh.
        state.prev_shaping = None;
        state.game_over = false;
        state.done = false;
        state
    }

    fn step_physics(&mut self, action: u8) {
        let angle = self.lander.angle;
        let tip = Vec2::new(angle.sin(), angle.cos());
        let side = Vec2::new(-tip.y, tip.x);

        let disp0: f32 = self.rng.gen_range(-1.0f32..1.0) / SCALE;
        let disp1: f32 = self.rng.gen_range(-1.0f32..1.0) / SCALE;

        let mut m_power = 0.0f32;
        let mut s_power = 0.0f32;

        // ── Engine impulses (applied directly to lander velocity) ─────────────
        if action == 2 {
            m_power = 1.0;
            let ox = tip.x * (MAIN_ENGINE_Y_LOCATION + 2.0 * disp0) + side.x * disp1;
            let oy = -tip.y * (MAIN_ENGINE_Y_LOCATION + 2.0 * disp0) - side.y * disp1;
            let imp_world = self.lander.pos + Vec2::new(ox, oy);
            let impulse = Vec2::new(-ox, -oy) * (MAIN_ENGINE_POWER * m_power);
            self.lander.apply_impulse_at(impulse, imp_world);
        }

        if action == 1 || action == 3 {
            s_power = 1.0;
            let direction = action as f32 - 2.0;
            let ox =
                tip.x * disp0 + side.x * (3.0 * disp1 + direction * SIDE_ENGINE_AWAY_S);
            let oy =
                -tip.y * disp0 - side.y * (3.0 * disp1 + direction * SIDE_ENGINE_AWAY_S);
            // 17/SCALE offset preserved from gymnasium for exact parity.
            let imp_world = self.lander.pos
                + Vec2::new(ox - tip.x * 17.0 / SCALE, oy + tip.y * SIDE_ENGINE_HEIGHT);
            let impulse = Vec2::new(-ox, -oy) * (SIDE_ENGINE_POWER * s_power);
            self.lander.apply_impulse_at(impulse, imp_world);
        }

        // ── Phase 1: integrate velocities (gravity) ───────────────────────────
        self.lander.integrate_velocity(GRAVITY);
        for leg in self.legs.iter_mut() {
            leg.integrate_velocity(GRAVITY);
        }

        // ── Phase 2: velocity-level constraint impulses ───────────────────────
        // Multiple iterations improve convergence for coupled constraints.
        for _ in 0..CONSTRAINT_ITERS {
            for i in 0..2 {
                self.apply_pin_joint(i);
                self.apply_motor_impulse(i);
                self.apply_angle_limit_impulse(i);
            }
        }

        // ── Phase 3: integrate positions ──────────────────────────────────────
        self.lander.integrate_position();
        for leg in self.legs.iter_mut() {
            leg.integrate_position();
        }

        // ── Phase 4: terrain collision ────────────────────────────────────────
        self.leg_contact = [false; 2];
        self.resolve_terrain_contacts();

        // ── Phase 5: sleep check ──────────────────────────────────────────────
        self.lander.check_sleep();

        // ── Phase 6: observation + reward ─────────────────────────────────────
        self.last_obs = self.compute_obs();
        let s = self.last_obs;

        let shaping = -100.0 * (s[0] * s[0] + s[1] * s[1]).sqrt()
            - 100.0 * (s[2] * s[2] + s[3] * s[3]).sqrt()
            - 100.0 * s[4].abs()
            + 10.0 * s[6]
            + 10.0 * s[7];

        let mut reward = match self.prev_shaping {
            Some(prev) => shaping - prev,
            None => 0.0,
        };
        self.prev_shaping = Some(shaping);

        reward -= m_power * 0.30;
        reward -= s_power * 0.03;

        if self.game_over || s[0].abs() >= 1.0 {
            self.done = true;
            reward = -100.0;
        }
        if self.lander.sleeping {
            self.done = true;
            reward = 100.0;
        }

        self.last_reward = reward;
    }

    /// Impulse-based revolute (pin) joint with Baumgarte position stabilization.
    ///
    /// Solves: v_A_at_anchor - v_B_at_anchor = 0 (velocity constraint)
    /// With bias to correct position drift each step.
    fn apply_pin_joint(&mut self, i: usize) {
        // World-space anchor positions.
        let wa = self.lander.world_point(self.anchor_lander[i]);
        let wb = self.legs[i].world_point(self.anchor_leg[i]);

        // Vectors from body centers to the anchor.
        let ra = wa - self.lander.pos;
        let rb = wb - self.legs[i].pos;

        // Relative velocity at constraint point (A - B).
        let va = self.lander.velocity_at_world(wa);
        let vb = self.legs[i].velocity_at_world(wb);
        let cdot = va - vb;

        // Baumgarte position-error bias (drives anchors together over time).
        let bias = (wa - wb) * (BAUMGARTE_BETA / DT);
        let rhs = -(cdot + bias);

        // 2×2 effective mass matrix K = J * M⁻¹ * Jᵀ
        let k00 = self.lander.inv_mass
            + self.legs[i].inv_mass
            + ra.y * ra.y * self.lander.inv_inertia
            + rb.y * rb.y * self.legs[i].inv_inertia;
        let k11 = self.lander.inv_mass
            + self.legs[i].inv_mass
            + ra.x * ra.x * self.lander.inv_inertia
            + rb.x * rb.x * self.legs[i].inv_inertia;
        let k01 = -(ra.x * ra.y * self.lander.inv_inertia
            + rb.x * rb.y * self.legs[i].inv_inertia);

        let det = k00 * k11 - k01 * k01;
        if det.abs() < 1e-12 {
            return;
        }

        // j = K⁻¹ * rhs
        let j = Vec2::new(
            (k11 * rhs.x - k01 * rhs.y) / det,
            (k00 * rhs.y - k01 * rhs.x) / det,
        );

        self.lander.vel += j * self.lander.inv_mass;
        self.lander.angvel += cross2(ra, j) * self.lander.inv_inertia;
        self.legs[i].vel -= j * self.legs[i].inv_mass;
        self.legs[i].angvel -= cross2(rb, j) * self.legs[i].inv_inertia;
    }

    /// Motor: drive relative angular velocity toward motor_speed, capped by max torque impulse.
    fn apply_motor_impulse(&mut self, i: usize) {
        let rel_angvel = self.legs[i].angvel - self.lander.angvel;
        let error = self.motor_speed[i] - rel_angvel;
        let inv_i_sum = self.lander.inv_inertia + self.legs[i].inv_inertia;
        let j = (error / inv_i_sum).clamp(-LEG_SPRING_TORQUE * DT, LEG_SPRING_TORQUE * DT);
        self.legs[i].angvel += j * self.legs[i].inv_inertia;
        self.lander.angvel -= j * self.lander.inv_inertia;
    }

    /// Angle limits: prevent relative angle going outside [lower, upper].
    fn apply_angle_limit_impulse(&mut self, i: usize) {
        let rel_angle = self.legs[i].angle - self.lander.angle;
        let rel_angvel = self.legs[i].angvel - self.lander.angvel;
        let inv_i_sum = self.lander.inv_inertia + self.legs[i].inv_inertia;
        let cap = LEG_SPRING_TORQUE * DT * 10.0;

        if rel_angle < self.angle_lower[i] && rel_angvel < 0.0 {
            let j = (-rel_angvel / inv_i_sum).min(cap);
            self.legs[i].angvel += j * self.legs[i].inv_inertia;
            self.lander.angvel -= j * self.lander.inv_inertia;
        } else if rel_angle > self.angle_upper[i] && rel_angvel > 0.0 {
            let j = (-rel_angvel / inv_i_sum).max(-cap);
            self.legs[i].angvel += j * self.legs[i].inv_inertia;
            self.lander.angvel -= j * self.lander.inv_inertia;
        }
    }

    fn resolve_terrain_contacts(&mut self) {
        // Leg contacts (do not trigger game_over).
        for i in 0..2 {
            let verts: Vec<Vec2> = LEG_POLY
                .iter()
                .map(|&(lx, ly)| self.legs[i].world_point(Vec2::new(lx, ly)))
                .collect();
            for v in &verts {
                let ty = self.terrain.height_at(v.x);
                if v.y < ty {
                    self.leg_contact[i] = true;
                    let depth = ty - v.y;
                    let normal = Vec2::Y;
                    let vel_at = self.legs[i].velocity_at_world(*v);
                    let vn = vel_at.dot(normal);
                    if vn < 0.0 {
                        let r = *v - self.legs[i].pos;
                        let inv_m = self.legs[i].inv_mass
                            + cross2(r, normal).powi(2) * self.legs[i].inv_inertia;
                        let j_n = (-(1.0 + GROUND_RESTITUTION) * vn / inv_m).max(0.0);
                        self.legs[i].apply_impulse_at(normal * j_n, *v);
                        let tangent = Vec2::X;
                        let vt = vel_at.dot(tangent);
                        let j_t = (-vt / inv_m)
                            .clamp(-GROUND_FRICTION * j_n, GROUND_FRICTION * j_n);
                        self.legs[i].apply_impulse_at(tangent * j_t, *v);
                    }
                    // Position correction: push leg out of ground.
                    self.legs[i].pos.y += depth * 0.5;
                }
            }
        }

        // Lander body contact → game_over.
        if !self.game_over {
            for &(lx, ly) in &LANDER_POLY {
                let v = self.lander.world_point(Vec2::new(lx, ly));
                if v.y < self.terrain.height_at(v.x) {
                    self.game_over = true;
                    break;
                }
            }
        }
    }

    fn compute_obs(&self) -> [f32; 8] {
        let pos = self.lander.pos;
        let vel = self.lander.vel;
        [
            (pos.x - W / 2.0) / (W / 2.0),
            (pos.y - (self.terrain.helipad_y + LEG_DOWN)) / (H / 2.0),
            vel.x * (W / 2.0) / FPS,
            vel.y * (H / 2.0) / FPS,
            self.lander.angle,
            20.0 * self.lander.angvel / FPS,
            if self.leg_contact[0] { 1.0 } else { 0.0 },
            if self.leg_contact[1] { 1.0 } else { 0.0 },
        ]
    }
}

// ── Public environment struct ─────────────────────────────────────────────────

pub struct LunarLanderEnv<B: Backend>
where
    B::Device: Clone,
{
    pub max_steps: usize,
    pub device: B::Device,
    state: RefCell<PhysicsState>,
    step_count: RefCell<usize>,
    running: AtomicBool,
}

impl<B: Backend> LunarLanderEnv<B>
where
    B::Device: Clone,
{
    pub fn new(max_steps: usize, device: B::Device) -> Self {
        let state = PhysicsState::build(12345);
        LunarLanderEnv {
            max_steps,
            device,
            state: RefCell::new(state),
            step_count: RefCell::new(0),
            running: AtomicBool::new(false),
        }
    }

    pub fn reset(&self) {
        let seed: u64 = self.state.borrow_mut().rng.random::<u64>();
        *self.state.borrow_mut() = PhysicsState::build(seed);
        *self.step_count.borrow_mut() = 0;
    }

    pub fn step(&self, _actor_idx: usize, action: u8) -> Result<(f32, bool), EnvironmentError> {
        *self.step_count.borrow_mut() += 1;
        self.state.borrow_mut().step_physics(action);
        let done = self.all_done() || self.is_max_steps_reached();
        let reward = self.state.borrow().last_reward;
        Ok((reward, done))
    }

    pub fn get_observation(&self, _actor_idx: usize) -> Vec<f32> {
        self.state.borrow().last_obs.to_vec()
    }

    pub fn get_last_reward(&self, _actor_idx: usize) -> f32 {
        self.state.borrow().last_reward
    }

    pub fn all_done(&self) -> bool {
        self.state.borrow().done
    }

    pub fn is_max_steps_reached(&self) -> bool {
        *self.step_count.borrow() >= self.max_steps
    }

    pub fn actor_count(&self) -> usize {
        1
    }
}

// ── EnvironmentTrait impls ────────────────────────────────────────────────────


impl<B: Backend> TrainingPerformanceReturnFn for LunarLanderEnv<B>
where
    B::Device: Clone,
{
    fn calculate_performance_return(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        let reward = self.state.borrow().last_reward;
        let tensor = Tensor::<B, 1, Float>::from_data(
            TensorData::new(vec![reward], [1usize]),
            &self.device,
        );
        Ok(Box::new(tensor))
    }
}
