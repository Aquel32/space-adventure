import { d, std } from "typegpu";
import type { v3f } from "typegpu/data";

export let GRAVITY_MULTIPLIER = 0.04;
export function SetGravityMultiplier(newG: number) {
  GRAVITY_MULTIPLIER = newG;
}

function calculateStableOrbitVelocity(distance: number, mass: number) {
  return std.sqrt((GRAVITY_MULTIPLIER * mass) / distance);
}

const SUN_MASS = 1.0;

const MERCURY_MASS = 1.66e-7;
const MERCURY_INITIAL_VELOCITY = d.vec3f(0, 0, 0.0321);

const VENUS_MASS = 2.45e-6;
const VENUS_INITIAL_VELOCITY = d.vec3f(0, 0, 0.0235);

const EARTH_MASS = 3.003e-6;
const EARTH_DISTANCE = 100;
const EARTH_ORBIT_VELOCITY = calculateStableOrbitVelocity(EARTH_DISTANCE, SUN_MASS);
const EARTH_INITIAL_VELOCITY = d.vec3f(0, 0, EARTH_ORBIT_VELOCITY);

const MOON_MASS = 3.69e-8;
const MOON_BASE_ORBIT_RADIUS = EARTH_DISTANCE * (384400 / 149597870);
// Visual exaggeration so the moon does not appear glued to Earth at current render scale.
// Keep this below ~3.9 to stay inside Earth's Hill sphere in this setup.
const MOON_ORBIT_DISTANCE_SCALE = 3;
const MOON_ORBIT_RADIUS = MOON_BASE_ORBIT_RADIUS * MOON_ORBIT_DISTANCE_SCALE;
const MOON_ORBIT_VELOCITY = calculateStableOrbitVelocity(MOON_ORBIT_RADIUS, EARTH_MASS);
const MOON_INITIAL_VELOCITY = d.vec3f(0, 0, EARTH_ORBIT_VELOCITY + MOON_ORBIT_VELOCITY);

const MARS_MASS = 3.23e-7;
const MARS_INITIAL_VELOCITY = d.vec3f(0, 0, 0.0162);

const JUPITER_MASS = 9.54e-4;
const JUPITER_INITIAL_VELOCITY = d.vec3f(0, 0, 0.00877);

const SATURN_MASS = 2.86e-4;
const SATURN_INITIAL_VELOCITY = d.vec3f(0, 0, 0.00646);

const URANUS_MASS = 4.37e-5;
const URANUS_INITIAL_VELOCITY = d.vec3f(0, 0, 0.00457);

const NEPTUNE_MASS = 5.15e-5;
const NEPTUNE_INITIAL_VELOCITY = d.vec3f(0, 0, 0.00365);

const EARTH_RENDER_RADIUS = 0.15;

function scaleRadiusFromEarth(realRadiusKm: number) {
  return (realRadiusKm / 6371) * EARTH_RENDER_RADIUS;
}

const SUN_RENDER_RADIUS = scaleRadiusFromEarth(696340);
const MERCURY_RENDER_RADIUS = scaleRadiusFromEarth(2439.7);
const VENUS_RENDER_RADIUS = scaleRadiusFromEarth(6051.8);
const EARTH_BODY_RENDER_RADIUS = scaleRadiusFromEarth(6371);
const MOON_RENDER_RADIUS = scaleRadiusFromEarth(1737.4);
const MARS_RENDER_RADIUS = scaleRadiusFromEarth(3389.5);
const JUPITER_RENDER_RADIUS = scaleRadiusFromEarth(69911);
const SATURN_RENDER_RADIUS = scaleRadiusFromEarth(58232);
const URANUS_RENDER_RADIUS = scaleRadiusFromEarth(25362);
const NEPTUNE_RENDER_RADIUS = scaleRadiusFromEarth(24622);

function calculateMomentumBalancedSunVelocity(
  otherBodies: ReadonlyArray<{ mass: number; initialVelocity: v3f }>,
  sunMass: number,
) {
  const totalMomentum = otherBodies.reduce(
    (momentum, body) => ({
      x: momentum.x + (body.mass * body.initialVelocity.x),
      y: momentum.y + (body.mass * body.initialVelocity.y),
      z: momentum.z + (body.mass * body.initialVelocity.z),
    }),
    { x: 0, y: 0, z: 0 },
  );

  return d.vec3f(
    -totalMomentum.x / sunMass,
    -totalMomentum.y / sunMass,
    -totalMomentum.z / sunMass,
  );
}

const SUN_INITIAL_VELOCITY = calculateMomentumBalancedSunVelocity(
  [
    { mass: MERCURY_MASS, initialVelocity: MERCURY_INITIAL_VELOCITY },
    { mass: VENUS_MASS, initialVelocity: VENUS_INITIAL_VELOCITY },
    { mass: EARTH_MASS, initialVelocity: EARTH_INITIAL_VELOCITY },
    { mass: MOON_MASS, initialVelocity: MOON_INITIAL_VELOCITY },
    { mass: MARS_MASS, initialVelocity: MARS_INITIAL_VELOCITY },
    { mass: JUPITER_MASS, initialVelocity: JUPITER_INITIAL_VELOCITY },
    { mass: SATURN_MASS, initialVelocity: SATURN_INITIAL_VELOCITY },
    { mass: URANUS_MASS, initialVelocity: URANUS_INITIAL_VELOCITY },
    { mass: NEPTUNE_MASS, initialVelocity: NEPTUNE_INITIAL_VELOCITY },
  ],
  SUN_MASS,
);


// 100 distance = 1 Au
// 1 velocity = 1,490km/s
// 1 mass = mass of sun

export const CelestianBody = d.struct({
  position: d.vec3f,
  radius: d.f32,
  color: d.vec4f,
  initialVelocity: d.vec3f,
  mass: d.f32,
})


export const INITIAL_BODIES = d.arrayOf(CelestianBody, 10)([
  // Sun
  { position: d.vec3f(0, 0, 0), radius: SUN_RENDER_RADIUS, color: d.vec4f(1.0, 0.85, 0.2, 1), initialVelocity: SUN_INITIAL_VELOCITY, mass: SUN_MASS },

  { position: d.vec3f(38.7, 0, 0), radius: MERCURY_RENDER_RADIUS, color: d.vec4f(0.72, 0.67, 0.62, 1), initialVelocity: MERCURY_INITIAL_VELOCITY, mass: MERCURY_MASS }, // Mercury
  { position: d.vec3f(72.3, 0, 0), radius: VENUS_RENDER_RADIUS, color: d.vec4f(0.95, 0.75, 0.42, 1), initialVelocity: VENUS_INITIAL_VELOCITY, mass: VENUS_MASS }, // Venus
  { position: d.vec3f(EARTH_DISTANCE, 0, 0), radius: EARTH_BODY_RENDER_RADIUS, color: d.vec4f(0.30, 0.60, 1.00, 1), initialVelocity: EARTH_INITIAL_VELOCITY, mass: EARTH_MASS }, // Earth
  { position: d.vec3f(EARTH_DISTANCE, MOON_ORBIT_RADIUS, 0), radius: MOON_RENDER_RADIUS, color: d.vec4f(0.80, 0.80, 0.84, 1), initialVelocity: MOON_INITIAL_VELOCITY, mass: MOON_MASS }, // Moon
  { position: d.vec3f(152.4, 0, 0), radius: MARS_RENDER_RADIUS, color: d.vec4f(0.89, 0.40, 0.24, 1), initialVelocity: MARS_INITIAL_VELOCITY, mass: MARS_MASS }, // Mars
  { position: d.vec3f(520.3, 0, 0), radius: JUPITER_RENDER_RADIUS, color: d.vec4f(0.83, 0.66, 0.43, 1), initialVelocity: JUPITER_INITIAL_VELOCITY, mass: JUPITER_MASS }, // Jupiter
  { position: d.vec3f(958.2, 0, 0), radius: SATURN_RENDER_RADIUS, color: d.vec4f(0.85, 0.79, 0.62, 1), initialVelocity: SATURN_INITIAL_VELOCITY, mass: SATURN_MASS }, // Saturn
  { position: d.vec3f(1918, 0, 0), radius: URANUS_RENDER_RADIUS, color: d.vec4f(0.52, 0.82, 0.91, 1), initialVelocity: URANUS_INITIAL_VELOCITY, mass: URANUS_MASS }, // Uranus
  { position: d.vec3f(3007, 0, 0), radius: NEPTUNE_RENDER_RADIUS, color: d.vec4f(0.28, 0.44, 0.93, 1), initialVelocity: NEPTUNE_INITIAL_VELOCITY, mass: NEPTUNE_MASS }, // Neptune
]);