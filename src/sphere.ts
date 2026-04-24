import tgpu, { d, std, type TgpuRoot } from "typegpu";
import type { v3f } from "typegpu/data";
import { select } from "typegpu/std";
import { CelestianBody } from "./simulation-data";
import { perlin2d, perlin3d } from '@typegpu/noise'

let strength = 0.3;
let epsilon = 0.2;

export function SetStrength(newStrength: number) {
  strength = newStrength;
}

export function SetEpsilon(newEpsilon: number) {
  epsilon = newEpsilon;
}

const cubeDirections = tgpu.const(d.arrayOf(d.vec3f, 6), [
  d.vec3f(1, 0, 0),
  d.vec3f(-1, 0, 0),
  d.vec3f(0, 1, 0),
  d.vec3f(0, -1, 0),
  d.vec3f(0, 0, 1),
  d.vec3f(0, 0, -1),
]);

export function getVertexAmount(divisions: number) {
  return cubeDirections.$.length * (4 ** divisions) * 6;
}

function perlinForPoint(point: v3f) {
  "use gpu";
  return (perlin3d.sample(point)) * 0.5;
  // return 1 + ((perlin3d.sample(point * 1) + 1) / 1);
}

const PerlinResult = d.struct({
  point: d.vec3f,
  normal: d.vec3f,
})

function getPointOnCubeFace(direction: v3f, divisions: number, x: number, y: number) {
  "use gpu";

  const relativeX = (x / divisions - 0.5) * 2;
  const relativeY = (y / divisions - 0.5) * 2;

  const xFaceDirection = d.vec3f(direction.y, direction.z, direction.x);
  const yFaceDirection = d.vec3f(direction.z, direction.x, direction.y);

  const point = std.normalize(direction.add(xFaceDirection.mul(relativeX)).add(yFaceDirection.mul(relativeY)));
  const strength = sphereComputeLayout.$.strength;
  const epsilon = sphereComputeLayout.$.epsilon;

  if (strength === 0 || epsilon === 0) {
    return PerlinResult({ point, normal: point });
  }

  const perlinOffset = sphereComputeLayout.$.perlinOffset;

  const perlinValue = perlinForPoint(point) * strength;
  const pointWithPerlin = point.add(perlinValue * point);

  let grad = d.vec3f(0, 0, 0);

  grad += perlin3d.sample(point + d.vec3f(1, 0, 0) * epsilon) / epsilon;
  grad -= perlin3d.sample(point - d.vec3f(1, 0, 0) * epsilon) / epsilon;

  grad += perlin3d.sample(point + d.vec3f(0, 1, 0) * epsilon) / epsilon;
  grad -= perlin3d.sample(point - d.vec3f(0, 1, 0) * epsilon) / epsilon;

  grad += perlin3d.sample(point + d.vec3f(0, 0, 1) * epsilon) / epsilon;
  grad -= perlin3d.sample(point - d.vec3f(0, 0, 1) * epsilon) / epsilon;

  grad *= strength;

  const tangentialGrad = grad - (std.dot(grad, point) * point);
  const normal = std.normalize(point - tangentialGrad);

  return PerlinResult({ point: pointWithPerlin, normal: normal });
}


const sphereComputeLayout = tgpu.bindGroupLayout({
  verticies: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  normals: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  currentDirection: { uniform: d.u32 },
  divisions: { uniform: d.u32 },
  strength: { uniform: d.f32 },
  epsilon: { uniform: d.f32 },
  perlinOffset: { uniform: d.f32 },
});

export function generateSphere(root: TgpuRoot, divisions: number, perlinOffset: number) {
  const verticies = root.createBuffer(d.arrayOf(d.vec3f, getVertexAmount(divisions))).$usage("vertex", "storage");
  const normals = root.createBuffer(d.arrayOf(d.vec3f, getVertexAmount(divisions))).$usage("vertex", "storage");

  const currentDirection = root.createBuffer(d.u32).$usage("uniform");
  const divisionsBuffer = root.createBuffer(d.u32).$usage("uniform");
  const strengthBuffer = root.createBuffer(d.f32).$usage("uniform");
  const epsilonBuffer = root.createBuffer(d.f32).$usage("uniform");
  const perlinOffsetBuffer = root.createBuffer(d.f32).$usage("uniform");

  divisionsBuffer.write(divisions);
  strengthBuffer.write(strength);
  epsilonBuffer.write(epsilon);
  perlinOffsetBuffer.write(perlinOffset);

  const sphereComputeBindGroup = root.createBindGroup(sphereComputeLayout, {
    verticies,
    normals,
    currentDirection,
    divisions: divisionsBuffer,
    strength: strengthBuffer,
    epsilon: epsilonBuffer,
    perlinOffset: perlinOffsetBuffer,
  });

  const createSphereComputePipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    "use gpu";

    const divisions = sphereComputeLayout.$.divisions;
    const directionIndex = sphereComputeLayout.$.currentDirection;
    const direction = cubeDirections.$[directionIndex];
    const index = (directionIndex * 6 * (4 ** divisions)) + (y * (2 ** divisions) * 6) + (x * 6);

    let result = getPointOnCubeFace(direction, divisions, x, y);
    sphereComputeLayout.$.verticies[index] = d.vec3f(result.point);
    sphereComputeLayout.$.normals[index] = d.vec3f(result.normal);

    result = getPointOnCubeFace(direction, divisions, x + 1, y);
    sphereComputeLayout.$.verticies[index + 1] = d.vec3f(result.point);
    sphereComputeLayout.$.normals[index + 1] = d.vec3f(result.normal);

    result = getPointOnCubeFace(direction, divisions, x, y + 1);
    sphereComputeLayout.$.verticies[index + 2] = d.vec3f(result.point);
    sphereComputeLayout.$.normals[index + 2] = d.vec3f(result.normal);

    result = getPointOnCubeFace(direction, divisions, x + 1, y);
    sphereComputeLayout.$.verticies[index + 3] = d.vec3f(result.point);
    sphereComputeLayout.$.normals[index + 3] = d.vec3f(result.normal);

    result = getPointOnCubeFace(direction, divisions, x + 1, y + 1);
    sphereComputeLayout.$.verticies[index + 4] = d.vec3f(result.point);
    sphereComputeLayout.$.normals[index + 4] = d.vec3f(result.normal);

    result = getPointOnCubeFace(direction, divisions, x, y + 1);
    sphereComputeLayout.$.verticies[index + 5] = d.vec3f(result.point);
    sphereComputeLayout.$.normals[index + 5] = d.vec3f(result.normal);
  });

  cubeDirections.$.forEach((_, direction) => {
    currentDirection.write(direction);

    createSphereComputePipeline.
      with(sphereComputeBindGroup).
      dispatchThreads(divisions, divisions);
  });

  return { verticies, normals };
}