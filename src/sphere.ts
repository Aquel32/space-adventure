import tgpu, { d, std } from "typegpu";
import type { v3f } from "typegpu/data";

const initialVerticies = [
  d.vec3f(0, 0, 0),
  d.vec3f(0, 0, 1),
  d.vec3f(0, 1, 0),
  d.vec3f(0, 1, 1),
  d.vec3f(1, 0, 0),
  d.vec3f(1, 0, 1),
  d.vec3f(1, 1, 0),
  d.vec3f(1, 1, 1),
].map(v => v.sub(0.5));

const initialFaces = [
  0, 1 ,2,
  4, 1, 0,
  0, 2, 4,
  3, 2, 1,
  1, 4, 5,
  5, 3, 1,
  2, 3, 6,
  6, 4, 2,
  7, 6, 3,
  3, 5, 7,
  6, 5, 4,
  5, 6, 7,
]

export function getVertexAmount(iterations: number) {
    return initialFaces.length * Math.pow(4, iterations);
}

export function generateSphere(position: v3f, radius: number, iterations: number) {
  let currentVerticies = initialFaces.map(i => initialVerticies[i]);

  for(let i = 0; i < iterations; i++)
  {
    currentVerticies = divideCube(currentVerticies);
  }

  return currentVerticies.map(v=>d.vec4f(v.add(position), 1));
}

function divideCube(verticies: v3f[]) {
  const newVerticies = [];

    for(let i = 0; i < verticies.length; i+=3)
    {
      const v1 = verticies[i];
      const v2 = verticies[i + 1];
      const v3 = verticies[i + 2];

      const mid12 = std.normalize(v1.add(v2).mul(0.5));
      const mid23 = std.normalize(v2.add(v3).mul(0.5));
      const mid31 = std.normalize(v3.add(v1).mul(0.5));

      newVerticies.push(v1, mid12, mid31);
      newVerticies.push(v2, mid23, mid12);
      newVerticies.push(v3, mid31, mid23);
      newVerticies.push(mid12, mid23, mid31);
    }

  return newVerticies;
}