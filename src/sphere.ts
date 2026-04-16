import tgpu, { d, std } from "typegpu";
import type { v3f } from "typegpu/data";

const cubeDirections = [
  d.vec3f(1, 0, 0),
  d.vec3f(-1, 0, 0),
  d.vec3f(0, 1, 0),
  d.vec3f(0, -1, 0),
  d.vec3f(0, 0, 1),
  d.vec3f(0, 0, -1),
];

export function getVertexAmount(divisions: number) {
    return cubeDirections.length * divisions * divisions * 6;
}

function getPointOnCubeFace(direction: v3f, divisions: number, x: number, y: number) {
  const relativeX = (x/divisions - 0.5) * 2;
  const relativeY = (y/divisions - 0.5) * 2;

  const xFaceDirection = d.vec3f(direction.y, direction.z, direction.x);
  const yFaceDirection = d.vec3f(direction.z, direction.x, direction.y);

  const point = direction.add(xFaceDirection.mul(relativeX)).add(yFaceDirection.mul(relativeY));
  return std.normalize(point);
}

export function generateSphere(divisions: number) {
  const vericies: v3f[] = [];
  cubeDirections.forEach((direction, i)=>{
    for(let x = 0; x < divisions; x++){
      for(let y = 0; y < divisions; y++){
        vericies.push(getPointOnCubeFace(direction, divisions, x, y));
        vericies.push(getPointOnCubeFace(direction, divisions, x+1, y));
        vericies.push(getPointOnCubeFace(direction, divisions, x, y+1));

        vericies.push(getPointOnCubeFace(direction, divisions, x+1, y));
        vericies.push(getPointOnCubeFace(direction, divisions, x+1, y+1));
        vericies.push(getPointOnCubeFace(direction, divisions, x, y+1));
      }
    }
  })

  return vericies.map(v=>d.vec4f(v, 1));
}