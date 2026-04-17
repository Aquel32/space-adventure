// oxlint-disable-next-line no-unassigned-import
import { cos, sin, step } from "typegpu/std";
import { Camera, setupFirstPersonCamera } from "./setup-first-person-camera";
import "./style.css";
import tgpu, { common, d, std, type TgpuBindGroup, type TgpuBuffer, type TgpuBufferMutable, type TgpuBufferUniform, type TgpuConst, type TgpuGuardedComputePipeline, type TgpuMutable, type TgpuRenderPipeline, type TgpuUniform } from "typegpu";
import { i32, type v3f, type v4f } from "typegpu/data";
import * as sphere from "./sphere";
import { SetUpControls } from "./ui-controls";

const CelestianBody = d.struct({
  position: d.vec3f,
  radius: d.f32,
  color: d.vec4f,
  initialVelocity: d.vec3f,
  mass: d.f32,
})

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<main>
  <canvas id="canvas" width="600" height="512"></canvas>
</main>
`;

const root = await tgpu.init();

export let GRAVITY_MULTIPLIER = 1e-9;
export function SetGravityMultiplier(newG: number) {
  GRAVITY_MULTIPLIER = newG;
}
const gravityMultiplierBuffer = root.createBuffer(d.f32).$usage("storage", "uniform");

function calculateStableOrbitVelocity(distance: number, mass: number) {
  return std.sqrt((GRAVITY_MULTIPLIER * mass) / distance);
}

const ORBIT_POINTS = 5000;
const ORBIT_POINTS_CONST = tgpu.const(d.i32, ORBIT_POINTS);

export const INITIAL_BODIES = d.arrayOf(CelestianBody, 2)([
  { position: d.vec3f(0, 0, 0), radius: 10, color: d.vec4f(1, 0, 0, 1), initialVelocity: d.vec3f(0 ,0 ,0), mass: 10000000 },
  { position: d.vec3f(100, 0, 0), radius: 5, color: d.vec4f(0, 1, 0, 1), initialVelocity: d.vec3f(0,0,calculateStableOrbitVelocity(10, 10000000)), mass: 100 },
]);
const BODY_COUNT_CONST = tgpu.const(d.i32, INITIAL_BODIES.length);

SetUpControls();

const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const context = root.configureContext({ canvas });

const cameraUniform = root.createUniform(Camera);
const { updatePosition } = setupFirstPersonCamera(
  canvas,
  {
    initPos: d.vec3f(0, 0, -25),
    speed: d.vec3f(0.001, 0.1, 1),
  },
  (props) => {
    cameraUniform.patch(props);
  },
);

let depthTexture = root.createTexture({
  size: [canvas.width, canvas.height, 1],
  format: 'depth24plus',
}).$usage("render");

const mainLayout = tgpu.bindGroupLayout({
  gravityMultiplier: { storage: d.f32, access: "readonly" },
  bodies: { storage: d.arrayOf(CelestianBody), access: "readonly" },
  masses: { storage: d.arrayOf(d.f32), access: "readonly" },
  offsets: { storage: d.arrayOf(d.vec3f), access:"readonly" },
  velocities: { storage: d.arrayOf(d.vec3f), access: "readonly" },
  currentBodyIndex: { storage: d.i32, access:"readonly" },
})

const computeLayout = tgpu.bindGroupLayout({
  gravityMultiplier: { storage: d.f32, access: "mutable" },
  bodies: { storage: d.arrayOf(CelestianBody), access: "mutable" },
  masses: { storage: d.arrayOf(d.f32), access: "mutable" },
  offsets: { storage: d.arrayOf(d.vec3f), access:"mutable" },
  velocities: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  currentBodyIndex: { storage: d.i32, access:"mutable" },
})

const orbitRenderLayout = tgpu.bindGroupLayout({
  currentBodyIndex: { storage: d.i32, access:"readonly" },
  vertecies: { storage: d.arrayOf(d.vec3f), access: "readonly" },
})

const orbitComputeLayout = tgpu.bindGroupLayout({
  orbitPointIndex: { storage: d.i32, access:"mutable" },
  offsets: { storage: d.arrayOf(d.vec3f), access:"mutable" },
  velocities: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  vertecies: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  initialVelocities: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  initialOffsets: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  masses: { storage: d.arrayOf(d.f32), access: "mutable" },
  gravityMultiplier: { storage: d.f32, access: "mutable" },
})

const bodiesBuffer = root.createBuffer(d.arrayOf(CelestianBody, INITIAL_BODIES.length)).$usage("storage", "uniform");
const massesBuffer = root.createBuffer(d.arrayOf(d.f32, INITIAL_BODIES.length)).$usage("storage");
const offsetsBuffer = root.createBuffer(d.arrayOf(d.vec3f, INITIAL_BODIES.length)).$usage("storage", "uniform");
const velocitiesBuffer = root.createBuffer(d.arrayOf(d.vec3f, INITIAL_BODIES.length)).$usage("storage", "uniform");
const currentBodyIndexBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

const mainRenderBindGroup = root.createBindGroup(mainLayout, {
  bodies: bodiesBuffer,
  masses: massesBuffer,
  offsets: offsetsBuffer,
  velocities: velocitiesBuffer,
  currentBodyIndex: currentBodyIndexBuffer,
  gravityMultiplier: gravityMultiplierBuffer
});

const mainComputeBindGroup = root.createBindGroup(computeLayout, {
  bodies: bodiesBuffer,
  masses: massesBuffer,
  offsets: offsetsBuffer,
  velocities: velocitiesBuffer,
  currentBodyIndex: currentBodyIndexBuffer,
  gravityMultiplier: gravityMultiplierBuffer
});

const data: {mainRenderPipeline: TgpuRenderPipeline<d.Vec4f>, verticies: TgpuConst<d.WgslArray<d.Vec4f>>, orbitRenderPipeline: TgpuRenderPipeline<d.Vec4f>}[] = [];

const SPHERE_DIVISIONS = 10;

export function SetUpBuffersAndData() {
  gravityMultiplierBuffer.write(GRAVITY_MULTIPLIER);

  massesBuffer.write(INITIAL_BODIES.map(b=>b.mass));
  offsetsBuffer.write(INITIAL_BODIES.map(b=>b.position));
  velocitiesBuffer.write(INITIAL_BODIES.map(b=>b.initialVelocity));
  currentBodyIndexBuffer.write(0);

  data.length = 0;

  INITIAL_BODIES.forEach((body, i) => {
    const mainRenderPipeline = root.createRenderPipeline({
      vertex:  tgpu.vertexFn({
        in: {vid:d.builtin.vertexIndex}, 
        out:{position: d.builtin.position, groundColor: d.vec4f, normal: d.vec4f, vid: d.interpolate("flat", d.i32)}})(({vid})=>{
          'use gpu';
          const camera = cameraUniform.$;
          const offset = mainLayout.$.offsets[mainLayout.$.currentBodyIndex];
          const normal = verticies.$[vid];
          const point = normal.xyz.mul(body.radius).add(offset);
          const position = camera.projection.mul(camera.view).mul(d.vec4f(point, 1));
          return {
            position: position,
            groundColor: body.color,
            normal: normal,
            vid,
          };
        }),
      fragment: ({ groundColor, normal, vid }) => {
        'use gpu';
        return std.abs(normal);
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const orbitRenderPipeline = root.createRenderPipeline({
      vertex:  tgpu.vertexFn({
        in: {vid:d.builtin.vertexIndex}, 
        out:{position: d.builtin.position, bodyIndex: d.interpolate("flat", d.i32)}})(({vid})=>{
          'use gpu';
          const bodyIndex = orbitRenderLayout.$.currentBodyIndex;
          const camera = cameraUniform.$;
          const point = orbitRenderLayout.$.vertecies[bodyIndex*ORBIT_POINTS_CONST.$ + vid];
          const position = camera.projection.mul(camera.view).mul(d.vec4f(point, 1));
          return {
            position,
            bodyIndex
          };
        }),
      fragment: ({ bodyIndex }) => {
        'use gpu';
        const val = bodyIndex / BODY_COUNT_CONST.$ ;
        return d.vec4f(val, 1, 1-val , 1);
      },
      primitive: {
        topology: "line-strip",
      }
    });
    
    const verticies = tgpu.const(d.arrayOf(d.vec4f, sphere.getVertexAmount(SPHERE_DIVISIONS)), sphere.generateSphere(SPHERE_DIVISIONS));
    data.push({ mainRenderPipeline, verticies, orbitRenderPipeline });
  });
}

SetUpBuffersAndData();


const bodiesVelocityPipeline = root.createGuardedComputePipeline((i)=>{
  'use gpu';
  const currentPosition = computeLayout.$.offsets[i];
  const currentMass = computeLayout.$.masses[i];

  let newVelocity = d.vec3f(0, 0, 0);

  for(let x = 0; x < computeLayout.$.bodies.length; x++)
  {
    if(x === i) continue;

    const otherPosition = computeLayout.$.offsets[x]
    const otherMass = computeLayout.$.masses[x]
    const distance = std.distance(currentPosition, otherPosition);
    const gravityForce =  computeLayout.$.gravityMultiplier * ((currentMass * otherMass) / (distance * distance));
    const direction = std.normalize(otherPosition.sub(currentPosition));
    newVelocity = newVelocity.add(direction.mul(gravityForce));
  }
  computeLayout.$.velocities[i] = computeLayout.$.velocities[i].add(newVelocity);
})

const bodiesOffsetPipeline = root.createGuardedComputePipeline((i)=>{
  'use gpu';
  computeLayout.$.offsets[i] = computeLayout.$.offsets[i].add(computeLayout.$.velocities[i]);
})

const orbitComputeVelocityPipeline = root.createGuardedComputePipeline((i)=>{
    'use gpu';
    const stepIndex = orbitComputeLayout.$.orbitPointIndex;

    if(stepIndex === 0)
    {
      orbitComputeLayout.$.velocities[i] = d.vec3f(orbitComputeLayout.$.initialVelocities[i]);
      return;
    }

    let newVelocity = d.vec3f(0, 0, 0);

    const currentPosition = orbitComputeLayout.$.offsets[i];
    const currentMass = orbitComputeLayout.$.masses[i];

      for(let x = 0; x < INITIAL_BODIES.length; x++)
      {
        if(x === i) continue;

        const otherPosition = orbitComputeLayout.$.offsets[x];
        const otherMass = orbitComputeLayout.$.masses[x];
        const distance = std.distance(currentPosition, otherPosition);
        const gravityForce =  orbitComputeLayout.$.gravityMultiplier * ((currentMass * otherMass) / (distance * distance));
        const direction = std.normalize(otherPosition.sub(currentPosition));
        newVelocity = newVelocity.add(direction.mul(gravityForce));
      }
    orbitComputeLayout.$.velocities[i] = orbitComputeLayout.$.velocities[i].add(newVelocity);
});

const orbitComputeOffsetPipeline = root.createGuardedComputePipeline((i)=>{
    'use gpu';
    const stepIndex = orbitComputeLayout.$.orbitPointIndex;
    const vertexIndex = i*ORBIT_POINTS_CONST.$ + stepIndex;

    if(stepIndex === 0)
    {
      orbitComputeLayout.$.offsets[i] = d.vec3f(orbitComputeLayout.$.initialOffsets[i]);
      orbitComputeLayout.$.vertecies[vertexIndex] = d.vec3f(orbitComputeLayout.$.initialOffsets[i]);
      return;
    }

    orbitComputeLayout.$.offsets[i] = orbitComputeLayout.$.offsets[i].add(orbitComputeLayout.$.velocities[i]);
    orbitComputeLayout.$.vertecies[vertexIndex] = d.vec3f(orbitComputeLayout.$.offsets[i]);
});

const orbitVerticiesBuffer = root.createBuffer(d.arrayOf(d.vec3f, ORBIT_POINTS * INITIAL_BODIES.length)).$usage("storage", "uniform");
const orbitComputeOffsetsBuffer = root.createBuffer(d.arrayOf(d.vec3f, INITIAL_BODIES.length)).$usage("storage", "uniform");
const orbitComputeVelocitiesBuffer = root.createBuffer(d.arrayOf(d.vec3f, INITIAL_BODIES.length)).$usage("storage", "uniform");
const orbitPointIndexBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

const orbitRenderBindGroup = root.createBindGroup(orbitRenderLayout, {
  currentBodyIndex: currentBodyIndexBuffer,
  vertecies: orbitVerticiesBuffer,
})

const orbitComputeBindGroup = root.createBindGroup(orbitComputeLayout, {
  orbitPointIndex: orbitPointIndexBuffer,
  offsets: orbitComputeOffsetsBuffer,
  velocities: orbitComputeVelocitiesBuffer,
  vertecies: orbitVerticiesBuffer,
  initialVelocities: velocitiesBuffer,
  initialOffsets: offsetsBuffer,
  masses: massesBuffer,
  gravityMultiplier: gravityMultiplierBuffer,
})

let frame = 0;

    for(let i = 0; i < ORBIT_POINTS; i++)
    {
      orbitPointIndexBuffer.patch(i);
      orbitComputeVelocityPipeline.with(orbitComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
    }

function render()
{
  bodiesVelocityPipeline.with(mainComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
  bodiesOffsetPipeline.with(mainComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);

  if(frame % 1000 === 0)
  {
    for(let i = 0; i < ORBIT_POINTS; i++)
    {
      orbitPointIndexBuffer.patch(i);
      orbitComputeVelocityPipeline.with(orbitComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
      orbitComputeOffsetPipeline.with(orbitComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
    }
  }


  data.forEach(async (item, i) => {
    currentBodyIndexBuffer.patch(i);

    item.mainRenderPipeline.
      withColorAttachment({ view: context, loadOp: i === 0 ? "clear" : "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } }).
      withDepthStencilAttachment({ 
        view: depthTexture, 
        depthLoadOp: i === 0 ? "clear" : "load",
        depthClearValue: 1, 
        depthStoreOp: "store"
      }).
      with(mainRenderBindGroup).
      draw(item.verticies.$.length, 1);

    item.orbitRenderPipeline.
      withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } }).
      // withDepthStencilAttachment({ 
      //   view: depthTexture, 
      //   depthLoadOp: i === 0 ? "clear" : "load",
      //   depthClearValue: 1, 
      //   depthStoreOp: "store"
      // }).
      with(orbitRenderBindGroup).
      draw(ORBIT_POINTS, 1);
  });
  
  updatePosition();

  frame++;
  requestAnimationFrame(render);
}

requestAnimationFrame(render);
