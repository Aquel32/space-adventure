// oxlint-disable-next-line no-unassigned-import
import { cos, select, sin, step } from "typegpu/std";
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
let frame = 0;

export let GRAVITY_MULTIPLIER = 0.04;
export function SetGravityMultiplier(newG: number) {
  GRAVITY_MULTIPLIER = newG;
}
const gravityMultiplierBuffer = root.createBuffer(d.f32).$usage("storage", "uniform");

function calculateStableOrbitVelocity(distance: number, mass: number) {
  return std.sqrt((GRAVITY_MULTIPLIER * mass) / distance);
}

const ORBIT_POINTS = 20000;
const ORBIT_POINTS_CONST = tgpu.const(d.i32, ORBIT_POINTS);

export const INITIAL_BODIES = d.arrayOf(CelestianBody, 10)([
  // Sun
  { position: d.vec3f(0, 0, 0), radius: 14, color: d.vec4f(1.0, 0.85, 0.2, 1), initialVelocity: d.vec3f(0, 0, 0), mass: 1.0 },

  { position: d.vec3f(38.7, 0, 0), radius: 0.96, color: d.vec4f(0.72, 0.67, 0.62, 1), initialVelocity: d.vec3f(0, 0, 0.0321), mass: 1.66e-7 }, // Mercury
  { position: d.vec3f(72.3, 0, 0), radius: 2.37, color: d.vec4f(0.95, 0.75, 0.42, 1), initialVelocity: d.vec3f(0, 0, 0.0235), mass: 2.45e-6 }, // Venus
  { position: d.vec3f(100, 0, 0), radius: 2.5, color: d.vec4f(0.30, 0.60, 1.00, 1), initialVelocity: d.vec3f(0, 0, 0.0200), mass: 3.00e-6 }, // Earth
  { position: d.vec3f(100, 6, 0), radius: 0.68, color: d.vec4f(0.80, 0.80, 0.84, 1), initialVelocity: d.vec3f(0.000141, 0, 0.0200), mass: 3.69e-8 }, // Moon
  { position: d.vec3f(152.4, 0, 0), radius: 1.33, color: d.vec4f(0.89, 0.40, 0.24, 1), initialVelocity: d.vec3f(0, 0, 0.0162), mass: 3.23e-7 }, // Mars
  { position: d.vec3f(520.3, 0, 0), radius: 8.5, color: d.vec4f(0.83, 0.66, 0.43, 1), initialVelocity: d.vec3f(0, 0, 0.00877), mass: 9.54e-4 }, // Jupiter
  { position: d.vec3f(958.2, 0, 0), radius: 7.1, color: d.vec4f(0.85, 0.79, 0.62, 1), initialVelocity: d.vec3f(0, 0, 0.00646), mass: 2.86e-4 }, // Saturn
  { position: d.vec3f(1918, 0, 0), radius: 5.0, color: d.vec4f(0.52, 0.82, 0.91, 1), initialVelocity: d.vec3f(0, 0, 0.00457), mass: 4.37e-5 }, // Uranus
  { position: d.vec3f(3007, 0, 0), radius: 4.9, color: d.vec4f(0.28, 0.44, 0.93, 1), initialVelocity: d.vec3f(0, 0, 0.00365), mass: 5.15e-5 }, // Neptune
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
    speed: d.vec3f(0.1, 1, 10),
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
        out:{position: d.builtin.position, point:d.vec3f, cameraPos: d.interpolate("flat", d.vec3f), emits:d.interpolate("flat", d.i32), sunPosition: d.vec3f, groundColor: d.vec4f, normal: d.vec4f, vid: d.interpolate("flat", d.i32)}})(({vid})=>{
          'use gpu';
          const camera = cameraUniform.$;
          const offset = mainLayout.$.offsets[mainLayout.$.currentBodyIndex];
          const normal = verticies.$[vid];
          const point = normal.xyz.mul(body.radius).add(offset);
          const position = camera.projection.mul(camera.view).mul(d.vec4f(point, 1));
          const sunPosition = mainLayout.$.offsets[0];
      
          let emits = 0;
          if(mainLayout.$.currentBodyIndex === 0)
          {
            emits = 1;
          }
          
          return {
            position: position,
            groundColor: body.color,
            normal: normal,
            vid,
            point,
            sunPosition,
            emits,
            cameraPos: camera.pos.xyz
          };
        }),
      fragment: ({ groundColor, normal, vid, sunPosition, point, emits, cameraPos }) => {
        'use gpu';
        if(emits === 1)
        {
          return groundColor;
        }

        const surfaceToLightDirection = std.normalize(sunPosition.sub(point));
        const light = std.dot(normal.xyz, surfaceToLightDirection);
        
        const surfaceToViewDirection = std.normalize(cameraPos.sub(point));
        const halfVector = std.normalize(surfaceToLightDirection.add(surfaceToViewDirection));
        let specular = std.dot(normal.xyz, halfVector);

        specular = select(0.0, std.pow(specular, 20), specular > 0);

        return groundColor.mul(light) + specular;
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

  frame = 0;
}

SetUpBuffersAndData();

const bodiesVelocityPipeline = root.createGuardedComputePipeline((i)=>{
  'use gpu';
  const currentPosition = computeLayout.$.offsets[i];

  let newVelocity = d.vec3f(0, 0, 0);

  for(let x = 0; x < computeLayout.$.bodies.length; x++)
  {
    if(x === i) continue;

    const otherPosition = computeLayout.$.offsets[x]
    const otherMass = computeLayout.$.masses[x]
    const distance = std.distance(currentPosition, otherPosition);
    const gravityForce =  computeLayout.$.gravityMultiplier * (otherMass / (distance * distance));
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

      for(let x = 0; x < INITIAL_BODIES.length; x++)
      {
        if(x === i) continue;

        const otherPosition = orbitComputeLayout.$.offsets[x];
        const otherMass = orbitComputeLayout.$.masses[x];
        const distance = std.distance(currentPosition, otherPosition);
        const gravityForce =  orbitComputeLayout.$.gravityMultiplier * (otherMass / (distance * distance));
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

function predictOrbits()
{
  for(let i = 0; i < ORBIT_POINTS; i++)
  {
    orbitPointIndexBuffer.patch(i);
    orbitComputeVelocityPipeline.with(orbitComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
    orbitComputeOffsetPipeline.with(orbitComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
  }
}

function render()
{
  bodiesVelocityPipeline.with(mainComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
  bodiesOffsetPipeline.with(mainComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);

  if(frame % (ORBIT_POINTS/2) === 0)
  {
    predictOrbits();
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
      //   depthLoadOp: "load",
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
