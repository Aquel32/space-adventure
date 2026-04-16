// oxlint-disable-next-line no-unassigned-import
import { cos, sin } from "typegpu/std";
import { Camera, setupFirstPersonCamera } from "./setup-first-person-camera";
import "./style.css";
import tgpu, { common, d, std, type TgpuBindGroup, type TgpuBuffer, type TgpuBufferMutable, type TgpuBufferUniform, type TgpuConst, type TgpuMutable, type TgpuRenderPipeline, type TgpuUniform } from "typegpu";
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

export const INITIAL_BODIES = d.arrayOf(CelestianBody, 2)([
  { position: d.vec3f(0, 0, 0), radius: 10, color: d.vec4f(1, 0, 0, 1), initialVelocity: d.vec3f(0 ,0 ,0), mass: 10000000 },
  { position: d.vec3f(100, 0, 0), radius: 5, color: d.vec4f(0, 1, 0, 1), initialVelocity: d.vec3f(0,0,calculateStableOrbitVelocity(10, 10000000)), mass: 100 },
]);

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

const orbitLayout = tgpu.bindGroupLayout({
  vertecies: { storage: d.arrayOf(d.vec4f), access: "readonly" },
})

const bodiesBuffer = root.createBuffer(d.arrayOf(CelestianBody, INITIAL_BODIES.length)).$usage("storage", "uniform");
const massesBuffer = root.createBuffer(d.arrayOf(d.f32, INITIAL_BODIES.length)).$usage("storage");
const offsetsBuffer = root.createBuffer(d.arrayOf(d.vec3f, INITIAL_BODIES.length)).$usage("storage", "uniform");
const velocitiesBuffer = root.createBuffer(d.arrayOf(d.vec3f, INITIAL_BODIES.length)).$usage("storage", "uniform");
const currentBodyIndexBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

const mainBindGroup = root.createBindGroup(mainLayout, {
  bodies: bodiesBuffer,
  masses: massesBuffer,
  offsets: offsetsBuffer,
  velocities: velocitiesBuffer,
  currentBodyIndex: currentBodyIndexBuffer,
  gravityMultiplier: gravityMultiplierBuffer
});

const computeBindGroup = root.createBindGroup(computeLayout, {
  bodies: bodiesBuffer,
  masses: massesBuffer,
  offsets: offsetsBuffer,
  velocities: velocitiesBuffer,
  currentBodyIndex: currentBodyIndexBuffer,
  gravityMultiplier: gravityMultiplierBuffer
});

const data: {pipeline: TgpuRenderPipeline<d.Vec4f>, verticies: TgpuConst<d.WgslArray<d.Vec4f>>, orbitPipeline: TgpuRenderPipeline<d.Vec4f>, orbitBindGroup: TgpuBindGroup<{
    vertecies: {
        storage: (elementCount: number) => d.WgslArray<d.Vec4f>;
        access: "readonly";
    };
}>}[] = [];

const SPHERE_DIVISIONS = 10;

export function SetUpBuffersAndData() {
  gravityMultiplierBuffer.write(GRAVITY_MULTIPLIER);

  massesBuffer.write(INITIAL_BODIES.map(b=>b.mass));
  offsetsBuffer.write(INITIAL_BODIES.map(b=>b.position));
  velocitiesBuffer.write(INITIAL_BODIES.map(b=>b.initialVelocity));
  currentBodyIndexBuffer.write(0);

  data.length = 0;
  const ORBIT_POINTS = 10000;

  INITIAL_BODIES.forEach((body, i) => {
    const pipeline = root.createRenderPipeline({
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

    const orbitPipeline = root.createRenderPipeline({
      vertex:  tgpu.vertexFn({
        in: {vid:d.builtin.vertexIndex}, 
        out:{position: d.builtin.position}})(({vid})=>{
          'use gpu';
          const i = std.floor(vid / 4);
          const camera = cameraUniform.$;
          const point = orbitLayout.$.vertecies[i];
          const position = camera.projection * (camera.view) * (point);
          return {
            position,
          };
        }),
      fragment: () => {
        'use gpu';
        return d.vec4f(1, 1, 1, 1);
      },
      primitive: {
        topology: "line-strip",
      }
    });

    let currentPosition = body.position;
    let currentVelocity = body.initialVelocity;
    const currentMass = body.mass;
    const prepareOrbitVertecies: Float32Array = new Float32Array(ORBIT_POINTS * 4);
    for(let j = 0; j < ORBIT_POINTS; j++){
      let newVelocity = d.vec3f(0, 0, 0);

      for(let x = 0; x < INITIAL_BODIES.length; x++)
      {
        if(x === i) continue;

        const otherPosition = INITIAL_BODIES[x].position
        const otherMass = INITIAL_BODIES[x].mass
        const distance = std.distance(currentPosition, otherPosition);
        const gravityForce =  GRAVITY_MULTIPLIER * ((currentMass * otherMass) / (distance * distance));
        const direction = std.normalize(otherPosition.sub(currentPosition));
        newVelocity = newVelocity.add(direction.mul(gravityForce));
      }
      currentVelocity = currentVelocity.add(newVelocity);
      currentPosition = currentPosition.add(currentVelocity);
      
      prepareOrbitVertecies[j*4] = currentPosition.x;
      prepareOrbitVertecies[j*4+1] = currentPosition.y;
      prepareOrbitVertecies[j*4+2] = currentPosition.z;
      prepareOrbitVertecies[j*4+3] = 1;
    }

    const orbitVerticiesBuffer = root.createBuffer(d.arrayOf(d.vec4f, ORBIT_POINTS)).$usage("storage");
    const orbitBindGroup = root.createBindGroup(orbitLayout, {
      vertecies: orbitVerticiesBuffer,
    });
    orbitVerticiesBuffer.write(prepareOrbitVertecies);
    
    const verticies = tgpu.const(d.arrayOf(d.vec4f, sphere.getVertexAmount(SPHERE_DIVISIONS)), sphere.generateSphere(SPHERE_DIVISIONS));
    data.push({ pipeline, verticies, orbitPipeline, orbitBindGroup });
  });
}

SetUpBuffersAndData();


const velocityPipeline = root.createGuardedComputePipeline((i)=>{
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

const offsetPipeline = root.createGuardedComputePipeline((i)=>{
  'use gpu';
  computeLayout.$.offsets[i] = computeLayout.$.offsets[i].add(computeLayout.$.velocities[i]);
})

function render()
{
  velocityPipeline.with(computeBindGroup).dispatchThreads(INITIAL_BODIES.length);
  offsetPipeline.with(computeBindGroup).dispatchThreads(INITIAL_BODIES.length);

  data.forEach(async (item, i) => {
    currentBodyIndexBuffer.patch(i);
    item.pipeline.
      withColorAttachment({ view: context, loadOp: i === 0 ? "clear" : "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } }).
      withDepthStencilAttachment({ 
        view: depthTexture, 
        depthLoadOp: i === 0 ? "clear" : "load",
        depthClearValue: 1, 
        depthStoreOp: "store"
      }).
      with(mainBindGroup).
      draw(item.verticies.$.length, 1);

    item.orbitPipeline.
      withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } }).
      // withDepthStencilAttachment({ 
      //   view: depthTexture, 
      //   depthLoadOp: i === 0 ? "clear" : "load",
      //   depthClearValue: 1, 
      //   depthStoreOp: "store"
      // }).
      with(item.orbitBindGroup).
      draw(10000*4, 1);
  });
  
  updatePosition();

  requestAnimationFrame(render);
}


requestAnimationFrame(render);