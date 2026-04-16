// oxlint-disable-next-line no-unassigned-import
import { cos, sin } from "typegpu/std";
import { Camera, setupFirstPersonCamera } from "./setup-first-person-camera";
import "./style.css";
import tgpu, { common, d, std, type TgpuBuffer, type TgpuBufferMutable, type TgpuBufferUniform, type TgpuConst, type TgpuMutable, type TgpuRenderPipeline, type TgpuUniform } from "typegpu";
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

export const INITIAL_BODIES = d.arrayOf(CelestianBody, 2)([
  { position: d.vec3f(0, 0, 0), radius: 4, color: d.vec4f(1, 0, 0, 1), initialVelocity: d.vec3f(0 ,0 ,0), mass: 100000 },
  { position: d.vec3f(10, 0, 0), radius: 0.1, color: d.vec4f(0, 1, 0, 1), initialVelocity: d.vec3f(-10000, 0, 1000000), mass: 100 },
]);

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<main>
  <canvas id="canvas" width="600" height="512"></canvas>
</main>
`;

// Setting up TypeGPU
const root = await tgpu.init();

export let GRAVITY_MULTIPLIER = 0.00000001;
export function SetGravityMultiplier(newG: number) {
  GRAVITY_MULTIPLIER = newG;
}
const gravityMultiplierBuffer = root.createBuffer(d.f32).$usage("storage", "uniform");
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

const data: {pipeline: TgpuRenderPipeline<d.Vec4f>, verticies: TgpuConst<d.WgslArray<d.Vec4f>>}[] = [];

const SPHERE_DIVISIONS = 10;

export function SetUpBuffersAndData() {
  console.log(GRAVITY_MULTIPLIER, INITIAL_BODIES);

  gravityMultiplierBuffer.write(GRAVITY_MULTIPLIER);

  massesBuffer.write(INITIAL_BODIES.map(b=>b.mass));
  offsetsBuffer.write(INITIAL_BODIES.map(b=>b.position));
  velocitiesBuffer.write(INITIAL_BODIES.map(b=>b.initialVelocity.mul(GRAVITY_MULTIPLIER)));
  currentBodyIndexBuffer.write(0);

  data.length = 0;

  INITIAL_BODIES.forEach((body, i) => {
    const pipeline = root.createRenderPipeline({
      vertex:  tgpu.vertexFn({
        in: {vid:d.builtin.vertexIndex}, 
        out:{position: d.builtin.position, groundColor: d.vec4f, normal: d.vec4f, vid: d.interpolate("flat", d.i32)}})(({vid})=>{
          'use gpu';
          const camera = cameraUniform.$;
          const offset = mainLayout.$.offsets[mainLayout.$.currentBodyIndex];
          const normal = verticies.$[vid];
          const point = normal.mul(body.radius).add(d.vec4f(offset, 1));
          const position = camera.projection.mul(camera.view).mul(point);
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

    const verticies = tgpu.const(d.arrayOf(d.vec4f, sphere.getVertexAmount(SPHERE_DIVISIONS)), sphere.generateSphere(body.position, body.radius, SPHERE_DIVISIONS));
    data.push({ pipeline, verticies});
  });
}

SetUpBuffersAndData();

const computePipeline = root.createGuardedComputePipeline((i)=>{
  'use gpu';
  const currentPosition = computeLayout.$.offsets[i];
  const currentVelocity = computeLayout.$.velocities[i];
  const currentMass = computeLayout.$.masses[i];
  const currentRadius = computeLayout.$.bodies[i].radius;

  let newVelocity = d.vec3f(0, 0, 0);

  for(let x = 0; x < computeLayout.$.bodies.length; x++)
  {
    if(x === i) continue;

    const otherPosition = computeLayout.$.offsets[x]
    const otherMass = computeLayout.$.masses[x]
    const otherRadius = computeLayout.$.bodies[x].radius

    const distance = std.max(
      currentRadius + otherRadius,
      std.distance(currentPosition, otherPosition)
    );

    const gravityForce = (currentMass * otherMass) / distance / distance;
    const direction = std.normalize(otherPosition.sub(currentPosition));

    newVelocity = newVelocity.add(direction.mul(gravityForce / currentMass).mul(computeLayout.$.gravityMultiplier));
  }
  computeLayout.$.velocities[i] = computeLayout.$.velocities[i].add(newVelocity);
  computeLayout.$.offsets[i] = computeLayout.$.offsets[i].add(computeLayout.$.velocities[i]);
})

function render()
{
  computePipeline.with(computeBindGroup).dispatchThreads(INITIAL_BODIES.length);

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
  });
  
  updatePosition();

  requestAnimationFrame(render);
}


requestAnimationFrame(render);