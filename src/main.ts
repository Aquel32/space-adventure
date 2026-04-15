// oxlint-disable-next-line no-unassigned-import
import { cos, sin } from "typegpu/std";
import { Camera, setupFirstPersonCamera } from "./setup-first-person-camera";
import "./style.css";
import tgpu, { common, d, std, type TgpuBuffer, type TgpuBufferMutable, type TgpuBufferUniform, type TgpuConst, type TgpuMutable, type TgpuRenderPipeline, type TgpuUniform } from "typegpu";
import { i32, type v3f, type v4f } from "typegpu/data";
import * as sphere from "./sphere";

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<section id="center">
  <div class="hero">
    <canvas id="canvas" width="600" height="512"></canvas>
  </div>
</section>
`;

// Setting up TypeGPU
const root = await tgpu.init();

const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const context = root.configureContext({ canvas });

const cameraUniform = root.createUniform(Camera);
const { updatePosition } = setupFirstPersonCamera(
  canvas,
  {
    initPos: d.vec3f(0, 0, -5),
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

const CelestianBody = d.struct({
  position: d.vec3f,
  radius: i32,
  color: d.vec4f,
  initialVelocity: d.vec3f,
  mass: d.f32,
})

const BODIES = tgpu.const(d.arrayOf(CelestianBody, 2), [
  { position: d.vec3f(-1, 0, 0), radius: 1, color: d.vec4f(1, 0, 0, 1), initialVelocity: d.vec3f(0.01, 0, 0), mass: 1000 },
  { position: d.vec3f(3, 0, 0), radius: 0.5, color: d.vec4f(0, 1, 0, 1), initialVelocity: d.vec3f(0, 0.002, 0), mass: 500 },
]);

const mainLayout = tgpu.bindGroupLayout({
  masses: { storage: d.arrayOf(d.f32), access: "readonly" },
  offsets: { storage: d.arrayOf(d.vec3f), access:"readonly" },
  velocities: { storage: d.arrayOf(d.vec3f), access: "readonly" },
  currentBodyIndex: { storage: d.i32, access:"readonly" },
})

const computeLayout = tgpu.bindGroupLayout({
  masses: { storage: d.arrayOf(d.f32), access: "mutable" },
  offsets: { storage: d.arrayOf(d.vec3f), access:"mutable" },
  velocities: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  currentBodyIndex: { storage: d.i32, access:"mutable" },
})

const massesBuffer = root.createBuffer(d.arrayOf(d.f32, BODIES.$.length)).$usage("storage");
const offsetsBuffer = root.createBuffer(d.arrayOf(d.vec3f, BODIES.$.length)).$usage("storage", "uniform");
const velocitiesBuffer = root.createBuffer(d.arrayOf(d.vec3f, BODIES.$.length)).$usage("storage", "uniform");
const currentBodyIndexBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

const mainBindGroup = root.createBindGroup(mainLayout, {
  masses: massesBuffer,
  offsets: offsetsBuffer,
  velocities: velocitiesBuffer,
  currentBodyIndex: currentBodyIndexBuffer
});

const computeBindGroup = root.createBindGroup(computeLayout, {
  masses: massesBuffer,
  offsets: offsetsBuffer,
  velocities: velocitiesBuffer,
  currentBodyIndex: currentBodyIndexBuffer
});


const data: {pipeline: TgpuRenderPipeline<d.Vec4f>, verticies: TgpuConst<d.WgslArray<d.Vec4f>>}[] = [];

const SPHERE_DIVISIONS = 10;

massesBuffer.patch(BODIES.$.map(b=>b.mass));
offsetsBuffer.patch(BODIES.$.map(b=>b.position));
velocitiesBuffer.patch(BODIES.$.map(b=>b.initialVelocity));
currentBodyIndexBuffer.patch(0);

BODIES.$.forEach((body, i) => {
  const pipeline = root.createRenderPipeline({
    vertex:  tgpu.vertexFn({
      in: {vid:d.builtin.vertexIndex}, 
      out:{position: d.builtin.position, groundColor: d.vec4f, vid: d.interpolate("flat", d.i32)}})(({vid})=>{
        'use gpu';
        const camera = cameraUniform.$;
        const offset = mainLayout.$.offsets[mainLayout.$.currentBodyIndex];
        const point = verticies.$[vid].add(d.vec4f(offset, 1));
        const position = camera.projection.mul(camera.view).mul(point);
        return {
          position: position,
          groundColor: body.color,
          vid,
        };
      }),
    fragment: ({ groundColor, vid }) => {
      'use gpu';
      return groundColor.mul(1 - vid%3/3);
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

const computePipeline = root.createGuardedComputePipeline((i)=>{
  'use gpu';
  // computeLayout.$.velocities[i] = computeLayout.$.velocities[i].add(d.vec3f(0.001, 0, 0));
  computeLayout.$.offsets[i] = computeLayout.$.offsets[i].add(computeLayout.$.velocities[i]);
})

function render()
{
  computePipeline.with(computeBindGroup).dispatchThreads(BODIES.$.length);

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
