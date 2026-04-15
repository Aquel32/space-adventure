// oxlint-disable-next-line no-unassigned-import
import { cos, sin } from "typegpu/std";
import { Camera, setupFirstPersonCamera } from "./setup-first-person-camera";
import "./style.css";
import tgpu, { common, d, std } from "typegpu";
import type { v3f, v4f } from "typegpu/data";
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
    cameraUniform.writePartial(props);
  },
);

const SPHERE_DIVISIONS = 10;
const RADIUS = 1;
const POSITION = d.vec3f(0, 0, 0);

const verticies = tgpu.const(d.arrayOf(d.vec4f, sphere.getVertexAmount(SPHERE_DIVISIONS)), sphere.generateSphere(POSITION, RADIUS, SPHERE_DIVISIONS));

const pipeline = root.createRenderPipeline({
  vertex: ({ $vertexIndex: vid, $instanceIndex: instanceid }) => {
    'use gpu';

    const point = verticies.$[vid];
    const camera = cameraUniform.$;
    const position = camera.projection.mul(camera.view).mul(point);
    return {
      $position: position,
      uv: d.vec2f(1, 1),
      vid: d.f32(vid),
    };
  },
  fragment: ({ uv, vid }) => {
    'use gpu';
    return d.vec4f(vid%3/3 , 0, 0, 1);
  }
});

function render()
{
  pipeline.withColorAttachment({ view: context }).draw(verticies.$.length, 1);

  updatePosition();

  requestAnimationFrame(render);
}

requestAnimationFrame(render);
