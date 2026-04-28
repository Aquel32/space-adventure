// oxlint-disable-next-line no-unassigned-import
import { cos, normalize, select, sin, step } from "typegpu/std";
import { Camera, setupFirstPersonCamera } from "./setup-first-person-camera";
import "./style.css";
import tgpu, {
  common,
  d,
  std,
  type StorageFlag,
  type TgpuBindGroup,
  type TgpuBuffer,
  type TgpuBufferMutable,
  type TgpuBufferUniform,
  type TgpuConst,
  type TgpuGuardedComputePipeline,
  type TgpuMutable,
  type TgpuRenderPipeline,
  type TgpuUniform,
  type VertexFlag,
} from "typegpu";
import * as sphere from "./sphere";
import { GenerateControls, SetUpControls } from "./ui-controls";
import { BODY_COUNT_CONST, CelestianBody, INITIAL_BODIES } from "./data/simulation-data";
import { ATTACHED_BODY_INDEX, DEBUG_NORMALS, GRAVITY_MULTIPLIER, ORBIT_PREDICTION_STEPS, ORBIT_PREDICTION_STEPS_CONST, RENDER_ORBITS, SetAttachedBody } from "./data/settings";
import { PrepareBloom } from "./postprocessing/bloom";
import { PrepareSimulation } from "./simulation";

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<main>
  <canvas id="canvas" width="2560" height="1440"></canvas>
</main>
`;

GenerateControls();

const root = await tgpu.init({
  device: { requiredLimits: { "maxBufferSize": 4294967292, "maxStorageBufferBindingSize": 4294967292 } },
});
const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const context = root.configureContext({ canvas });

let frame = 0;

export const pixelScaleUniform = root.createUniform(d.f32);

const cameraUniform = root.createUniform(Camera);
const camera = setupFirstPersonCamera(
  canvas,
  {
    initPos: d.vec3f(20, 0, 0),
    speed: d.vec3f(0.01, 0.1, 5),
  },
  (props) => {
    cameraUniform.patch(props);
  },
);

const positionVertexLayout = tgpu.vertexLayout(d.disarrayOf(d.float16x4));
const normalVertexLayout = tgpu.vertexLayout(d.disarrayOf(d.float16x4));
const normalDebugVertexLayout = tgpu.vertexLayout(d.disarrayOf(d.float16x4));

const mainLayout = tgpu.bindGroupLayout({
  bodies: { storage: d.arrayOf(CelestianBody), access: "readonly" },
  currentBodyIndex: { storage: d.i32, access: "readonly" },
});

let bodies = INITIAL_BODIES;
const bodiesBuffer = root
  .createBuffer(d.arrayOf(CelestianBody, INITIAL_BODIES.length))
  .$usage("storage", "uniform");
const currentBodyIndexBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

const mainRenderBindGroup = root.createBindGroup(mainLayout, {
  bodies: bodiesBuffer,
  currentBodyIndex: currentBodyIndexBuffer,
});

const bodiesRenderData: {
  verticies: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag & VertexFlag;
  normals: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag & VertexFlag;
  trickVerticies: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
  trickNormals: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
  debugNormalVerticies: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
  trickDebugNormalVerticies: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
}[] = [];

export function SetUpBodiesRenderData() {
  bodiesBuffer.write(bodies);

  bodiesRenderData.length = 0;
  bodies.forEach((body, i) => {
    const { verticies, normals, trickVerticies, trickNormals, debugNormalVerticies, trickDebugNormalVerticies } = sphere.generateSphere(root, i, body.isSphere);
    bodiesRenderData.push({ verticies, normals, trickVerticies, trickNormals, debugNormalVerticies, trickDebugNormalVerticies });
  });

  frame = 0;
}

const depthTexture = root
  .createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "depth24plus",
  })
  .$usage("render");

const mainRenderPipeline = root.createRenderPipeline({
  attribs: { inVertex: positionVertexLayout.attrib, inNormal: normalVertexLayout.attrib },
  vertex: tgpu.vertexFn({
    in: { vid: d.builtin.vertexIndex, inVertex: d.vec4f, inNormal: d.vec4f },
    out: {
      position: d.builtin.position,
      point: d.vec3f,
      cameraPos: d.interpolate("flat", d.vec3f),
      emits: d.interpolate("flat", d.i32),
      sunPosition: d.vec3f,
      groundColor: d.vec4f,
      normal: d.vec3f,
      vid: d.interpolate("flat", d.i32),
      bodyId: d.interpolate("flat", d.i32),
    },
  })(({ vid, inVertex, inNormal }) => {
    "use gpu";
    const body = mainLayout.$.bodies[mainLayout.$.currentBodyIndex];
    const camera = cameraUniform.$;
    const offset = body.position;
    const vertex = inVertex.xyz;
    const normal = inNormal.xyz;

    const point = vertex.mul(body.radius).add(offset);
    const position = camera.projection.mul(camera.view).mul(d.vec4f(point, 1));
    const sunPosition = mainLayout.$.bodies[0].position;

    const height = std.length(vertex);
    const colors = body.colors;
    let groundColor = d.vec4f(body.colors[0].color);

    for (let i = 0; i < colors.length; i++) {
      if (height >= colors[i].height) {
        groundColor = d.vec4f(colors[i].color);
      }
    }

    let emits = 0;
    if (mainLayout.$.currentBodyIndex === 0) {
      emits = 1;
    }

    return {
      position: position,
      groundColor,
      normal,
      vid,
      point,
      sunPosition,
      emits,
      cameraPos: camera.pos.xyz,
      bodyId: mainLayout.$.currentBodyIndex,
    };
  }),
  fragment: ({ $position, groundColor, normal, vid, sunPosition, point, emits, cameraPos, bodyId }) => {
    "use gpu";
    if (emits === 1) {
      return {
        color: groundColor,
        emission: d.vec4f(1, 1, 1, 1),
      };
    }

    const surfaceToLightDirection = std.normalize(sunPosition.sub(point));
    const diffuse = std.max(0, std.dot(normal, surfaceToLightDirection));

    const reflectionDirection = std.reflect(surfaceToLightDirection.mul(-1), normal.xyz);
    const surfaceToViewDirection = std.normalize(cameraPos.sub(point));
    const specular = std.pow(std.max(0, std.dot(reflectionDirection, surfaceToViewDirection)), 150) * 0.5;

    const finalColor = groundColor * diffuse + specular;
    // const finalColor = groundColor;

    let emission = d.vec4f(0, 0, 0, 1);
    const treshold = 0.8;
    if (finalColor.r > treshold || finalColor.g > treshold || finalColor.b > treshold) {
      const val = (diffuse + specular - treshold) / (treshold);
      emission = d.vec4f(val, val, val, 1);
    }

    return {
      color: finalColor,
      emission,
    };
  },
  depthStencil: {
    format: "depth24plus",
    depthWriteEnabled: true,
    depthCompare: "less",
  },
  targets: {
    emission: { format: "rgba8unorm" },
  },
});

const normalDebugPipeline = root.createRenderPipeline({
  attribs: { inVertex: normalDebugVertexLayout.attrib },
  vertex: tgpu.vertexFn({
    in: { vid: d.builtin.vertexIndex, inVertex: d.vec4f },
    out: {
      position: d.builtin.position,
    },
  })(({ vid, inVertex }) => {
    "use gpu";
    const body = mainLayout.$.bodies[mainLayout.$.currentBodyIndex];
    const camera = cameraUniform.$;
    const offset = body.position;
    const vertex = inVertex.xyz;

    const point = vertex.mul(body.radius).add(offset);
    const position = camera.projection.mul(camera.view).mul(d.vec4f(point, 1));

    return {
      position,
    };
  }),
  fragment: ({ $position }) => {
    "use gpu";

    return d.vec4f(1, 1, 1, 1);
  },
  depthStencil: {
    format: "depth24plus",
    depthWriteEnabled: true,
    depthCompare: "less",
  },
  primitive: {
    topology: "line-list"
  },
});

export function moveCameraToAttachedObject() {
  if (ATTACHED_BODY_INDEX === -1) return;
  if (ATTACHED_BODY_INDEX < 0 || ATTACHED_BODY_INDEX >= INITIAL_BODIES.length) return;

  camera.setPosition(bodies[ATTACHED_BODY_INDEX].position.sub(d.vec3f(0, 0, bodies[ATTACHED_BODY_INDEX].radius * 3)));
}

function alignCameraToAttachedObject() {
  if (ATTACHED_BODY_INDEX === -1) return;

  const attachedBody = bodies[ATTACHED_BODY_INDEX];
  const cameraPosition = camera.state.pos;

  const newCameraPos = cameraPosition.add(attachedBody.velocity)
  camera.setPosition(newCameraPos);
}

const simulation = PrepareSimulation(root, canvas, context, cameraUniform);
const bloomEffect = PrepareBloom(root, canvas, context, pixelScaleUniform);

function render() {
  camera.updatePosition();

  bodies = simulation.simulateGravity(bodies);
  bodiesBuffer.write(bodies);

  alignCameraToAttachedObject();

  if (frame % (ORBIT_PREDICTION_STEPS / 2) === 0) {
    simulation.predictOrbits(bodies);
  }

  bodiesRenderData.forEach((item, i) => {
    currentBodyIndexBuffer.write(i);

    mainRenderPipeline
      .withColorAttachment({
        color: {
          view: context,
          loadOp: i === 0 ? "clear" : "load",
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        },
        emission: { view: bloomEffect.emmisionTexture.createView("render", { mipLevelCount: 1, baseMipLevel: 0 }), loadOp: i === 0 ? "clear" : "load", clearValue: { r: 0, g: 0, b: 0, a: 0 } },
      })
      .withDepthStencilAttachment({
        view: depthTexture,
        depthLoadOp: i === 0 ? "clear" : "load",
        depthClearValue: 1,
        depthStoreOp: "store",
      })
      .with(positionVertexLayout, item.trickVerticies)
      .with(normalVertexLayout, item.trickNormals)
      .with(mainRenderBindGroup)
      .draw(sphere.getVertexAmount(), 1);

    simulation.prepareOrbitRender(i);
  });

  bloomEffect.applyGausianBlur();
  bloomEffect.render();

  if (RENDER_ORBITS) {
    simulation.renderOrbits();
  }

  if (DEBUG_NORMALS) {
    bodiesRenderData.forEach((item, i) => {
      currentBodyIndexBuffer.write(i);
      normalDebugPipeline
        .withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } })
        .withDepthStencilAttachment({
          view: depthTexture,
          depthLoadOp: "load",
          depthClearValue: 1,
          depthStoreOp: "store",
        })
        .with(normalDebugVertexLayout, item.trickDebugNormalVerticies)
        .with(mainRenderBindGroup)
        .draw(sphere.getVertexAmount() * 2, 1);
    });
  }

  frame++;
  requestAnimationFrame(render);
}

SetUpControls();
SetUpBodiesRenderData();
SetAttachedBody(3); // attach to earth

requestAnimationFrame(render);
