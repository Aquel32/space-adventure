// oxlint-disable-next-line no-unassigned-import
import { cos, normalize, select, sin, step } from "typegpu/std";
import { calculateProj, calculateView, Camera, setupFirstPersonCamera } from "./setup-first-person-camera";
import { perlin2d, perlin3d } from '@typegpu/noise'
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
import { i32, type v3f, type v4f } from "typegpu/data";
import * as sphere from "./sphere";
import { SetUpControls } from "./ui-controls";
import { CelestianBody, GRAVITY_MULTIPLIER, INITIAL_BODIES } from "./simulation-data";
import { cubeDirections } from "./sphere";
import * as m from "wgpu-matrix";

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<main>
  <canvas id="canvas" width="2560" height="1440"></canvas>
</main>
`;

const root = await tgpu.init({
  device: { requiredLimits: { "maxBufferSize": 4294967292, "maxStorageBufferBindingSize": 4294967292 } },
});
let frame = 0;

const gravityMultiplierBuffer = root.createBuffer(d.f32).$usage("storage", "uniform");

export const PIXEL_SCALE_BUFFER = root.createUniform(d.f32);
PIXEL_SCALE_BUFFER.write(1);
export let GAUSIAN_ITERATIONS = d.f32(5);
export function SetBlurIterations(newbi: number) {
  GAUSIAN_ITERATIONS = newbi;
}

export let RENDER_ORBITS = true;
export function SetRenderOrbits(newRo: boolean) {
  RENDER_ORBITS = newRo;
}

export let DEBUG_NORMALS = false;
export function SetDebugNormals(newDn: boolean) {
  DEBUG_NORMALS = newDn;
}

const ORBIT_POINTS = 10000;
const ORBIT_POINTS_CONST = tgpu.const(d.i32, ORBIT_POINTS);

const BODY_COUNT_CONST = tgpu.const(d.i32, INITIAL_BODIES.length);

SetUpControls();

const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const context = root.configureContext({ canvas });

const cameraBuffer = root.createBuffer(Camera).$usage("storage", "uniform");
const cameraMutable = cameraBuffer.as("mutable");
const cameraUniform = cameraBuffer.as("uniform");

const { cameraState, updatePosition, setPosition } = setupFirstPersonCamera(
  canvas,
  {
    initPos: d.vec3f(20, 0, 0),
    speed: d.vec3f(0.01, 0.1, 5),
  },
  (props) => {
    cameraBuffer.patch(props);
  },
);
let bodies = INITIAL_BODIES;

let attachedBodyIndex = -1;
export function UpdateAttachedBody(newIndex: number) {
  if (newIndex == -1) {
    attachedBodyIndex = -1;
    return;
  }

  if (newIndex < 0 || newIndex >= bodies.length) return;

  attachedBodyIndex = newIndex;
  setPosition(bodies[newIndex].position.sub(d.vec3f(0, 0, bodies[newIndex].radius * 3)));
}
UpdateAttachedBody(3)

let depthTexture = root
  .createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "depth24plus",
  })
  .$usage("render");

const mainSampler = root.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
});

const positionVertexLayout = tgpu.vertexLayout(d.disarrayOf(d.float16x4));
const normalVertexLayout = tgpu.vertexLayout(d.disarrayOf(d.float16x4));
const normalDebugVertexLayout = tgpu.vertexLayout(d.disarrayOf(d.float16x4));

const mainLayout = tgpu.bindGroupLayout({
  gravityMultiplier: { storage: d.f32, access: "readonly" },
  bodies: { storage: d.arrayOf(CelestianBody), access: "readonly" },
  currentBodyIndex: { storage: d.i32, access: "readonly" },
});

const orbitPrepareRenderLayout = tgpu.bindGroupLayout({
  currentBodyIndex: { storage: d.i32, access: "readonly" },
  vertecies: { storage: d.arrayOf(d.vec3f), access: "readonly" },
});

const orbitFinalRenderLayout = tgpu.bindGroupLayout({
  texture: { texture: d.texture2d(), access: "readonly" },
  sampler: { sampler: "filtering", access: "readonly" },
});

const bodiesBuffer = root
  .createBuffer(d.arrayOf(CelestianBody, INITIAL_BODIES.length))
  .$usage("storage", "uniform");
const currentBodyIndexBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

const data: {
  mainRenderPipeline: TgpuRenderPipeline<{ color: d.Vec4f; emission: d.Vec4f }>;
  mainRenderBindGroup: TgpuBindGroup;
  verticies: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag & VertexFlag;
  normals: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag & VertexFlag;
  trickVerticies: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
  trickNormals: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
  orbitRenderPipeline: TgpuRenderPipeline<d.Vec4f>;
  normalDebugPipeline: TgpuRenderPipeline<d.Vec4f>;
  debugNormalVerticies: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
  trickDebugNormalVerticies: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
}[] = [];

const SPHERE_DIVISIONS = 8;

export function SetUpBuffersAndData() {
  gravityMultiplierBuffer.write(GRAVITY_MULTIPLIER);

  bodiesBuffer.write(bodies);

  data.length = 0;

  bodies.forEach((body, i) => {
    const { verticies, normals, trickVerticies, trickNormals, debugNormalVerticies, trickDebugNormalVerticies } = sphere.generateSphere(root, SPHERE_DIVISIONS, i, body.isSphere);

    const mainRenderBindGroup = root.createBindGroup(mainLayout, {
      bodies: bodiesBuffer,
      currentBodyIndex: currentBodyIndexBuffer,
      gravityMultiplier: gravityMultiplierBuffer,
    });

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

    const orbitRenderPipeline = root.createRenderPipeline({
      vertex: tgpu.vertexFn({
        in: { vid: d.builtin.vertexIndex },
        out: { position: d.builtin.position, bodyIndex: d.interpolate("flat", d.i32) },
      })(({ vid }) => {
        "use gpu";
        const bodyIndex = orbitPrepareRenderLayout.$.currentBodyIndex;
        const camera = cameraUniform.$;
        const point = orbitPrepareRenderLayout.$.vertecies[bodyIndex * ORBIT_POINTS_CONST.$ + vid];
        const position = camera.projection.mul(camera.view).mul(d.vec4f(point, 1));
        return {
          position,
          bodyIndex,
        };
      }),
      fragment: ({ bodyIndex }) => {
        "use gpu";
        const val = bodyIndex / BODY_COUNT_CONST.$;
        return d.vec4f(val, 1, 1 - val, 1);
      },
      primitive: {
        topology: "line-strip",
      },
      targets: {
        format: "rgba8unorm",
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

    data.push({ mainRenderPipeline, verticies, normals, trickVerticies, trickNormals, normalDebugPipeline, debugNormalVerticies, trickDebugNormalVerticies, orbitRenderPipeline, mainRenderBindGroup });
  });

  frame = 0;
}

SetUpBuffersAndData();

const orbitVerticiesBuffer = root
  .createBuffer(d.arrayOf(d.vec3f, ORBIT_POINTS * INITIAL_BODIES.length))
  .$usage("storage", "uniform");

const orbitRenderTexture = root
  .createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba8unorm",
  })
  .$usage("render", "sampled");

const orbitPrepareRenderBindGroup = root.createBindGroup(orbitPrepareRenderLayout, {
  currentBodyIndex: currentBodyIndexBuffer,
  vertecies: orbitVerticiesBuffer,
});

const orbitFinalRenderBindGroup = root.createBindGroup(orbitFinalRenderLayout, {
  texture: orbitRenderTexture,
  sampler: mainSampler
});

function predictOrbits() {
  let currentBodies = bodies.map((b) => (CelestianBody(b)));
  for (let i = 0; i < ORBIT_POINTS; i++) {
    currentBodies = simulateGravity(currentBodies);

    currentBodies.forEach((body, index) => {
      const vertexIndex = index * ORBIT_POINTS + i;
      orbitVerticiesBuffer.patch({ [vertexIndex]: body.position });
    });
  }
}

const finalOrbitRenderPipeline = root.createRenderPipeline({
  vertex: tgpu.vertexFn({
    in: { vid: d.builtin.vertexIndex },
    out: { position: d.builtin.position, uv: d.vec2f },
  })(({ vid }) => {
    "use gpu";
    const positions = [
      d.vec3f(-1, 1, 0),
      d.vec3f(-1, -1, 0),
      d.vec3f(1, -1, 0),
      d.vec3f(-1, 1, 0),
      d.vec3f(1, 1, 0),
      d.vec3f(1, -1, 0),
    ];
    const uvX = positions[vid].x * 0.5 + 0.5;
    const uvY = 1.0 - (positions[vid].y * 0.5 + 0.5);
    return {
      position: d.vec4f(positions[vid], 1),
      uv: d.vec2f(uvX, uvY),
    };
  }),
  fragment: ({ uv }) => {
    "use gpu";
    return std.textureSampleLevel(
      orbitFinalRenderLayout.$.texture,
      orbitFinalRenderLayout.$.sampler,
      uv,
      1
    );
  },
  targets: {
    blend: {
      color: {
        operation: 'add',
        srcFactor: 'one-minus-dst-alpha',
        dstFactor: 'one',
      },
      alpha: {
        operation: 'add',
        srcFactor: 'one-minus-dst-alpha',
        dstFactor: 'one',
      },
    },
  }
});

const emmisionTexture = root
  .createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba8unorm",
    mipLevelCount: 4
  })
  .$usage("render", "sampled");

const postProccessTarget = root
  .createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba8unorm",
  })
  .$usage("render", "sampled");

const postProccessBindGroupLayout = tgpu.bindGroupLayout({
  isHorizontal: { storage: d.i32, access: "readonly" },
  targetTexture: { texture: d.texture2d() },
  emissionTexture: { texture: d.texture2d() },
  sampler: { sampler: "filtering" },
});

const isBlurHorizontalBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

const postProccessBindGroup = root.createBindGroup(postProccessBindGroupLayout, {
  isHorizontal: isBlurHorizontalBuffer,
  targetTexture: postProccessTarget,
  emissionTexture: emmisionTexture,
  sampler: mainSampler,
});

const blurRenderPipeline = root.createRenderPipeline({
  vertex: tgpu.vertexFn({
    in: { vid: d.builtin.vertexIndex },
    out: { position: d.builtin.position, uv: d.vec2f },
  })(({ vid }) => {
    "use gpu";
    const positions = [
      d.vec3f(-1, 1, 0),
      d.vec3f(-1, -1, 0),
      d.vec3f(1, -1, 0),
      d.vec3f(-1, 1, 0),
      d.vec3f(1, 1, 0),
      d.vec3f(1, -1, 0),
    ];
    const uvX = positions[vid].x * 0.5 + 0.5;
    const uvY = 1.0 - (positions[vid].y * 0.5 + 0.5);
    return {
      position: d.vec4f(positions[vid], 1),
      uv: d.vec2f(uvX, uvY),
    };
  }),
  fragment: ({ uv }) => {
    "use gpu";
    const weights = [0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216];

    let pixelSize = 1.0 / std.textureDimensions(postProccessBindGroupLayout.$.emissionTexture).x;
    pixelSize *= PIXEL_SCALE_BUFFER.$;

    let result = std
      .textureSampleLevel(
        postProccessBindGroupLayout.$.emissionTexture,
        postProccessBindGroupLayout.$.sampler,
        uv,
        1
      )

    let sum = d.f32(0);

    result *= select(weights[0], 1, postProccessBindGroupLayout.$.isHorizontal === 0);

    for (let i = -weights.length + 1; i < weights.length; i++) {
      sum += weights[std.abs(i)];

      if (i === 0) continue;

      const vec = std.select(d.vec2f(pixelSize * i, 0), d.vec2f(0, pixelSize * i), postProccessBindGroupLayout.$.isHorizontal === 0);

      result +=
        std.textureSampleLevel(
          postProccessBindGroupLayout.$.emissionTexture,
          postProccessBindGroupLayout.$.sampler,
          uv.add(vec),
          1
        ).mul(weights[std.abs(i)])
    }

    return result / sum;
  },
  targets: {
    format: "rgba8unorm",
  },
});

const currentBlurPassTarget = root
  .createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba8unorm",
    mipLevelCount: 4,
  })
  .$usage("render", "sampled");

const currentBlurPassBindGroup = root.createBindGroup(postProccessBindGroupLayout, {
  isHorizontal: isBlurHorizontalBuffer,
  targetTexture: postProccessTarget,
  emissionTexture: currentBlurPassTarget,
  sampler: mainSampler,
});

function gausianBlur(passes: number) {
  emmisionTexture.generateMipmaps();

  for (let i = 0; i < passes * 2; i++) {
    const target = i % 2 === 0 ? currentBlurPassTarget : emmisionTexture;

    isBlurHorizontalBuffer.write(i % 2);
    blurRenderPipeline
      .withColorAttachment({
        view: target.createView("render", { mipLevelCount: 1, baseMipLevel: 1 }),
        loadOp: "load",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      })
      .with(i % 2 === 0 ? postProccessBindGroup : currentBlurPassBindGroup)
      .draw(6, 1);
  }
}

const finalBloomRenderPipeline = root.createRenderPipeline({
  vertex: tgpu.vertexFn({
    in: { vid: d.builtin.vertexIndex },
    out: { position: d.builtin.position, uv: d.vec2f },
  })(({ vid }) => {
    "use gpu";
    const positions = [
      d.vec3f(-1, 1, 0),
      d.vec3f(-1, -1, 0),
      d.vec3f(1, -1, 0),
      d.vec3f(-1, 1, 0),
      d.vec3f(1, 1, 0),
      d.vec3f(1, -1, 0),
    ];
    const uvX = positions[vid].x * 0.5 + 0.5;
    const uvY = 1.0 - (positions[vid].y * 0.5 + 0.5);
    return {
      position: d.vec4f(positions[vid], 1),
      uv: d.vec2f(uvX, uvY),
    };
  }),
  fragment: ({ uv }) => {
    "use gpu";
    return std.textureSampleLevel(
      postProccessBindGroupLayout.$.emissionTexture,
      postProccessBindGroupLayout.$.sampler,
      uv,
      1
    );
  },
  targets: {
    blend: {
      color: {
        srcFactor: "one",
        dstFactor: "one",
        operation: "add",
      },
      alpha: {
        srcFactor: "one",
        dstFactor: "one",
        operation: "add",
      }
    },
  }
});

const mult = 1;
function simulateGravity(lastBodies: d.Infer<typeof CelestianBody>[]) {
  const finalBodies = lastBodies.map((b) => (CelestianBody(b)));

  finalBodies.forEach((body, i) => {
    let newVelocity = d.vec3f(0, 0, 0);
    finalBodies.forEach((otherBody, x) => {
      if (x === i) return;
      const otherPosition = otherBody.position;
      const otherMass = otherBody.mass;
      const distance = std.distance(body.position, otherPosition);
      const gravityForce = GRAVITY_MULTIPLIER * (otherMass / (distance * distance));
      const direction = std.normalize(otherPosition.sub(body.position));
      newVelocity = newVelocity.add(direction.mul(gravityForce));
    });
    body.velocity = body.velocity.add(newVelocity.mul(mult));
  });

  finalBodies.forEach((body, i) => {
    body.position = body.position.add(body.velocity.mul(mult));
  });

  return finalBodies;
}

function alignCameraToAttachedObject() {
  if (attachedBodyIndex === -1) return;

  const attachedBody = bodies[attachedBodyIndex];
  const cameraPosition = cameraState.pos;

  const newCameraPos = cameraPosition.add(attachedBody.velocity)
  setPosition(newCameraPos);
}

function render() {
  updatePosition();

  bodies = simulateGravity(bodies);
  bodiesBuffer.write(bodies);

  alignCameraToAttachedObject();

  if (frame % (ORBIT_POINTS / 2) === 0) {
    predictOrbits();
  }

  data.forEach((item, i) => {
    currentBodyIndexBuffer.write(i);

    item.mainRenderPipeline
      .withColorAttachment({
        color: {
          view: context,
          loadOp: i === 0 ? "clear" : "load",
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        },
        emission: { view: emmisionTexture.createView("render", { mipLevelCount: 1, baseMipLevel: 0 }), loadOp: i === 0 ? "clear" : "load", clearValue: { r: 0, g: 0, b: 0, a: 0 } },
      })
      .withDepthStencilAttachment({
        view: depthTexture,
        depthLoadOp: i === 0 ? "clear" : "load",
        depthClearValue: 1,
        depthStoreOp: "store",
      })
      .with(positionVertexLayout, item.trickVerticies)
      .with(normalVertexLayout, item.trickNormals)
      .with(item.mainRenderBindGroup)
      .draw(sphere.getVertexAmount(SPHERE_DIVISIONS), 1);

    item.orbitRenderPipeline.
      withColorAttachment({ view: orbitRenderTexture, loadOp: i === 0 ? "clear" : "load", clearValue: { r: 0, g: 0, b: 0, a: 0 } }).
      with(orbitPrepareRenderBindGroup).
      draw(ORBIT_POINTS, 1);
  });

  gausianBlur(GAUSIAN_ITERATIONS);

  finalBloomRenderPipeline
    .withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } })
    .with(postProccessBindGroup)
    .draw(6, 1);

  if (RENDER_ORBITS) {
    finalOrbitRenderPipeline
      .withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } })
      .with(orbitFinalRenderBindGroup)
      .draw(6, 1);
  }

  if (DEBUG_NORMALS) {
    data.forEach((item, i) => {
      currentBodyIndexBuffer.write(i);
      item.normalDebugPipeline
        .withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } })
        .withDepthStencilAttachment({
          view: depthTexture,
          depthLoadOp: "load",
          depthClearValue: 1,
          depthStoreOp: "store",
        })
        .with(normalDebugVertexLayout, item.trickDebugNormalVerticies)
        .with(item.mainRenderBindGroup)
        .draw(sphere.getVertexAmount(SPHERE_DIVISIONS) * 2, 1);
    });
  }

  frame++;
  requestAnimationFrame(render);
}

requestAnimationFrame(render);
