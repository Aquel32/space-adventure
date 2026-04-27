// oxlint-disable-next-line no-unassigned-import
import { cos, normalize, select, sin, step } from "typegpu/std";
import { Camera, setupFirstPersonCamera } from "./setup-first-person-camera";
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

const attachedObjectIndexUniform = root.createUniform(i32);

const cameraBuffer = root.createBuffer(Camera).$usage("storage", "uniform");
const cameraMutable = cameraBuffer.as("mutable");
const cameraUniform = cameraBuffer.as("uniform");

const { updatePosition, setPosition } = setupFirstPersonCamera(
  canvas,
  {
    initPos: d.vec3f(0, 0, 0),
    speed: d.vec3f(0.01, 0.1, 5),
  },
  (props) => {
    cameraBuffer.patch(props);
  },
);

export function UpdateAttachedBody(newIndex: number) {
  if (newIndex < 0 || newIndex >= INITIAL_BODIES.length) return;

  attachedObjectIndexUniform.write(newIndex);
  setPosition(d.vec3f(-INITIAL_BODIES[newIndex].radius - 1, 0, 0));
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
  masses: { storage: d.arrayOf(d.f32), access: "readonly" },
  offsets: { storage: d.arrayOf(d.vec3f), access: "readonly" },
  velocities: { storage: d.arrayOf(d.vec3f), access: "readonly" },
  currentBodyIndex: { storage: d.i32, access: "readonly" },
});

const computeLayout = tgpu.bindGroupLayout({
  gravityMultiplier: { storage: d.f32, access: "mutable" },
  bodies: { storage: d.arrayOf(CelestianBody), access: "mutable" },
  masses: { storage: d.arrayOf(d.f32), access: "mutable" },
  offsets: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  velocities: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  currentBodyIndex: { storage: d.i32, access: "mutable" },
});

const orbitPrepareRenderLayout = tgpu.bindGroupLayout({
  currentBodyIndex: { storage: d.i32, access: "readonly" },
  vertecies: { storage: d.arrayOf(d.vec3f), access: "readonly" },
});

const orbitFinalRenderLayout = tgpu.bindGroupLayout({
  texture: { texture: d.texture2d(), access: "readonly" },
  sampler: { sampler: "filtering", access: "readonly" },
});

const orbitComputeLayout = tgpu.bindGroupLayout({
  orbitPointIndex: { storage: d.i32, access: "mutable" },
  offsets: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  velocities: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  vertecies: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  initialVelocities: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  initialOffsets: { storage: d.arrayOf(d.vec3f), access: "mutable" },
  masses: { storage: d.arrayOf(d.f32), access: "mutable" },
  gravityMultiplier: { storage: d.f32, access: "mutable" },
});

const bodiesBuffer = root
  .createBuffer(d.arrayOf(CelestianBody, INITIAL_BODIES.length))
  .$usage("storage", "uniform");
const massesBuffer = root.createBuffer(d.arrayOf(d.f32, INITIAL_BODIES.length)).$usage("storage");
const offsetsBuffer = root
  .createBuffer(d.arrayOf(d.vec3f, INITIAL_BODIES.length))
  .$usage("storage", "uniform");
const velocitiesBuffer = root
  .createBuffer(d.arrayOf(d.vec3f, INITIAL_BODIES.length))
  .$usage("storage", "uniform");
const currentBodyIndexBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

const mainComputeBindGroup = root.createBindGroup(computeLayout, {
  bodies: bodiesBuffer,
  masses: massesBuffer,
  offsets: offsetsBuffer,
  velocities: velocitiesBuffer,
  currentBodyIndex: currentBodyIndexBuffer,
  gravityMultiplier: gravityMultiplierBuffer,
});

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

  bodiesBuffer.write(INITIAL_BODIES);
  massesBuffer.write(INITIAL_BODIES.map((b) => b.mass));
  offsetsBuffer.write(INITIAL_BODIES.map((b) => b.position));
  velocitiesBuffer.write(INITIAL_BODIES.map((b) => b.initialVelocity));
  currentBodyIndexBuffer.write(0);

  data.length = 0;

  INITIAL_BODIES.forEach((body, i) => {
    const { verticies, normals, trickVerticies, trickNormals, debugNormalVerticies, trickDebugNormalVerticies } = sphere.generateSphere(root, SPHERE_DIVISIONS, i, body.isSphere);

    const mainRenderBindGroup = root.createBindGroup(mainLayout, {
      bodies: bodiesBuffer,
      masses: massesBuffer,
      offsets: offsetsBuffer,
      velocities: velocitiesBuffer,
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
        const camera = cameraUniform.$;
        const offset = mainLayout.$.offsets[mainLayout.$.currentBodyIndex];
        const vertex = inVertex.xyz;
        const normal = inNormal.xyz;

        const point = vertex.mul(body.radius).add(offset - camera.pos.xyz);
        const position = camera.projection.mul(camera.view).mul(d.vec4f(point, 1));
        const sunPosition = mainLayout.$.offsets[0] - camera.pos.xyz;

        const height = std.length(vertex);
        const colors = mainLayout.$.bodies[mainLayout.$.currentBodyIndex].colors;
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
        const surfaceToViewDirection = std.normalize(point * - 1);
        const specular = std.pow(std.max(0, std.dot(reflectionDirection, surfaceToViewDirection)), 150) * 0.5;

        const finalColor = groundColor * diffuse + specular;

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
        const point = orbitPrepareRenderLayout.$.vertecies[bodyIndex * ORBIT_POINTS_CONST.$ + vid] - camera.pos.xyz;
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
        const camera = cameraUniform.$;
        const offset = mainLayout.$.offsets[mainLayout.$.currentBodyIndex];
        const vertex = inVertex.xyz;

        const point = vertex.mul(body.radius).add(offset - camera.pos.xyz);
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

const bodiesVelocityPipeline = root.createGuardedComputePipeline((i) => {
  "use gpu";
  const currentPosition = computeLayout.$.offsets[i];

  let newVelocity = d.vec3f(0, 0, 0);

  for (let x = 0; x < computeLayout.$.bodies.length; x++) {
    if (x === i) continue;

    const otherPosition = computeLayout.$.offsets[x];
    const otherMass = computeLayout.$.masses[x];
    const distance = std.distance(currentPosition, otherPosition);
    const gravityForce = computeLayout.$.gravityMultiplier * (otherMass / (distance * distance));
    const direction = std.normalize(otherPosition.sub(currentPosition));
    newVelocity = newVelocity.add(direction.mul(gravityForce));
  }
  computeLayout.$.velocities[i] = computeLayout.$.velocities[i].add(newVelocity);
});

const bodiesOffsetPipeline = root.createGuardedComputePipeline((i) => {
  "use gpu";
  computeLayout.$.offsets[i] = computeLayout.$.offsets[i].add(computeLayout.$.velocities[i]);
});

const orbitComputeVelocityPipeline = root.createGuardedComputePipeline((i) => {
  "use gpu";
  const stepIndex = orbitComputeLayout.$.orbitPointIndex;

  if (stepIndex === 0) {
    orbitComputeLayout.$.velocities[i] = d.vec3f(orbitComputeLayout.$.initialVelocities[i]);
    return;
  }

  let newVelocity = d.vec3f(0, 0, 0);

  const currentPosition = orbitComputeLayout.$.offsets[i];

  for (let x = 0; x < INITIAL_BODIES.length; x++) {
    if (x === i) continue;

    const otherPosition = orbitComputeLayout.$.offsets[x];
    const otherMass = orbitComputeLayout.$.masses[x];
    const distance = std.distance(currentPosition, otherPosition);
    const gravityForce =
      orbitComputeLayout.$.gravityMultiplier * (otherMass / (distance * distance));
    const direction = std.normalize(otherPosition.sub(currentPosition));
    newVelocity = newVelocity.add(direction.mul(gravityForce));
  }
  orbitComputeLayout.$.velocities[i] = orbitComputeLayout.$.velocities[i].add(newVelocity);
});

const orbitComputeOffsetPipeline = root.createGuardedComputePipeline((i) => {
  "use gpu";
  const stepIndex = orbitComputeLayout.$.orbitPointIndex;
  const vertexIndex = i * ORBIT_POINTS_CONST.$ + stepIndex;

  if (stepIndex === 0) {
    orbitComputeLayout.$.offsets[i] = d.vec3f(orbitComputeLayout.$.initialOffsets[i]);
    orbitComputeLayout.$.vertecies[vertexIndex] = d.vec3f(orbitComputeLayout.$.initialOffsets[i]);
    return;
  }

  orbitComputeLayout.$.offsets[i] = orbitComputeLayout.$.offsets[i].add(
    orbitComputeLayout.$.velocities[i],
  );
  orbitComputeLayout.$.vertecies[vertexIndex] = d.vec3f(orbitComputeLayout.$.offsets[i]);
});

const orbitVerticiesBuffer = root
  .createBuffer(d.arrayOf(d.vec3f, ORBIT_POINTS * INITIAL_BODIES.length))
  .$usage("storage", "uniform");
const orbitComputeOffsetsBuffer = root
  .createBuffer(d.arrayOf(d.vec3f, INITIAL_BODIES.length))
  .$usage("storage", "uniform");
const orbitComputeVelocitiesBuffer = root
  .createBuffer(d.arrayOf(d.vec3f, INITIAL_BODIES.length))
  .$usage("storage", "uniform");
const orbitPointIndexBuffer = root.createBuffer(d.i32).$usage("storage", "uniform");

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

const orbitComputeBindGroup = root.createBindGroup(orbitComputeLayout, {
  orbitPointIndex: orbitPointIndexBuffer,
  offsets: orbitComputeOffsetsBuffer,
  velocities: orbitComputeVelocitiesBuffer,
  vertecies: orbitVerticiesBuffer,
  initialVelocities: velocitiesBuffer,
  initialOffsets: offsetsBuffer,
  masses: massesBuffer,
  gravityMultiplier: gravityMultiplierBuffer,
});

function predictOrbits() {
  for (let i = 0; i < ORBIT_POINTS; i++) {
    orbitPointIndexBuffer.patch(i);
    orbitComputeVelocityPipeline.with(orbitComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
    orbitComputeOffsetPipeline.with(orbitComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
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

const computeCameraPositionPipeline = root.createGuardedComputePipeline(() => {
  "use gpu";
  const attachedObjectIndex = attachedObjectIndexUniform.$;

  if (attachedObjectIndex === -1) {
    return;
  }

  const currentCameraPosition = cameraMutable.$.pos.xyz;
  const attachedObjectPosition = computeLayout.$.offsets[attachedObjectIndex];

  cameraMutable.$.pos = d.vec4f(currentCameraPosition + attachedObjectPosition, 1);
});

function render() {
  updatePosition();

  bodiesVelocityPipeline.with(mainComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
  bodiesOffsetPipeline.with(mainComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
  computeCameraPositionPipeline.with(mainComputeBindGroup).dispatchThreads();

  if (frame % (ORBIT_POINTS / 2) === 0) {
    predictOrbits();
  }

  data.forEach(async (item, i) => {
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
