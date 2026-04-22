// oxlint-disable-next-line no-unassigned-import
import { cos, select, sin, step } from "typegpu/std";
import { Camera, setupFirstPersonCamera } from "./setup-first-person-camera";
import "./style.css";
import tgpu, {
  common,
  d,
  std,
  type TgpuBindGroup,
  type TgpuBuffer,
  type TgpuBufferMutable,
  type TgpuBufferUniform,
  type TgpuConst,
  type TgpuGuardedComputePipeline,
  type TgpuMutable,
  type TgpuRenderPipeline,
  type TgpuUniform,
} from "typegpu";
import { i32, type v3f, type v4f } from "typegpu/data";
import * as sphere from "./sphere";
import { SetUpControls } from "./ui-controls";
import { CelestianBody, GRAVITY_MULTIPLIER, INITIAL_BODIES } from "./simulation-data";

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<main>
  <canvas id="canvas" width="600" height="600"></canvas>
</main>
`;

const root = await tgpu.init();
let frame = 0;

const gravityMultiplierBuffer = root.createBuffer(d.f32).$usage("storage", "uniform");

export const PIXEL_SCALE_BUFFER = root.createUniform(d.f32);
PIXEL_SCALE_BUFFER.write(1);
export let GAUSIAN_ITERATIONS = d.f32(5);
export function SetBlurIterations(newbi: number) {
  GAUSIAN_ITERATIONS = newbi;
}

const ORBIT_POINTS = 10000;
const ORBIT_POINTS_CONST = tgpu.const(d.i32, ORBIT_POINTS);

const BODY_COUNT_CONST = tgpu.const(d.i32, INITIAL_BODIES.length);

SetUpControls();

const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const context = root.configureContext({ canvas });

const cameraUniform = root.createUniform(Camera);
const { updatePosition } = setupFirstPersonCamera(
  canvas,
  {
    initPos: d.vec3f(100, 0, 0),
    speed: d.vec3f(0.01, 0.1, 5),
  },
  (props) => {
    cameraUniform.patch(props);
  },
);

let depthTexture = root
  .createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "depth24plus",
  })
  .$usage("render");

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

const orbitRenderLayout = tgpu.bindGroupLayout({
  currentBodyIndex: { storage: d.i32, access: "readonly" },
  vertecies: { storage: d.arrayOf(d.vec3f), access: "readonly" },
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

const mainRenderBindGroup = root.createBindGroup(mainLayout, {
  bodies: bodiesBuffer,
  masses: massesBuffer,
  offsets: offsetsBuffer,
  velocities: velocitiesBuffer,
  currentBodyIndex: currentBodyIndexBuffer,
  gravityMultiplier: gravityMultiplierBuffer,
});

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
  verticies: TgpuConst<d.WgslArray<d.Vec4f>>;
  orbitRenderPipeline: TgpuRenderPipeline<d.Vec4f>;
}[] = [];

const SPHERE_DIVISIONS = 10;

export function SetUpBuffersAndData() {
  gravityMultiplierBuffer.write(GRAVITY_MULTIPLIER);

  massesBuffer.write(INITIAL_BODIES.map((b) => b.mass));
  offsetsBuffer.write(INITIAL_BODIES.map((b) => b.position));
  velocitiesBuffer.write(INITIAL_BODIES.map((b) => b.initialVelocity));
  currentBodyIndexBuffer.write(0);

  data.length = 0;

  INITIAL_BODIES.forEach((body, i) => {
    const mainRenderPipeline = root.createRenderPipeline({
      vertex: tgpu.vertexFn({
        in: { vid: d.builtin.vertexIndex },
        out: {
          position: d.builtin.position,
          point: d.vec3f,
          cameraPos: d.interpolate("flat", d.vec3f),
          emits: d.interpolate("flat", d.i32),
          sunPosition: d.vec3f,
          groundColor: d.vec4f,
          normal: d.vec4f,
          vid: d.interpolate("flat", d.i32),
        },
      })(({ vid }) => {
        "use gpu";
        const camera = cameraUniform.$;
        const offset = mainLayout.$.offsets[mainLayout.$.currentBodyIndex];
        const normal = verticies.$[vid];
        const point = normal.xyz.mul(body.radius).add(offset);
        const position = camera.projection.mul(camera.view).mul(d.vec4f(point, 1));
        const sunPosition = mainLayout.$.offsets[0];

        let emits = 0;
        if (mainLayout.$.currentBodyIndex === 0) {
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
          cameraPos: camera.pos.xyz,
        };
      }),
      fragment: ({ groundColor, normal, vid, sunPosition, point, emits, cameraPos }) => {
        "use gpu";
        if (emits === 1) {
          return {
            color: groundColor,
            emission: d.vec4f(1, 1, 1, 1),
          };
        }

        const surfaceToLightDirection = std.normalize(sunPosition.sub(point));
        const light = std.dot(normal.xyz, surfaceToLightDirection);
        // console.log(light);
        const surfaceToViewDirection = std.normalize(cameraPos.sub(point));
        const halfVector = std.normalize(surfaceToLightDirection.add(surfaceToViewDirection));
        let specular = std.dot(normal.xyz, halfVector);
        specular = select(0.0, std.pow(specular, 150), specular > 0);
        const finalColor = groundColor.mul(light) + specular;

        let emission = d.vec4f(0, 0, 0, 1);
        const treshold = 0.8;
        if (finalColor.r > treshold || finalColor.g > treshold || finalColor.b > treshold) {
          const val = (light + specular - treshold) / (treshold);
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
        const bodyIndex = orbitRenderLayout.$.currentBodyIndex;
        const camera = cameraUniform.$;
        const point = orbitRenderLayout.$.vertecies[bodyIndex * ORBIT_POINTS_CONST.$ + vid];
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
    });

    const verticies = tgpu.const(
      d.arrayOf(d.vec4f, sphere.getVertexAmount(SPHERE_DIVISIONS)),
      sphere.generateSphere(SPHERE_DIVISIONS),
    );
    data.push({ mainRenderPipeline, verticies, orbitRenderPipeline });
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

const orbitRenderBindGroup = root.createBindGroup(orbitRenderLayout, {
  currentBodyIndex: currentBodyIndexBuffer,
  vertecies: orbitVerticiesBuffer,
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

const postProccessSampler = root.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
});

const emmisionTexture = root
  .createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba8unorm",
    mipLevelCount: 3
  })
  .$usage("render", "sampled");

const postProccessTarget = root
  .createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba8unorm",
  })
  .$usage("render", "sampled");

const postProccessBindGroupLayout = tgpu.bindGroupLayout({
  targetTexture: { texture: d.texture2d() },
  emissionTexture: { texture: d.texture2d() },
  sampler: { sampler: "filtering" },
});

const postProccessBindGroup = root.createBindGroup(postProccessBindGroupLayout, {
  targetTexture: postProccessTarget,
  emissionTexture: emmisionTexture,
  sampler: postProccessSampler,
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
    // const weights = [0.227027, 0.1945946, 0.16, 0.1216216, 0.09, 0.07, 0.054054, 0.04, 0.03, 0.02, 0.016216, 0.012, 0.01, 0.007];
    const weights = [0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216];

    let pixelSize = 1.0 / std.textureDimensions(postProccessBindGroupLayout.$.emissionTexture).x;
    pixelSize *= PIXEL_SCALE_BUFFER.$;

    let result = std
      .textureSampleLevel(
        postProccessBindGroupLayout.$.emissionTexture,
        postProccessBindGroupLayout.$.sampler,
        uv,
        2
      )

    let sum = d.f32(0);

    result *= weights[0];
    for (let i = -weights.length + 1; i < weights.length; i++) {
      for (let j = -weights.length + 1; j < weights.length; j++) {
        sum += weights[std.abs(i)] * weights[std.abs(j)];

        if (i === 0 && j === 0) continue;

        result += std
          .textureSampleLevel(
            postProccessBindGroupLayout.$.emissionTexture,
            postProccessBindGroupLayout.$.sampler,
            uv.add(d.vec2f(pixelSize * i, pixelSize * j)),
            2
          )
          .mul(weights[std.abs(i)] * weights[std.abs(j)]);
      }
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
    mipLevelCount: 3,
  })
  .$usage("render", "sampled");

const currentBlurPassBindGroup = root.createBindGroup(postProccessBindGroupLayout, {
  targetTexture: postProccessTarget,
  emissionTexture: currentBlurPassTarget,
  sampler: postProccessSampler,
});

function gausianBlur(passes: number) {
  emmisionTexture.generateMipmaps();

  for (let i = 0; i < passes; i++) {
    const target = i % 2 === 0 ? currentBlurPassTarget : emmisionTexture;

    blurRenderPipeline
      .withColorAttachment({
        view: target.createView("render", { mipLevelCount: 1, baseMipLevel: 2 }),
        loadOp: "load",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      })
      .with(i % 2 === 0 ? postProccessBindGroup : currentBlurPassBindGroup)
      .draw(6, 1);
  }

  return passes % 2 === 1 ? currentBlurPassBindGroup : postProccessBindGroup;
}

const finalRenderPipeline = root.createRenderPipeline({
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
      2
    );
  },
  // targets: {
  //   blend: {
  //     color: {
  //       srcFactor: "one",
  //       dstFactor: "one",
  //       operation: "add",
  //     },
  //     alpha: {
  //       srcFactor: "one",
  //       dstFactor: "one",
  //       operation: "add",
  //     }
  //   },
  // }
});

function render() {
  // bodiesVelocityPipeline.with(mainComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);
  // bodiesOffsetPipeline.with(mainComputeBindGroup).dispatchThreads(INITIAL_BODIES.length);

  if (frame % (ORBIT_POINTS / 2) === 0) {
    // predictOrbits();
  }

  data.forEach(async (item, i) => {
    currentBodyIndexBuffer.write(i);

    item.mainRenderPipeline
      .withColorAttachment({
        color: {
          view: context,
          loadOp: i === 0 ? "clear" : "load",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
        emission: { view: emmisionTexture.createView("render", { mipLevelCount: 1, baseMipLevel: 0 }), loadOp: i === 0 ? "clear" : "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      })
      .withDepthStencilAttachment({
        view: depthTexture,
        depthLoadOp: i === 0 ? "clear" : "load",
        depthClearValue: 1,
        depthStoreOp: "store",
      })
      .with(mainRenderBindGroup)
      .draw(item.verticies.$.length, 1);

    // item.orbitRenderPipeline.
    //   withColorAttachment({ view: context, loadOp: i === 0 ? "clear" : "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } }).
    //   // withDepthStencilAttachment({
    //   //   view: depthTexture,
    //   //   depthLoadOp: "load",
    //   //   depthClearValue: 1,
    //   //   depthStoreOp: "store"
    //   // }).
    //   with(orbitRenderBindGroup).
    //   draw(ORBIT_POINTS, 1);
  });

  const finalBindGroup = gausianBlur(GAUSIAN_ITERATIONS);

  finalRenderPipeline
    .withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } })
    .with(finalBindGroup)
    .draw(6, 1);

  updatePosition();

  frame++;
  requestAnimationFrame(render);
}

requestAnimationFrame(render);
