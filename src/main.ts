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
import { BODY_COUNT_CONST, CelestianBody, INITIAL_BODIES } from "./data/simulation-data";
import { ATTACHED_BODY_INDEX, DEBUG_NORMALS, DEBUG_SHADOWS, NORMAL_OFFSET, ORBIT_PREDICTION_STEPS, RENDER_ORBITS, SetAttachedBody, SHOW_DEPTH_CUBE, SIMULATION_SPEED } from "./data/settings";
import { PrepareBloom } from "./postprocessing/bloom";
import { bodiesToArrays, getBodyRotationSpeedInAngle, PrepareSimulation } from "./simulation";
import { PrepareUI } from "./ui-controls";
import { writeSoA } from "typegpu/common";
import { PrepareShadows } from "./shadows";
import { sample } from "./cpuPerlin";
import * as m from "wgpu-matrix";
import { PrepareAtmosphere } from "./atmosphere";

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<main>
  <canvas id="canvas" width="2560" height="1440"></canvas>
  <p id="veloasdasdadadada"></p>
</main>
`;

const ui = PrepareUI();

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
    speed: d.vec3f(0.001, 0.01, 5),
  },
  (props) => {
    cameraUniform.patch(props);
  },
);

const positionVertexLayout = tgpu.vertexLayout(d.disarrayOf(d.float16x4));
const normalVertexLayout = tgpu.vertexLayout(d.disarrayOf(d.float16x4));
const normalDebugVertexLayout = tgpu.vertexLayout(d.disarrayOf(d.float16x4));

const mainBindGroupLayout = tgpu.bindGroupLayout({
  positions: { storage: d.arrayOf(d.f32), access: "readonly" },
  velocities: { storage: d.arrayOf(d.f32), access: "readonly" },
  rotationMatricies: { storage: d.arrayOf(d.mat4x4f), access: "readonly" },
});

const shadowsLayout = tgpu.bindGroupLayout({
  texture: { texture: d.textureDepthCube() },
  sampler: { sampler: "comparison" }
});

const debugShadowsUniform = root.createUniform(d.i32);
const normalOffsetUniform = root.createUniform(d.f32);

let bodies = INITIAL_BODIES;
const bodiesUniform = root.createUniform(d.arrayOf(CelestianBody, INITIAL_BODIES.length));
const bodiesPositionsBuffer = root.createBuffer(d.arrayOf(d.f32, INITIAL_BODIES.length * 3)).$usage("storage");
const bodiesVelocitiesBuffer = root.createBuffer(d.arrayOf(d.f32, INITIAL_BODIES.length * 3)).$usage("storage");
const bodiesRotationMatriciesBuffer = root.createBuffer(d.arrayOf(d.mat4x4f, INITIAL_BODIES.length)).$usage("storage");
const positionsArray = new Float32Array(INITIAL_BODIES.length * 3); // CPU BUFFER
const velocitiesArray = new Float32Array(INITIAL_BODIES.length * 3); // CPU BUFFER
const currentRotationArray = new Float32Array(INITIAL_BODIES.length); // CPU BUFFER
const rotationMatricesArray = new Float32Array(INITIAL_BODIES.length * 16); // CPU BUFFER

const mainBindGroup = root.createBindGroup(mainBindGroupLayout, {
  positions: bodiesPositionsBuffer,
  velocities: bodiesVelocitiesBuffer,
  rotationMatricies: bodiesRotationMatriciesBuffer,
});

const currentBodyIndexUniform = root.createUniform(d.u32);

const bodiesRenderData: {
  verticies: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag & VertexFlag;
  normals: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag & VertexFlag;
  trickVerticies: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
  trickNormals: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
  debugNormalVerticies: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
  trickDebugNormalVerticies: TgpuBuffer<d.Disarray<d.float16x4>> & VertexFlag;
}[] = [];

export function SetUpBodiesRenderData() {
  bodiesUniform.write(bodies);
  const { positions: initialPositionsArray, velocities: initialVelocitiesArray } = bodiesToArrays(bodies);
  positionsArray.set(initialPositionsArray);
  velocitiesArray.set(initialVelocitiesArray);
  bodiesPositionsBuffer.write(initialPositionsArray);
  bodiesVelocitiesBuffer.write(initialVelocitiesArray);
  debugShadowsUniform.write(DEBUG_SHADOWS ? 1 : 0);
  normalOffsetUniform.write(NORMAL_OFFSET);

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
  .$usage("render", "sampled");

const mainRenderPipeline = root.createRenderPipeline({
  attribs: { inVertex: positionVertexLayout.attrib, inNormal: normalVertexLayout.attrib },
  vertex: tgpu.vertexFn({
    in: { vid: d.builtin.vertexIndex, inVertex: d.vec4f, inNormal: d.vec4f },
    out: {
      position: d.builtin.position,
      point: d.vec3f,
      cameraPos: d.interpolate("flat", d.vec3f),
      emits: d.interpolate("flat", d.i32),
      groundColor: d.vec4f,
      normal: d.vec3f,
      vid: d.interpolate("flat", d.u32),
      bodyId: d.interpolate("flat", d.u32),
    },
  })(({ vid, inVertex, inNormal }) => {
    "use gpu";
    const bodyIndex = currentBodyIndexUniform.$;
    const body = bodiesUniform.$[bodyIndex];
    const camera = cameraUniform.$;
    const offset = d.vec3f(
      mainBindGroupLayout.$.positions[bodyIndex * 3],
      mainBindGroupLayout.$.positions[bodyIndex * 3 + 1],
      mainBindGroupLayout.$.positions[bodyIndex * 3 + 2],
    );
    const rotationMatrix = mainBindGroupLayout.$.rotationMatricies[bodyIndex];

    const vertex = inVertex.xyz;
    const normal = inNormal.xyz;

    const rotatedNormal = rotationMatrix.mul(d.vec4f(normal, 1)).xyz;

    const rotatedPoint = rotationMatrix.mul(d.vec4f(vertex, 1)).xyz;
    const finalPoint = rotatedPoint.mul(body.radius).add(offset);
    const position = camera.projection.mul(camera.view).mul(d.vec4f(finalPoint, 1));

    const height = std.length(vertex);
    const colors = body.colors;
    let groundColor = d.vec4f(body.colors[0].color);

    for (let i = 0; i < colors.length; i++) {
      if (height >= colors[i].height) {
        groundColor = d.vec4f(colors[i].color);
      }
    }

    let emits = 0;
    if (currentBodyIndexUniform.$ === 0) {
      emits = 1;
    }

    return {
      position: position,
      groundColor,
      normal: rotatedNormal,
      vid,
      point: finalPoint,
      emits,
      cameraPos: camera.pos.xyz,
      bodyId: currentBodyIndexUniform.$,
    };
  }),
  fragment: ({ $position, groundColor, normal, vid, point, emits, cameraPos, bodyId }) => {
    "use gpu";
    if (emits === 1) {
      return {
        color: groundColor,
        emission: d.vec4f(1, 1, 1, 1),
      };
    }

    const sunPosition = d.vec3f(
      mainBindGroupLayout.$.positions[0],
      mainBindGroupLayout.$.positions[1],
      mainBindGroupLayout.$.positions[2],
    )

    const surfaceToLightDirection = std.normalize(sunPosition.sub(point));
    const diffuse = std.max(0, std.dot(normal, surfaceToLightDirection));

    const reflectionDirection = std.reflect(surfaceToLightDirection.mul(-1), normal.xyz);
    const surfaceToViewDirection = std.normalize(cameraPos.sub(point));
    const specular = std.pow(std.max(0, std.dot(reflectionDirection, surfaceToViewDirection)), 150) * 0.5;

    const normalOffset = normalOffsetUniform.$;

    const toLight = point.add(normal.mul(normalOffset).mul(std.dot(normal, surfaceToLightDirection))).sub(sunPosition);
    const dist = std.length(toLight);
    const normalMove = 1;
    const lightDir = toLight.add(normal.mul(normalMove / 1000)).div(dist).mul(d.vec3f(-1, 1, 1));
    const depthRef = (dist - normalOffset) / 1000;

    let inShadow = std.textureSampleCompareLevel(shadowsLayout.$.texture, shadowsLayout.$.sampler, lightDir, depthRef);
    inShadow = 0.4 + inShadow * 0.6; // ambient term
    let finalColor = (groundColor * (0.4 + diffuse * 0.6) + specular) * inShadow;

    if (debugShadowsUniform.$ === 1) {
      finalColor = d.vec4f(d.vec3f(inShadow), 1);
    }

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
    color: { format: "rgba8unorm" },
    emission: { format: "rgba8unorm" },
  },
});

const normalDebugPipeline = root.createRenderPipeline({
  attribs: { inNormal: normalDebugVertexLayout.attrib },
  vertex: tgpu.vertexFn({
    in: { vid: d.builtin.vertexIndex, inNormal: d.vec4f },
    out: {
      position: d.builtin.position,
    },
  })(({ vid, inNormal }) => {
    "use gpu";
    const bodyIndex = currentBodyIndexUniform.$;
    const body = bodiesUniform.$[bodyIndex];
    const camera = cameraUniform.$;

    const offset = d.vec3f(
      mainBindGroupLayout.$.positions[bodyIndex * 3],
      mainBindGroupLayout.$.positions[bodyIndex * 3 + 1],
      mainBindGroupLayout.$.positions[bodyIndex * 3 + 2],
    );
    const rotationMatrix = mainBindGroupLayout.$.rotationMatricies[bodyIndex];

    const normal = inNormal.xyz;
    const rotatedNormal = rotationMatrix.mul(d.vec4f(normal, 1)).xyz;

    const point = rotatedNormal.mul(body.radius).add(offset);
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

  const attachedBodyPosition = d.vec3f(
    positionsArray[ATTACHED_BODY_INDEX * 3],
    positionsArray[ATTACHED_BODY_INDEX * 3 + 1],
    positionsArray[ATTACHED_BODY_INDEX * 3 + 2],
  );

  camera.setPosition(attachedBodyPosition.sub(d.vec3f(0, 0, bodies[ATTACHED_BODY_INDEX].radius * 3)));
}

function moveCameraWithAttachedObjectVelocity(collisionInfo: {
  collides: boolean;
  distance: number;
  normal: d.v3f;
}) {
  if (ATTACHED_BODY_INDEX === -1) return;


  const attachedBodyVelocity = d.vec3f(
    velocitiesArray[ATTACHED_BODY_INDEX * 3],
    velocitiesArray[ATTACHED_BODY_INDEX * 3 + 1],
    velocitiesArray[ATTACHED_BODY_INDEX * 3 + 2],
  );

  const cameraPosition = camera.state.pos.add(attachedBodyVelocity.mul(SIMULATION_SPEED));

  if (collisionInfo.distance === -1 || collisionInfo.distance > bodies[ATTACHED_BODY_INDEX].radius + 2) {
    camera.setPosition(cameraPosition);
    return;
  }

  // pozycja stara (przed przesunieciem symulacji tej klatki)
  const attachedBodyPos = d.vec3f(
    positionsArray[ATTACHED_BODY_INDEX * 3],
    positionsArray[ATTACHED_BODY_INDEX * 3 + 1],
    positionsArray[ATTACHED_BODY_INDEX * 3 + 2],
  ).add(attachedBodyVelocity.mul(-SIMULATION_SPEED));

  // to samo niezaleznie od rotacij
  const direction = camera.state.pos.sub(attachedBodyPos);

  // ile sie obroci w tej klatce
  const rotationAngle = getBodyRotationSpeedInAngle(bodies[ATTACHED_BODY_INDEX]);
  const attachedBodyRotationMatrix = m.mat4.rotationY(rotationAngle, d.mat4x4f());

  // direction po rotacji tej klatki
  const rotatedDirection = attachedBodyRotationMatrix.mul(d.vec4f(direction, 1)).xyz;

  const newCameraPos = cameraPosition.sub(direction).add(rotatedDirection);
  camera.setPosition(newCameraPos);
  camera.state.bodyMatrix = attachedBodyRotationMatrix.mul(camera.state.bodyMatrix);
}

function checkForCollision() {
  "use gpu";

  if (ATTACHED_BODY_INDEX === -1) {
    return {
      collides: false,
      distance: -1,
      normal: d.vec3f(1, 1, 1),
      surfaceHeight: -1,
    }
  }

  const attachedBodyPos = d.vec3f(
    positionsArray[ATTACHED_BODY_INDEX * 3],
    positionsArray[ATTACHED_BODY_INDEX * 3 + 1],
    positionsArray[ATTACHED_BODY_INDEX * 3 + 2],
  );

  const perlinOffset = d.vec3f(ATTACHED_BODY_INDEX, 0, 0);

  const planetAngleStep = getBodyRotationSpeedInAngle(bodies[ATTACHED_BODY_INDEX]);
  const planetAngle = currentRotationArray[ATTACHED_BODY_INDEX] + planetAngleStep;
  const reverseRotationMatrix = m.mat4.rotationY(-planetAngle, d.mat4x4f());
  const rotationMatrix = m.mat4.rotationY(planetAngle, d.mat4x4f());

  const direction = camera.state.pos.sub(attachedBodyPos);
  const normalizedDirection = std.normalize(direction);
  const distance = std.length(direction);

  const pointOnInitialSphere = reverseRotationMatrix.mul(d.vec4f(normalizedDirection, 1)).xyz;

  const perlinValue = 1 + (sample(pointOnInitialSphere + perlinOffset) * sphere.strength);
  const pointWithPerlin = pointOnInitialSphere * perlinValue;

  const ex = d.vec3f(sphere.epsilon, 0, 0);
  const ey = d.vec3f(0, sphere.epsilon, 0);
  const ez = d.vec3f(0, 0, sphere.epsilon);

  const dhdx = (sample(pointOnInitialSphere + perlinOffset + ex) - sample(pointOnInitialSphere + perlinOffset - ex)) / (2 * sphere.epsilon);
  const dhdy = (sample(pointOnInitialSphere + perlinOffset + ey) - sample(pointOnInitialSphere + perlinOffset - ey)) / (2 * sphere.epsilon);
  const dhdz = (sample(pointOnInitialSphere + perlinOffset + ez) - sample(pointOnInitialSphere + perlinOffset - ez)) / (2 * sphere.epsilon);

  const grad = d.vec3f(dhdx, dhdy, dhdz) * sphere.strength;
  const tangentialGrad = grad - (std.dot(grad, pointOnInitialSphere) * pointOnInitialSphere);
  const normal = rotationMatrix.mul(d.vec4f(std.normalize(pointWithPerlin - tangentialGrad), 1)).xyz;

  const surfaceHeight = perlinValue * bodies[ATTACHED_BODY_INDEX].radius;

  if (distance - 0.02 <= surfaceHeight) {
    return {
      collides: true,
      distance,
      normal: normal,
      surfaceHeight,
    };
  }

  return {
    collides: false,
    distance,
    normal: normal,
    surfaceHeight,
  }
}

function findClosestBody() {
  let closestBodyIndex = ATTACHED_BODY_INDEX;
  let smallestDistance = Infinity;
  bodies.forEach((body, i) => {
    const bodyPos = d.vec3f(
      positionsArray[i * 3],
      positionsArray[i * 3 + 1],
      positionsArray[i * 3 + 2],
    );

    const distance = std.length(camera.state.pos.sub(bodyPos));
    if (distance < body.radius + 10 && distance < smallestDistance) {
      smallestDistance = distance;
      closestBodyIndex = i;
    }
  });

  SetAttachedBody(closestBodyIndex, false);
}

const mainTexture = root.createTexture({
  size: [canvas.width, canvas.height, 1],
  format: "rgba8unorm",
})
  .$usage("render", "sampled");


const simulation = PrepareSimulation(root, canvas, context, cameraUniform, bodies, rotationMatricesArray, bodiesRotationMatriciesBuffer, currentRotationArray);
const shadows = PrepareShadows(root, canvas, context, positionVertexLayout, bodiesRenderData, cameraUniform, bodiesUniform, mainBindGroupLayout, mainBindGroup, positionsArray, 0);
const bloomEffect = PrepareBloom(root, canvas, context, pixelScaleUniform);
const atmosphere = PrepareAtmosphere(root, canvas, context, cameraUniform, bodies, bodiesUniform, bodiesPositionsBuffer, depthTexture, mainTexture);

const shadowsBindGroup = root.createBindGroup(shadowsLayout, {
  sampler: shadows.sampler,
  texture: shadows.shadowMap,
});

export function ReloadSettings() {
  debugShadowsUniform.write(DEBUG_SHADOWS ? 1 : 0);
  normalOffsetUniform.write(NORMAL_OFFSET);
  shadows.reloadSettings();
  atmosphere.reloadSettings();
}

function render() {
  camera.updatePosition();

  findClosestBody();
  const collisionInfo = checkForCollision();

  // simulation.pullTowardsAttachedBody(positionsArray, camera, collisionInfo);

  simulation.simulateGravity(positionsArray, velocitiesArray, bodies);
  simulation.simulateRotation();

  bodiesPositionsBuffer.write(positionsArray);
  bodiesVelocitiesBuffer.write(velocitiesArray);

  if (collisionInfo.distance !== -1 && collisionInfo.distance < bodies[ATTACHED_BODY_INDEX].radius + 2) {
    camera.setUp(collisionInfo.normal);
  }
  else {
    camera.setUp(d.vec3f(0, 1, 0));
  }

  moveCameraWithAttachedObjectVelocity(collisionInfo);

  if (collisionInfo.collides) {
    const upVector = camera.state.bodyMatrix.columns[1].xyz;
    camera.setPosition(camera.state.pos.add(upVector.mul(collisionInfo.surfaceHeight - collisionInfo.distance + 0.02)));
  }

  if (frame % (ORBIT_PREDICTION_STEPS / 2) === 0) {
    simulation.predictOrbits(positionsArray, velocitiesArray, bodies);
  }

  bodiesRenderData.forEach((item, i) => {
    currentBodyIndexUniform.write(i);

    mainRenderPipeline
      .withColorAttachment({
        color: {
          view: mainTexture,
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
      .with(mainBindGroup)
      .with(positionVertexLayout, item.trickVerticies)
      .with(normalVertexLayout, item.trickNormals)
      .with(shadowsBindGroup)
      .draw(sphere.getVertexAmount(), 1);
  });


  atmosphere.render();

  // bloomEffect.applyGausianBlur();
  // bloomEffect.render();

  // if (RENDER_ORBITS) {
  //   simulation.renderOrbits(bodies);
  // }

  // if (DEBUG_NORMALS) {
  //   bodiesRenderData.forEach((item, i) => {
  //     currentBodyIndexUniform.write(i);
  //     normalDebugPipeline
  //       .withColorAttachment({ view: context, loadOp: "load", clearValue: { r: 0, g: 0, b: 0, a: 1 } })
  //       .withDepthStencilAttachment({
  //         view: depthTexture,
  //         depthLoadOp: "load",
  //         depthClearValue: 1,
  //         depthStoreOp: "store",
  //       })
  //       .with(mainBindGroup)
  //       .with(normalDebugVertexLayout, item.trickDebugNormalVerticies)
  //       .draw(sphere.getVertexAmount() * 2, 1);
  //   });
  // }

  // shadows.renderShadowMaps();


  if (SHOW_DEPTH_CUBE) {
    shadows.debugRender();
  }


  frame++;
  requestAnimationFrame(render);
}

ui.SetUpControls();
SetUpBodiesRenderData();
SetAttachedBody(3); // attach to earth

requestAnimationFrame(render);
