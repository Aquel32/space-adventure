import * as m from "wgpu-matrix";
import { d, std } from "typegpu";

export const Camera = d.struct({
  pos: d.vec4f,
  view: d.mat4x4f,
  projection: d.mat4x4f,
  viewInverse: d.mat4x4f,
  projectionInverse: d.mat4x4f,
});

export interface CameraOptions {
  initPos?: d.v3f;
  /**
   * Scrolling accelerates/decelerates the movement.
   * `d.vec3f(minimum, initial, maximum)`
   */
  speed?: d.v3f;
}

const cameraDefaults: Partial<CameraOptions> = {
  initPos: d.vec3f(0, 0, 0),
  speed: d.vec3f(1, 1, 1),
};

/**
 * Sets up a first person camera.
 * Calls the callback on scroll events, canvas clicks/touches and resizes.
 * Also, calls the callback during the setup with an initial camera.
 */
export function setupFirstPersonCamera(
  canvas: HTMLCanvasElement,
  partialOptions: CameraOptions,
  callback: (updatedProps: Partial<d.Infer<typeof Camera>>) => void,
) {
  const options = { ...cameraDefaults, ...partialOptions } as Required<CameraOptions>;

  // `runCallback` creates a Camera object based on the `cameraState` and passes it to the callback
  const cameraState = {
    pos: options.initPos,
    yaw: 0,
    pitch: 0,
    bodyMatrix: m.mat4.identity(d.mat4x4f()),
  };

  function runCallback() {
    const position = cameraState.pos;
    const pitch = cameraState.pitch;

    const headMatrix = m.mat4.axisRotate(cameraState.bodyMatrix, d.vec3f(1, 0, 0), pitch, d.mat4x4f());

    const translationMatrix = m.mat4.translation(position, d.mat4x4f());
    const viewInverse = m.mat4.mul(translationMatrix, headMatrix, d.mat4x4f());
    const projection = calculateProj(canvas.clientWidth / canvas.clientHeight);

    callback(
      Camera({
        pos: d.vec4f(position, 1),
        view: invertMat(viewInverse),
        projection,
        viewInverse: viewInverse,
        projectionInverse: invertMat(projection),
      }),
    );
  }

  function setUp(newUp: d.v3f) {
    const lastUp = cameraState.bodyMatrix.columns[1].xyz;

    if (std.abs(std.dot(lastUp, newUp)) > 0.999 && !std.allEq(newUp, d.vec3f(0, 1, 0))) {
      // handle going head first towards a planet
      return;
    }

    const axis = std.normalize(std.cross(lastUp, newUp));

    // clamp the dot product to ensure value is within the valid range for acos
    const angle = Math.acos(std.clamp(std.dot(lastUp, newUp), -1, 1)) * 0.1;

    const rotationMat = m.mat4.axisRotation(axis, angle, d.mat4x4f());
    m.mat4.mul(rotationMat, cameraState.bodyMatrix, cameraState.bodyMatrix);

    runCallback();
  }

  function rotateCamera(dx: number, dy: number) {
    const orbitSensitivity = 0.005;
    cameraState.yaw -= dx * orbitSensitivity;
    cameraState.pitch -= dy * orbitSensitivity;
    cameraState.pitch = std.clamp(cameraState.pitch, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);

    // const upVector = cameraState.bodyMatrix.columns[1].xyz;
    const upVector = d.vec3f(0, 1, 0);
    m.mat4.axisRotate(cameraState.bodyMatrix, upVector, -dx * orbitSensitivity, cameraState.bodyMatrix);

    runCallback();
  }

  function rotateCameraByAngle(angle: number, upVector: d.v3f = cameraState.bodyMatrix.columns[1].xyz) {
    m.mat4.axisRotate(cameraState.bodyMatrix, upVector, angle, cameraState.bodyMatrix);
    runCallback();
  }

  // resize observer
  const resizeObserver = new ResizeObserver(() => {
    runCallback();
  });
  resizeObserver.observe(canvas);

  // Variables for interaction.
  const pressedKeys = new Set<string>();
  let moveSpeed = options.speed.y;

  // keyboard events
  const keyDownEventListener = (event: KeyboardEvent) => {
    pressedKeys.add(event.key.toLowerCase());
  };
  window.addEventListener("keydown", keyDownEventListener);

  const keyUpEventListener = (event: KeyboardEvent) => {
    pressedKeys.delete(event.key.toLowerCase());
  };
  window.addEventListener("keyup", keyUpEventListener);

  // mouse events
  canvas.addEventListener("mousedown", () => {
    void canvas.requestPointerLock();
  });

  canvas.addEventListener("mousemove", (event: MouseEvent) => {
    if (document.pointerLockElement !== canvas) {
      return;
    }
    const dx = event.movementX;
    const dy = event.movementY;
    rotateCamera(dx, dy);
  });

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      moveSpeed = std.clamp(moveSpeed * (1 - e.deltaY * 0.0005), options.speed.x, options.speed.z);
    },
    { passive: false },
  );

  function cleanupCamera() {
    window.removeEventListener("keydown", keyDownEventListener);
    window.removeEventListener("keyup", keyUpEventListener);
    resizeObserver.unobserve(canvas);
  }

  // update position function
  const updatePosition = () => {
    const up = cameraState.bodyMatrix.columns[1].xyz.mul(-moveSpeed);
    const forward = cameraState.bodyMatrix.columns[2].xyz.mul(-moveSpeed);
    const left = cameraState.bodyMatrix.columns[0].xyz.mul(-moveSpeed);

    if (pressedKeys.has("w")) {
      cameraState.pos = cameraState.pos.add(forward);
    }
    if (pressedKeys.has("s")) {
      cameraState.pos = cameraState.pos.sub(forward);
    }
    if (pressedKeys.has("a")) {
      cameraState.pos = cameraState.pos.add(left);
    }
    if (pressedKeys.has("d")) {
      cameraState.pos = cameraState.pos.sub(left);
    }
    if (pressedKeys.has("shift")) {
      cameraState.pos = cameraState.pos.add(up);
    }
    if (pressedKeys.has(" ")) {
      cameraState.pos = cameraState.pos.sub(up);
    }
    runCallback();
  };

  const setPosition = (newPosition: d.v3f) => {
    cameraState.pos = newPosition;
    runCallback();
  };

  runCallback();
  return { state: cameraState, cleanupCamera, updatePosition, setPosition, setUp, rotateCameraByAngle };
}

export function calculateView(position: d.v3f, target: d.v3f, up: d.v3f) {
  return m.mat4.lookAt(position, target, up, d.mat4x4f());
}

export function calculateProj(aspectRatio: number, fov: number = Math.PI / 4, near: number = 0.001, far: number = 1000) {
  return m.mat4.perspective(fov, aspectRatio, near, far, d.mat4x4f());
}

function invertMat(matrix: d.m4x4f) {
  return m.mat4.invert(matrix, d.mat4x4f());
}
