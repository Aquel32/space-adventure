import { DEBUG_NORMALS, DEBUG_SHADOWS, DEPTH_BIAS, GAUSIAN_ITERATIONS, GRAVITY_MULTIPLIER, NORMAL_OFFSET, PIXEL_SCALE, RENDER_ORBITS, SetAttachedBody, SetDebugNormals, SetDebugShadows, SetDepthBias, SetGausianIterations, SetGravityMultiplier, SetNormalOffset, SetPixelScale, SetRenderOrbits, SetShowDepthCube, SetSimulationSpeed, SHOW_DEPTH_CUBE, SIMULATION_SPEED } from "./data/settings";
import { INITIAL_BODIES } from "./data/simulation-data";
import { ReloadSettings, SetUpBodiesRenderData } from "./main";
import { SetEpsilon, SetStrength } from "./sphere";

export function PrepareUI() {
  let controlsSetUp = false;

  document.querySelector("main")!.innerHTML += `<section id="controls">
        <div class="main-controls">
          <p>simulation</p>
          <label>Gravity Multiplier: <input name="gravity" type="number" class="g reload" value="${GRAVITY_MULTIPLIER}" /></label>
          <label>Simulation Speed: <input name="simulation-speed" type="number" class="ss" value="${SIMULATION_SPEED}" /></label>
          <label>Attached Body: <input name="attached-body" type="number" class="ab" value="${3}" /></label>
          <p>bloom</p>
          <label>Gaussian Iterations: <input name="gaussian-iterations" type="number" class="bi" value="${GAUSIAN_ITERATIONS}" /></label>
          <label>Pixel Scale: <input name="pixel-scale" type="number" class="ps" value="${PIXEL_SCALE}" /></label>
          <p>orbit prediction</p>
          <label>Render Orbits: <input name="render-orbits" type="checkbox" class="ro" ${RENDER_ORBITS ? "checked" : ""} /></label>
          <p>sphere</p>
          <label>Perlin Strength: <input name="strength" type="number" class="str reload" value="${0.1}" /></label>
          <label>Epsilon: <input name="epsilon" type="number" class="eps reload" value="${0.001}" /></label>
          <label>Debug Normals: <input name="debug-normals" type="checkbox" class="dn" ${DEBUG_NORMALS ? "checked" : ""} /></label>
          <p>shadows</p>
          <label>Debug Shadows: <input name="debug-shadows" type="checkbox" class="ds" ${DEBUG_SHADOWS ? "checked" : ""} /></label>
          <label>Show Depth Cube: <input name="show-depth-cube" type="checkbox" class="sdc" ${SHOW_DEPTH_CUBE ? "checked" : ""} /></label>
          <label>Depth Bias: <input name="depth-bias" type="number" class="db" value="${DEPTH_BIAS}" /></label>
           <label>Normal Offset: <input name="normal-offset" type="number" class="no" value="${NORMAL_OFFSET}" /></label>
          </div>
        <div class="body-controls">

        </div>
    <section>`;

  // document.querySelector(".body-controls")!.innerHTML += INITIAL_BODIES.map(
  //   (body, i) => `
  //           <div class="body">
  //               <h2>Body ${i}</h2>
  //               <label>Mass: <input type="number" class="mass" value="${body.mass}" /></label>
  //               <label>Radius: <input type="number" class="radius" value="${body.radius}" /></label>
  //               <label>Initial Position: 
  //               <input type="number" class="position-x" value="${body.position.x}" step="0.1" />
  //               <input type="number" class="position-y" value="${body.position.y}" step="0.1" />
  //               <input type="number" class="position-z" value="${body.position.z}" step="0.1" />
  //               </label>
  //               <label>Initial Velocity: 
  //               <input type="number" class="velocity-x" value="${body.initialVelocity.x}" step="0.01" />
  //               <input type="number" class="velocity-y" value="${body.initialVelocity.y}" step="0.01" />
  //               <input type="number" class="velocity-z" value="${body.initialVelocity.z}" step="0.01" />
  //               </label>
  //           </div>
  //       `,
  // ).join("");

  function SetUpControls() {
    if (controlsSetUp) return;

    document.querySelector(".g")!.addEventListener("change", (e) => {
      const newGravityMultiplier = parseFloat((e.target as HTMLInputElement).value);
      SetGravityMultiplier(newGravityMultiplier);
    });

    document.querySelector(".ss")!.addEventListener("change", (e) => {
      const newSimulationSpeed = parseFloat((e.target as HTMLInputElement).value);
      SetSimulationSpeed(newSimulationSpeed);
    });

    document.querySelector(".bi")!.addEventListener("change", (e) => {
      const newGausianIterations = parseFloat((e.target as HTMLInputElement).value);
      SetGausianIterations(newGausianIterations);
    });

    document.querySelector(".ps")!.addEventListener("change", (e) => {
      const newPixelScale = parseFloat((e.target as HTMLInputElement).value);
      SetPixelScale(newPixelScale);
    });

    document.querySelector(".ab")!.addEventListener("change", (e) => {
      const newAttachedBodyIndex = parseFloat((e.target as HTMLInputElement).value);
      SetAttachedBody(newAttachedBodyIndex);
    });

    document.querySelector(".ro")!.addEventListener("change", (e) => {
      const newRenderOrbits = (e.target as HTMLInputElement).checked;
      SetRenderOrbits(newRenderOrbits);
    });

    document.querySelector(".str")!.addEventListener("change", (e) => {
      const newStr = parseFloat((e.target as HTMLInputElement).value);
      SetStrength(newStr);
    });

    document.querySelector(".eps")!.addEventListener("change", (e) => {
      const newEps = parseFloat((e.target as HTMLInputElement).value);
      SetEpsilon(newEps);
    });

    document.querySelector(".dn")!.addEventListener("change", (e) => {
      const newDebugNormals = (e.target as HTMLInputElement).checked;
      SetDebugNormals(newDebugNormals);
    });

    document.querySelector(".ds")!.addEventListener("change", (e) => {
      const newDebugShadows = (e.target as HTMLInputElement).checked;
      SetDebugShadows(newDebugShadows);
    });

    document.querySelector(".sdc")!.addEventListener("change", (e) => {
      const newShowDepthCube = (e.target as HTMLInputElement).checked;
      SetShowDepthCube(newShowDepthCube);
    });

    document.querySelector(".db")!.addEventListener("change", (e) => {
      const newDepthBias = parseFloat((e.target as HTMLInputElement).value);
      SetDepthBias(newDepthBias);
    });

    document.querySelector(".no")!.addEventListener("change", (e) => {
      const newNormalOffset = parseFloat((e.target as HTMLInputElement).value);
      SetNormalOffset(newNormalOffset);
    });

    document.querySelectorAll(".body").forEach((control, i) => {
      const massInput = control.querySelector(".mass") as HTMLInputElement;
      const radiusInput = control.querySelector(".radius") as HTMLInputElement;
      const positionXInput = control.querySelector(".position-x") as HTMLInputElement;
      const positionYInput = control.querySelector(".position-y") as HTMLInputElement;
      const positionZInput = control.querySelector(".position-z") as HTMLInputElement;
      const velocityXInput = control.querySelector(".velocity-x") as HTMLInputElement;
      const velocityYInput = control.querySelector(".velocity-y") as HTMLInputElement;
      const velocityZInput = control.querySelector(".velocity-z") as HTMLInputElement;

      massInput.addEventListener("change", () => {
        INITIAL_BODIES[i].mass = parseFloat(massInput.value);
      });

      radiusInput.addEventListener("change", () => {
        INITIAL_BODIES[i].radius = parseFloat(radiusInput.value);
      });

      positionXInput.addEventListener("change", () => {
        INITIAL_BODIES[i].position.x = parseFloat(positionXInput.value);
      });

      positionYInput.addEventListener("change", () => {
        INITIAL_BODIES[i].position.y = parseFloat(positionYInput.value);
      });

      positionZInput.addEventListener("change", () => {
        INITIAL_BODIES[i].position.z = parseFloat(positionZInput.value);
      });

      velocityXInput.addEventListener("change", () => {
        INITIAL_BODIES[i].velocity.x = parseFloat(velocityXInput.value);
      });

      velocityYInput.addEventListener("change", () => {
        INITIAL_BODIES[i].velocity.y = parseFloat(velocityYInput.value);
      });

      velocityZInput.addEventListener("change", () => {
        INITIAL_BODIES[i].velocity.z = parseFloat(velocityZInput.value);
      });
    });

    document.querySelectorAll("input.reload").forEach((input) => {
      input.addEventListener("change", SetUpBodiesRenderData);
    });

    document.querySelectorAll("input").forEach((input) => {
      input.addEventListener("change", ReloadSettings);
    });

    controlsSetUp = true;

  }


  return {
    SetUpControls
  }
}
