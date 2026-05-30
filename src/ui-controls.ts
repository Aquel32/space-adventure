export function PrepareUI(
  settings: any,
) {
  let controlsSetUp = false;
  document.querySelector("main")!.innerHTML += `<section id="controls">
        <div class="other">
          <p id="fps">FPS: 000</p>
          <button>I</button>
        </div>
        <div class="main-controls">
          <label>Camera speed: <input name="camera-speed" type="number" class="cs" value="1" disabled /></label>
          <button class="tab-button" tab="simulation">Gravity</button>
          <div class="tab" id="simulation">
            <label>Gravity Multiplier: <input name="gravity" type="number" class="g reload" value="${settings.GRAVITY_MULTIPLIER}" /></label>
            <label>Simulation Speed: <input name="simulation-speed" type="number" class="ss" value="${settings.SIMULATION_SPEED}" /></label>
            <label>Attached Body: <input name="attached-body" type="number" class="ab" value="${settings.ATTACHED_BODY_INDEX}" /></label>
            <label>Pull Camera: <input name="pull-camera" type="checkbox" class="pc" ${settings.PULL_CAMERA ? "checked" : ""} /></label>
          </div>
          <button class="tab-button" tab="bloom">Bloom</button>
          <div class="tab" id="bloom">
            <label>Gaussian Iterations: <input name="gaussian-iterations" type="number" class="bi" value="${settings.GAUSIAN_ITERATIONS}" /></label>
            <label>Pixel Scale: <input name="pixel-scale" type="number" class="ps" value="${settings.PIXEL_SCALE}" /></label>
          </div>
          <button class="tab-button" tab="orbit">Orbit prediction</button>
          <div class="tab" id="orbit">
            <label>Render Orbits: <input name="render-orbits" type="checkbox" class="ro" ${settings.RENDER_ORBITS ? "checked" : ""} /></label>
          </div>
          <button class="tab-button" tab="sphere">Sphere generator</button>
          <div class="tab" id="sphere">
            <label>Perlin Strength: <input name="strength" type="number" class="str reload" value="${settings.PERLIN_STRENGTH}" /></label>
            <label>Epsilon: <input name="epsilon" type="number" class="eps reload" value="${settings.PERLIN_EPSILON}" /></label>
            <label>Debug Normals: <input name="debug-normals" type="checkbox" class="dn" ${settings.DEBUG_NORMALS ? "checked" : ""} /></label>
          </div>
          <button class="tab-button" tab="shadows">Shadows</button>
          <div class="tab" id="shadows">
            <label>Debug Shadows: <input name="debug-shadows" type="checkbox" class="ds" ${settings.DEBUG_SHADOWS ? "checked" : ""} /></label>
            <label>Show Depth Cube: <input name="show-depth-cube" type="checkbox" class="sdc" ${settings.SHOW_DEPTH_CUBE ? "checked" : ""} /></label>
            <label>Depth Bias: <input name="depth-bias" type="number" class="db" value="${settings.DEPTH_BIAS}" /></label>
            <label>Normal Offset: <input name="normal-offset" type="number" class="no" value="${settings.NORMAL_OFFSET}" /></label>
          </div>
          <button class="tab-button" tab="atmosphere">Atmosphere</button>
          <div class="tab" id="atmosphere">
           <label>Step Count: <input name="atmosphere-step-count" type="number" class="asc" value="${settings.ATMOSPHERE_STEP_COUNT}" /></label>
              <label>Show Prebaked Depth: <input name="show-prebaked-depth" type="checkbox" class="spd" ${settings.ATMOSPHERE_SHOW_PREBAKED_DEPTH ? "checked" : ""} /></label>
           </div>
        <button class="tab-button" tab="bodies">Initial bodies data</button>
        <div class="tab body-controls" id="bodies">

        </div>
    <section>`;


  document.querySelector(".body-controls")!.innerHTML += settings.INITIAL_BODIES.map(
    (body:any, i:number) => `
            <div class="body">
                <h2>Body ${i}</h2>
                <label>Mass: <input type="number" class="mass" value="${body.mass}" /></label>
                <label>Radius: <input type="number" class="radius" value="${body.radius}" /></label>
                <label class="vector">Initial Position: 
                  <input type="number" class="position-x" value="${body.position.x}" step="0.1" />
                  <input type="number" class="position-y" value="${body.position.y}" step="0.1" />
                  <input type="number" class="position-z" value="${body.position.z}" step="0.1" />
                </label>
                <p>Atmosphere</p>
                <label>Enabled: <input name="atmosphere-enabled" type="checkbox" class="ae" ${body.atmosphere.enabled === 1 ? "checked" : ""} /></label>
                <label>Radius: <input name="atmosphere-radius" type="number" class="ar" value="${body.atmosphere.atmosphereRadius}" /></label>
                <label>Falloff: <input name="atmosphere-falloff" type="number" class="af" value="${body.atmosphere.densityFalloff}" /></label>
                <label>Scattering Strength: <input name="atmosphere-scattering-strength" type="number" class="as" value="${body.atmosphere.scatteringStrength}" /></label>
                <label class="vector">Wavelengths: 
                  <input name="atmosphere-wavelengths-r" type="number" class="aw-r" value="${body.atmosphere.wavelengths.x}" />
                  <input name="atmosphere-wavelengths-g" type="number" class="aw-g" value="${body.atmosphere.wavelengths.y}" />
                  <input name="atmosphere-wavelengths-b" type="number" class="aw-b" value="${body.atmosphere.wavelengths.z}" />
                </label>
            </div>
        `,
  ).join("");

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.add("hidden");
  })

  document.querySelectorAll("button.tab-button").forEach((tabButton) => {
    tabButton.addEventListener("click", () => {
      const tabName = tabButton.getAttribute("tab");
      document.querySelector(`.tab#${tabName}`)?.classList.toggle("hidden");
    })
  })

  const infoBox = document.querySelector("#info") as HTMLDivElement; 
  document.querySelector(".other > button")!.addEventListener("click", () => {
    infoBox.style.display = "flex";
  });
  document.querySelector("#info-content > button")!.addEventListener("click", () => {
    infoBox.style.display = "none";
  });

  const visited = localStorage.getItem("visited");
  if (!visited) {
    infoBox.style.display = "flex";
    localStorage.setItem("visited", "true");
  }


  function SetUpControls() {
    if (controlsSetUp) return;

    document.querySelector(".g")!.addEventListener("change", (e) => {
      const newGravityMultiplier = parseFloat((e.target as HTMLInputElement).value);
      settings.SetGravityMultiplier(newGravityMultiplier);
    });

    document.querySelector(".ss")!.addEventListener("change", (e) => {
      const newSimulationSpeed = parseFloat((e.target as HTMLInputElement).value);
      settings.SetSimulationSpeed(newSimulationSpeed);
    });

    document.querySelector(".bi")!.addEventListener("change", (e) => {
      const newGausianIterations = parseFloat((e.target as HTMLInputElement).value);
      settings.SetGausianIterations(newGausianIterations);
    });

    document.querySelector(".ps")!.addEventListener("change", (e) => {
      const newPixelScale = parseFloat((e.target as HTMLInputElement).value);
      settings.SetPixelScale(newPixelScale);
    });

    document.querySelector(".ab")!.addEventListener("change", (e) => {
      const newAttachedBodyIndex = parseFloat((e.target as HTMLInputElement).value);
      settings.SetAttachedBody(newAttachedBodyIndex, settings.INITIAL_BODIES.length - 1);
    });

    document.querySelector(".ro")!.addEventListener("change", (e) => {
      const newRenderOrbits = (e.target as HTMLInputElement).checked;
      settings.SetRenderOrbits(newRenderOrbits);
    });

    document.querySelector(".str")!.addEventListener("change", (e) => {
      const newStr = parseFloat((e.target as HTMLInputElement).value);
      settings.SetStrength(newStr);
    });

    document.querySelector(".eps")!.addEventListener("change", (e) => {
      const newEps = parseFloat((e.target as HTMLInputElement).value);
      settings.SetEpsilon(newEps);
    });

    document.querySelector(".dn")!.addEventListener("change", (e) => {
      const newDebugNormals = (e.target as HTMLInputElement).checked;
      settings.SetDebugNormals(newDebugNormals);
    });

    document.querySelector(".ds")!.addEventListener("change", (e) => {
      const newDebugShadows = (e.target as HTMLInputElement).checked;
      settings.SetDebugShadows(newDebugShadows);
    });

    document.querySelector(".sdc")!.addEventListener("change", (e) => {
      const newShowDepthCube = (e.target as HTMLInputElement).checked;
      settings.SetShowDepthCube(newShowDepthCube);
    });

    document.querySelector(".db")!.addEventListener("change", (e) => {
      const newDepthBias = parseFloat((e.target as HTMLInputElement).value);
      settings.SetDepthBias(newDepthBias);
    });

    document.querySelector(".no")!.addEventListener("change", (e) => {
      const newNormalOffset = parseFloat((e.target as HTMLInputElement).value);
      settings.SetNormalOffset(newNormalOffset);
    });

    document.querySelector(".asc")!.addEventListener("change", (e) => {
      const newAtmosphereStepCount = parseFloat((e.target as HTMLInputElement).value);
      settings.SetAtmosphereStepCount(newAtmosphereStepCount);
    });

    document.querySelector(".spd")!.addEventListener("change", (e) => {
      const newShowPrebakedDepth = (e.target as HTMLInputElement).checked;
      settings.SetShowPrebakedDepth(newShowPrebakedDepth);
    });

    document.querySelector(".pc")!.addEventListener("change", (e) => {
      const newPullCamera = (e.target as HTMLInputElement).checked;
      settings.SetPullCamera(newPullCamera);
    });

    document.querySelectorAll(".body").forEach((control, i) => {
      const massInput = control.querySelector(".mass") as HTMLInputElement;
      const radiusInput = control.querySelector(".radius") as HTMLInputElement;
      const positionXInput = control.querySelector(".position-x") as HTMLInputElement;
      const positionYInput = control.querySelector(".position-y") as HTMLInputElement;
      const positionZInput = control.querySelector(".position-z") as HTMLInputElement;
      const atmosphereEnabledInput = control.querySelector(".ae") as HTMLInputElement;
      const atmosphereRadiusInput = control.querySelector(".ar") as HTMLInputElement;
      const atmosphereFalloffInput = control.querySelector(".af") as HTMLInputElement;
      const atmosphereScatteringStrengthInput = control.querySelector(".as") as HTMLInputElement;
      const atmosphereWavelengthsRInput = control.querySelector(".aw-r") as HTMLInputElement;
      const atmosphereWavelengthsGInput = control.querySelector(".aw-g") as HTMLInputElement;
      const atmosphereWavelengthsBInput = control.querySelector(".aw-b") as HTMLInputElement;

      atmosphereEnabledInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].atmosphere.enabled = atmosphereEnabledInput.checked ? 1 : 0;
      });

      atmosphereRadiusInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].atmosphere.atmosphereRadius = parseFloat(atmosphereRadiusInput.value);
      });

      atmosphereFalloffInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].atmosphere.densityFalloff = parseFloat(atmosphereFalloffInput.value);
      });

      atmosphereScatteringStrengthInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].atmosphere.scatteringStrength = parseFloat(atmosphereScatteringStrengthInput.value);
      });

      atmosphereWavelengthsRInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].atmosphere.wavelengths.x = parseFloat(atmosphereWavelengthsRInput.value);
      });

      atmosphereWavelengthsGInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].atmosphere.wavelengths.y = parseFloat(atmosphereWavelengthsGInput.value);
      });

      atmosphereWavelengthsBInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].atmosphere.wavelengths.z = parseFloat(atmosphereWavelengthsBInput.value);
      });

      massInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].mass = parseFloat(massInput.value);
      });

      radiusInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].radius = parseFloat(radiusInput.value);
      });

      positionXInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].position.x = parseFloat(positionXInput.value);
      });

      positionYInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].position.y = parseFloat(positionYInput.value);
      });

      positionZInput.addEventListener("change", () => {
        settings.INITIAL_BODIES[i].position.z = parseFloat(positionZInput.value);
      });
    });

    controlsSetUp = true;
  }

  function setUpCallbacks(setUpBodiesRenderData: () => void, reloadSettings: () => void) {
    document.querySelectorAll("input.reload").forEach((input) => {
      input.addEventListener("change", setUpBodiesRenderData);
    });

    document.querySelectorAll("input").forEach((input) => {
      input.addEventListener("change", reloadSettings);
    });
  }

  return {
    SetUpControls,
    setUpCallbacks,
  }
}
