import { GRAVITY_MULTIPLIER, INITIAL_BODIES, SetGravityMultiplier, SetUpBuffersAndData } from "./main";

let controlsReady = false;

export function SetUpControls()
{
    if(controlsReady) return;

    document.querySelector("main")!.innerHTML += `<section id="controls">
        <label>G: <input type="number" class="g" value="${GRAVITY_MULTIPLIER}" /></label>
        <div class="body-controls">
        ${INITIAL_BODIES.map((body, i) => `
            <div class="body">
                <h2>Body ${i}</h2>
                <label>Mass: <input type="number" class="mass" value="${body.mass}" /></label>
                <label>Radius: <input type="number" class="radius" value="${body.radius}" /></label>
                <label>Initial Position: 
                <input type="number" class="position-x" value="${body.position.x}" step="0.1" />
                <input type="number" class="position-y" value="${body.position.y}" step="0.1" />
                <input type="number" class="position-z" value="${body.position.z}" step="0.1" />
                </label>
                <label>Initial Velocity: 
                <input type="number" class="velocity-x" value="${body.initialVelocity.x}" step="0.01" />
                <input type="number" class="velocity-y" value="${body.initialVelocity.y}" step="0.01" />
                <input type="number" class="velocity-z" value="${body.initialVelocity.z}" step="0.01" />
                </label>
            </div>
        `).join("")}
        </div>
    <section>`;

    document.querySelector(".g")!.addEventListener("change", (e) => {
        const newG = parseFloat((e.target as HTMLInputElement).value);
        SetGravityMultiplier(newG);
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
            INITIAL_BODIES[i].initialVelocity.x = parseFloat(velocityXInput.value);
        });

        velocityYInput.addEventListener("change", () => {
            INITIAL_BODIES[i].initialVelocity.y = parseFloat(velocityYInput.value);
        });

        velocityZInput.addEventListener("change", () => {
            INITIAL_BODIES[i].initialVelocity.z = parseFloat(velocityZInput.value);
        });
    });

    document.querySelectorAll("input").forEach(input => {
        input.addEventListener("change", SetUpBuffersAndData);
    });

    controlsReady = true;
}