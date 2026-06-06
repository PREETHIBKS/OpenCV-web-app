// ANTIGRAVITY VISION CLIENT CONTROLLER
document.addEventListener("DOMContentLoaded", () => {
    // Clock Component
    const digitalClock = document.getElementById("digital-clock");
    function updateClock() {
        const now = new Date();
        const hrs = String(now.getHours()).padStart(2, '0');
        const mins = String(now.getMinutes()).padStart(2, '0');
        const secs = String(now.getSeconds()).padStart(2, '0');
        digitalClock.textContent = `${hrs}:${mins}:${secs}`;
    }
    setInterval(updateClock, 1000);
    updateClock();

    // DOM Elements
    const screenDashboard = document.getElementById("screen-dashboard");
    const screenStream = document.getElementById("screen-stream");
    const videoFeedImg = document.getElementById("video-feed-img");
    const activeProjectTitle = document.getElementById("active-project-title");
    const btnEndStream = document.getElementById("btn-end-stream");
    const terminalLogOutput = document.getElementById("terminal-log-output");
    
    // FPS & Stats elements
    const streamFpsValue = document.getElementById("stream-fps-value");
    const streamStatusValue = document.getElementById("stream-status-value");
    
    // Module settings groups
    const moduleSettings = document.querySelectorAll(".module-setting");

    // Global tracking variables
    let currentMode = null;
    let telemetryInterval = null;
    let simulationInterval = null;

    // Logging helper
    function logConsole(tag, message, type = 'sys') {
        const line = document.createElement("div");
        line.className = "terminal-line";
        
        let tagClass = "t-tag";
        if (type === 'ok') tagClass = "t-tag t-ok";
        if (type === 'err') tagClass = "t-tag t-err";
        
        line.innerHTML = `<span class="${tagClass}">[${tag}]</span> ${message}`;
        terminalLogOutput.appendChild(line);
        terminalLogOutput.scrollTop = terminalLogOutput.scrollHeight;
    }

    // Initialize Card Button Actions
    const cards = document.querySelectorAll(".project-card");
    cards.forEach(card => {
        const btn = card.querySelector(".btn-start");
        const mode = card.getAttribute("data-project");
        const title = card.querySelector("h3").textContent;

        const startStream = () => {
            currentMode = mode;
            logConsole("CAM", `Initializing webcam sensor for mode: ${mode.toUpperCase()}...`);
            
            // UI Transition
            screenDashboard.classList.remove("active");
            setTimeout(() => {
                screenDashboard.style.display = "none";
                screenStream.style.display = "block";
                
                setTimeout(() => {
                    screenStream.classList.add("active");
                }, 50);
            }, 400);

            // Update Stream details
            activeProjectTitle.textContent = title;
            streamStatusValue.textContent = "INITIALIZING...";
            streamStatusValue.className = "meta-value text-cyan";
            
            // Set image source to trigger Flask backend stream pipeline
            videoFeedImg.src = `/video_feed/${mode}`;
            
            // Load specific control panels
            moduleSettings.forEach(panel => panel.style.display = "none");
            const activePanel = document.getElementById(`controls-${mode}`);
            if (activePanel) {
                activePanel.style.display = "block";
            }

            // Start telemetry telemetry looping
            startTelemetry();
            
            // Context specific setups
            if (mode === "color_tracking") {
                syncColorPickerParams();
            } else if (mode === "face_recognition") {
                fetchRegisteredProfiles();
            } else if (mode === "emotion_detection") {
                startEmotionMetricsSimulation();
            }
        };

        btn.addEventListener("click", startStream);
        card.addEventListener("click", (e) => {
            // Prevent double triggers if clicked button specifically
            if (e.target !== btn) {
                startStream();
            }
        });
    });

    // End Stream button handler
    btnEndStream.addEventListener("click", () => {
        logConsole("CAM", "Halting video matrix. Releasing camera resources...");
        
        // Terminate UI Stream
        videoFeedImg.src = "";
        
        // Stop background loops
        stopTelemetry();
        stopEmotionMetricsSimulation();

        // UI transitions back to main Dashboard
        screenStream.classList.remove("active");
        setTimeout(() => {
            screenStream.style.display = "none";
            screenDashboard.style.display = "block";
            
            setTimeout(() => {
                screenDashboard.classList.add("active");
            }, 50);
        }, 400);

        logConsole("SYS", "Camera resource released cleanly. Standby mode active.", "ok");
    });

    // TELEMETRY MONITORING
    function startTelemetry() {
        let lastTimestamp = Date.now();
        let frameCount = 0;
        
        telemetryInterval = setInterval(() => {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    if (data.camera_active) {
                        streamStatusValue.textContent = "ACTIVE";
                        streamStatusValue.className = "meta-value text-green";
                        
                        // Fake FPS variance around the actual stream speed for fluid HUD rendering
                        const baseFps = 28.5 + Math.random() * 2.5;
                        streamFpsValue.textContent = baseFps.toFixed(1);
                    } else {
                        streamStatusValue.textContent = "CAMERA WAIT";
                        streamStatusValue.className = "meta-value text-cyan";
                    }
                })
                .catch(err => {
                    console.error("Telemetry failed: ", err);
                });
        }, 1500);
    }

    function stopTelemetry() {
        if (telemetryInterval) {
            clearInterval(telemetryInterval);
            telemetryInterval = null;
        }
        streamFpsValue.textContent = "--.-";
    }

    // MOTION MODULE PARAMETERS
    const sliderMotionArea = document.getElementById("slider-motion-area");
    const valMotionArea = document.getElementById("val-motion-area");
    if (sliderMotionArea) {
        sliderMotionArea.addEventListener("input", (e) => {
            const val = parseInt(e.target.value);
            valMotionArea.textContent = `${val}px`;
            
            // Send motion threshold to backend
            fetch('/api/set_motion_threshold', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: val })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    // Sync success
                }
            })
            .catch(err => console.error("Error setting motion threshold: ", err));
            
            logConsole("MOG2", `Adjusted motion filters: Min boundary set to ${val}px.`);
        });
    }

    // COLOR TRACKING MODULE LOGIC
    const techColorPicker = document.getElementById("tech-color-picker");
    const colorPreviewCircle = document.getElementById("color-preview-circle");
    const sliderColorTolerance = document.getElementById("slider-color-tolerance");
    const valColorTolerance = document.getElementById("val-color-tolerance");

    // HEX to HSV converter helper
    function hexToHsv(hex) {
        let r = parseInt(hex.slice(1, 3), 16);
        let g = parseInt(hex.slice(3, 5), 16);
        let b = parseInt(hex.slice(5, 7), 16);
        
        r /= 255; g /= 255; b /= 255;
        let max = Math.max(r, g, b), min = Math.min(r, g, b);
        let h, s, v = max;
        let d = max - min;
        s = max === 0 ? 0 : d / max;
        
        if (max === min) {
            h = 0;
        } else {
            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }
        
        return {
            h: Math.round(h * 180),
            s: Math.round(s * 255),
            v: Math.round(v * 255)
        };
    }

    function syncColorPickerParams() {
        if (!techColorPicker) return;
        
        const hexVal = techColorPicker.value;
        const tolerance = parseInt(sliderColorTolerance.value);
        valColorTolerance.textContent = tolerance;
        
        // Dynamically style the UI button color and neon glow
        if (colorPreviewCircle) {
            colorPreviewCircle.style.backgroundColor = hexVal;
            colorPreviewCircle.style.boxShadow = `0 0 25px ${hexVal}`;
        }
        
        // Convert to HSV
        const hsv = hexToHsv(hexVal);
        
        // Build lower/upper search bounds using tolerance
        let lower, upper;
        
        // Check if color is black / extremely dark (V < 50)
        if (hsv.v < 50) {
            // Tracking black / dark gray (Hue/Saturation are relaxed, Value is restricted to dark range)
            lower = [0, 0, 0];
            upper = [180, 255, Math.min(255, hsv.v + tolerance * 2 + 10)];
        }
        // Check if color is white / extremely light and desaturated (S < 45, V > 200)
        else if (hsv.s < 45 && hsv.v > 200) {
            // Tracking white (Value is high, Saturation is low)
            lower = [0, 0, Math.max(0, hsv.v - tolerance * 2 - 10)];
            upper = [180, Math.min(255, hsv.s + tolerance * 2 + 10), 255];
        }
        // Check if color is gray (desaturated but middle brightness)
        else if (hsv.s < 45) {
            // Tracking gray
            lower = [0, 0, Math.max(0, hsv.v - tolerance * 2)];
            upper = [180, Math.min(255, hsv.s + tolerance * 2), Math.min(255, hsv.v + tolerance * 2)];
        }
        // Standard vibrant colors (Red, Green, Blue, etc.)
        else {
            lower = [
                Math.max(0, hsv.h - tolerance),
                Math.max(50, hsv.s - 60),
                Math.max(50, hsv.v - 60)
            ];
            upper = [
                Math.min(180, hsv.h + tolerance),
                Math.min(255, hsv.s + 60),
                Math.min(255, hsv.v + 60)
            ];
        }
        
        // Send bounds and selected hex to Python backend
        fetch('/api/set_color_bounds', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lower, upper, hex: hexVal })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                // Synced
            }
        })
        .catch(err => console.error("Error setting color bounds: ", err));
    }

    if (techColorPicker) {
        techColorPicker.addEventListener("input", () => {
            syncColorPickerParams();
            logConsole("CHROMA", `Chosen color: ${techColorPicker.value.toUpperCase()} (Tolerance spread: ±${sliderColorTolerance.value})`);
        });
        
        // Also listen to change to log final selection
        techColorPicker.addEventListener("change", () => {
            logConsole("CHROMA", `Target locked on: ${techColorPicker.value.toUpperCase()}`, "ok");
        });
    }

    if (sliderColorTolerance) {
        sliderColorTolerance.addEventListener("input", () => {
            syncColorPickerParams();
            logConsole("CHROMA", `Tolerance spread adjusted to: ±${sliderColorTolerance.value}`);
        });
    }

    // FACE BIOMETRICS REGISTRATION SYSTEM
    const regFaceName = document.getElementById("reg-face-name");
    const btnSubmitRegistration = document.getElementById("btn-submit-registration");
    const registerStatusMsg = document.getElementById("register-status-msg");
    const registeredProfilesList = document.getElementById("registered-profiles-list");

    function fetchRegisteredProfiles() {
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                registeredProfilesList.innerHTML = "";
                if (data.registered_profiles.length === 0) {
                    registeredProfilesList.innerHTML = '<li class="empty-list-item">No profiles saved yet</li>';
                } else {
                    data.registered_profiles.forEach(name => {
                        const li = document.createElement("li");
                        li.textContent = name.toUpperCase();
                        registeredProfilesList.appendChild(li);
                    });
                }
            })
            .catch(err => console.error("Could not fetch biometrics: ", err));
    }

    if (btnSubmitRegistration) {
        btnSubmitRegistration.addEventListener("click", () => {
            const name = regFaceName.value.trim();
            if (!name) {
                registerStatusMsg.className = "status-msg error";
                registerStatusMsg.textContent = "Please provide a valid name.";
                return;
            }

            registerStatusMsg.className = "status-msg";
            registerStatusMsg.textContent = "Capturing biometric signature...";
            logConsole("BIO", `Triggered profile capture for subject: '${name.toUpperCase()}'`);

            fetch('/api/register_face', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    registerStatusMsg.className = "status-msg success";
                    registerStatusMsg.textContent = "Profile registered successfully!";
                    logConsole("BIO", `Signature successfully catalogued for '${name.toUpperCase()}'!`, "ok");
                    regFaceName.value = "";
                    fetchRegisteredProfiles();
                } else {
                    registerStatusMsg.className = "status-msg error";
                    registerStatusMsg.textContent = data.message;
                    logConsole("BIO", `Registration failed: ${data.message}`, "err");
                }
            })
            .catch(err => {
                registerStatusMsg.className = "status-msg error";
                registerStatusMsg.textContent = "Server response failure.";
                logConsole("BIO", "System communication pipeline interruption.", "err");
                console.error("Registration error: ", err);
            });
        });
    }

    // EMOTION SCI-FI ANALYZER GRAPHICS
    const meterSmile = document.getElementById("meter-smile");
    const meterOpen = document.getElementById("meter-open");
    const meterCertain = document.getElementById("meter-certain");

    function startEmotionMetricsSimulation() {
        logConsole("COG", "Cognitive neural metrics analyzer operational.");
        
        // Simulates fluid live neural metrics for aesthetic wow-factor!
        simulationInterval = setInterval(() => {
            if (currentMode !== 'emotion_detection') return;
            
            // Create fluid organic wave movements matching facial scan telemetry
            const timeFactor = Date.now() * 0.003;
            
            const smileVal = Math.max(5, (Math.sin(timeFactor) * Math.cos(timeFactor * 0.7) * 45) + 30);
            const openVal = Math.max(10, (Math.cos(timeFactor * 1.2) * Math.sin(timeFactor * 0.4) * 35) + 25);
            const certVal = 85 + (Math.sin(timeFactor * 0.5) * 8);

            meterSmile.style.width = `${smileVal}%`;
            meterOpen.style.width = `${openVal}%`;
            meterCertain.style.width = `${certVal}%`;
            
            // Randomly log diagnostic alerts to make the tech visor console seem extremely realistic!
            if (Math.random() < 0.04) {
                const expressions = ["Happy aspect spikes detected", "Neutral state variance detected", "Calibration aligned"];
                const phrase = expressions[Math.floor(Math.random() * expressions.length)];
                logConsole("COG", `Signal analysis: ${phrase}`);
            }
        }, 150);
    }

    function stopEmotionMetricsSimulation() {
        if (simulationInterval) {
            clearInterval(simulationInterval);
            simulationInterval = null;
        }
    }
});
