module.exports = [
"[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/app/actions/video-processing.ts [app-rsc] (ecmascript)", ((__turbopack_context__) => {
"use strict";

/* __next_internal_action_entry_do_not_use__ [{"004f205e1ed11677fe1713a7912bd9dd0e7d66486c":"validateBackendConnection","00f69f0ec98f8a627ca04268e4802224d4ad218d4f":"startWebcamStream","409b90f73597c21d775523eac5658ba545ff87aa21":"processVideoFile"},"",""] */ __turbopack_context__.s([
    "processVideoFile",
    ()=>processVideoFile,
    "startWebcamStream",
    ()=>startWebcamStream,
    "validateBackendConnection",
    ()=>validateBackendConnection
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$node_modules$2f$next$2f$dist$2f$build$2f$webpack$2f$loaders$2f$next$2d$flight$2d$loader$2f$server$2d$reference$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/node_modules/next/dist/build/webpack/loaders/next-flight-loader/server-reference.js [app-rsc] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$node_modules$2f$next$2f$dist$2f$build$2f$webpack$2f$loaders$2f$next$2d$flight$2d$loader$2f$action$2d$validate$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/node_modules/next/dist/build/webpack/loaders/next-flight-loader/action-validate.js [app-rsc] (ecmascript)");
;
async function processVideoFile(formData) {
    try {
        const video = formData.get("video");
        if (!video) {
            return {
                success: false,
                error: "No video file provided"
            };
        }
        // Convert file to buffer
        const bytes = await video.arrayBuffer();
        const buffer = Buffer.from(bytes);
        // Get Python backend URL from environment
        const backendUrl = process.env.PYTHON_BACKEND_URL || "http://localhost:5000";
        // Send to Python backend
        const response = await fetch(`${backendUrl}/api/detect`, {
            method: "POST",
            headers: {
                "Content-Type": "application/octet-stream",
                "Content-Length": buffer.length.toString()
            },
            body: buffer,
            timeout: 300000
        });
        if (!response.ok) {
            const errorText = await response.text();
            return {
                success: false,
                error: `Backend error: ${response.statusText} - ${errorText}`
            };
        }
        const result = await response.json();
        return {
            success: true,
            stats: {
                maleIn: result.maleIn || 0,
                maleOut: result.maleOut || 0,
                femaleIn: result.femaleIn || 0,
                femaleOut: result.femaleOut || 0,
                currentCount: result.currentCount || 0,
                fps: result.fps || 0,
                duration: result.duration || 0
            }
        };
    } catch (error) {
        console.error("Video processing error:", error);
        return {
            success: false,
            error: error instanceof Error ? error.message : "Unknown error occurred"
        };
    }
}
async function startWebcamStream() {
    try {
        const backendUrl = process.env.PYTHON_BACKEND_URL || "http://localhost:5000";
        const response = await fetch(`${backendUrl}/api/webcam-start`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });
        if (!response.ok) {
            return {
                success: false,
                error: "Failed to start webcam stream"
            };
        }
        return {
            success: true
        };
    } catch (error) {
        console.error("Webcam start error:", error);
        return {
            success: false,
            error: error instanceof Error ? error.message : "Unknown error occurred"
        };
    }
}
async function validateBackendConnection() {
    try {
        const backendUrl = process.env.PYTHON_BACKEND_URL || "http://localhost:5000";
        const response = await fetch(`${backendUrl}/health`, {
            method: "GET",
            timeout: 5000
        });
        return response.ok;
    } catch (error) {
        console.error("Backend connection validation failed:", error);
        return false;
    }
}
;
(0, __TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$node_modules$2f$next$2f$dist$2f$build$2f$webpack$2f$loaders$2f$next$2d$flight$2d$loader$2f$action$2d$validate$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["ensureServerEntryExports"])([
    processVideoFile,
    startWebcamStream,
    validateBackendConnection
]);
(0, __TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$node_modules$2f$next$2f$dist$2f$build$2f$webpack$2f$loaders$2f$next$2d$flight$2d$loader$2f$server$2d$reference$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["registerServerReference"])(processVideoFile, "409b90f73597c21d775523eac5658ba545ff87aa21", null);
(0, __TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$node_modules$2f$next$2f$dist$2f$build$2f$webpack$2f$loaders$2f$next$2d$flight$2d$loader$2f$server$2d$reference$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["registerServerReference"])(startWebcamStream, "00f69f0ec98f8a627ca04268e4802224d4ad218d4f", null);
(0, __TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$node_modules$2f$next$2f$dist$2f$build$2f$webpack$2f$loaders$2f$next$2d$flight$2d$loader$2f$server$2d$reference$2e$js__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["registerServerReference"])(validateBackendConnection, "004f205e1ed11677fe1713a7912bd9dd0e7d66486c", null);
}),
"[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/.next-internal/server/app/page/actions.js { ACTIONS_MODULE0 => \"[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/app/actions/video-processing.ts [app-rsc] (ecmascript)\" } [app-rsc] (server actions loader, ecmascript) <locals>", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([]);
var __TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$app$2f$actions$2f$video$2d$processing$2e$ts__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/app/actions/video-processing.ts [app-rsc] (ecmascript)");
;
;
}),
"[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/.next-internal/server/app/page/actions.js { ACTIONS_MODULE0 => \"[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/app/actions/video-processing.ts [app-rsc] (ecmascript)\" } [app-rsc] (server actions loader, ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "00f69f0ec98f8a627ca04268e4802224d4ad218d4f",
    ()=>__TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$app$2f$actions$2f$video$2d$processing$2e$ts__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["startWebcamStream"],
    "409b90f73597c21d775523eac5658ba545ff87aa21",
    ()=>__TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$app$2f$actions$2f$video$2d$processing$2e$ts__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__["processVideoFile"]
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f2e$next$2d$internal$2f$server$2f$app$2f$page$2f$actions$2e$js__$7b$__ACTIONS_MODULE0__$3d3e$__$225b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$app$2f$actions$2f$video$2d$processing$2e$ts__$5b$app$2d$rsc$5d$__$28$ecmascript$2922$__$7d$__$5b$app$2d$rsc$5d$__$28$server__actions__loader$2c$__ecmascript$29$__$3c$locals$3e$__ = __turbopack_context__.i('[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/.next-internal/server/app/page/actions.js { ACTIONS_MODULE0 => "[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/app/actions/video-processing.ts [app-rsc] (ecmascript)" } [app-rsc] (server actions loader, ecmascript) <locals>');
var __TURBOPACK__imported__module__$5b$project$5d2f$OneDrive__$2d$__Bina__Nusantara$2f$Sem__5$2f$Deep__Learning$2f$DL_Projects$2f$frontend$2f$app$2f$actions$2f$video$2d$processing$2e$ts__$5b$app$2d$rsc$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/app/actions/video-processing.ts [app-rsc] (ecmascript)");
}),
"[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/node_modules/next/dist/build/webpack/loaders/next-flight-loader/server-reference.js [app-rsc] (ecmascript)", ((__turbopack_context__, module, exports) => {
"use strict";

/* eslint-disable import/no-extraneous-dependencies */ Object.defineProperty(exports, "__esModule", {
    value: true
});
Object.defineProperty(exports, "registerServerReference", {
    enumerable: true,
    get: function() {
        return _server.registerServerReference;
    }
});
const _server = __turbopack_context__.r("[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/node_modules/next/dist/server/route-modules/app-page/vendored/rsc/react-server-dom-turbopack-server.js [app-rsc] (ecmascript)"); //# sourceMappingURL=server-reference.js.map
}),
"[project]/OneDrive - Bina Nusantara/Sem 5/Deep Learning/DL_Projects/frontend/node_modules/next/dist/build/webpack/loaders/next-flight-loader/action-validate.js [app-rsc] (ecmascript)", ((__turbopack_context__, module, exports) => {
"use strict";

// This function ensures that all the exported values are valid server actions,
// during the runtime. By definition all actions are required to be async
// functions, but here we can only check that they are functions.
Object.defineProperty(exports, "__esModule", {
    value: true
});
Object.defineProperty(exports, "ensureServerEntryExports", {
    enumerable: true,
    get: function() {
        return ensureServerEntryExports;
    }
});
function ensureServerEntryExports(actions) {
    for(let i = 0; i < actions.length; i++){
        const action = actions[i];
        if (typeof action !== 'function') {
            throw Object.defineProperty(new Error(`A "use server" file can only export async functions, found ${typeof action}.\nRead more: https://nextjs.org/docs/messages/invalid-use-server-value`), "__NEXT_ERROR_CODE", {
                value: "E352",
                enumerable: false,
                configurable: true
            });
        }
    }
} //# sourceMappingURL=action-validate.js.map
}),
];

//# sourceMappingURL=OneDrive%20-%20Bina%20Nusantara_Sem%205_Deep%20Learning_DL_Projects_frontend_82a92e42._.js.map