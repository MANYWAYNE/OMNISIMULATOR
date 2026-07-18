import { app } from "../../scripts/app.js";

const THEME = {
	title:  "#232", // LiteGraph "green" preset (matches rgthree-style default nodes)
	body:   "#353", // LiteGraph "green" preset
	accent: "#FFFFFF", // White text
};

const REDUCE = (typeof window !== "undefined" && window.matchMedia)
	? window.matchMedia("(prefers-reduced-motion: reduce)").matches : false;

function animate() { return !REDUCE && (typeof window === "undefined" || window.OMNISIMULATOR_THEME_ANIMATE !== false); }
function _lg() { return (typeof window !== "undefined" && window.LiteGraph) || null; }

function isOmniNode(nodeData) {
	const cat  = (nodeData && nodeData.category) || "";
	const name = (nodeData && nodeData.name) || "";
	return cat === "OmniSimulator" || cat.indexOf("OmniSimulator/") === 0 || name.indexOf("OmniSimulator_") === 0;
}

// Draws one "OMNIFLEX" label.
// pulse (0..1): 0 = static; otherwise the fraction it shrinks by before returning to full
// size — e.g. 0.6 = shrinks down to 40% size, then grows back to 100%, on a smooth loop.
// phase: shifts the cycle so different instances don't beat in sync. Never rotates.
function drawOmniLabel(ctx, cx, cy, alpha, pulse = 0, phase = 0) {
	ctx.save();
	ctx.translate(cx, cy);

	if (pulse > 0) {
		const cycle = (1 - Math.cos(performance.now() / 900 + phase)) / 2; // smooth 0..1
		const scale = 1 - pulse * cycle; // 1 (normal) -> (1-pulse) (shrunk) -> back to 1
		ctx.scale(scale, scale);
	}

	ctx.globalAlpha = alpha;
	ctx.fillStyle = THEME.accent;
	ctx.font = "bold 10px Arial"; // 9.1px + 10%
	ctx.textAlign = "center";
	ctx.textBaseline = "middle";
	ctx.fillText("OMNIFLEX", 0, 0);
	ctx.restore();
}

app.registerExtension({
	name: "Comfy.OmniSimulator.Theme",

	async setup() {
		if (!animate()) return;
		let last = 0;
		const tick = (now) => {
			requestAnimationFrame(tick);
			if (document.hidden || !animate()) return;
			if (now - last < 33) return;
			last = now;
			const g = app.graph;
			if (!g || !g._nodes) return;
			for (const n of g._nodes) {
				if (n.__omni_themed && !(n.flags && n.flags.collapsed)) {
					try { app.canvas.setDirty(true, false); } catch (e) {}
					return;
				}
			}
		};
		requestAnimationFrame(tick);
	},

	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (!isOmniNode(nodeData)) return;

		const onCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onCreated ? onCreated.apply(this, arguments) : undefined;
			try {
				this.color    = THEME.title;
				this.bgcolor  = THEME.body;
				this.color_text = THEME.accent;
				this.boxcolor = THEME.accent;
				this.__omni_themed = true;
			} catch (e) { }
			return r;
		};

		const onDrawBG = nodeType.prototype.onDrawBackground;
		nodeType.prototype.onDrawBackground = function(ctx) {
			if (onDrawBG) onDrawBG.apply(this, arguments);
			if (this.flags && this.flags.collapsed) return;
			ctx.fillStyle = THEME.body;
			ctx.fillRect(0, 0, this.size[0], this.size[1]);
		};

		const onDraw = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (ctx) {
			if (onDraw) onDraw.apply(this, arguments);
			if (this.flags && this.flags.collapsed) return;
			try {
				const on = animate();
				const t  = on ? performance.now() / 1000 : 0;
				const w  = this.size[0];
				const h  = this.size[1];
				const LG = _lg();
				const th = (LG && LG.NODE_TITLE_HEIGHT) || 30;
				const A  = THEME.accent;

				ctx.save();
				ctx.globalAlpha = 0.85;
				ctx.strokeStyle = A;
				ctx.lineWidth = 2;
				ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(w, 0); ctx.stroke();
				ctx.restore();

				if (on) {
					const sx = ((t * 95) % (w + 120)) - 60;
					const grad = ctx.createLinearGradient(sx - 55, 0, sx + 55, 0);
					grad.addColorStop(0.0, "rgba(255, 255, 255, 0)");
					grad.addColorStop(0.5, "rgba(255, 255, 255, 0.95)");
					grad.addColorStop(1.0, "rgba(255, 255, 255, 0)");
					ctx.save();
					ctx.strokeStyle = grad;
					ctx.lineWidth = 2.5;
					ctx.beginPath();
					ctx.moveTo(Math.max(0, sx - 55), 0);
					ctx.lineTo(Math.min(w, sx + 55), 0);
					ctx.stroke();
					ctx.restore();
				}

				// Header brand label: stays fixed in the corner, just pulses in place.
				drawOmniLabel(ctx, w - 30, -th * 0.5, 0.95, on ? 0.6 : 0, 0);

				if (h > 46) {
					drawOmniLabel(ctx, w - 30, h - 15, 0.16, on ? 0.6 : 0, 1.4);
				}

				if (on && h > 40) {
					ctx.save();
					ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip();
					const seeds = this.__omni_seeds ||
						(this.__omni_seeds = [[0.1, 0.5, 0.0], [0.3, 0.9, 1.7], [0.5, 0.6, 3.4], [0.7, 1.2, 0.8], [0.9, 0.7, 2.5]]);
					const span = h + 24;
					for (const s of seeds) {
						// Straight vertical fall (no side-to-side sway) that loops top-to-bottom.
						const py = ((t * (12 * s[1])) + s[2] * 40) % span - 12;
						drawOmniLabel(ctx, s[0] * w, py, 0.22, 0.6, s[2]);
					}
					ctx.restore();
				}
			} catch (e) { }
		};
	},
});