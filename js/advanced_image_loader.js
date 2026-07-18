// ---
// Filename: ../ComfyUI_OmniSimulator/js/advanced_image_loader.js
// ---

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Auto-migrate images from old per-node folders to central pool (runs once on load)
(async function migrateOldImages() {
	try {
		const response = await fetch("/omnisimulator/migrate_images", { method: "POST" });
		const result = await response.json();
		if (result.success && result.migrated > 0) {
			console.log(`[OmniSimulator] ✅ Migrated ${result.migrated} images to central pool`);
		}
	} catch (e) {
		// Silent fail - migration is optional
	}
})();

app.registerExtension({
	name: "Comfy.OmniSimulator.AdvancedImageLoader",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "OmniSimulator_AdvancedImageLoader") {
			// Store original onConfigure to ensure saved data is properly restored
			const origOnConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function(data) {
				// Restore properties from saved data FIRST
				if (data.properties) {
					this.properties = this.properties || {};
					Object.assign(this.properties, data.properties);
					console.log(`[OmniSimulator AIL ${this.id}] Restored properties from saved data:`, {
						hasBatchData: !!this.properties.batch_data,
						batchDataLength: this.properties.batch_data?.length
					});
				}
				// Also restore from widgets_values if batch_data widget was serialized
				if (data.widgets_values) {
					const batchDataWidget = this.widgets?.find(w => w.name === "batch_data");
					if (batchDataWidget) {
						const widgetIdx = this.widgets.indexOf(batchDataWidget);
						if (data.widgets_values[widgetIdx]) {
							this.properties.batch_data = data.widgets_values[widgetIdx];
							console.log(`[OmniSimulator AIL ${this.id}] Restored batch_data from widget value`);
						}
					}
				}
				origOnConfigure?.apply(this, arguments);
			};

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				// Only set defaults if not already set (don't overwrite loaded data)
				if (!this.properties) this.properties = {};
				if (!this.properties.batch_data) {
					this.properties.batch_data = JSON.stringify({ images: [], order: [], total_count: 0 });
				}
				if (!this.properties.txt2img_data_backup) {
					this.properties.txt2img_data_backup = JSON.stringify({ latents: [], order: [], total_count: 0 });
				}
				if (!this.properties.img2img_data_backup) {
					this.properties.img2img_data_backup = JSON.stringify({ images: [], order: [], total_count: 0 });
				}

				const node = this;
				let cachedHeight = 300;
				let isUpdatingHeight = false;
				let currentDetectedMode = null; // Track detected mode to minimize re-renders
				let modeCheckInterval = null; // For periodic mode checking

				// Selection mode state (for multi-select delete)
				let selectionMode = false;
				const selectedImages = new Set(); // Track which image IDs are selected

				const container = document.createElement("div");
				container.className = "omnisimulator-adv-loader-container";
				container.style.width = "100%";
				container.style.boxSizing = "border-box";
				container.style.overflow = "hidden";
				container.style.height = `${cachedHeight}px`;

				const updateCachedHeight = () => {
					if (isUpdatingHeight) return;
					isUpdatingHeight = true;

					// Safety timeout to prevent stuck flag
					const safetyTimeout = setTimeout(() => {
						isUpdatingHeight = false;
					}, 500);

					container.style.overflow = "visible";
					container.style.height = "auto";

					requestAnimationFrame(() => {
						requestAnimationFrame(() => {
							clearTimeout(safetyTimeout);

							// Use scrollHeight as it measures full content even during transitions
							const newHeight = Math.max(container.offsetHeight, container.scrollHeight);

							if (newHeight > 0 && Math.abs(newHeight - cachedHeight) > 2) {
								cachedHeight = newHeight;
								container.style.height = `${newHeight}px`;
								const sz = node.computeSize();
								node.size[1] = sz[1];
								node.onResize?.(sz);
								app.graph.setDirtyCanvas(true, false);
							} else {
								container.style.height = `${cachedHeight}px`;
							}
							container.style.overflow = "hidden";
							isUpdatingHeight = false;
						});
					});
				};

				// Periodic height sync - keep running, only clear when node is removed
				const heightSyncInterval = setInterval(() => {
					if (!node || !container) {
						clearInterval(heightSyncInterval);
						return;
					}
					updateCachedHeight();
				}, 2000);
				node._heightSyncInterval = heightSyncInterval;

				// Fast initial syncs for page load
				[100, 300, 500, 1000, 1500, 2000].forEach(delay => {
					setTimeout(() => {
						if (container) {
							updateCachedHeight();
						}
					}, delay);
				});

				const syncBatchDataWidget = () => {
					const batchDataWidget = node.widgets?.find((w) => w.name === "batch_data");
					if (batchDataWidget) {
						batchDataWidget.value = node.properties.batch_data;
					} else {
						const widget = node.addWidget("text", "batch_data", node.properties.batch_data, () => {}, { serialize: true });
						widget.hidden = true;
					}
				};

				/**
				 * Computes aspect ratio node outputs locally (for multi-output nodes).
				 * Aspect ratio nodes compute width/height in Python, but we replicate that logic here
				 * to properly read values from the correct output slot.
				 */
				const getAspectRatioOutput = (aspectRatioNode, slotIndex) => {
					// Read the dropdown selection from the aspect ratio node
					const selection = aspectRatioNode.widgets?.[0]?.value;
					if (!selection) {
						console.warn(`[OmniSimulator AIL ${node.id}] Aspect ratio node has no selection`);
						return null;
					}

					// Handle Nano Banana Pro Aspect Ratio node (has 5 outputs)
					if (aspectRatioNode.type === "OmniSimulator_NanoBananaAspectRatio") {
						const resolutionWidget = aspectRatioNode.widgets?.[1];
						const resolution = resolutionWidget?.value || "1K";

						// Nano Banana aspect ratio mappings
						const NANO_RATIOS = {
							"1:1 (Square)": { ratio: "1:1", w: 1, h: 1 },
							"3:2 (Landscape)": { ratio: "3:2", w: 3, h: 2 },
							"2:3 (Portrait)": { ratio: "2:3", w: 2, h: 3 },
							"3:4 (Portrait)": { ratio: "3:4", w: 3, h: 4 },
							"4:3 (Landscape)": { ratio: "4:3", w: 4, h: 3 },
							"4:5 (Portrait)": { ratio: "4:5", w: 4, h: 5 },
							"5:4 (Landscape)": { ratio: "5:4", w: 5, h: 4 },
							"9:16 (Tall Portrait)": { ratio: "9:16", w: 9, h: 16 },
							"16:9 (Wide Landscape)": { ratio: "16:9", w: 16, h: 9 },
							"21:9 (Ultrawide)": { ratio: "21:9", w: 21, h: 9 },
						};

						const RESOLUTION_BASE = { "1K": 1024, "2K": 2048, "4K": 4096 };

						const config = NANO_RATIOS[selection];
						if (!config) {
							console.warn(`[OmniSimulator AIL ${node.id}] Unknown Nano Banana aspect ratio: ${selection}`);
							return null;
						}

						// Calculate dimensions (same logic as Python)
						const baseSize = RESOLUTION_BASE[resolution] || 1024;
						let width, height;
						if (config.w >= config.h) {
							height = baseSize;
							width = Math.floor(baseSize * config.w / config.h);
						} else {
							width = baseSize;
							height = Math.floor(baseSize * config.h / config.w);
						}
						// Round to nearest 64
						width = Math.floor(width / 64) * 64;
						height = Math.floor(height / 64) * 64;

						// Output slots: 0=aspect_ratio, 1=resolution, 2=width, 3=height, 4=aspect_label
						if (slotIndex === 0) return config.ratio;
						if (slotIndex === 1) return resolution;
						if (slotIndex === 2) return width;
						if (slotIndex === 3) return height;
						if (slotIndex === 4) return selection;

						return null;
					}

					// Aspect ratio mappings (must match Python ASPECT_RATIOS dicts exactly)
					const WAN_RATIOS = {
						"3:4 (Portrait)": { width: 720, height: 960, label: "3:4" },
						"9:16 (Tall Portrait)": { width: 540, height: 960, label: "9:16" },
						"1:1 (Square)": { width: 960, height: 960, label: "1:1" },
						"16:9 (Landscape)": { width: 960, height: 540, label: "16:9" }
					};

					const SDXL_RATIOS = {
						"3:4 (Portrait)": { width: 896, height: 1152, label: "3:4" },
						"9:16 (Tall Portrait)": { width: 768, height: 1344, label: "9:16" },
						"1:1 (Square)": { width: 1024, height: 1024, label: "1:1" },
						"16:9 (Landscape)": { width: 1344, height: 768, label: "16:9" }
					};

					const ratios = aspectRatioNode.type === "OmniSimulator_WANAspectRatio"
						? WAN_RATIOS
						: SDXL_RATIOS;

					const config = ratios[selection];
					if (!config) {
						console.warn(`[OmniSimulator AIL ${node.id}] Unknown aspect ratio selection: ${selection}`);
						return null;
					}

					// Return based on output slot (0=width, 1=height, 2=aspect_label)
					if (slotIndex === 0) return config.width;
					if (slotIndex === 1) return config.height;
					if (slotIndex === 2) return config.label;

					return null;
				};

				/**
				 * Retrieves the final value of an input by traversing connected nodes.
				 * Enhanced to properly handle multi-output nodes like aspect ratio nodes.
				 */
				const getFinalInputValue = (inputName, defaultValue) => {
					if (!node.inputs || node.inputs.length === 0) {
						const widget = node.widgets?.find(w => w.name === inputName);
						return widget ? widget.value : defaultValue;
					}

					const input = node.inputs.find(i => i.name === inputName);
					if (!input || input.link == null) {
						const widget = node.widgets?.find(w => w.name === inputName);
						return widget ? widget.value : defaultValue;
					}

					const link = app.graph.links[input.link];
					if (!link) return defaultValue;

					const originNode = app.graph.getNodeById(link.origin_id);
					if (!originNode) return defaultValue;

					// SPECIAL HANDLING: For aspect ratio nodes, compute the output locally
					// because they have multiple outputs and don't store computed values in widgets
					if (originNode.type === "OmniSimulator_WANAspectRatio" ||
					    originNode.type === "OmniSimulator_SDXLAspectRatio" ||
					    originNode.type === "OmniSimulator_NanoBananaAspectRatio") {
						const output = getAspectRatioOutput(originNode, link.origin_slot);
						if (output !== null) return output;
					}

					// For other nodes, read from widgets
					if (originNode.widgets && originNode.widgets.length > 0) {
						return originNode.widgets[0].value;
					}

					if (originNode.properties && originNode.properties.value !== undefined) {
						return originNode.properties.value;
					}

					return defaultValue;
				};

				/**
				 * Detects if we're in txt2img mode by reading enable_img2img from connected nodes.
				 * Returns true for txt2img mode, false for img2img mode.
				 */
				const isTxt2ImgMode = () => {
					const enableImg2Img = getFinalInputValue("enable_img2img", true);
					// console.log(`[OmniSimulator AIL ${node.id}] enable_img2img value:`, enableImg2Img, `(type: ${typeof enableImg2Img})`);
					const result = enableImg2Img === false || enableImg2Img === "false";
					// console.log(`[OmniSimulator AIL ${node.id}] isTxt2ImgMode result:`, result);
					return result;
				};

				/**
				 * Switches to img2img mode by setting enable_img2img to true.
				 * Works with both connected boolean nodes and local widgets.
				 */
				const switchToImg2ImgMode = () => {
					console.log(`[OmniSimulator AIL ${node.id}] === Switching to img2img mode ===`);

					// First, try to find and update a connected boolean node
					const input = node.inputs?.find(i => i.name === "enable_img2img");
					console.log(`[OmniSimulator AIL ${node.id}] enable_img2img input:`, input);

					if (input && input.link != null) {
						console.log(`[OmniSimulator AIL ${node.id}] Input is linked, link ID:`, input.link);
						const link = app.graph.links[input.link];
						console.log(`[OmniSimulator AIL ${node.id}] Link object:`, link);

						if (link) {
							const booleanNode = app.graph.getNodeById(link.origin_id);
							console.log(`[OmniSimulator AIL ${node.id}] Boolean node:`, booleanNode);
							console.log(`[OmniSimulator AIL ${node.id}] Boolean node type:`, booleanNode?.type);
							console.log(`[OmniSimulator AIL ${node.id}] Boolean node widgets:`, booleanNode?.widgets);

							if (booleanNode && booleanNode.widgets && booleanNode.widgets.length > 0) {
								const widget = booleanNode.widgets[0];
								console.log(`[OmniSimulator AIL ${node.id}] Widget name:`, widget.name);
								console.log(`[OmniSimulator AIL ${node.id}] Widget type:`, widget.type);
								const oldValue = widget.value;
								console.log(`[OmniSimulator AIL ${node.id}] Old value:`, oldValue);

								// Update the boolean node's widget value
								widget.value = true;
								console.log(`[OmniSimulator AIL ${node.id}] New value set:`, widget.value);

								// Trigger the widget's callback if it exists
								if (widget.callback) {
									console.log(`[OmniSimulator AIL ${node.id}] Calling widget callback...`);
									widget.callback.call(booleanNode, true, null, booleanNode, null, oldValue);
								} else {
									console.log(`[OmniSimulator AIL ${node.id}] ⚠️ Widget has no callback!`);
								}

								console.log(`[OmniSimulator AIL ${node.id}] ✅ Updated connected boolean node #${booleanNode.id} to true`);
								app.graph.setDirtyCanvas(true, true);
								return true;
							}
						}
					}

					// If not connected, try to update local widget
					const widget = node.widgets?.find(w => w.name === "enable_img2img");
					if (widget) {
						const oldValue = widget.value;
						widget.value = true;

						// Trigger the widget's callback if it exists
						if (widget.callback) {
							widget.callback.call(node, true, null, node, null, oldValue);
						}

						console.log(`[OmniSimulator AIL ${node.id}] Updated local enable_img2img widget to true`);
						app.graph.setDirtyCanvas(true, true);
						return true;
					}

					console.warn(`[OmniSimulator AIL ${node.id}] Could not find enable_img2img input or widget`);
					return false;
				};

				/**
				 * Gets width, height, and aspect_label from connected nodes for txt2img mode.
				 */
				const getTxt2ImgDimensions = () => {
					// console.log(`[OmniSimulator AIL ${node.id}] === Reading dimensions from connected nodes ===`);

					const widthRaw = getFinalInputValue("width", 960);
					const heightRaw = getFinalInputValue("height", 960);
					const aspect_label_raw = getFinalInputValue("aspect_label", null);

					// console.log(`[OmniSimulator AIL ${node.id}] Raw values:`, { widthRaw, heightRaw, aspect_label_raw });

					const width = parseInt(widthRaw) || 960;
					const height = parseInt(heightRaw) || 960;
					const aspect_label = aspect_label_raw || getAspectLabel(width, height);

					// console.log(`[OmniSimulator AIL ${node.id}] Final dimensions:`, { width, height, aspect_label });
					// console.log(`[OmniSimulator AIL ${node.id}] Expected tensor size: ${width}×${height} (${(width * height / 1000000).toFixed(2)}MP)`);

					return { width, height, aspect_label };
				};

				/**
				 * Generates a simple UUID v4.
				 */
				const generateUUID = () => {
					return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
						const r = Math.random() * 16 | 0;
						const v = c === 'x' ? r : (r & 0x3 | 0x8);
						return v.toString(16);
					});
				};

				/**
				 * Calculates aspect ratio label from width and height.
				 */
				const getAspectLabel = (width, height) => {
					const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
					const divisor = gcd(width, height);
					const w = width / divisor;
					const h = height / divisor;

					// Common aspect ratios
					if (w === h) return "1:1";
					if (w === 3 && h === 4) return "3:4";
					if (w === 4 && h === 3) return "4:3";
					if (w === 9 && h === 16) return "9:16";
					if (w === 16 && h === 9) return "16:9";
					if (w === 2 && h === 3) return "2:3";
					if (w === 3 && h === 2) return "3:2";

					return `${w}:${h}`;
				};

				const renderGallery = () => {
					const detectedMode = isTxt2ImgMode();
					console.log(`[OmniSimulator AIL ${node.id}] renderGallery - currentDetectedMode:`, currentDetectedMode, `detectedMode:`, detectedMode);

					// Handle mode switching
					if (currentDetectedMode !== null && currentDetectedMode !== detectedMode) {
						console.log(`[OmniSimulator AIL ${node.id}] MODE SWITCH DETECTED! Switching to ${detectedMode ? 'txt2img' : 'img2img'}`);
						if (detectedMode) {
							// Switching to txt2img mode
							node.properties.img2img_data_backup = node.properties.batch_data;
							// Only restore backup if it has actual items that exist
							const backup = node.properties.txt2img_data_backup;
							let backupData = null;
							try { backupData = backup ? JSON.parse(backup) : null; } catch (e) { backupData = null; }
							// Validate backup: check if order items actually exist in latents
							const latents = backupData?.latents || [];
							const order = backupData?.order || [];
							const validOrder = order.filter(id => latents.some(l => l.id === id));
							const hasValidBackup = validOrder.length > 0;
							if (hasValidBackup) {
								// Clean up the backup data to only include valid items
								backupData.order = validOrder;
								backupData.total_count = validOrder.reduce((sum, id) => {
									const latent = latents.find(l => l.id === id);
									return sum + (latent?.repeat_count || 1);
								}, 0);
								node.properties.batch_data = JSON.stringify(backupData);
							} else {
								node.properties.batch_data = JSON.stringify({ latents: [], order: [], total_count: 0 });
							}
						} else {
							// Switching to img2img mode
							node.properties.txt2img_data_backup = node.properties.batch_data;
							// Only restore backup if it has actual items that exist
							const backup = node.properties.img2img_data_backup;
							let backupData = null;
							try { backupData = backup ? JSON.parse(backup) : null; } catch (e) { backupData = null; }
							// Validate backup: check if order items actually exist in images
							const images = backupData?.images || [];
							const order = backupData?.order || [];
							const validOrder = order.filter(id => images.some(img => img.id === id));
							const hasValidBackup = validOrder.length > 0;
							if (hasValidBackup) {
								// Clean up the backup data to only include valid items
								backupData.order = validOrder;
								backupData.total_count = validOrder.reduce((sum, id) => {
									const image = images.find(img => img.id === id);
									return sum + (image?.repeat_count || 1);
								}, 0);
								node.properties.batch_data = JSON.stringify(backupData);
							} else {
								node.properties.batch_data = JSON.stringify({ images: [], order: [], total_count: 0 });
							}
						}
						syncBatchDataWidget();
					}
					currentDetectedMode = detectedMode;

					// console.log(`[OmniSimulator AIL ${node.id}] Rendering ${detectedMode ? 'txt2img' : 'img2img'} gallery`);
					if (detectedMode) {
						renderTxt2ImgGallery();
					} else {
						renderImg2ImgGallery();
					}
				};

				const renderImg2ImgGallery = () => {
					let batchData = JSON.parse(node.properties.batch_data || "{}");
					let images = batchData.images || [];
					let order = batchData.order || [];

					// Validate and clean up stale data (order items that don't exist in images)
					const validOrder = order.filter(id => images.some(img => img.id === id));
					if (validOrder.length !== order.length) {
						console.log(`[OmniSimulator AIL ${node.id}] Cleaning up stale img2img data: ${order.length} -> ${validOrder.length} items`);
						batchData.order = validOrder;
						batchData.total_count = validOrder.reduce((sum, id) => {
							const image = images.find(img => img.id === id);
							return sum + (image?.repeat_count || 1);
						}, 0);
						order = validOrder;
						node.properties.batch_data = JSON.stringify(batchData);
						syncBatchDataWidget();
					}

					const modeWidget = node.widgets?.find((w) => w.name === "mode");
					const currentMode = modeWidget?.value || "Batch Tensor";
					const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
					const currentIndex = node._processingIndex !== undefined ? node._processingIndex : batchIndexWidget?.value || 0;

					container.innerHTML = `
                        <div class="omnisimulator-adv-loader-brand-row">
                            <img src="/extensions/ComfyUI_OmniSimulator_NODES/omnisimulator.svg" alt="OmniSimulator" class="omnisimulator-adv-loader-logo" />
                            <span class="omnisimulator-adv-loader-version">AIL V2.0</span>
                        </div>
                        <div class="omnisimulator-adv-loader-topbar">
                            <div class="omnisimulator-adv-loader-topbar-left">
                                <span class="omnisimulator-adv-loader-mode-badge omnisimulator-adv-loader-mode-img2img">🖼️ IMG2IMG</span>
                                <div class="omnisimulator-adv-loader-mode-select">
                                    <select class="omnisimulator-adv-loader-mode-dropdown">
                                        <option value="Batch Tensor" ${currentMode === "Batch Tensor" ? "selected" : ""}>🎯 Batch Tensor</option>
                                        <option value="Sequential" ${currentMode === "Sequential" ? "selected" : ""}>📑 Sequential</option>
                                    </select>
                                </div>
                                ${currentMode === "Sequential" ? `<span class="omnisimulator-adv-loader-progress-badge">${currentIndex}/${batchData.total_count || 0}</span>` : ""}
                            </div>
                            <div class="omnisimulator-adv-loader-topbar-right">
                                <span class="omnisimulator-adv-loader-count">${images.length} image${images.length !== 1 ? "s" : ""}</span>
                                <span class="omnisimulator-adv-loader-separator">·</span>
                                <span class="omnisimulator-adv-loader-total">${batchData.total_count || 0} total</span>
                            </div>
                        </div>
                        <div class="omnisimulator-adv-loader-actions">
                            ${selectionMode ? `
                                <button class="omnisimulator-adv-loader-select-all-btn" title="Select all images">☑ Select All</button>
                                <button class="omnisimulator-adv-loader-deselect-all-btn" title="Deselect all">☐ Deselect</button>
                                <button class="omnisimulator-adv-loader-delete-selected-btn" title="Delete selected images" ${selectedImages.size === 0 ? 'disabled' : ''}>🗑️ Delete (${selectedImages.size})</button>
                                <button class="omnisimulator-adv-loader-cancel-selection-btn" title="Exit selection mode">✖ Cancel</button>
                            ` : `
                                <button class="omnisimulator-adv-loader-upload-btn" title="Upload images">📁 Upload Images</button>
                                ${currentMode === "Sequential" ? `<button class="omnisimulator-adv-loader-queue-all-btn" title="Queue all images">🎬 Queue All</button>` : ""}
                                ${images.length > 0 ? `<button class="omnisimulator-adv-loader-duplicate-btn" title="Duplicate last image">⧉ Duplicate</button>` : ""}
                                ${images.length > 0 ? `<button class="omnisimulator-adv-loader-sync-btn" title="Smart sync: Match counts across AILs">🔄 Sync</button>` : ""}
                                ${images.length > 0 ? `<button class="omnisimulator-adv-loader-enter-selection-btn" title="Select multiple images">☑ Select</button>` : ""}
                                ${images.length > 0 ? `<button class="omnisimulator-adv-loader-delete-all-btn" title="Delete all images">🗑️ Clear</button>` : ""}
                            `}
                        </div>
                        <div class="omnisimulator-adv-loader-gallery">
                            ${order.length === 0 ? `<div class="omnisimulator-adv-loader-empty"><p>No images loaded</p><p class="omnisimulator-adv-loader-hint">Click "Upload Images" to get started</p></div>` : (() => {
								let currentIdx = 0;
								return order.map((imgId) => {
									const img = images.find((i) => i.id === imgId);
									if (!img) return "";
									const thumbUrl = `/omnisimulator/view/${img.thumbnail}`;
									const repeatCount = img.repeat_count || 1;
									const startIdx = currentIdx;
									const endIdx = currentIdx + repeatCount - 1;
									currentIdx += repeatCount;
									const isActive = currentMode === "Sequential" && currentIndex >= startIdx && currentIndex <= endIdx;
									const isPast = currentMode === "Sequential" && currentIndex > endIdx;
									const isProcessing = node._isProcessing && isActive;
									return `<div class="omnisimulator-adv-loader-item ${isActive ? "omnisimulator-adv-loader-item-active" : ""} ${isPast ? "omnisimulator-adv-loader-item-done" : ""} ${isProcessing ? "omnisimulator-adv-loader-item-processing" : ""} ${selectionMode ? "omnisimulator-adv-loader-item-selection" : ""} ${selectedImages.has(String(img.id)) ? "omnisimulator-adv-loader-item-selected" : ""}" data-id="${img.id}" draggable="${!selectionMode}">
                                        ${selectionMode ? `
                                            <label class="omnisimulator-adv-loader-selection-checkbox">
                                                <input type="checkbox" class="omnisimulator-adv-loader-image-checkbox" data-id="${img.id}" ${selectedImages.has(String(img.id)) ? 'checked' : ''} />
                                            </label>
                                        ` : ''}
                                        ${currentMode === "Sequential" && !selectionMode ? `<div class="omnisimulator-adv-loader-index-badge">${repeatCount === 1 ? `#${startIdx}` : `#${startIdx}-${endIdx}`}</div>` : ""}
                                        <div class="omnisimulator-adv-loader-thumb">
                                            <img src="${thumbUrl}" alt="${img.original_name}" style="background: rgba(0,0,0,0.3);" onerror="this.style.opacity='0.3'; this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22%3E%3Cpath fill=%22%239ca3af%22 d=%22M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z%22/%3E%3C/svg%3E';" />
                                            ${isProcessing ? '<div class="omnisimulator-adv-loader-processing-indicator">⚡ PROCESSING...</div>' : isActive && !selectionMode ? '<div class="omnisimulator-adv-loader-active-indicator">▶ NEXT</div>' : ""}
                                            ${isPast && !selectionMode ? '<div class="omnisimulator-adv-loader-done-indicator">✓</div>' : ""}
                                        </div>
                                        <div class="omnisimulator-adv-loader-controls">
                                            <label>×</label>
                                            <input type="number" class="omnisimulator-adv-loader-repeat-input" value="${img.repeat_count || 1}" min="1" max="99" data-id="${img.id}" ${selectionMode ? 'disabled' : ''} />
                                            <button class="omnisimulator-adv-loader-delete-btn" data-id="${img.id}" title="Delete" ${selectionMode ? 'disabled' : ''}>×</button>
                                        </div>
                                        <div class="omnisimulator-adv-loader-info">
                                            <div class="omnisimulator-adv-loader-filename" title="${img.original_name}">${img.original_name}</div>
                                            <div class="omnisimulator-adv-loader-dimensions">${img.width}×${img.height} (original)</div>
                                        </div>
                                    </div>`;
								}).join("");
							})()}
                        </div>`;
					setupEventHandlers();
					setupDragAndDrop();
					setupFileDropZone();
					updateCachedHeight();

					// Dispatch update event for other nodes (e.g., RPG)
					window.dispatchEvent(new CustomEvent("OmniSimulator_AIL_UPDATED", {
						detail: {
							nodeId: node.id,
							mode: "img2img",
							enable_img2img: true,  // NEW: explicit boolean for easier detection
							images: order.map(imgId => {
								const img = images.find(i => i.id === imgId);
								if (!img) return null;
								const thumbUrl = `/omnisimulator/view/${img.thumbnail}`;
								const width = img.width || 1024;
								const height = img.height || 1024;
								return {
									url: thumbUrl,
									index: order.indexOf(imgId),
									id: imgId,
									repeat_count: img.repeat_count || 1,
									width: width,
									height: height,
									aspect_label: getAspectLabel(width, height)
								};
							}).filter(i => i !== null),
							total: batchData.total_count || 0
						}
					}));
				};

				const renderTxt2ImgGallery = () => {
					// Preserve batch count input value
					const existingInput = container.querySelector(".omnisimulator-adv-loader-batch-count-input");
					const preservedBatchCount = existingInput?.value || node._batchAddCount || 5;
					node._batchAddCount = preservedBatchCount;

					let batchData = JSON.parse(node.properties.batch_data || "{}");
					let latents = batchData.latents || [];
					let order = batchData.order || [];

					// Validate and clean up stale data (order items that don't exist in latents)
					const validOrder = order.filter(id => latents.some(l => l.id === id));
					if (validOrder.length !== order.length) {
						console.log(`[OmniSimulator AIL ${node.id}] Cleaning up stale txt2img data: ${order.length} -> ${validOrder.length} items`);
						batchData.order = validOrder;
						batchData.total_count = validOrder.reduce((sum, id) => {
							const latent = latents.find(l => l.id === id);
							return sum + (latent?.repeat_count || 1);
						}, 0);
						order = validOrder;
						node.properties.batch_data = JSON.stringify(batchData);
						syncBatchDataWidget();
					}
					const modeWidget = node.widgets?.find((w) => w.name === "mode");
					const currentMode = modeWidget?.value || "Batch Tensor";
					const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
					const currentIndex = node._processingIndex !== undefined ? node._processingIndex : batchIndexWidget?.value || 0;

					// Get current dimensions from aspect ratio selector (for live updates)
					const dimensions = getTxt2ImgDimensions();

					// Debug: Log dimensions for debugging
					console.log(`[OmniSimulator AIL ${node.id}] renderTxt2ImgGallery - dimensions:`, dimensions);

					container.innerHTML = `
                        <div class="omnisimulator-adv-loader-brand-row">
                            <img src="/extensions/ComfyUI_OmniSimulator_NODES/omnisimulator.svg" alt="OmniSimulator" class="omnisimulator-adv-loader-logo" />
                            <span class="omnisimulator-adv-loader-version">AIL V2.0</span>
                        </div>
                        <div class="omnisimulator-adv-loader-topbar">
                            <div class="omnisimulator-adv-loader-topbar-left">
                                <span class="omnisimulator-adv-loader-mode-badge omnisimulator-adv-loader-mode-txt2img">🎨 TXT2IMG</span>
                                <div class="omnisimulator-adv-loader-mode-select">
                                    <select class="omnisimulator-adv-loader-mode-dropdown">
                                        <option value="Batch Tensor" ${currentMode === "Batch Tensor" ? "selected" : ""}>🎯 Batch Tensor</option>
                                        <option value="Sequential" ${currentMode === "Sequential" ? "selected" : ""}>📑 Sequential</option>
                                    </select>
                                </div>
                                ${currentMode === "Sequential" ? `<span class="omnisimulator-adv-loader-progress-badge">${currentIndex}/${batchData.total_count || 0}</span>` : ""}
                            </div>
                            <div class="omnisimulator-adv-loader-topbar-right">
                                <span class="omnisimulator-adv-loader-count">${latents.length} latent${latents.length !== 1 ? "s" : ""}</span>
                                <span class="omnisimulator-adv-loader-separator">·</span>
                                <span class="omnisimulator-adv-loader-total">${batchData.total_count || 0} total</span>
                            </div>
                        </div>
                        <div class="omnisimulator-adv-loader-actions">
                            <button class="omnisimulator-adv-loader-add-latent-btn" title="Add empty latent">➕ Add Latent</button>
                            <div class="omnisimulator-adv-loader-batch-add-controls">
                                <input type="number" class="omnisimulator-adv-loader-batch-count-input" value="${preservedBatchCount}" min="1" max="100" />
                                <button class="omnisimulator-adv-loader-batch-add-btn" title="Batch add empty latents">📦 Add N</button>
                            </div>
                            ${currentMode === "Sequential" ? `<button class="omnisimulator-adv-loader-queue-all-btn" title="Queue all latents">🎬 Queue All</button>` : ""}
                            ${latents.length > 0 ? `<button class="omnisimulator-adv-loader-duplicate-btn" title="Duplicate last latent">⧉ Duplicate</button>` : ""}
                            ${latents.length > 0 ? `<button class="omnisimulator-adv-loader-sync-btn" title="Smart sync: Match counts across AILs">🔄 Sync</button>` : ""}
                            ${latents.length > 0 ? `<button class="omnisimulator-adv-loader-delete-all-btn" title="Delete all latents">🗑️ Clear</button>` : ""}
                        </div>
                        <div class="omnisimulator-adv-loader-gallery">
                            ${order.length === 0 ? `<div class="omnisimulator-adv-loader-empty"><p>No latents added</p><p class="omnisimulator-adv-loader-hint">Click "Add Empty Latent" to get started (txt2img mode)</p></div>` : (() => {
								let currentIdx = 0;
								return order.map((latentId) => {
									const latent = latents.find((l) => l.id === latentId);
									if (!latent) return "";

									// Debug: Log what latent contains
									console.log(`[OmniSimulator AIL ${node.id}] Rendering latent:`, latent);

									const repeatCount = latent.repeat_count || 1;
									const startIdx = currentIdx;
									const endIdx = currentIdx + repeatCount - 1;
									currentIdx += repeatCount;
									const isActive = currentMode === "Sequential" && currentIndex >= startIdx && currentIndex <= endIdx;
									const isPast = currentMode === "Sequential" && currentIndex > endIdx;
									const isProcessing = node._isProcessing && isActive;

									// Use CURRENT dimensions from aspect ratio selector (not stored dimensions)
									// This ensures live updates when aspect ratio selector changes
									const width = dimensions.width;
									const height = dimensions.height;
									const aspectRatio = width / height;
									const aspectLabel = dimensions.aspect_label;

									return `<div class="omnisimulator-adv-loader-item ${isActive ? "omnisimulator-adv-loader-item-active" : ""} ${isPast ? "omnisimulator-adv-loader-item-done" : ""} ${isProcessing ? "omnisimulator-adv-loader-item-processing" : ""}" data-id="${latent.id}" draggable="true">
                                        ${currentMode === "Sequential" ? `<div class="omnisimulator-adv-loader-index-badge">${repeatCount === 1 ? `#${startIdx}` : `#${startIdx}-${endIdx}`}</div>` : ""}
                                        <div class="omnisimulator-adv-loader-latent-thumb">
                                            <div class="omnisimulator-adv-loader-aspect-preview" style="aspect-ratio: ${aspectRatio};">
                                                <div class="omnisimulator-adv-loader-aspect-content">
                                                    <div style="font-size: 24px;">📐</div>
                                                    <div style="font-size: 11px; font-weight: 600;">${aspectLabel}</div>
                                                </div>
                                            </div>
                                            ${isProcessing ? '<div class="omnisimulator-adv-loader-processing-indicator">⚡ PROCESSING...</div>' : isActive ? '<div class="omnisimulator-adv-loader-active-indicator">▶ NEXT</div>' : ""}
                                            ${isPast ? '<div class="omnisimulator-adv-loader-done-indicator">✓</div>' : ""}
                                        </div>
                                        <div class="omnisimulator-adv-loader-controls">
                                            <label>×</label>
                                            <input type="number" class="omnisimulator-adv-loader-repeat-input" value="${latent.repeat_count || 1}" min="1" max="99" data-id="${latent.id}" />
                                            <button class="omnisimulator-adv-loader-delete-btn" data-id="${latent.id}" title="Delete">×</button>
                                        </div>
                                        <div class="omnisimulator-adv-loader-info">
                                            <div class="omnisimulator-adv-loader-filename" title="${latent.id}">Latent ${latent.id.substring(0, 8)}</div>
                                            <div class="omnisimulator-adv-loader-dimensions">${width}×${height}</div>
                                        </div>
                                    </div>`;
								}).join("");
							})()}
                        </div>`;
					setupTxt2ImgEventHandlers();
					setupDragAndDrop();
					setupFileDropZone();
					updateCachedHeight();

					// Dispatch update event for other nodes (e.g., RPG)
					window.dispatchEvent(new CustomEvent("OmniSimulator_AIL_UPDATED", {
						detail: {
							nodeId: node.id,
							mode: "txt2img",
							enable_img2img: false,  // NEW: explicit boolean for easier detection
							latents: order.map(latentId => {
								const latent = latents.find(l => l.id === latentId);
								if (!latent) return null;
								return {
									id: latent.id,
									width: latent.width,
									height: latent.height,
									aspect_label: latent.aspect_label,
									repeat_count: latent.repeat_count || 1,
									index: order.indexOf(latentId)
								};
							}).filter(l => l !== null),
							total: batchData.total_count || 0
						}
					}));
				};

				const addEmptyLatent = () => {
					const dimensions = getTxt2ImgDimensions();
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					batchData.latents = batchData.latents || [];
					batchData.order = batchData.order || [];

					// Check for aspect ratio mismatch
					if (batchData.latents.length > 0) {
						const existingAspect = batchData.latents[0].aspect_label;
						if (existingAspect !== dimensions.aspect_label) {
							const confirmed = confirm(
								`Current batch has ${existingAspect} latents.\n` +
								`Switch to ${dimensions.aspect_label}?\n\n` +
								`This will clear all existing latents.`
							);
							if (!confirmed) {
								return; // Abort
							}
							// Clear existing latents
							batchData.latents = [];
							batchData.order = [];
						}
					}

					const newLatent = {
						id: generateUUID(),
						width: dimensions.width,
						height: dimensions.height,
						repeat_count: 1,
						aspect_label: dimensions.aspect_label  // Use aspect_label from connected node
					};

					console.log(`[OmniSimulator AIL ${node.id}] Adding latent:`, newLatent);

					batchData.latents.push(newLatent);
					batchData.order.push(newLatent.id);
					batchData.total_count = batchData.latents.reduce((sum, l) => sum + (l.repeat_count || 1), 0);

					node.properties.batch_data = JSON.stringify(batchData);
					syncBatchDataWidget();
					renderGallery();
				};

				const batchAddLatents = (count) => {
					const dimensions = getTxt2ImgDimensions();
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					batchData.latents = batchData.latents || [];
					batchData.order = batchData.order || [];

					// Check for aspect ratio mismatch
					if (batchData.latents.length > 0) {
						const existingAspect = batchData.latents[0].aspect_label;
						if (existingAspect !== dimensions.aspect_label) {
							const confirmed = confirm(
								`Current batch has ${existingAspect} latents.\n` +
								`Switch to ${dimensions.aspect_label}?\n\n` +
								`This will clear all existing latents.`
							);
							if (!confirmed) {
								return; // Abort
							}
							// Clear existing latents
							batchData.latents = [];
							batchData.order = [];
						}
					}

					console.log(`[OmniSimulator AIL ${node.id}] Batch adding ${count} latents with dimensions:`, dimensions);

					for (let i = 0; i < count; i++) {
						const newLatent = {
							id: generateUUID(),
							width: dimensions.width,
							height: dimensions.height,
							repeat_count: 1,
							aspect_label: dimensions.aspect_label  // Use aspect_label from connected node
						};
						batchData.latents.push(newLatent);
						batchData.order.push(newLatent.id);
					}

					batchData.total_count = batchData.latents.reduce((sum, l) => sum + (l.repeat_count || 1), 0);
					node.properties.batch_data = JSON.stringify(batchData);
					syncBatchDataWidget();
					renderGallery();
				};

				const deleteLatent = (latentId) => {
					if (!confirm("Delete this latent?")) return;
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					batchData.latents = batchData.latents.filter((l) => l.id !== latentId);
					batchData.order = batchData.order.filter((id) => id !== latentId);
					batchData.total_count = batchData.latents.reduce((sum, l) => sum + (l.repeat_count || 1), 0);
					node.properties.batch_data = JSON.stringify(batchData);
					syncBatchDataWidget();
					renderGallery();
				};

				const deleteAllLatents = () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const latentCount = batchData.latents?.length || 0;
					if (latentCount === 0 || !confirm(`Delete all ${latentCount} latent${latentCount !== 1 ? "s" : ""}?`)) return;
					node.properties.batch_data = JSON.stringify({ latents: [], order: [], total_count: 0 });
					// Clear backup to prevent old data from reappearing on mode switch
					node.properties.txt2img_data_backup = JSON.stringify({ latents: [], order: [], total_count: 0 });
					syncBatchDataWidget();
					renderGallery();
				};

				const updateLatentRepeatCount = (latentId, newCount) => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const latent = batchData.latents.find((l) => l.id === latentId);
					if (latent) {
						latent.repeat_count = Math.max(1, Math.min(99, newCount));
						batchData.total_count = batchData.latents.reduce((sum, l) => sum + (l.repeat_count || 1), 0);
						node.properties.batch_data = JSON.stringify(batchData);
						syncBatchDataWidget();
						const statsEl = container.querySelector(".omnisimulator-adv-loader-total");
						if (statsEl) statsEl.textContent = `Total: ${batchData.total_count} (with repeats)`;
						// Re-render to trigger update event for RPG
						renderGallery();
					}
				};

				const queueAllLatents = async () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const totalCount = batchData.total_count || 0;
					if (totalCount === 0 || !confirm(`Queue ${totalCount} workflow executions?`)) return;
					try {
						const prompt = await app.graphToPrompt();
						for (let i = 0; i < totalCount; i++) {
							const promptCopy = JSON.parse(JSON.stringify(prompt.output));
							if (promptCopy[node.id] && promptCopy[node.id].inputs) {
								promptCopy[node.id].inputs.batch_index = i;
							}
							await fetch("/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: promptCopy, client_id: app.clientId }) });
							if (i < totalCount - 1) await new Promise((resolve) => setTimeout(resolve, 50));
						}
					} catch (error) {
						alert(`Queue error: ${error.message}`);
					}
				};

				const setupTxt2ImgEventHandlers = () => {
					// === Canvas Panning & Zoom Event Forwarding ===
					// Allow dragging/zooming canvas even when mouse is over custom UI
					const isInteractiveElement = (element) => {
						if (!element) return false;
						const tagName = element.tagName?.toLowerCase();
						if (['button', 'input', 'select', 'textarea', 'a'].includes(tagName)) return true;
						const className = element.className || '';
						if (typeof className === 'string' && (
							className.includes('-btn') || className.includes('-input') ||
							className.includes('-select') || className.includes('repeat-input') ||
							className.includes('gallery-item') || className.includes('checkbox')
						)) return true;
						if (element.onclick || element.onmousedown || element.draggable) return true;
						return false;
					};

					const isInteractiveOrChildOfInteractive = (target) => {
						let element = target;
						while (element && element !== container) {
							if (isInteractiveElement(element)) return true;
							element = element.parentElement;
						}
						return false;
					};

					// Track drag state for manual panning
					if (!container._ailDragState) {
						container._ailDragState = { isDragging: false, startX: 0, startY: 0, offsetX: 0, offsetY: 0 };
					}
					const dragState = container._ailDragState;

					const canvasForwardHandler = (e) => {
						if (e._ailForwarded) return;

						if (e.type === 'mousedown') {
							if (isInteractiveOrChildOfInteractive(e.target)) {
								dragState.isDragging = false;
								return;
							}
							dragState.isDragging = true;
							dragState.startX = e.clientX;
							dragState.startY = e.clientY;
							const canvas = app?.canvas || app?.graph?.list_of_graphcanvas?.[0];
							if (canvas) {
								dragState.offsetX = canvas.ds.offset[0];
								dragState.offsetY = canvas.ds.offset[1];
							}
							e.preventDefault();
							e.stopPropagation();
						}

						if (e.type === 'mousemove' && dragState.isDragging) {
							const deltaX = e.clientX - dragState.startX;
							const deltaY = e.clientY - dragState.startY;
							const canvas = app?.canvas || app?.graph?.list_of_graphcanvas?.[0];
							if (canvas) {
								const scale = canvas.ds.scale || 1;
								canvas.ds.offset[0] = dragState.offsetX + (deltaX / scale);
								canvas.ds.offset[1] = dragState.offsetY + (deltaY / scale);
								canvas.setDirty(true, true);
							}
							e.preventDefault();
							e.stopPropagation();
						}

						if (e.type === 'mouseup' && dragState.isDragging) {
							dragState.isDragging = false;
							e.preventDefault();
							e.stopPropagation();
						}
					};

					const wheelForwardHandler = (e) => {
						if (e._ailForwarded) return;
						if (isInteractiveOrChildOfInteractive(e.target)) return;
						const canvas = document.querySelector('#graph-canvas') ||
									  document.querySelector('canvas.litegraph') ||
									  document.querySelector('.litegraph canvas');
						if (canvas) {
							e.preventDefault();
							e.stopPropagation();
							const newEvent = new WheelEvent('wheel', {
								bubbles: true, cancelable: true, view: window,
								clientX: e.clientX, clientY: e.clientY,
								deltaX: e.deltaX, deltaY: e.deltaY, deltaZ: e.deltaZ, deltaMode: e.deltaMode,
								ctrlKey: e.ctrlKey, shiftKey: e.shiftKey, altKey: e.altKey, metaKey: e.metaKey
							});
							newEvent._ailForwarded = true;
							canvas.dispatchEvent(newEvent);
						}
					};

					// Remove old listeners if exist
					if (container._canvasForwardHandler) {
						container.removeEventListener('mousedown', container._canvasForwardHandler, true);
						document.removeEventListener('mousemove', container._canvasForwardHandler, true);
						document.removeEventListener('mouseup', container._canvasForwardHandler, true);
					}
					if (container._wheelForwardHandler) {
						container.removeEventListener('wheel', container._wheelForwardHandler, true);
					}

					container._canvasForwardHandler = canvasForwardHandler;
					container._wheelForwardHandler = wheelForwardHandler;

					container.addEventListener('mousedown', canvasForwardHandler, true);
					document.addEventListener('mousemove', canvasForwardHandler, true);
					document.addEventListener('mouseup', canvasForwardHandler, true);
					container.addEventListener('wheel', wheelForwardHandler, true);

					const modeDropdown = container.querySelector(".omnisimulator-adv-loader-mode-dropdown");
					if (modeDropdown)
						modeDropdown.onchange = (e) => {
							const modeWidget = node.widgets?.find((w) => w.name === "mode");
							if (modeWidget) {
								modeWidget.value = e.target.value;
								renderGallery();
							}
						};
					const queueAllBtn = container.querySelector(".omnisimulator-adv-loader-queue-all-btn");
					if (queueAllBtn) queueAllBtn.onclick = queueAllLatents;
					const addLatentBtn = container.querySelector(".omnisimulator-adv-loader-add-latent-btn");
					if (addLatentBtn) addLatentBtn.onclick = addEmptyLatent;
					const batchAddBtn = container.querySelector(".omnisimulator-adv-loader-batch-add-btn");
					if (batchAddBtn) {
						batchAddBtn.onclick = () => {
							const countInput = container.querySelector(".omnisimulator-adv-loader-batch-count-input");
							const count = parseInt(countInput?.value || 5);
							if (count > 0 && count <= 100) {
								batchAddLatents(count);
							}
						};
					}
					const deleteAllBtn = container.querySelector(".omnisimulator-adv-loader-delete-all-btn");
					if (deleteAllBtn) deleteAllBtn.onclick = deleteAllLatents;
					container.querySelectorAll(".omnisimulator-adv-loader-delete-btn").forEach((btn) => (btn.onclick = (e) => { e.stopPropagation(); deleteLatent(btn.dataset.id); }));
					container.querySelectorAll(".omnisimulator-adv-loader-repeat-input").forEach((input) => {
						input.onchange = (e) => updateLatentRepeatCount(input.dataset.id, parseInt(input.value) || 1);
						input.onmousedown = (e) => e.stopPropagation();
					});
				};

				const handleFileSelect = async (e) => {
					const files = Array.from(e.target.files);
					if (files.length === 0) return;
					const uploadBtn = container.querySelector(".omnisimulator-adv-loader-upload-btn");
					const originalText = uploadBtn.textContent;
					uploadBtn.textContent = "⏳ Uploading...";
					uploadBtn.disabled = true;
					try {
						const formData = new FormData();
						formData.append("node_id", node.id);
						files.forEach((file) => formData.append("files", file));
						const response = await fetch("/omnisimulator/batch_upload", { method: "POST", body: formData });
						const result = await response.json();
						console.log(`[OmniSimulator AIL ${node.id}] 📤 Upload result:`, result.success);
						if (result.success) {
							// Auto-switch to img2img mode FIRST if currently in txt2img mode
							const wasInTxt2ImgMode = isTxt2ImgMode();
							console.log(`[OmniSimulator AIL ${node.id}] 📊 Current mode - isTxt2ImgMode:`, wasInTxt2ImgMode);
							if (wasInTxt2ImgMode) {
								console.log(`[OmniSimulator AIL ${node.id}] 🔄 Auto-switching to img2img mode BEFORE adding images`);

								// Properly swap data (backup txt2img, restore img2img)
								node.properties.txt2img_data_backup = node.properties.batch_data;
								node.properties.batch_data = node.properties.img2img_data_backup || JSON.stringify({ images: [], order: [], total_count: 0 });
								console.log(`[OmniSimulator AIL ${node.id}] 💾 Backed up txt2img data, restored img2img data`);

								// Update the boolean
								switchToImg2ImgMode();

								// Manually update currentDetectedMode to prevent mode switch detection in renderGallery
								currentDetectedMode = false; // false = img2img mode
								console.log(`[OmniSimulator AIL ${node.id}] ✅ Updated currentDetectedMode to img2img`);
							} else {
								console.log(`[OmniSimulator AIL ${node.id}] ✅ Already in img2img mode, no switch needed`);
							}

							// NOW add images to batch_data (after mode switch completes)
							const batchData = JSON.parse(node.properties.batch_data || "{}");
							batchData.images = batchData.images || [];
							batchData.order = batchData.order || [];
							result.images.forEach((img) => {
								batchData.images.push(img);
								batchData.order.push(img.id);
							});
							batchData.total_count = batchData.images.reduce((sum, img) => sum + (img.repeat_count || 1), 0);
							node.properties.batch_data = JSON.stringify(batchData);

							syncBatchDataWidget();
							renderGallery();
						} else {
							alert(`Upload failed: ${result.error}`);
						}
					} catch (error) {
						alert(`Upload error: ${error.message}`);
					} finally {
						uploadBtn.textContent = originalText;
						uploadBtn.disabled = false;
					}
				};

				const deleteImage = async (imageId) => {
					if (!confirm("Delete this image?")) return;
					try {
						const batchData = JSON.parse(node.properties.batch_data || "{}");
						const imageIdStr = String(imageId);

						// Check if this is a duplicated image (ID starts with "dup_" or is numeric)
						const isDuplicate = imageIdStr.startsWith("dup_") || /^\d+(\.\d+)?$/.test(imageIdStr);

						// Only call server delete for original images (not duplicates)
						if (!isDuplicate) {
							try {
								await fetch(`/omnisimulator/batch_delete/${node.id}/${imageId}`, { method: "DELETE" });
							} catch (serverError) {
								// Server delete failed, but continue with local removal
								console.warn(`[AIL] Server delete failed for ${imageId}, removing locally`);
							}
						}

						// Remove from local batch data (compare as strings to handle type mismatches)
						batchData.images = batchData.images.filter((img) => String(img.id) !== imageIdStr);
						batchData.order = batchData.order.filter((id) => String(id) !== imageIdStr);
						batchData.total_count = batchData.images.reduce((sum, img) => sum + (img.repeat_count || 1), 0);
						node.properties.batch_data = JSON.stringify(batchData);
						syncBatchDataWidget();
						renderGallery();
					} catch (error) {
						alert(`Delete error: ${error.message}`);
					}
				};

				const duplicateLastItem = () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const isTxt2Img = isTxt2ImgMode();
					const items = isTxt2Img ? (batchData.latents || []) : (batchData.images || []);
					const order = batchData.order || [];

					if (items.length === 0 || order.length === 0) return;

					// Get last item ID from order
					const lastItemId = order[order.length - 1];
					const lastItem = items.find(item => item.id === lastItemId);

					if (!lastItem) return;

					// Create deep copy with new unique ID
					const duplicatedItem = {
						...lastItem,
						id: Date.now() + Math.random(),
						repeat_count: 1 // Reset repeat count for duplicate
					};

					// Add to batch data
					items.push(duplicatedItem);
					order.push(duplicatedItem.id);

					// Update total count
					if (isTxt2Img) {
						batchData.latents = items;
					} else {
						batchData.images = items;
					}
					batchData.order = order;
					batchData.total_count = items.reduce((sum, item) => sum + (item.repeat_count || 1), 0);

					// Save and re-render (renderGallery already dispatches OmniSimulator_AIL_UPDATED)
					node.properties.batch_data = JSON.stringify(batchData);
					syncBatchDataWidget();
					renderGallery();
				};

				const deleteAllImages = async () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const imageCount = batchData.images?.length || 0;
					if (imageCount === 0 || !confirm(`Delete all ${imageCount} image${imageCount !== 1 ? "s" : ""}?`)) return;
					try {
						await Promise.all(batchData.images.map((img) => fetch(`/omnisimulator/batch_delete/${node.id}/${img.id}`, { method: "DELETE" })));
						node.properties.batch_data = JSON.stringify({ images: [], order: [], total_count: 0 });
						// Clear backup to prevent old data from reappearing on mode switch
						node.properties.img2img_data_backup = JSON.stringify({ images: [], order: [], total_count: 0 });
						syncBatchDataWidget();
						renderGallery();
					} catch (error) {
						alert(`Delete all error: ${error.message}`);
					}
				};

				const queueAllImages = async () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const totalCount = batchData.total_count || 0;
					if (totalCount === 0 || !confirm(`Queue ${totalCount} workflow executions?`)) return;
					try {
						const prompt = await app.graphToPrompt();
						for (let i = 0; i < totalCount; i++) {
							const promptCopy = JSON.parse(JSON.stringify(prompt.output));
							if (promptCopy[node.id] && promptCopy[node.id].inputs) {
								promptCopy[node.id].inputs.batch_index = i;
							}
							await fetch("/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: promptCopy, client_id: app.clientId }) });
							if (i < totalCount - 1) await new Promise((resolve) => setTimeout(resolve, 50));
						}
					} catch (error) {
						alert(`Queue error: ${error.message}`);
					}
				};

				const updateRepeatCount = (imageId, newCount) => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const img = batchData.images.find((i) => i.id === imageId);
					if (img) {
						img.repeat_count = Math.max(1, Math.min(99, newCount));
						batchData.total_count = batchData.images.reduce((sum, i) => sum + (i.repeat_count || 1), 0);
						node.properties.batch_data = JSON.stringify(batchData);
						syncBatchDataWidget();
						const statsEl = container.querySelector(".omnisimulator-adv-loader-total");
						if (statsEl) statsEl.textContent = `Total: ${batchData.total_count} (with repeats)`;
						// Trigger re-render to update RPG
						renderGallery();
					}
				};

				// Smart sync function - finds connected RPG and syncs accordingly
				const smartSync = () => {
					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const isTxt2Img = isTxt2ImgMode();
					const items = isTxt2Img ? (batchData.latents || []) : (batchData.images || []);
					const order = batchData.order || [];
					const myCount = order.length;

					if (myCount === 0) {
						alert("No items to sync! Add images/latents first.");
						return;
					}

					// Find connected RPG node by checking our output connections
					let connectedRPG = null;
					let promptCount = 0;
					let promptRepeats = [];

					if (node.outputs && node.outputs.length > 0) {
						for (const output of node.outputs) {
							if (output.links && output.links.length > 0) {
								for (const linkId of output.links) {
									const link = app.graph.links[linkId];
									if (link) {
										const targetNode = app.graph.getNodeById(link.target_id);
										if (targetNode && targetNode.type === "OmniSimulator_RealityPromptGenerator") {
											connectedRPG = targetNode;
											// Get prompt queue from RPG
											try {
												const promptData = JSON.parse(targetNode.properties.prompt_batch_data || "[]");
												if (Array.isArray(promptData) && promptData.length > 0) {
													promptCount = promptData.length;
													promptRepeats = promptData.map(p => p.repeat_count || 1);
												}
											} catch (e) {
												console.warn("[AIL] Failed to parse RPG prompt data:", e);
											}
											break;
										}
									}
								}
							}
							if (connectedRPG) break;
						}
					}

					if (promptCount > 0) {
						// Prompts exist - prompts are king, sync this AIL to match
						console.log(`[AIL ${node.id}] 🔄 Prompts are king: Syncing to ${promptCount} prompts`);

						if (myCount === promptCount) {
							// Same count, just sync repeats
							window.dispatchEvent(new CustomEvent("OmniSimulator_SYNC_AIL_REPEATS", {
								detail: {
									targetNodeId: node.id,
									mode: isTxt2Img ? "txt2img" : "img2img",
									repeats: promptRepeats
								}
							}));
							alert(`✅ Synced repeat counts to match ${promptCount} prompts`);
						} else if (myCount < promptCount) {
							// Need to add items
							const toAdd = promptCount - myCount;
							window.dispatchEvent(new CustomEvent("OmniSimulator_DUPLICATE_LAST_N", {
								detail: { targetNodeId: node.id, count: toAdd }
							}));
							// Then sync repeats after a short delay
							setTimeout(() => {
								window.dispatchEvent(new CustomEvent("OmniSimulator_SYNC_AIL_REPEATS", {
									detail: {
										targetNodeId: node.id,
										mode: isTxt2Img ? "txt2img" : "img2img",
										repeats: promptRepeats
									}
								}));
							}, 100);
							alert(`✅ Added ${toAdd} items and synced to match ${promptCount} prompts`);
						} else {
							// Need to remove items - just update repeats for first N
							alert(`⚠️ You have ${myCount} items but only ${promptCount} prompts. Remove ${myCount - promptCount} items manually, or prompts will only use the first ${promptCount}.`);
						}
					} else {
						// No prompts - find connected AILs and sync to the one with most images
						console.log(`[AIL ${node.id}] 🔍 No prompts: Looking for other AILs to sync with`);

						// Find all AILs connected to the same RPG or BIG
						const findSiblingAILs = () => {
							const siblings = [];

							// Check our output connections to find RPG/BIG
							if (node.outputs && node.outputs.length > 0) {
								for (const output of node.outputs) {
									if (output.links && output.links.length > 0) {
										for (const linkId of output.links) {
											const link = app.graph.links[linkId];
											if (link) {
												const targetNode = app.graph.getNodeById(link.target_id);
												if (targetNode && (targetNode.type === "OmniSimulator_RealityPromptGenerator" || targetNode.type === "OmniSimulator_BatchImageGenerator")) {
													// Find all AILs connected to this target node
													const imageInputs = ["images", "images2", "images3", "images4"];
													for (const inputName of imageInputs) {
														const input = targetNode.inputs?.find(i => i.name === inputName);
														if (input?.link) {
															const ailLink = app.graph.links[input.link];
															if (ailLink) {
																const ailNode = app.graph.getNodeById(ailLink.origin_id);
																if (ailNode && ailNode.type === "OmniSimulator_AdvancedImageLoader" && ailNode.id !== node.id) {
																	try {
																		const ailData = JSON.parse(ailNode.properties.batch_data || "{}");
																		const ailCount = (ailData.order || []).length;
																		if (ailCount > 0) {
																			siblings.push({ node: ailNode, count: ailCount, inputName });
																		}
																	} catch (e) {}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
							return siblings;
						};

						const siblingAILs = findSiblingAILs();

						if (siblingAILs.length === 0) {
							alert("No other AILs with images found. Connect this AIL to RPG or BIG with other AILs.");
							return;
						}

						// Find the AIL with the most images
						const masterAIL = siblingAILs.reduce((max, ail) => ail.count > max.count ? ail : max, siblingAILs[0]);

						if (myCount === masterAIL.count) {
							alert(`✅ Already synced! Both have ${myCount} items.`);
							return;
						}

						const action = myCount < masterAIL.count ? "duplicate" : "trim";
						const diff = Math.abs(masterAIL.count - myCount);

						if (!confirm(`Sync this AIL to match ${masterAIL.count} items?\n\n${action === "duplicate" ? `📋 Duplicate last image ${diff} time(s)` : `🗑️ Remove ${diff} image(s)`}\n\nContinue?`)) {
							return;
						}

						if (myCount < masterAIL.count) {
							// Duplicate to match
							window.dispatchEvent(new CustomEvent("OmniSimulator_DUPLICATE_LAST_N", {
								detail: { targetNodeId: node.id, count: diff }
							}));
							alert(`✅ Duplicated ${diff} images to match ${masterAIL.count} items`);
						} else {
							// Trim to match
							window.dispatchEvent(new CustomEvent("OmniSimulator_TRIM_AIL_IMAGES", {
								detail: { targetNodeId: node.id, targetCount: masterAIL.count }
							}));
							alert(`✅ Trimmed to ${masterAIL.count} items`);
						}
					}
				};

				const setupEventHandlers = () => {
					// === Canvas Panning & Zoom Event Forwarding ===
					// Allow dragging/zooming canvas even when mouse is over custom UI
					const isInteractiveElement = (element) => {
						if (!element) return false;
						const tagName = element.tagName?.toLowerCase();
						if (['button', 'input', 'select', 'textarea', 'a'].includes(tagName)) return true;
						const className = element.className || '';
						if (typeof className === 'string' && (
							className.includes('-btn') || className.includes('-input') ||
							className.includes('-select') || className.includes('repeat-input') ||
							className.includes('gallery-item') || className.includes('checkbox')
						)) return true;
						if (element.onclick || element.onmousedown || element.draggable) return true;
						return false;
					};

					const isInteractiveOrChildOfInteractive = (target) => {
						let element = target;
						while (element && element !== container) {
							if (isInteractiveElement(element)) return true;
							element = element.parentElement;
						}
						return false;
					};

					// Track drag state for manual panning
					if (!container._ailDragState) {
						container._ailDragState = { isDragging: false, startX: 0, startY: 0, offsetX: 0, offsetY: 0 };
					}
					const dragState = container._ailDragState;

					const canvasForwardHandler = (e) => {
						if (e._ailForwarded) return;

						if (e.type === 'mousedown') {
							if (isInteractiveOrChildOfInteractive(e.target)) {
								dragState.isDragging = false;
								return;
							}
							dragState.isDragging = true;
							dragState.startX = e.clientX;
							dragState.startY = e.clientY;
							const canvas = app?.canvas || app?.graph?.list_of_graphcanvas?.[0];
							if (canvas) {
								dragState.offsetX = canvas.ds.offset[0];
								dragState.offsetY = canvas.ds.offset[1];
							}
							e.preventDefault();
							e.stopPropagation();
						}

						if (e.type === 'mousemove' && dragState.isDragging) {
							const deltaX = e.clientX - dragState.startX;
							const deltaY = e.clientY - dragState.startY;
							// Divide by scale to match native ComfyUI drag behavior at any zoom level
							const canvas = app?.canvas || app?.graph?.list_of_graphcanvas?.[0];
							if (canvas) {
								const scale = canvas.ds.scale || 1;
								canvas.ds.offset[0] = dragState.offsetX + (deltaX / scale);
								canvas.ds.offset[1] = dragState.offsetY + (deltaY / scale);
								canvas.setDirty(true, true);
							}
							e.preventDefault();
							e.stopPropagation();
						}

						if (e.type === 'mouseup' && dragState.isDragging) {
							dragState.isDragging = false;
							e.preventDefault();
							e.stopPropagation();
						}
					};

					const wheelForwardHandler = (e) => {
						if (e._ailForwarded) return;
						if (isInteractiveOrChildOfInteractive(e.target)) return;
						const canvas = document.querySelector('#graph-canvas') ||
									  document.querySelector('canvas.litegraph') ||
									  document.querySelector('.litegraph canvas');
						if (canvas) {
							e.preventDefault();
							e.stopPropagation();
							const newEvent = new WheelEvent('wheel', {
								bubbles: true, cancelable: true, view: window,
								clientX: e.clientX, clientY: e.clientY,
								deltaX: e.deltaX, deltaY: e.deltaY, deltaZ: e.deltaZ, deltaMode: e.deltaMode,
								ctrlKey: e.ctrlKey, shiftKey: e.shiftKey, altKey: e.altKey, metaKey: e.metaKey
							});
							newEvent._ailForwarded = true;
							canvas.dispatchEvent(newEvent);
						}
					};

					// Remove old listeners if exist
					if (container._canvasForwardHandler) {
						container.removeEventListener('mousedown', container._canvasForwardHandler, true);
						document.removeEventListener('mousemove', container._canvasForwardHandler, true);
						document.removeEventListener('mouseup', container._canvasForwardHandler, true);
					}
					if (container._wheelForwardHandler) {
						container.removeEventListener('wheel', container._wheelForwardHandler, true);
					}

					container._canvasForwardHandler = canvasForwardHandler;
					container._wheelForwardHandler = wheelForwardHandler;

					container.addEventListener('mousedown', canvasForwardHandler, true);
					document.addEventListener('mousemove', canvasForwardHandler, true);
					document.addEventListener('mouseup', canvasForwardHandler, true);
					container.addEventListener('wheel', wheelForwardHandler, true);

					const modeDropdown = container.querySelector(".omnisimulator-adv-loader-mode-dropdown");
					if (modeDropdown)
						modeDropdown.onchange = (e) => {
							const modeWidget = node.widgets?.find((w) => w.name === "mode");
							if (modeWidget) {
								modeWidget.value = e.target.value;
								renderGallery();
							}
						};
					const queueAllBtn = container.querySelector(".omnisimulator-adv-loader-queue-all-btn");
					if (queueAllBtn) queueAllBtn.onclick = queueAllImages;
					const uploadBtn = container.querySelector(".omnisimulator-adv-loader-upload-btn");
					if (uploadBtn)
						uploadBtn.onclick = () => {
							const input = document.createElement("input");
							input.type = "file";
							input.multiple = true;
							input.accept = "image/*";
							input.onchange = handleFileSelect;
							input.click();
						};
					const duplicateBtn = container.querySelector(".omnisimulator-adv-loader-duplicate-btn");
					if (duplicateBtn) duplicateBtn.onclick = duplicateLastItem;
					const syncBtn = container.querySelector(".omnisimulator-adv-loader-sync-btn");
					if (syncBtn) syncBtn.onclick = smartSync;
					const deleteAllBtn = container.querySelector(".omnisimulator-adv-loader-delete-all-btn");
					if (deleteAllBtn) deleteAllBtn.onclick = deleteAllImages;
					container.querySelectorAll(".omnisimulator-adv-loader-delete-btn").forEach((btn) => (btn.onclick = (e) => { e.stopPropagation(); deleteImage(btn.dataset.id); }));
					container.querySelectorAll(".omnisimulator-adv-loader-repeat-input").forEach((input) => {
						input.onchange = (e) => updateRepeatCount(input.dataset.id, parseInt(input.value) || 1);
						input.onmousedown = (e) => e.stopPropagation();
					});

					// Selection mode handlers
					const enterSelectionBtn = container.querySelector(".omnisimulator-adv-loader-enter-selection-btn");
					if (enterSelectionBtn) {
						enterSelectionBtn.onclick = () => {
							selectionMode = true;
							selectedImages.clear();
							renderGallery();
							console.log(`[OmniSimulator AIL ${node.id}] Entered selection mode`);
						};
					}

					const cancelSelectionBtn = container.querySelector(".omnisimulator-adv-loader-cancel-selection-btn");
					if (cancelSelectionBtn) {
						cancelSelectionBtn.onclick = () => {
							selectionMode = false;
							selectedImages.clear();
							renderGallery();
							console.log(`[OmniSimulator AIL ${node.id}] Exited selection mode`);
						};
					}

					const selectAllBtn = container.querySelector(".omnisimulator-adv-loader-select-all-btn");
					if (selectAllBtn) {
						selectAllBtn.onclick = () => {
							const batchData = JSON.parse(node.properties.batch_data || "{}");
							const order = batchData.order || [];
							order.forEach(id => selectedImages.add(String(id)));
							renderGallery();
							console.log(`[OmniSimulator AIL ${node.id}] Selected all: ${selectedImages.size} images`);
						};
					}

					const deselectAllBtn = container.querySelector(".omnisimulator-adv-loader-deselect-all-btn");
					if (deselectAllBtn) {
						deselectAllBtn.onclick = () => {
							selectedImages.clear();
							renderGallery();
							console.log(`[OmniSimulator AIL ${node.id}] Deselected all`);
						};
					}

					const deleteSelectedBtn = container.querySelector(".omnisimulator-adv-loader-delete-selected-btn");
					if (deleteSelectedBtn) {
						deleteSelectedBtn.onclick = async () => {
							if (selectedImages.size === 0) return;

							const confirmMsg = `Delete ${selectedImages.size} selected image${selectedImages.size === 1 ? '' : 's'}?\n\nThis cannot be undone.`;
							if (!confirm(confirmMsg)) return;

							try {
								const batchData = JSON.parse(node.properties.batch_data || "{}");
								const images = batchData.images || [];
								const order = batchData.order || [];

								// Remove selected images (selectedImages contains strings, img.id/order may be numbers)
								const selectedIds = Array.from(selectedImages);
								selectedIds.forEach(id => {
									// Compare as strings to handle type mismatch
									const imgIndex = images.findIndex(i => String(i.id) === id);
									if (imgIndex !== -1) images.splice(imgIndex, 1);
									const orderIndex = order.findIndex(orderId => String(orderId) === id);
									if (orderIndex !== -1) order.splice(orderIndex, 1);
								});

								// Recalculate total count
								batchData.total_count = images.reduce((sum, img) => sum + (img.repeat_count || 1), 0);

								node.properties.batch_data = JSON.stringify(batchData);
								syncBatchDataWidget();

								console.log(`[OmniSimulator AIL ${node.id}] Deleted ${selectedIds.length} images`);

								// Exit selection mode
								selectionMode = false;
								selectedImages.clear();
								renderGallery();

								alert(`Successfully deleted ${selectedIds.length} image${selectedIds.length === 1 ? '' : 's'}`);
							} catch (error) {
								console.error(`[OmniSimulator AIL ${node.id}] Error deleting selected images:`, error);
								alert(`Error deleting images: ${error.message}`);
							}
						};
					}

					// Checkbox handlers
					container.querySelectorAll(".omnisimulator-adv-loader-image-checkbox").forEach((checkbox) => {
						checkbox.onchange = (e) => {
							e.stopPropagation();
							const imageId = checkbox.dataset.id;

							if (checkbox.checked) {
								selectedImages.add(imageId);
							} else {
								selectedImages.delete(imageId);
							}

							renderGallery();
							console.log(`[OmniSimulator AIL ${node.id}] ${checkbox.checked ? 'Selected' : 'Deselected'} image ${imageId}. Total: ${selectedImages.size}`);
						};
						checkbox.onclick = (e) => e.stopPropagation();
					});

					// Allow clicking on the card to toggle selection in selection mode
					if (selectionMode) {
						container.querySelectorAll(".omnisimulator-adv-loader-item").forEach((item) => {
							item.onclick = (e) => {
								if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
								const imageId = item.dataset.id;
								if (selectedImages.has(imageId)) {
									selectedImages.delete(imageId);
								} else {
									selectedImages.add(imageId);
								}
								renderGallery();
							};
						});
					}
				};

				const setupDragAndDrop = () => {
					const items = container.querySelectorAll(".omnisimulator-adv-loader-item");
					let draggedItem = null;
					items.forEach((item) => {
						item.addEventListener("dragstart", (e) => {
							draggedItem = item;
							item.style.opacity = "0.5";
							e.dataTransfer.effectAllowed = "move";
							e.stopPropagation();
							e.dataTransfer.setData("text/plain", "omnisimulator-reorder");
						});
						item.addEventListener("dragend", () => {
							item.style.opacity = "1";
							items.forEach((i) => i.classList.remove("omnisimulator-adv-loader-drop-before", "omnisimulator-adv-loader-drop-after"));
						});
						item.addEventListener("dragover", (e) => {
							e.preventDefault();
							if (draggedItem === item) return;
							e.dataTransfer.dropEffect = "move";
							const rect = item.getBoundingClientRect();
							const midpoint = rect.top + rect.height / 2;
							items.forEach((i) => i.classList.remove("omnisimulator-adv-loader-drop-before", "omnisimulator-adv-loader-drop-after"));
							item.classList.add(e.clientY < midpoint ? "omnisimulator-adv-loader-drop-before" : "omnisimulator-adv-loader-drop-after");
						});
						item.addEventListener("drop", (e) => {
							e.preventDefault();
							if (draggedItem === item) return;
							const draggedId = draggedItem.dataset.id;
							const targetId = item.dataset.id;
							const batchData = JSON.parse(node.properties.batch_data || "{}");
							const order = batchData.order;
							const draggedIndex = order.indexOf(draggedId);
							order.splice(draggedIndex, 1);
							const rect = item.getBoundingClientRect();
							const insertAfter = e.clientY > rect.top + rect.height / 2;
							const newTargetIndex = order.indexOf(targetId);
							order.splice(insertAfter ? newTargetIndex + 1 : newTargetIndex, 0, draggedId);
							node.properties.batch_data = JSON.stringify(batchData);
							syncBatchDataWidget();
							renderGallery();
						});
					});
				};

				const setupFileDropZone = () => {
					// Prevent duplicate listeners
					if (container._hasFileDropListeners) return;
					container._hasFileDropListeners = true;

					let dragCounter = 0; // Track nested drag events
					let dragTimeout = null; // Timeout to auto-clear highlight

					const clearDragHighlight = () => {
						dragCounter = 0;
						container.classList.remove("omnisimulator-adv-loader-drag-over");
						if (dragTimeout) {
							clearTimeout(dragTimeout);
							dragTimeout = null;
						}
					};

					const handleFileDrop = async (files) => {
						console.log(`[OmniSimulator AIL ${node.id}] 📂 Files dropped:`, files.length);
						if (files.length === 0) return;

						const uploadBtn = container.querySelector(".omnisimulator-adv-loader-upload-btn");
						let originalText = null;

						// Update button state if it exists (might not exist in txt2img mode)
						if (uploadBtn) {
							originalText = uploadBtn.textContent;
							uploadBtn.textContent = "⏳ Uploading...";
							uploadBtn.disabled = true;
						} else {
							console.log(`[OmniSimulator AIL ${node.id}] ℹ️ No upload button (probably in txt2img mode), proceeding anyway...`);
						}

						try {
							const formData = new FormData();
							formData.append("node_id", node.id);
							files.forEach((file) => formData.append("files", file));
							const response = await fetch("/omnisimulator/batch_upload", { method: "POST", body: formData });
							const result = await response.json();
							console.log(`[OmniSimulator AIL ${node.id}] 📤 Upload result:`, result.success);
							console.log(`[OmniSimulator AIL ${node.id}] 📤 Images returned:`, result.images?.length || 0);
							if (result.success) {
								// Auto-switch to img2img mode FIRST if currently in txt2img mode
								const wasInTxt2ImgMode = isTxt2ImgMode();
								console.log(`[OmniSimulator AIL ${node.id}] 📊 Current mode - isTxt2ImgMode:`, wasInTxt2ImgMode);
								if (wasInTxt2ImgMode) {
									console.log(`[OmniSimulator AIL ${node.id}] 🔄 Auto-switching to img2img mode BEFORE adding images`);

									// Properly swap data (backup txt2img, restore img2img)
									node.properties.txt2img_data_backup = node.properties.batch_data;
									node.properties.batch_data = node.properties.img2img_data_backup || JSON.stringify({ images: [], order: [], total_count: 0 });
									console.log(`[OmniSimulator AIL ${node.id}] 💾 Backed up txt2img data, restored img2img data`);

									// Update the boolean
									switchToImg2ImgMode();

									// Manually update currentDetectedMode to prevent mode switch detection in renderGallery
									currentDetectedMode = false; // false = img2img mode
									console.log(`[OmniSimulator AIL ${node.id}] ✅ Updated currentDetectedMode to img2img`);
								} else {
									console.log(`[OmniSimulator AIL ${node.id}] ✅ Already in img2img mode, no switch needed`);
								}

								// NOW add images to batch_data (after mode switch completes)
								const batchData = JSON.parse(node.properties.batch_data || "{}");
								console.log(`[OmniSimulator AIL ${node.id}] 📦 Batch data before:`, batchData);
								batchData.images = batchData.images || [];
								batchData.order = batchData.order || [];
								result.images.forEach((img) => {
									batchData.images.push(img);
									batchData.order.push(img.id);
								});
								batchData.total_count = batchData.images.reduce((sum, img) => sum + (img.repeat_count || 1), 0);
								node.properties.batch_data = JSON.stringify(batchData);
								console.log(`[OmniSimulator AIL ${node.id}] 📦 Batch data after:`, JSON.parse(node.properties.batch_data));

								syncBatchDataWidget();
								renderGallery();
							} else {
								alert(`Upload failed: ${result.error}`);
							}
						} catch (error) {
							alert(`Upload error: ${error.message}`);
						} finally {
							// Reset button state if it exists
							if (uploadBtn && originalText) {
								uploadBtn.textContent = originalText;
								uploadBtn.disabled = false;
							}
						}
					};

					container.addEventListener("dragenter", (e) => {
						e.preventDefault();
						e.stopPropagation();
						dragCounter++;
						if (e.dataTransfer.types.includes("Files")) {
							container.classList.add("omnisimulator-adv-loader-drag-over");
							// Set timeout to auto-clear if drag is abandoned
							if (dragTimeout) clearTimeout(dragTimeout);
							dragTimeout = setTimeout(clearDragHighlight, 200);
						}
					});

					container.addEventListener("dragover", (e) => {
						e.preventDefault();
						e.stopPropagation();
						if (e.dataTransfer.types.includes("Files")) {
							e.dataTransfer.dropEffect = "copy";
							// Reset timeout while actively dragging over
							if (dragTimeout) clearTimeout(dragTimeout);
							dragTimeout = setTimeout(clearDragHighlight, 200);
						}
					});

					container.addEventListener("dragleave", (e) => {
						e.preventDefault();
						e.stopPropagation();
						dragCounter--;
						if (dragCounter === 0) {
							clearDragHighlight();
						}
					});

					container.addEventListener("drop", (e) => {
						e.preventDefault();
						e.stopPropagation();
						clearDragHighlight();

						// Only handle file drops, not reordering
						if (e.dataTransfer.getData("text/plain") === "omnisimulator-reorder") {
							return; // Let the reordering handler deal with this
						}

						const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith("image/"));
						if (files.length > 0) {
							handleFileDrop(files);
						}
					});

					// Global dragend listener to reset highlight when drag is cancelled
					document.addEventListener("dragend", clearDragHighlight);
					document.addEventListener("drop", clearDragHighlight);

					// Store cleanup function on node
					node._cleanupDragListeners = () => {
						document.removeEventListener("dragend", clearDragHighlight);
						document.removeEventListener("drop", clearDragHighlight);
						if (dragTimeout) clearTimeout(dragTimeout);
					};
				};

				const widget = node.addDOMWidget("batch_display", "batchloader", container, { getValue: () => node.properties.batch_data, setValue: (v) => { node.properties.batch_data = v; renderGallery(); }, serialize: false });
				widget.computeSize = (width) => [width, cachedHeight + 6];
				node._updateCachedHeight = updateCachedHeight;
				node._renderGallery = renderGallery;

				// Update existing latents with new dimensions
				const updateLatentsWithNewDimensions = () => {
					if (!isTxt2ImgMode()) return; // Only for txt2img mode

					const currentDims = getTxt2ImgDimensions();
					const batchData = JSON.parse(node.properties.batch_data || "{}");

					if (batchData.latents && batchData.latents.length > 0) {
						// Check if dimensions actually changed
						const firstLatent = batchData.latents[0];
						if (firstLatent.width !== currentDims.width ||
						    firstLatent.height !== currentDims.height ||
						    firstLatent.aspect_label !== currentDims.aspect_label) {

							console.log(`[OmniSimulator AIL ${node.id}] 🔄 Updating ${batchData.latents.length} latents: ${firstLatent.width}×${firstLatent.height} (${firstLatent.aspect_label}) → ${currentDims.width}×${currentDims.height} (${currentDims.aspect_label})`);

							batchData.latents.forEach(latent => {
								latent.width = currentDims.width;
								latent.height = currentDims.height;
								latent.aspect_label = currentDims.aspect_label;
							});
							node.properties.batch_data = JSON.stringify(batchData);
							syncBatchDataWidget();
						}
					}
				};

				// Add widget change callbacks to automatically refresh
				const setupWidgetCallbacks = () => {
					const modeWidget = node.widgets?.find((w) => w.name === "mode");
					if (modeWidget && !modeWidget._omnisimulator_callback_added) {
						const originalCallback = modeWidget.callback;
						modeWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							renderGallery();
						};
						modeWidget._omnisimulator_callback_added = true;
					}

					const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
					if (batchIndexWidget && !batchIndexWidget._omnisimulator_callback_added) {
						const originalCallback = batchIndexWidget.callback;
						batchIndexWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							renderGallery();
						};
						batchIndexWidget._omnisimulator_callback_added = true;
					}

					// Aspect ratio widgets - update latents AND re-render when dimensions change
					const widthWidget = node.widgets?.find((w) => w.name === "width");
					if (widthWidget && !widthWidget._omnisimulator_callback_added) {
						const originalCallback = widthWidget.callback;
						widthWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							updateLatentsWithNewDimensions();
							renderGallery();
						};
						widthWidget._omnisimulator_callback_added = true;
					}

					const heightWidget = node.widgets?.find((w) => w.name === "height");
					if (heightWidget && !heightWidget._omnisimulator_callback_added) {
						const originalCallback = heightWidget.callback;
						heightWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							updateLatentsWithNewDimensions();
							renderGallery();
						};
						heightWidget._omnisimulator_callback_added = true;
					}

					const aspectLabelWidget = node.widgets?.find((w) => w.name === "aspect_label");
					if (aspectLabelWidget && !aspectLabelWidget._omnisimulator_callback_added) {
						const originalCallback = aspectLabelWidget.callback;
						aspectLabelWidget.callback = function() {
							if (originalCallback) originalCallback.apply(this, arguments);
							updateLatentsWithNewDimensions();
							renderGallery();
						};
						aspectLabelWidget._omnisimulator_callback_added = true;
					}
				};

				// Periodic mode check - checks every 2 seconds if mode changed
				const startModeCheck = () => {
					if (modeCheckInterval) clearInterval(modeCheckInterval);
					modeCheckInterval = setInterval(() => {
						const detectedMode = isTxt2ImgMode();
						if (currentDetectedMode !== null && currentDetectedMode !== detectedMode) {
							renderGallery();
						}
					}, 2000);
					// Store on node for cleanup
					node._modeCheckInterval = modeCheckInterval;
				};

				// Periodic dimension check - checks every 2 seconds if dimensions changed
				let dimensionCheckInterval = null;
				let lastDimensions = null;
				const startDimensionCheck = () => {
					if (dimensionCheckInterval) clearInterval(dimensionCheckInterval);
					dimensionCheckInterval = setInterval(() => {
						// Only check dimensions in txt2img mode (where aspect ratio matters for latent display)
						if (isTxt2ImgMode()) {
							const currentDims = getTxt2ImgDimensions();
							const dimsKey = `${currentDims.width}x${currentDims.height}:${currentDims.aspect_label}`;
							if (lastDimensions !== null && lastDimensions !== dimsKey) {
								console.log(`[OmniSimulator AIL ${node.id}] Dimensions changed: ${lastDimensions} -> ${dimsKey}`);
								updateLatentsWithNewDimensions();
								renderGallery();
							}
							lastDimensions = dimsKey;
						}
					}, 2000);
					// Store on node for cleanup
					node._dimensionCheckInterval = dimensionCheckInterval;
				};

				const handleBatchUpdate = (event) => {
					const data = event.detail;
					if (data && data.node_id == node.id) {
						const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
						if (batchIndexWidget) batchIndexWidget.value = data.next_index;
						node._processingIndex = data.next_index;
						node._isProcessing = false;
						if (node._renderGallery) node._renderGallery();
						app.graph.setDirtyCanvas(true, false);
					}
				};
				api.addEventListener("omnisimulator_adv_loader_update", handleBatchUpdate);

				// Listen for Sync requests from RPG
				window.addEventListener("OmniSimulator_SYNC_AIL_LATENTS", (event) => {
					const { targetNodeId, latentSpecs, dimensions } = event.detail;
					if (node.id !== targetNodeId) return; // Not for this node

					console.log(`[OmniSimulator AIL ${node.id}] Received sync request: Create ${latentSpecs.length} empty latents with repeat counts`);

					// Get current dimensions or use provided
					const currentDimensions = getTxt2ImgDimensions();
					const width = dimensions?.width || currentDimensions.width;
					const height = dimensions?.height || currentDimensions.height;
					const aspect_label = dimensions?.aspect_label || currentDimensions.aspect_label;

					// Ensure we're working with a clean txt2img batch_data structure
					// If AIL hasn't swapped data yet, we need to do it now
					const oldBatchData = JSON.parse(node.properties.batch_data || "{}");
					if (oldBatchData.images && oldBatchData.images.length > 0) {
						// Still has img2img data - backup and swap
						console.log(`[OmniSimulator AIL ${node.id}] Sync: Swapping from img2img to txt2img data`);
						node.properties.img2img_data_backup = node.properties.batch_data;
					}

					// Create fresh txt2img batch_data (clear any stale images)
					const batchData = {
						latents: [],
						order: [],
						total_count: 0
					};

					// Create latents with repeat counts matching prompts
					let totalCount = 0;
					for (let i = 0; i < latentSpecs.length; i++) {
						const spec = latentSpecs[i];
						const newLatent = {
							id: generateUUID(),
							width: width,
							height: height,
							repeat_count: spec.repeat_count || 1,
							aspect_label: aspect_label
						};
						batchData.latents.push(newLatent);
						batchData.order.push(newLatent.id);
						totalCount += newLatent.repeat_count;
					}

					batchData.total_count = totalCount;
					node.properties.batch_data = JSON.stringify(batchData);

					// Update mode tracking so renderGallery doesn't re-swap
					currentDetectedMode = true; // true = txt2img mode

					syncBatchDataWidget();
					renderGallery();

					console.log(`[OmniSimulator AIL ${node.id}] Created ${latentSpecs.length} latents (${totalCount} total generations) (${width}×${height})`);
				});

				// Listen for Repeat Sync requests from RPG
				window.addEventListener("OmniSimulator_SYNC_AIL_REPEATS", (event) => {
					const { targetNodeId, mode, repeats } = event.detail;
					if (node.id !== targetNodeId) return; // Not for this node

					console.log(`[OmniSimulator AIL ${node.id}] Received repeat sync request: Update ${repeats.length} items`);

					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const items = mode === "img2img" ? batchData.images : batchData.latents;
					const order = batchData.order || [];

					if (!items || items.length === 0) {
						console.warn(`[OmniSimulator AIL ${node.id}] No ${mode === "img2img" ? "images" : "latents"} in AIL to sync (mode: ${mode})`);
						return;
					}

					// Update repeat counts to match prompts, respecting display order
					let totalCount = 0;
					repeats.forEach((repeatCount, idx) => {
						// Find item by order, not by raw array index
						const itemId = order[idx];
						if (itemId) {
							const item = items.find(i => i.id === itemId);
							if (item) {
								item.repeat_count = repeatCount;
								totalCount += repeatCount;
							}
						}
					});

					batchData.total_count = totalCount;
					node.properties.batch_data = JSON.stringify(batchData);

					syncBatchDataWidget();
					renderGallery();

					console.log(`[OmniSimulator AIL ${node.id}] Synced repeat counts: ${repeats.length} items, ${totalCount} total`);
				});

				// Listen for Duplicate Last N requests from RPG
				window.addEventListener("OmniSimulator_DUPLICATE_LAST_N", (event) => {
					const { targetNodeId, count } = event.detail;
					if (node.id !== targetNodeId) return; // Not for this node

					console.log(`[OmniSimulator AIL ${node.id}] Received duplicate last N request: Duplicate last item ${count} times`);

					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const isTxt2Img = isTxt2ImgMode();
					const items = isTxt2Img ? (batchData.latents || []) : (batchData.images || []);
					const order = batchData.order || [];

					if (items.length === 0 || order.length === 0) {
						console.warn(`[OmniSimulator AIL ${node.id}] No items to duplicate!`);
						return;
					}

					// Get last item ID from order
					const lastItemId = order[order.length - 1];
					const lastItem = items.find(item => item.id === lastItemId);

					if (!lastItem) {
						console.warn(`[OmniSimulator AIL ${node.id}] Last item not found!`);
						return;
					}

					// Duplicate N times
					for (let i = 0; i < count; i++) {
						const duplicatedItem = {
							...lastItem,
							id: Date.now() + Math.random() + i,
							repeat_count: 1
						};
						items.push(duplicatedItem);
						order.push(duplicatedItem.id);
					}

					// Update batch data
					if (isTxt2Img) {
						batchData.latents = items;
					} else {
						batchData.images = items;
					}
					batchData.order = order;
					batchData.total_count = items.reduce((sum, item) => sum + (item.repeat_count || 1), 0);

					node.properties.batch_data = JSON.stringify(batchData);
					syncBatchDataWidget();
					renderGallery();  // Already dispatches OmniSimulator_AIL_UPDATED with proper format

					console.log(`[OmniSimulator AIL ${node.id}] Duplicated last item ${count} times, total items: ${items.length}, total count: ${batchData.total_count}`);
				});

				// Listen for Trim Images requests from RPG
				window.addEventListener("OmniSimulator_TRIM_AIL_IMAGES", (event) => {
					const { targetNodeId, targetCount } = event.detail;
					if (node.id !== targetNodeId) return; // Not for this node

					console.log(`[OmniSimulator AIL ${node.id}] Received trim request: Trim to ${targetCount} images`);

					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const images = batchData.images || [];
					const order = batchData.order || [];

					if (images.length === 0 || order.length === 0) {
						console.warn(`[OmniSimulator AIL ${node.id}] No images to trim!`);
						return;
					}

					if (targetCount >= order.length) {
						console.warn(`[OmniSimulator AIL ${node.id}] Target count ${targetCount} >= current count ${order.length}, no trimming needed`);
						return;
					}

					// Remove excess images from the end
					const excessCount = order.length - targetCount;
					const removedIds = order.splice(targetCount, excessCount);

					// Remove the actual image objects
					batchData.images = images.filter(img => !removedIds.includes(img.id));

					// Update order and total count
					batchData.order = order;
					batchData.total_count = batchData.images.reduce((sum, img) => sum + (img.repeat_count || 1), 0);

					node.properties.batch_data = JSON.stringify(batchData);
					syncBatchDataWidget();
					renderGallery();  // Already dispatches OmniSimulator_AIL_UPDATED with proper format

					console.log(`[OmniSimulator AIL ${node.id}] Trimmed ${excessCount} image${excessCount > 1 ? 's' : ''}, total images: ${batchData.images.length}, total count: ${batchData.total_count}`);
				});

				// Listen for Sync from Master AIL broadcast (when no prompts, another AIL is master)
				window.addEventListener("OmniSimulator_SYNC_FROM_MASTER_AIL", (event) => {
					const { masterNodeId, count, repeats, mode } = event.detail;
					if (node.id === masterNodeId) return; // Don't sync to ourselves

					console.log(`[OmniSimulator AIL ${node.id}] Received sync from master AIL ${masterNodeId}: ${count} items`);

					const batchData = JSON.parse(node.properties.batch_data || "{}");
					const isTxt2Img = isTxt2ImgMode();
					const items = isTxt2Img ? (batchData.latents || []) : (batchData.images || []);
					const order = batchData.order || [];
					const myCount = order.length;

					if (myCount === 0) {
						console.log(`[OmniSimulator AIL ${node.id}] No items to sync, skipping`);
						return;
					}

					// Sync to match the master AIL's count
					if (myCount === count) {
						// Same count, sync repeats
						window.dispatchEvent(new CustomEvent("OmniSimulator_SYNC_AIL_REPEATS", {
							detail: {
								targetNodeId: node.id,
								mode: isTxt2Img ? "txt2img" : "img2img",
								repeats: repeats
							}
						}));
						console.log(`[OmniSimulator AIL ${node.id}] Synced repeat counts to match master AIL`);
					} else if (myCount < count) {
						// Need to add items
						const toAdd = count - myCount;
						window.dispatchEvent(new CustomEvent("OmniSimulator_DUPLICATE_LAST_N", {
							detail: { targetNodeId: node.id, count: toAdd }
						}));
						// Then sync repeats
						setTimeout(() => {
							window.dispatchEvent(new CustomEvent("OmniSimulator_SYNC_AIL_REPEATS", {
								detail: {
									targetNodeId: node.id,
									mode: isTxt2Img ? "txt2img" : "img2img",
									repeats: repeats
								}
							}));
						}, 100);
						console.log(`[OmniSimulator AIL ${node.id}] Added ${toAdd} items to match master AIL`);
					} else {
						// Need to trim items
						const targetCount = count;
						window.dispatchEvent(new CustomEvent("OmniSimulator_TRIM_AIL_IMAGES", {
							detail: { targetNodeId: node.id, targetCount: targetCount }
						}));
						// Then sync repeats
						setTimeout(() => {
							window.dispatchEvent(new CustomEvent("OmniSimulator_SYNC_AIL_REPEATS", {
								detail: {
									targetNodeId: node.id,
									mode: isTxt2Img ? "txt2img" : "img2img",
									repeats: repeats
								}
							}));
						}, 100);
						console.log(`[OmniSimulator AIL ${node.id}] Trimmed to ${targetCount} items to match master AIL`);
					}
				});

				setTimeout(() => {
					const batchIndexWidget = node.widgets?.find((w) => w.name === "batch_index");
					if (batchIndexWidget && batchIndexWidget.value !== undefined) node._processingIndex = batchIndexWidget.value;
					syncBatchDataWidget();
					setupWidgetCallbacks();
					startModeCheck();
					startDimensionCheck();
					renderGallery();
				}, 100);
			};

			const onResize = nodeType.prototype.onResize;
			nodeType.prototype.onResize = function (size) {
				onResize?.apply(this, arguments);
				if (this._updateCachedHeight) {
					clearTimeout(this._resizeTimeout);
					this._resizeTimeout = setTimeout(() => this._updateCachedHeight(), 50);
				}
			};

			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function (data) {
				onConfigure?.apply(this, arguments);
				setTimeout(() => {
					// VALIDATION: Sanitize corrupted dimensions in loaded batch_data
					try {
						const batchData = JSON.parse(this.properties.batch_data || "{}");
						let needsFixing = false;

						// Check and fix latents (txt2img mode)
						if (batchData.latents && Array.isArray(batchData.latents)) {
							batchData.latents.forEach(latent => {
								if (latent.width < 64 || latent.height < 64) {
									console.warn(`[OmniSimulator AIL ${this.id}] ⚠️ Corrupted latent dimensions detected: ${latent.width}x${latent.height}`);
									latent.width = 512;
									latent.height = 512;
									latent.aspect_label = "1:1";
									needsFixing = true;
								}
							});
						}

						if (needsFixing) {
							console.log(`[OmniSimulator AIL ${this.id}] ✅ Fixed corrupted dimensions in batch_data`);
							this.properties.batch_data = JSON.stringify(batchData);
						}
					} catch (e) {
						console.error(`[OmniSimulator AIL ${this.id}] Error validating batch_data:`, e);
					}

					const batchDataWidget = this.widgets?.find((w) => w.name === "batch_data");
					if (batchDataWidget) batchDataWidget.value = this.properties.batch_data || "{}";
					const batchIndexWidget = this.widgets?.find((w) => w.name === "batch_index");
					if (batchIndexWidget && batchIndexWidget.value !== undefined) this._processingIndex = batchIndexWidget.value;
					if (this._renderGallery) this._renderGallery();

					// Force multiple height recalculations to ensure sync
					if (this._updateCachedHeight) {
						setTimeout(() => this._updateCachedHeight(), 100);
						setTimeout(() => this._updateCachedHeight(), 500);
						setTimeout(() => this._updateCachedHeight(), 1000);
					}
				}, 200);
			};

			const onRemoved = nodeType.prototype.onRemoved;
			nodeType.prototype.onRemoved = function () {
				// Clean up the periodic mode check interval
				if (this._modeCheckInterval) {
					clearInterval(this._modeCheckInterval);
					this._modeCheckInterval = null;
				}
				// Clean up the periodic dimension check interval
				if (this._dimensionCheckInterval) {
					clearInterval(this._dimensionCheckInterval);
					this._dimensionCheckInterval = null;
				}
				// Clean up the height sync interval
				if (this._heightSyncInterval) {
					clearInterval(this._heightSyncInterval);
					this._heightSyncInterval = null;
				}
				onRemoved?.apply(this, arguments);
			};
		}
	},
});