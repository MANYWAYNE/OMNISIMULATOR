import os
import uuid
from pathlib import Path
from PIL import Image
import folder_paths

POOL_DIR = os.path.join(folder_paths.get_input_directory(), "omni_pool")

def setup_routes():
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        print("[OmniSimulator] Could not set up server routes (server module not available).")
        return

    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        print("[OmniSimulator] PromptServer instance is not ready yet.")
        return

    routes = PromptServer.instance.routes
    os.makedirs(POOL_DIR, exist_ok=True)

    @routes.post("/omni/batch_upload")
    async def batch_upload(request):
        os.makedirs(POOL_DIR, exist_ok=True)
        try:
            reader = await request.multipart()
            images = []
            while True:
                part = await reader.next()
                if part is None: 
                    break
                if part.name in ("files", "file"):
                    filename = part.filename or "upload.jpg"
                    data = await part.read()
                    img_id = str(uuid.uuid4())
                    ext = Path(filename).suffix.lower() or ".jpg"
                    save_name = f"{img_id}{ext}"
                    save_path = os.path.join(POOL_DIR, save_name)
                    with open(save_path, 'wb') as f:
                        f.write(data)
                    try:
                        pil = Image.open(save_path).convert("RGB")
                        w, h = pil.size
                        resample_method = getattr(Image, "LANCZOS", None) or Image.Resampling.LANCZOS
                        pil.thumbnail((512, 512), resample_method)
                        thumb_name = f"thumb_{img_id}.jpg"
                        pil.save(os.path.join(POOL_DIR, thumb_name), "JPEG", quality=85)
                        images.append({
                            "id": img_id, 
                            "thumbnail": thumb_name,
                            "original_name": filename, 
                            "width": w, 
                            "height": h, 
                            "repeat_count": 1
                        })
                    except Exception as e:
                        if os.path.exists(save_path): 
                            os.remove(save_path)
                        print(f"[OmniSimulator] Upload error: {e}")
            return web.json_response({"success": True, "images": images})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.get("/omni/view/{filename}")
    async def view_image(request):
        filename = os.path.basename(request.match_info["filename"])
        file_path = os.path.join(POOL_DIR, filename)
        if not os.path.exists(file_path):
            raise web.HTTPNotFound()
        ext = Path(filename).suffix.lower()
        content_type = {
            ".jpg": "image/jpeg", 
            ".jpeg": "image/jpeg", 
            ".png": "image/png",
            ".webp": "image/webp"
        }.get(ext, "image/jpeg")
        with open(file_path, 'rb') as f:
            data = f.read()
        return web.Response(body=data, content_type=content_type, headers={"Cache-Control": "public, max-age=3600"})

    @routes.delete("/omni/batch_delete/{node_id}/{image_id}")
    async def batch_delete(request):
        image_id = request.match_info["image_id"]
        removed = 0
        if os.path.exists(POOL_DIR):
            for f in os.listdir(POOL_DIR):
                if image_id in f:
                    try: 
                        os.remove(os.path.join(POOL_DIR, f))
                        removed += 1
                    except: 
                        pass
        return web.json_response({"success": True, "removed": removed})

    @routes.post("/omni/migrate_images")
    async def migrate_images(request):
        import shutil
        migrated = 0
        input_dir = folder_paths.get_input_directory()
        os.makedirs(POOL_DIR, exist_ok=True)
        if os.path.exists(input_dir):
            for folder in os.listdir(input_dir):
                if folder.startswith("instaraw_") or folder.startswith("omni_node_"):
                    old_dir = os.path.join(input_dir, folder)
                    if os.path.isdir(old_dir):
                        for fn in os.listdir(old_dir):
                            src = os.path.join(old_dir, fn)
                            dst = os.path.join(POOL_DIR, fn)
                            if not os.path.exists(dst):
                                shutil.move(src, dst)
                                migrated += 1
        return web.json_response({"success": True, "migrated": migrated})

    print("[OmniSimulator] Server routes successfully registered: /omni/batch_upload, /omni/view, /omni/batch_delete, /omni/migrate_images")