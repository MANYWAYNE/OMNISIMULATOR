import sys
import os
import time
import json
from typing import Optional, List, Dict
from server import PromptServer
from aiohttp import web
from comfy.model_management import InterruptProcessingException, throw_exception_if_processing_interrupted

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
if _ROOT not in sys.path: 
    sys.path.insert(0, _ROOT)

REQUEST_RESHOW = "-1"
CANCEL = "-3"
WAITING_FOR_RESPONSE = "-9"

SPECIALS = [REQUEST_RESHOW, CANCEL, WAITING_FOR_RESPONSE]

class Response:
    def __init__(self, selection: Optional[List[str]] = None, text: Optional[str] = None,
                 masked_image: Optional[str] = None, masked_data: Optional[str] = None,
                 extras: Optional[List[str]] = None, crop: Optional[Dict] = None):
        self.selection: List[int] = [int(x) for x in selection] if selection else []
        self.text: Optional[str] = text
        self.masked_image: Optional[str] = masked_image
        self.masked_data: Optional[str] = masked_data  
        self.extras: Optional[List[str]] = extras
        self.crop: Optional[Dict] = crop

    def get_extras(self, defaults: list[str]) -> list[str]:
        return self.extras or defaults  

class TimeoutResponse(Response): pass
class CancelledResponse(Response): pass
class RequestResponse(Response): pass

class MessageState:
    _latest: 'Optional[MessageState]' = None
    unique_expected = None

    def __init__(self, data: dict | str = {}):
        data_dict: dict = data if isinstance(data, dict) else json.loads(data)
        self.unique: str = data_dict.pop('unique', None)
        self.special: Optional[str] = data_dict.pop('special', None)
        self.response: Response = Response(**data_dict)

    @classmethod
    def latest(cls) -> 'MessageState': 
        if cls._latest is None: 
            cls._latest = cls()
        return cls._latest 
    
    @classmethod
    def set_latest(cls, latest: 'MessageState'):
        cls._latest = latest

    @classmethod
    def waiting_state(cls): 
        return MessageState(data={'special': WAITING_FOR_RESPONSE})

    @classmethod
    def request_state(cls): 
        return MessageState(data={'special': REQUEST_RESHOW})

    @classmethod
    def start_waiting(cls, unique): 
        cls._latest = cls.waiting_state()
        cls.unique_expected = unique

    @classmethod
    def get_response(cls) -> Response:  
        if cls.waiting(): 
            return TimeoutResponse()
        if cls.latest().cancelled: 
            return CancelledResponse()
        if cls.latest().request: 
            return RequestResponse()
        return cls.latest().response

    @classmethod
    def stop_waiting(cls): 
        cls._latest = MessageState()

    @classmethod
    def waiting(cls) -> bool: 
        return cls.latest().special == WAITING_FOR_RESPONSE

    @property
    def cancelled(self) -> bool: 
        return self.special == CANCEL

    @property
    def request(self) -> bool: 
        return self.special == REQUEST_RESHOW

    @property
    def real(self) -> bool: 
        return self.special is None


@PromptServer.instance.routes.post('/omni/interactive_message')
async def cg_image_filter_message(request):
    post = await request.post()
    response = post.get("response")
    message = MessageState(response)

    if str(MessageState.unique_expected) == str(message.unique):
        if MessageState.waiting():
            MessageState.set_latest(message)
        else:
            print(f"[OmniSimulator] Ignoring response {response} because not waiting for one")
    else:
        print(f"[OmniSimulator] Ignoring mismatched response {response}")

    return web.json_response({})


@PromptServer.instance.routes.post('/omni/clear_text_filter_cache')
async def clear_text_filter_cache(request):
    try:
        data = await request.json()
        uid = data.get('uid')
        print(f"[OmniSimulator] OMNI API: Received request to clear text filter cache from node UID {uid}.")

        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        if not os.path.isdir(cache_dir):
            print("[OmniSimulator] OMNI API: Cache directory does not exist. Nothing to clear.")
            return web.json_response({"status": "success", "cleared_count": 0, "message": "Cache directory not found."})

        cleared_count = 0
        files_cleared = []
        
        for filename in os.listdir(cache_dir):
            if filename.endswith("_text_edit.json"):
                file_path = os.path.join(cache_dir, filename)
                try:
                    os.remove(file_path)
                    files_cleared.append(filename)
                    cleared_count += 1
                except OSError as e:
                    print(f"[OmniSimulator] OMNI API: Error clearing cache file {file_path}: {e}")
        
        print(f"[OmniSimulator] OMNI API: Successfully cleared {cleared_count} text filter cache file(s).")
        return web.json_response({"status": "success", "cleared_count": cleared_count, "files": files_cleared})

    except Exception as e:
        print(f"[OmniSimulator] OMNI API: An unexpected error occurred while clearing cache: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)


def wait_for_response(secs, uid, unique) -> Response:
    MessageState.start_waiting(unique)
    try:
        end_time = time.monotonic() + secs
        grace_period_end = time.monotonic() + 2.0

        while time.monotonic() < end_time and MessageState.waiting():
            if time.monotonic() > grace_period_end:
                throw_exception_if_processing_interrupted()
            PromptServer.instance.send_sync("omni-interactive-images", {"tick": int(end_time - time.monotonic()), "uid": uid, "unique": unique})
            time.sleep(0.5)
        if MessageState.waiting():
            PromptServer.instance.send_sync("omni-interactive-images", {"timeout": True, "uid": uid, "unique": unique})
        return MessageState.get_response()
    finally: 
        MessageState.stop_waiting()
    
def send_and_wait(payload, timeout, uid, unique) -> Response:
    payload['uid'] = uid
    payload['unique'] = unique

    while True:
        PromptServer.instance.send_sync("omni-interactive-images", payload)
        r = wait_for_response(timeout, uid, unique)
        if isinstance(r, CancelledResponse): 
            raise InterruptProcessingException()
        if not isinstance(r, RequestResponse): 
            return r