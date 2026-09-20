## A session in Godot's own process, through the `OrlaSession` class of
## the `orla-godot` GDExtension (`cargo build --release -p orla-godot`),
## wrapped to answer like `OrlaClient`: `request(op, fields)` returns the
## `ok` payload or {"error": message}, so the view and the demo drive
## either. What differs is the frame: `evaluate(input)` and `image()`
## hand the picture over as an RGBAF Image built on the Host's words, no
## socket and no encoding between.
class_name OrlaLocal
extends RefCounted

const EXTENSION := "res://orla.gdextension"

## The native session, or null.
var native: RefCounted


## Whether the extension is loaded, loading it from its descriptor when
## it is not yet (a project run from the editor has it loaded already;
## a headless `-s` run does not).
static func available() -> bool:
	if ClassDB.class_exists("OrlaSession"):
		return true
	if not FileAccess.file_exists(EXTENSION):
		return false
	var status := GDExtensionManager.load_extension(EXTENSION)
	return status == GDExtensionManager.LOAD_STATUS_OK or status == GDExtensionManager.LOAD_STATUS_ALREADY_LOADED


## Opens a scene (a `res://` or file path to a `.osn`) on a backend; an
## error is returned as a String, an empty String on success.
func open(scene: String, backend: String = "jit") -> String:
	if not available():
		return "the orla-godot extension is not built (cargo build --release -p orla-godot)"
	native = ClassDB.instantiate("OrlaSession")
	if not native.open(scene, backend):
		var error: String = native.error()
		native = null
		return error
	return ""


func is_open() -> bool:
	return native != null and native.is_open()


func close() -> void:
	if native:
		native.close()
	native = null


## One request of the protocol, answered like OrlaClient.
func request(op: String, fields: Dictionary = {}) -> Dictionary:
	if not is_open():
		return {"error": "no scene is open"}
	var line := fields.duplicate()
	line["op"] = op
	var parsed = JSON.parse_string(native.request(JSON.stringify(line)))
	if typeof(parsed) != TYPE_DICTIONARY:
		return {"error": "the session answered something that is not JSON"}
	if parsed.has("error"):
		var detail = parsed["error"]
		var result := {"error": str(detail.get("message", detail))}
		if detail is Dictionary:
			if detail.has("line"):
				result["line"] = int(detail["line"])
			if detail.has("column"):
				result["column"] = int(detail["column"])
		return result
	var ok = parsed.get("ok", {})
	return ok if ok is Dictionary else {"value": ok}


## One evaluation under `input` (the keys `OrlaSession.evaluate` takes:
## time, pointer, buttons, wheel, keys, window, scale); the report as a
## Dictionary, or {"error": message}.
func evaluate(input: Dictionary) -> Dictionary:
	if not is_open():
		return {"error": "no scene is open"}
	if not native.evaluate(input):
		return {"error": str(native.error())}
	var report = JSON.parse_string(native.report())
	return report if report is Dictionary else {}


## The first graph output as an RGBAF Image, null when it is not a picture.
func image(name: String = "") -> Image:
	if not is_open():
		return null
	return native.output_image(name)


## A node's port as a Dictionary of `words` (PackedByteArray) and its
## layout; pin the node first when fusion would inline it.
func node(node_name: String, port: String = "") -> Dictionary:
	if not is_open():
		return {}
	return native.node(node_name, port)
