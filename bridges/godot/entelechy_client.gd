## Entelechy's creation protocol from Godot: JSON lines over a TCP socket to a
## session serving with `--listen`, the same lines Blender, a browser or a
## shell would send. A client of the public contract and nothing more;
## nothing here knows how Entelechy is built.
##
## Requests block until the session answers (it answers one line per
## request, in order), which keeps a frame's evaluate-then-read simple; a
## request that fails returns a Dictionary with an "error" key holding
## the message and, when the session gave one, "line" and "column".
class_name EntelechyClient
extends RefCounted

var peer: StreamPeerTCP
var _buffer := PackedByteArray()
## Milliseconds a request may take before it is given up as an error.
var timeout_ms := 10000


## Connects to `host:port` and waits for the socket; an error is returned
## as a String, an empty String on success.
func open(host: String, port: int) -> String:
	peer = StreamPeerTCP.new()
	var err := peer.connect_to_host(host, port)
	if err != OK:
		return "cannot connect to %s:%d (%s)" % [host, port, error_string(err)]
	var deadline := Time.get_ticks_msec() + timeout_ms
	while peer.get_status() == StreamPeerTCP.STATUS_CONNECTING and Time.get_ticks_msec() < deadline:
		peer.poll()
		OS.delay_msec(2)
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return "cannot connect to %s:%d" % [host, port]
	# A frame is two small requests; they go out at once.
	peer.set_no_delay(true)
	return ""


func is_open() -> bool:
	return peer != null and peer.get_status() == StreamPeerTCP.STATUS_CONNECTED


func close() -> void:
	if peer:
		if is_open():
			peer.put_data((JSON.stringify({"op": "quit"}) + "\n").to_utf8_buffer())
		peer.disconnect_from_host()
	peer = null


## One request: the `ok` payload as a Dictionary, or {"error": message}.
func request(op: String, fields: Dictionary = {}) -> Dictionary:
	if not is_open():
		return {"error": "not connected"}
	var line := fields.duplicate()
	line["op"] = op
	var sent := peer.put_data((JSON.stringify(line) + "\n").to_utf8_buffer())
	if sent != OK:
		return {"error": "the connection closed"}
	var text := _read_line()
	if text.is_empty():
		return {"error": "no answer from the session"}
	var parsed = JSON.parse_string(text)
	if typeof(parsed) != TYPE_DICTIONARY:
		return {"error": "the session answered something that is not JSON: " + text}
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


## Whether a result is a failure.
static func failed(result: Dictionary) -> bool:
	return result.has("error")


func _read_line() -> String:
	var deadline := Time.get_ticks_msec() + timeout_ms
	while Time.get_ticks_msec() < deadline:
		var at := _buffer.find(10)
		if at >= 0:
			var line := _buffer.slice(0, at).get_string_from_utf8()
			_buffer = _buffer.slice(at + 1)
			return line
		peer.poll()
		if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			return ""
		var available := peer.get_available_bytes()
		if available > 0:
			var got := peer.get_partial_data(available)
			if got[0] == OK:
				_buffer.append_array(got[1])
		else:
			OS.delay_usec(200)
	return ""


# ---- Decoders of what `read` returns. ----


## A `read` in `ppm` format as an RGB8 Image.
static func decode_ppm(base64: String) -> Image:
	var bytes := Marshalls.base64_to_raw(base64)
	# "P6\n<width> <height>\n255\n" then the pixels.
	var fields: Array[String] = []
	var cursor := 0
	while fields.size() < 4 and cursor < bytes.size():
		while cursor < bytes.size() and _is_space(bytes[cursor]):
			cursor += 1
		var start := cursor
		while cursor < bytes.size() and not _is_space(bytes[cursor]):
			cursor += 1
		fields.append(bytes.slice(start, cursor).get_string_from_ascii())
	cursor += 1
	if fields.size() < 4 or fields[0] != "P6":
		push_error("not a binary PPM")
		return null
	var width := int(fields[1])
	var height := int(fields[2])
	return Image.create_from_data(width, height, false, Image.FORMAT_RGB8, bytes.slice(cursor, cursor + width * height * 3))


static func _is_space(byte: int) -> bool:
	return byte == 32 or byte == 10 or byte == 13 or byte == 9


## A `read` of a Field that is not pixels: its words, little-endian
## 32-bit values, read with `float_at`, `int_at` and `bool_at`.
static func decode_words(base64: String) -> PackedByteArray:
	return Marshalls.base64_to_raw(base64)


static func float_at(words: PackedByteArray, index: int) -> float:
	return words.decode_float(index * 4)


static func int_at(words: PackedByteArray, index: int) -> int:
	return words.decode_s32(index * 4)


static func bool_at(words: PackedByteArray, index: int) -> bool:
	return words.decode_u32(index * 4) != 0


## Where a frame of `size` lands in a `target` rectangle, aspect kept,
## centered: the Player's placement.
static func placement(size: Vector2i, target: Vector2i) -> Rect2i:
	if size.x <= 0 or size.y <= 0 or target.x <= 0 or target.y <= 0:
		return Rect2i()
	var scale := minf(float(target.x) / size.x, float(target.y) / size.y)
	var w := maxi(1, int(size.x * scale))
	var h := maxi(1, int(size.y * scale))
	return Rect2i(int((target.x - w) / 2.0), int((target.y - h) / 2.0), w, h)


## Two nodes' rows as an ArrayMesh: vertices of eight or nine floats
## (position, normal, uv, as the mesh and model examples produce) and
## triangles of three ints. The nodes are pinned first so fusion keeps
## them, then evaluated once. Returns null on failure, the message pushed.
func mesh_of(vertex_node: String, triangle_node: String, vertex_port := "result", triangle_port := "result") -> ArrayMesh:
	for node in [vertex_node, triangle_node]:
		var pinned := request("pin", {"node": node})
		if failed(pinned):
			push_error(pinned["error"])
			return null
	var evaluated := request("evaluate", {"nodes": [vertex_node, triangle_node]})
	if failed(evaluated):
		push_error(evaluated["error"])
		return null
	var vertices := request("read", {"node": vertex_node, "port": vertex_port, "format": "words"})
	var triangles := request("read", {"node": triangle_node, "port": triangle_port, "format": "words"})
	if failed(vertices) or failed(triangles):
		push_error(str(vertices.get("error", triangles.get("error"))))
		return null
	var stride := str(vertices["schema"]).count(",") + 1
	if stride < 8:
		push_error("vertex rows of %d floats; eight or nine expected" % stride)
		return null
	var words := decode_words(vertices["data"])
	var count := int(vertices["count"])
	var positions := PackedVector3Array()
	var normals := PackedVector3Array()
	var uvs := PackedVector2Array()
	positions.resize(count)
	normals.resize(count)
	uvs.resize(count)
	# Nine floats are clip x y z w, normal, uv; eight are position, normal, uv.
	var normal_at := 4 if stride >= 9 else 3
	for i in count:
		var base := i * stride
		positions[i] = Vector3(float_at(words, base), float_at(words, base + 1), float_at(words, base + 2))
		normals[i] = Vector3(float_at(words, base + normal_at), float_at(words, base + normal_at + 1), float_at(words, base + normal_at + 2))
		uvs[i] = Vector2(float_at(words, base + normal_at + 3), float_at(words, base + normal_at + 4))
	var tri_words := decode_words(triangles["data"])
	var tri_count := int(triangles["count"])
	var indices := PackedInt32Array()
	indices.resize(tri_count * 3)
	for t in tri_count:
		# Entelechy winds counter-clockwise as glTF does; Godot's front face is
		# clockwise, so two corners swap.
		indices[t * 3] = int_at(tri_words, t * 3)
		indices[t * 3 + 1] = int_at(tri_words, t * 3 + 2)
		indices[t * 3 + 2] = int_at(tri_words, t * 3 + 1)
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = positions
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_INDEX] = indices
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return mesh
