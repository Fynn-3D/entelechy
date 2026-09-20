## Checks the bridge headless against three sessions it serves itself:
##
##   cargo build --release
##   godot --headless --path bridges/godot -s test.gd [-- --orla <path>]
##
## The text example's picture is read and its background republished, the
## voxel editor is driven by an OrlaView with a window and a pointer, and
## the mesh example's torus comes out as an ArrayMesh. Prints one line per
## check and ends with `RESULT: ok`, or names the first failure.
extends SceneTree

const Client = preload("res://orla_client.gd")
const Local = preload("res://orla_local.gd")
const View = preload("res://orla_view.gd")

var _sessions: Array[Dictionary] = []


func _initialize() -> void:
	# From the first frame on, so a scene added to the root gets its
	# `_ready` as it would in a running game.
	process_frame.connect(_first_frame, CONNECT_ONE_SHOT)


func _first_frame() -> void:
	var args := OS.get_cmdline_user_args()
	var orla := ProjectSettings.globalize_path("res://../../target/release/orla")
	var at := args.find("--orla")
	if at >= 0 and at + 1 < args.size():
		orla = args[at + 1]
	var result := "RESULT: ok"
	var failure := _run(orla)
	if not failure.is_empty():
		result = "RESULT: failed: " + failure
	for session in _sessions:
		OS.kill(session["pid"])
	print(result)
	quit(0 if failure.is_empty() else 1)


## Every check; the first failure's message, or an empty String.
func _run(orla: String) -> String:
	var text_port := _serve(orla, "examples/text/scene.osn", "cpu")
	var voxel_port := _serve(orla, "examples/voxel/scene.osn", "gpu")
	var mesh_port := _serve(orla, "examples/mesh/scene.osn", "cpu")
	if text_port == 0 or voxel_port == 0 or mesh_port == 0:
		return "a session did not start"
	for check in [_text.bind(text_port), _voxel.bind(voxel_port), _mesh.bind(mesh_port), _main.bind(mesh_port), _local]:
		var failure = check.call()
		if typeof(failure) != TYPE_STRING:
			return "a check stopped on a script error (see above)"
		if not failure.is_empty():
			return failure
	return ""


## Starts `orla serve` on a free port and returns the port, 0 on failure.
func _serve(orla: String, scene: String, backend: String) -> int:
	var path := ProjectSettings.globalize_path("res://../../" + scene)
	var started := OS.execute_with_pipe(orla, ["serve", path, "--backend", backend, "--listen", "127.0.0.1:0"], false)
	if started.is_empty():
		print("cannot start ", orla)
		return 0
	_sessions.append(started)
	var stderr: FileAccess = started["stderr"]
	var pattern := RegEx.create_from_string("listening on 127\\.0\\.0\\.1:(\\d+)")
	var deadline := Time.get_ticks_msec() + 30000
	while Time.get_ticks_msec() < deadline:
		var line := stderr.get_line()
		if line.is_empty():
			OS.delay_msec(10)
			continue
		var found := pattern.search(line)
		if found:
			return int(found.get_string(1))
	print("no port from ", scene)
	return 0


func _connect(port: int) -> Client:
	var client := Client.new()
	var error := client.open("127.0.0.1", port)
	if not error.is_empty():
		push_error(error)
		return null
	return client


static func _near(a: float, b: float, tolerance: float) -> bool:
	return absf(a - b) <= tolerance


static func _colour_is(image: Image, x: int, y: int, rgb: Color, tolerance: float) -> bool:
	var got := image.get_pixel(x, y)
	return _near(got.r, rgb.r, tolerance) and _near(got.g, rgb.g, tolerance) and _near(got.b, rgb.b, tolerance)


func _text(port: int) -> String:
	var client := _connect(port)
	if client == null:
		return "text: cannot connect"
	var status := client.request("status")
	if not str(status.get("scene", "")).ends_with("scene.osn"):
		return "text: status " + JSON.stringify(status)
	var kernels := client.request("kernels")
	var ids: Array = kernels.get("kernels", []).map(func(k): return k["id"])
	if not ids.has("text.draw@1"):
		return "text: kernels " + str(ids)
	# The picture as the replay pins it: the title's O at (8, 6) twice the size.
	client.request("evaluate", {"time": 0.0})
	var read := client.request("read", {"format": "ppm"})
	if int(read.get("width", 0)) != 192 or int(read.get("height", 0)) != 88:
		return "text: size %sx%s" % [read.get("width"), read.get("height")]
	var frame := Client.decode_ppm(read["data"])
	if not _colour_is(frame, 10, 6, Color(1.0, 0.85, 0.45), 0.02):
		return "text: the O's top is " + str(frame.get_pixel(10, 6))
	if not _colour_is(frame, 12, 12, Color(0.09, 0.1, 0.13), 0.02):
		return "text: the O's hollow is " + str(frame.get_pixel(12, 12))
	print("text: picture read")
	# The background is the node's inline Kernel: its source comes back as
	# publish takes it, so a recoloured copy publishes over it.
	var back := client.request("kernel", {"kernel": "scene.back@1"})
	var source: String = str(back.get("source", ""))
	if not source.begins_with("kernel scene.back@1 {") or source.contains("domain ="):
		return "text: source as publish takes it: " + source
	var recoloured := source.replace("vec4(0.09, 0.1, 0.13, 1.0)", "vec4(0.3, 0.1, 0.1, 1.0)")
	if recoloured == source:
		return "text: the background colour is where expected"
	var refused := client.request("prepare", {"source": recoloured.replace("vec4(", "vec5(")})
	if not Client.failed(refused) or int(refused.get("line", 0)) <= 0 or not str(refused["error"]).contains("vec5"):
		return "text: a diagnostic names the problem and its line: " + JSON.stringify(refused)
	print("text: a broken Kernel is refused with a position")
	var published := client.request("publish", {"source": recoloured, "expect": "any"})
	if str(published.get("id", "")) != "scene.back@1":
		return "text: published " + JSON.stringify(published)
	client.request("evaluate", {"time": 0.1})
	var again := Client.decode_ppm(client.request("read", {"format": "ppm"})["data"])
	if not _colour_is(again, 12, 12, Color(0.3, 0.1, 0.1), 0.02):
		return "text: the recoloured hollow is " + str(again.get_pixel(12, 12))
	if not _colour_is(again, 10, 6, Color(1.0, 0.85, 0.45), 0.02):
		return "text: the title changed"
	print("text: background republished")
	client.close()
	return ""


func _voxel(port: int) -> String:
	var client := _connect(port)
	if client == null:
		return "voxel: cannot connect"
	# An OrlaView is the window: the replay's 240 by 135, and its size goes
	# with every evaluate. Headless, the scale is one, so the bar is laid
	# out full size and the pointer aims at the picture.
	var view := View.new()
	view.size = Vector2(240, 135)
	root.add_child(view)
	view.client = client
	client.request("pin", {"node": "state"})
	var report := view.step()
	if Client.failed(report):
		return "voxel: " + str(report["error"])
	if view.frame_size != Vector2i(240, 135):
		return "voxel: frame " + str(view.frame_size)
	view.set_pointer(Vector2(120, 78))
	view.step()
	var state := client.request("read", {"node": "state", "port": "result"})
	var words := Client.decode_words(state["data"])
	# Edit { yaw, pitch, zoom, pan(3), px, py, down, right, middle, on_bar,
	# painting, erasing, stroked, tool, colour, hover, found, hit(3), ... }
	if Client.float_at(words, 6) != 120.0 or Client.float_at(words, 7) != 78.0:
		return "voxel: pointer %s, %s" % [Client.float_at(words, 6), Client.float_at(words, 7)]
	if not Client.bool_at(words, 18):
		return "voxel: the pointer finds no voxel"
	if Client.float_at(words, 19) != 27.0 or Client.float_at(words, 20) != 11.0 or Client.float_at(words, 21) != 27.0:
		return "voxel: hit %s %s %s" % [Client.float_at(words, 19), Client.float_at(words, 20), Client.float_at(words, 21)]
	print("voxel: pointer finds the tree")
	# The left button places a voxel on the face seen: (27, 12, 27) is 62811.
	view.press_button(0, true)
	view.step()
	view.press_button(0, false)
	client.request("pin", {"node": "world"})
	view.step()
	var world := Client.decode_words(client.request("read", {"node": "world", "port": "result"})["data"])
	if Client.int_at(world, 62811) != 1:
		return "voxel: placed voxel is %d" % Client.int_at(world, 62811)
	print("voxel: a click places a voxel")
	# A wider view is a resize: the picture follows the window.
	view.size = Vector2(320, 180)
	view.step()
	if view.frame_size != Vector2i(320, 180):
		return "voxel: resized frame " + str(view.frame_size)
	print("voxel: the picture follows the window")
	# Key codes as a Kernel reads them.
	var arrow := InputEventKey.new()
	arrow.keycode = KEY_LEFT
	var letter := InputEventKey.new()
	letter.keycode = KEY_Q
	if View.key_code(arrow) != 37 or View.key_code(letter) != 81:
		return "voxel: key codes %d %d" % [View.key_code(arrow), View.key_code(letter)]
	client.close()
	view.queue_free()
	return ""


func _mesh(port: int) -> String:
	var client := _connect(port)
	if client == null:
		return "mesh: cannot connect"
	# The torus's rows as a mesh: 1152 vertices, 2304 triangles, flat in
	# Orla's xz plane, so its y extent is the smallest.
	var mesh := client.mesh_of("torus", "faces")
	if mesh == null or mesh.get_surface_count() != 1:
		return "mesh: no surface"
	var arrays := mesh.surface_get_arrays(0)
	var positions: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
	if positions.size() != 1152 or indices.size() != 2304 * 3:
		return "mesh: %d vertices, %d indices" % [positions.size(), indices.size()]
	var lo := positions[0]
	var hi := positions[0]
	for p in positions:
		lo = lo.min(p)
		hi = hi.max(p)
	var extent := hi - lo
	if not (extent.y < extent.x and extent.y < extent.z):
		return "mesh: extent " + str(extent)
	print("mesh: the torus is an ArrayMesh")
	# A param that sizes the graph resizes the running scene; the picture
	# still reads at the raster's size.
	var set := client.request("param", {"name": "level", "value": 2})
	if Client.failed(set):
		return "mesh: param " + str(set["error"])
	client.request("evaluate", {"time": 0.0})
	var read := client.request("read", {"format": "ppm"})
	if int(read.get("width", 0)) != 320 or int(read.get("height", 0)) != 240:
		return "mesh: picture %sx%s" % [read.get("width"), read.get("height")]
	print("mesh: level set and the picture read")
	client.close()
	return ""


## The demo scene itself: connected to the mesh session, it frames and
## builds a control for the graph's one param.
func _main(port: int) -> String:
	var main: Control = load("res://main.tscn").instantiate()
	root.add_child(main)
	main.url.text = "127.0.0.1:%d" % port
	main._connect()
	if not str(main.status.text).begins_with("scene.osn on cpu"):
		return "main: status " + main.status.text
	if main.params_box.get_child_count() != 1:
		return "main: %d param controls" % main.params_box.get_child_count()
	main.view.size = Vector2(320, 240)
	var report: Dictionary = main.view.step()
	if Client.failed(report):
		return "main: " + str(report["error"])
	if main.view.frame_size != Vector2i(320, 240) or not str(main.status.text).contains(" ms a frame"):
		return "main: frame %s, status %s" % [main.view.frame_size, main.status.text]
	print("main: connected, framed and built the params")
	main.client.close()
	main.queue_free()
	return ""


## The session in this process, when the extension is built: the text
## example's picture as an Image on the Host's words, a Kernel published
## through the same door, a node's words, and the voxel editor driven by
## an OrlaView over the local session.
func _local() -> String:
	if not Local.available():
		print("local: skipped (cargo build --release -p orla-godot)")
		return ""
	var local := Local.new()
	var error := local.open("res://../../examples/text/scene.osn", "jit")
	if not error.is_empty():
		return "local: " + error
	var report := local.evaluate({"time": 0.0})
	if Client.failed(report):
		return "local: " + str(report["error"])
	var image := local.image()
	if image == null or image.get_size() != Vector2i(192, 88) or image.get_format() != Image.FORMAT_RGBAF:
		return "local: image %s" % [image]
	if not _colour_is(image, 10, 6, Color(1.0, 0.85, 0.45), 0.02):
		return "local: the O's top is " + str(image.get_pixel(10, 6))
	print("local: picture as an Image")
	var back := local.request("kernel", {"kernel": "scene.back@1"})
	var recoloured := str(back.get("source", "")).replace("vec4(0.09, 0.1, 0.13, 1.0)", "vec4(0.3, 0.1, 0.1, 1.0)")
	var published := local.request("publish", {"source": recoloured, "expect": "any"})
	if str(published.get("id", "")) != "scene.back@1":
		return "local: published " + JSON.stringify(published)
	local.evaluate({"time": 0.1})
	if not _colour_is(local.image(), 12, 12, Color(0.3, 0.1, 0.1), 0.02):
		return "local: the recoloured hollow is " + str(local.image().get_pixel(12, 12))
	local.request("pin", {"node": "codes"})
	local.evaluate({"time": 0.2})
	var codes := local.node("codes")
	if int(codes.get("count", 0)) != 95 or codes["words"].decode_s32(0) != 32:
		return "local: codes " + JSON.stringify(codes.keys())
	print("local: republished and a node's words read")
	local.close()
	# The voxel editor through an OrlaView over the local session.
	local = Local.new()
	error = local.open("res://../../examples/voxel/scene.osn", "jit")
	if not error.is_empty():
		return "local: " + error
	var view := View.new()
	view.size = Vector2(240, 135)
	root.add_child(view)
	view.client = local
	local.request("pin", {"node": "state"})
	local.request("pin", {"node": "world"})
	report = view.step()
	if Client.failed(report) or view.frame_size != Vector2i(240, 135):
		return "local: voxel frame %s %s" % [view.frame_size, report.get("error", "")]
	view.set_pointer(Vector2(120, 78))
	view.step()
	var state := local.node("state")
	var words: PackedByteArray = state["words"]
	if not Client.bool_at(words, 18) or Client.float_at(words, 19) != 27.0:
		return "local: the pointer finds no voxel"
	view.press_button(0, true)
	view.step()
	view.press_button(0, false)
	view.step()
	var world: PackedByteArray = local.node("world")["words"]
	if Client.int_at(world, 62811) != 1:
		return "local: placed voxel is %d" % Client.int_at(world, 62811)
	print("local: the voxel editor runs in this process")
	local.close()
	view.queue_free()
	return ""
