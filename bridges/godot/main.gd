## The bridge's demo: connect to a serving session (host:port) or open a
## scene in this process (a path to a `.osn`, with the orla-godot
## extension built), run it frame after frame in an OrlaView, set the
## graph's params from a side panel, reset the state, and see the round
## trip. The view is the window either way.
extends Control

const Client = preload("res://orla_client.gd")
const Local = preload("res://orla_local.gd")
const View = preload("res://orla_view.gd")

## An OrlaClient or an OrlaLocal.
var client: RefCounted = Client.new()
var view: View
var url: LineEdit
var backend: OptionButton
var status: Label
var params_box: VBoxContainer
var run_button: Button


func _ready() -> void:
	var split := HSplitContainer.new()
	split.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(split)
	var side := VBoxContainer.new()
	side.custom_minimum_size = Vector2(260, 0)
	split.add_child(side)
	view = View.new()
	view.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	split.add_child(view)
	view.framed.connect(_on_framed)
	view.stopped.connect(_on_stopped)

	url = LineEdit.new()
	url.text = "127.0.0.1:7788"
	url.placeholder_text = "host:port of `orla serve --listen`, or a scene.osn path"
	side.add_child(url)
	var row := HBoxContainer.new()
	side.add_child(row)
	backend = OptionButton.new()
	for name in ["jit", "cpu", "gpu"]:
		backend.add_item(name)
	backend.tooltip_text = "The backend when a scene opens in this process"
	row.add_child(backend)
	var connect_button := Button.new()
	connect_button.text = "Connect"
	connect_button.pressed.connect(_connect)
	row.add_child(connect_button)
	run_button = Button.new()
	run_button.text = "Run"
	run_button.pressed.connect(_toggle_run)
	row.add_child(run_button)
	var reset_button := Button.new()
	reset_button.text = "Reset state"
	reset_button.pressed.connect(func(): _report(client.request("reset")))
	row.add_child(reset_button)
	status = Label.new()
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	status.text = "Not connected."
	side.add_child(status)
	var heading := Label.new()
	heading.text = "Graph params"
	side.add_child(heading)
	params_box = VBoxContainer.new()
	side.add_child(params_box)


func _connect() -> void:
	view.stop()
	run_button.text = "Run"
	client.close()
	var error := ""
	if url.text.strip_edges().ends_with(".osn"):
		client = Local.new()
		error = client.open(url.text.strip_edges(), backend.get_item_text(backend.selected))
	else:
		client = Client.new()
		var parts := url.text.split(":")
		error = client.open(parts[0], int(parts[1]) if parts.size() > 1 else 7788)
	if not error.is_empty():
		status.text = error
		return
	view.client = client
	var info: Dictionary = client.request("status")
	if Client.failed(info):
		status.text = str(info["error"])
		return
	var where := "in this process" if client is Local else "over the socket"
	status.text = "%s on %s, %s: %d nodes, %d kernels" % [str(info["scene"]).get_file(), info["backend"], where, info["nodes"], info["kernels"]]
	_build_params()
	view.grab_focus()


func _toggle_run() -> void:
	if view.running:
		view.stop()
		run_button.text = "Run"
	elif client.is_open():
		view.start()
		run_button.text = "Stop"


func _on_framed(report: Dictionary, millis: float) -> void:
	status.text = "%.1f ms a frame, %d ran, %d reused, %dx%d" % [millis, report.get("executed", []).size(), report.get("reused", []).size(), view.frame_size.x, view.frame_size.y]


func _on_stopped(message: String) -> void:
	run_button.text = "Run"
	status.text = message


func _report(result: Dictionary) -> void:
	if Client.failed(result):
		status.text = str(result["error"])


## One control per graph param, by its type: a SpinBox for Float and
## Int, a CheckBox for Bool, a LineEdit for a vector.
func _build_params() -> void:
	for child in params_box.get_children():
		child.queue_free()
	var graph: Dictionary = client.request("graph")
	if Client.failed(graph):
		return
	for param in graph.get("params", []):
		var name: String = param["name"]
		var row := HBoxContainer.new()
		var label := Label.new()
		label.text = name
		label.custom_minimum_size = Vector2(90, 0)
		row.add_child(label)
		var value = param["value"]
		match str(param["type"]):
			"Float", "Int":
				var spin := SpinBox.new()
				spin.allow_greater = true
				spin.allow_lesser = true
				spin.step = 1.0 if param["type"] == "Int" else 0.01
				spin.value = float(value)
				spin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
				spin.value_changed.connect(func(v: float): _set_param(name, int(v) if param["type"] == "Int" else v))
				row.add_child(spin)
			"Bool":
				var check := CheckBox.new()
				check.button_pressed = bool(value)
				check.toggled.connect(func(on: bool): _set_param(name, on))
				row.add_child(check)
			_:
				var edit := LineEdit.new()
				edit.text = JSON.stringify(value)
				edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
				edit.text_submitted.connect(func(text: String): _set_param(name, JSON.parse_string(text)))
				row.add_child(edit)
		params_box.add_child(row)


func _set_param(name: String, value) -> void:
	_report(client.request("param", {"name": name, "value": value}))
