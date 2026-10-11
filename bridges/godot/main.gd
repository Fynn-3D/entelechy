## The bridge's demo: connect to a serving session, run it frame after
## frame in an EntelechyView, set the graph's params from a side panel, reset
## the state, and see the round trip. Everything here goes through
## EntelechyClient; the view is the window.
extends Control

const Client = preload("res://entelechy_client.gd")
const View = preload("res://entelechy_view.gd")

var client: Client = Client.new()
var view: View
var url: LineEdit
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
	url.placeholder_text = "host:port of `entelechy serve --listen`"
	side.add_child(url)
	var row := HBoxContainer.new()
	side.add_child(row)
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
	client = Client.new()
	var parts := url.text.split(":")
	var error := client.open(parts[0], int(parts[1]) if parts.size() > 1 else 7788)
	if not error.is_empty():
		status.text = error
		return
	view.client = client
	var info := client.request("status")
	if Client.failed(info):
		status.text = str(info["error"])
		return
	status.text = "%s on %s: %d nodes, %d kernels" % [str(info["scene"]).get_file(), info["backend"], info["nodes"], info["kernels"]]
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
	var graph := client.request("graph")
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
