## A Control that is a session's window: its size in device pixels and
## the screen's scale go with every `evaluate`, the pointer, buttons,
## wheel and keys over it are the input, and each frame is an `evaluate`
## followed by a `read` shown as its texture, letterboxed like the Player.
class_name EntelechyView
extends Control

## The client's script, by path rather than by its class name, so this
## script loads before the editor has catalogued the classes.
const Client = preload("res://entelechy_client.gd")

signal framed(report: Dictionary, millis: float)
signal stopped(message: String)

var client: Client
## Whether `_process` steps the session frame after frame.
var running := false
## The last frame's size in output pixels.
var frame_size := Vector2i.ZERO
## Where the last frame was placed inside this Control, in its own pixels.
var placed := Rect2i()
## Milliseconds the last frame's round trip took.
var last_millis := 0.0

var _texture := ImageTexture.new()
var _pointer := Vector2.ZERO
var _buttons := {}
var _keys := {}
var _wheel := Vector2.ZERO
var _started := 0
var _stepping := false


func _ready() -> void:
	focus_mode = Control.FOCUS_ALL
	mouse_filter = Control.MOUSE_FILTER_STOP
	_started = Time.get_ticks_msec()


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Color.BLACK)
	if frame_size.x > 0 and _texture.get_size().x > 0:
		draw_texture_rect(_texture, Rect2(placed.position, placed.size), false)


## The window the session sees: this Control in device pixels.
func window_size() -> Vector2i:
	var factor := window_scale()
	return Vector2i(maxi(1, int(size.x * factor)), maxi(1, int(size.y * factor)))


func window_scale() -> float:
	var window := get_window()
	if window:
		return window.content_scale_factor * DisplayServer.screen_get_scale()
	return 1.0


## The pointer in output pixels: this Control's position mapped through
## where the frame is placed, clamped to the frame like the Player.
func set_pointer(local: Vector2) -> void:
	if placed.size.x <= 0 or placed.size.y <= 0 or frame_size.x <= 0:
		_pointer = Vector2.ZERO
		return
	var fx := (local.x - placed.position.x) * frame_size.x / placed.size.x
	var fy := (local.y - placed.position.y) * frame_size.y / placed.size.y
	_pointer = Vector2(clampf(fx, 0.0, frame_size.x - 1), clampf(fy, 0.0, frame_size.y - 1))


func press_button(button: int, down: bool) -> void:
	if down:
		_buttons[button] = true
	else:
		_buttons.erase(button)


func press_key(code: int, down: bool) -> void:
	if down:
		_keys[code] = true
	else:
		_keys.erase(code)


func scroll(delta: Vector2) -> void:
	_wheel += delta


func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		set_pointer(event.position)
	elif event is InputEventMouseButton:
		var button := -1
		match event.button_index:
			MOUSE_BUTTON_LEFT:
				button = 0
			MOUSE_BUTTON_RIGHT:
				button = 1
			MOUSE_BUTTON_MIDDLE:
				button = 2
			MOUSE_BUTTON_WHEEL_UP:
				scroll(Vector2(0, 1))
			MOUSE_BUTTON_WHEEL_DOWN:
				scroll(Vector2(0, -1))
			MOUSE_BUTTON_WHEEL_LEFT:
				scroll(Vector2(-1, 0))
			MOUSE_BUTTON_WHEEL_RIGHT:
				scroll(Vector2(1, 0))
		if button >= 0:
			if event.pressed:
				grab_focus()
			press_button(button, event.pressed)
			set_pointer(event.position)
		accept_event()
	elif event is InputEventKey:
		var code := key_code(event)
		if code >= 0:
			press_key(code, event.pressed)
			accept_event()


func _notification(what: int) -> void:
	if what == NOTIFICATION_FOCUS_EXIT or what == NOTIFICATION_MOUSE_EXIT:
		_keys.clear()
		if what == NOTIFICATION_FOCUS_EXIT:
			_buttons.clear()


## The key codes a Kernel reads (`entelechy help key`): a letter or a
## digit by its ASCII uppercase, a punctuation key by the browser's code
## for it on a US keyboard, a named key by the browser's code; -1 for a
## key the protocol does not name.
static func key_code(event: InputEventKey) -> int:
	var key := event.keycode
	if key >= KEY_A and key <= KEY_Z or key >= KEY_0 and key <= KEY_9 or key == KEY_SPACE:
		return key
	match key:
		KEY_SEMICOLON:
			return 186
		KEY_EQUAL:
			return 187
		KEY_COMMA:
			return 188
		KEY_MINUS:
			return 189
		KEY_PERIOD:
			return 190
		KEY_SLASH:
			return 191
		KEY_QUOTELEFT:
			return 192
		KEY_BRACKETLEFT:
			return 219
		KEY_BACKSLASH:
			return 220
		KEY_BRACKETRIGHT:
			return 221
		KEY_APOSTROPHE:
			return 222
		KEY_PAGEUP:
			return 33
		KEY_PAGEDOWN:
			return 34
		KEY_END:
			return 35
		KEY_HOME:
			return 36
		KEY_DELETE:
			return 46
		KEY_CTRL:
			return 17
		KEY_ALT:
			return 18
		KEY_META:
			return 91
		KEY_LEFT:
			return 37
		KEY_UP:
			return 38
		KEY_RIGHT:
			return 39
		KEY_DOWN:
			return 40
		KEY_ENTER, KEY_KP_ENTER:
			return 13
		KEY_SHIFT:
			return 16
		KEY_ESCAPE:
			return 27
		KEY_TAB:
			return 9
		KEY_BACKSPACE:
			return 8
	return -1


## One frame: evaluate under the current input, read the picture, show
## it. Returns the evaluation's report, or {"error": message}.
func step() -> Dictionary:
	if client == null or not client.is_open():
		return {"error": "not connected"}
	var wheel := _wheel
	_wheel = Vector2.ZERO
	var began := Time.get_ticks_usec()
	var report := client.request("evaluate", {
		"time": (Time.get_ticks_msec() - _started) / 1000.0,
		"pointer": [_pointer.x, _pointer.y],
		"buttons": _buttons.keys(),
		"wheel": [wheel.x, wheel.y],
		"keys": _keys.keys(),
		"window": [window_size().x, window_size().y],
		"scale": window_scale(),
	})
	if Client.failed(report):
		return report
	for output in report.get("outputs", []):
		if int(output.get("width", 0)) > 0:
			var read := client.request("read", {"output": output["name"], "format": "ppm"})
			if Client.failed(read):
				return read
			var image := Client.decode_ppm(read["data"])
			if image:
				show_image(image)
			break
	last_millis = (Time.get_ticks_usec() - began) / 1000.0
	framed.emit(report, last_millis)
	return report


func show_image(image: Image) -> void:
	var image_size := image.get_size()
	if image_size != frame_size or _texture.get_size() != Vector2(image_size):
		frame_size = image_size
		_texture = ImageTexture.create_from_image(image)
	else:
		_texture.update(image)
	placed = Client.placement(frame_size, Vector2i(size))
	queue_redraw()


func _process(_delta: float) -> void:
	if running and not _stepping:
		_stepping = true
		var report := step()
		_stepping = false
		if Client.failed(report):
			running = false
			stopped.emit(str(report["error"]))


func start() -> void:
	_started = Time.get_ticks_msec()
	running = true


func stop() -> void:
	running = false
