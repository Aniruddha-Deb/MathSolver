var canvas = null;

function main() {
	console.log("loaded");
	canvas = new fabric.Canvas('c1');
	canvas.isDrawingMode = true;
	canvas.freeDrawingBrush.width = 6;
}

function clear_canvas() {
	canvas.clear();
}

function calculate() {
	var img = canvas.toDataURL()
	print(img)
}