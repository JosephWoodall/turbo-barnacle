class ExistingClass {
    constructor(canvas) {
        this.canvas = canvas;
    }

    drawSquare(x, y, size) {
        // draws a square on the canvas
    }
}

class Adapter {
    constructor(canvas) {
        this.canvas = canvas;
    }

    drawTriangle(x, y, size) {
        const x1 = x;
        const y1 = y;
        const x2 = x + size;
        const y2 = y;
        const x3 = x + (size / 2);
        const y3 = y - size;

        this.canvas.beginPath();
        this.canvas.moveTo(x1, y1);
        this.canvas.lineTo(x2, y2);
        this.canvas.lineTo(x3, y3);
        this.canvas.closePath();
        this.canvas.stroke();
    }
}

// example usage
const canvas = document.getElementById('canvas');
const existing = new ExistingClass(canvas);
const adapter = new Adapter(canvas);

existing.drawSquare(10, 10, 50);
adapter.drawTriangle(100, 100, 50);
