function Prototype() {
    this.objects = {};
}

Prototype.prototype.register = function (name, obj) {
    this.objects[name] = obj;
}

Prototype.prototype.unregister = function (name) {
    delete this.objects[name];
}

Prototype.prototype.clone = function (name, attr) {
    const obj = Object.create(this.objects[name]);
    obj.attr = attr;
    return obj;
}

function Shape() {
    this.id = null;
    this.type = null;
}

Shape.prototype.clone = function () { }

function Rectangle() {
    Shape.call(this);
    this.type = "rectangle";
}

Rectangle.prototype = Object.create(Shape.prototype);
Rectangle.prototype.constructor = Rectangle;

Rectangle.prototype.clone = function () {
    return Object.create(this);
}

function Circle() {
    Shape.call(this);
    this.type = "circle";
}

Circle.prototype = Object.create(Shape.prototype);
Circle.prototype.constructor = Circle;

Circle.prototype.clone = function () {
    return Object.create(this);
}

const prototype = new Prototype();
prototype.register("rectangle", new Rectangle());
prototype.register("circle", new Circle());

const rectangle = prototype.clone("rectangle", { attr: "value" });
const circle = prototype.clone("circle", { attr: "value" });
console.log(rectangle.type); // Output: rectangle
console.log(circle.type); // Output: circle
