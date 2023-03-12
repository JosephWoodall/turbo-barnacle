// Abstraction
class Shape {
    constructor(color) {
      this.color = color;
    }
  
    getColor() {
      return this.color.getColor();
    }
  
    setColor(color) {
      this.color = color;
    }
  
    draw() {
      throw new Error('Method not implemented');
    }
  }
  
  // Concrete Abstraction
  class Circle extends Shape {
    constructor(radius, color) {
      super(color);
      this.radius = radius;
    }
  
    draw() {
      console.log(`Drawing a circle with radius ${this.radius} and color ${this.color.getColor()}`);
    }
  }
  
  // Concrete Abstraction
  class Rectangle extends Shape {
    constructor(width, height, color) {
      super(color);
      this.width = width;
      this.height = height;
    }
  
    draw() {
      console.log(`Drawing a rectangle with width ${this.width}, height ${this.height} and color ${this.color.getColor()}`);
    }
  }
  
  // Implementation
  class Color {
    constructor(value) {
      this.value = value;
    }
  
    getColor() {
      return this.value;
    }
  }
  
  // Concrete Implementation
  class RedColor extends Color {
    constructor() {
      super('red');
    }
  }
  
  // Concrete Implementation
  class BlueColor extends Color {
    constructor() {
      super('blue');
    }
  }
  
  // Usage
  const red = new RedColor();
  const blue = new BlueColor();
  
  const circle = new Circle(10, red);
  const rectangle = new Rectangle(20, 30, blue);
  
  circle.draw(); // Drawing a circle with radius 10 and color red
  rectangle.draw(); // Drawing a rectangle with width 20, height 30 and color blue
  
  circle.setColor(blue);
  rectangle.setColor(red);
  
  circle.draw(); // Drawing a circle with radius 10 and color blue
  rectangle.draw(); // Drawing a rectangle with width 20, height 30 and color red
  