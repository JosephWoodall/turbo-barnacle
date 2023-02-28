using System;
using System.Collections.Generic;

namespace PrototypePattern
{
    class Program
    {
        static void Main(string[] args)
        {
            var prototype = new Prototype();
            prototype.Register("rectangle", new Rectangle());
            prototype.Register("circle", new Circle());

            var rectangle = prototype.Clone("rectangle");
            var circle = prototype.Clone("circle");
            Console.WriteLine(rectangle.Type); // Output: rectangle
            Console.WriteLine(circle.Type); // Output: circle
        }
    }

    public class Prototype
    {
        private readonly Dictionary<string, Shape> _objects = new Dictionary<string, Shape>();

        public void Register(string name, Shape obj)
        {
            _objects[name] = obj;
        }

        public void Unregister(string name)
        {
            _objects.Remove(name);
        }

        public Shape Clone(string name)
        {
            return _objects[name].Clone();
        }
    }

    public abstract class Shape
    {
        public int Id { get; set; }
        public string Type { get; set; }

        public abstract Shape Clone();
    }

    public class Rectangle : Shape
{
    public int Width { get; set; }
    public int Height { get; set; }

    public Rectangle(int x, int y, int width, int height) : base(x, y)
    {
        Width = width;
        Height = height;
    }

    public override void Draw()
    {
        Console.WriteLine($"Drawing a rectangle at ({X}, {Y}), width = {Width}, height = {Height}");
    }

    public override object Clone()
    {
        // Perform a shallow copy of this rectangle
        return MemberwiseClone();
    }
}

public class Circle : Shape
{
    public int Radius { get; set; }

    public Circle(int x, int y, int radius) : base(x, y)
    {
        Radius = radius;
    }

    public override void Draw()
    {
        Console.WriteLine($"Drawing a circle at ({X}, {Y}), radius = {Radius}");
    }

    public override object Clone()
    {
        // Perform a shallow copy of this circle
        return MemberwiseClone();
    }
}

public class PrototypeDemo
{
    static void Main(string[] args)
    {
        // Create a rectangle prototype
        var rectanglePrototype = new Rectangle(10, 20, 30, 40);

        // Clone the rectangle prototype
        var clonedRectangle = rectanglePrototype.Clone() as Rectangle;

        // Update the cloned rectangle
        clonedRectangle.X = 50;
        clonedRectangle.Y = 60;

        // Draw the original rectangle and the cloned rectangle
        rectanglePrototype.Draw();
        clonedRectangle.Draw();

        // Create a circle prototype
        var circlePrototype = new Circle(70, 80, 90);

        // Clone the circle prototype
        var clonedCircle = circlePrototype.Clone() as Circle;

        // Update the cloned circle
        clonedCircle.X = 100;
        clonedCircle.Y = 110;

        // Draw the original circle and the cloned circle
        circlePrototype.Draw();
        clonedCircle.Draw();
    }
}

