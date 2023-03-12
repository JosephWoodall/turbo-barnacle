// Abstraction
abstract class Shape
{
    protected IColor color;

    public Shape(IColor color)
    {
        this.color = color;
    }

    public abstract void Draw();
}

// Concrete Abstraction
class Circle : Shape
{
    private int radius;

    public Circle(int radius, IColor color) : base(color)
    {
        this.radius = radius;
    }

    public override void Draw()
    {
        Console.WriteLine($"Drawing a circle with radius {radius} and color {color.GetColor()}");
    }
}

// Concrete Abstraction
class Rectangle : Shape
{
    private int width;
    private int height;

    public Rectangle(int width, int height, IColor color) : base(color)
    {
        this.width = width;
        this.height = height;
    }

    public override void Draw()
    {
        Console.WriteLine($"Drawing a rectangle with width {width}, height {height} and color {color.GetColor()}");
    }
}

// Implementation
interface IColor
{
    string GetColor();
}

// Concrete Implementation
class RedColor : IColor
{
    public string GetColor()
    {
        return "red";
    }
}

// Concrete Implementation
class BlueColor : IColor
{
    public string GetColor()
    {
        return "blue";
    }
}
/*
// Usage
var red = new RedColor();
var blue = new BlueColor();

var circle = new Circle(10, red);
var rectangle = new Rectangle(20, 30, blue);

circle.Draw(); // Drawing a circle with radius 10 and color red
rectangle.Draw(); // Drawing a rectangle with width 20, height 30 and color blue

circle.color = blue;
rectangle.color = red;

circle.Draw(); // Drawing a circle with radius 10 and color blue
rectangle.Draw(); // Drawing a rectangle with width 20, height 30 and color red
*/