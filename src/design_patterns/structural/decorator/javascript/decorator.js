// Component
class Component {
    operation() {
      throw new Error("operation() must be implemented");
    }
  }
  
  // Concrete Component
  class ConcreteComponent extends Component {
    operation() {
      return "ConcreteComponent";
    }
  }
  
  // Decorator
  class Decorator extends Component {
    constructor(component) {
      super();
      this._component = component;
    }
  
    operation() {
      return this._component.operation();
    }
  }
  
  // Concrete Decorator A
  class ConcreteDecoratorA extends Decorator {
    operation() {
      return `ConcreteDecoratorA(${super.operation()})`;
    }
  }
  
  // Concrete Decorator B
  class ConcreteDecoratorB extends Decorator {
    operation() {
      return `ConcreteDecoratorB(${super.operation()})`;
    }
  }
  
  // Usage
  const component = new ConcreteComponent();
  const decorator1 = new ConcreteDecoratorA(component);
  const decorator2 = new ConcreteDecoratorB(decorator1);
  console.log(decorator2.operation());  // Output: ConcreteDecoratorB(ConcreteDecoratorA(ConcreteComponent))
  