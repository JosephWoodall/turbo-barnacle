// Component
class Component {
    operation() {
      throw new Error("operation() must be implemented");
    }
  }
  
  // Leaf
  class Leaf extends Component {
    operation() {
      return "Leaf";
    }
  }
  
  // Composite
  class Composite extends Component {
    constructor() {
      super();
      this._children = [];
    }
  
    add(component) {
      this._children.push(component);
    }
  
    remove(component) {
      const index = this._children.indexOf(component);
      if (index !== -1) {
        this._children.splice(index, 1);
      }
    }
  
    operation() {
      const results = [];
      for (let child of this._children) {
        results.push(child.operation());
      }
      return `Composite(${results.join("+")})`;
    }
  }
  
  // Usage
  const leaf1 = new Leaf();
  const leaf2 = new Leaf();
  const composite1 = new Composite();
  composite1.add(leaf1);
  composite1.add(leaf2);
  const composite2 = new Composite();
  composite2.add(composite1);
  console.log(composite2.operation());  // Output: Composite(Leaf+Leaf)
  