class SubsystemA {
    operationA() {
      return "Subsystem A operation";
    }
  }
  
  class SubsystemB {
    operationB() {
      return "Subsystem B operation";
    }
  }
  
  class Facade {
    constructor(subsystemA, subsystemB) {
      this._subsystemA = subsystemA;
      this._subsystemB = subsystemB;
    }
  
    operation() {
      let results = [];
      results.push("Facade initializes subsystems:");
      results.push(this._subsystemA.operationA());
      results.push(this._subsystemB.operationB());
      return results.join("\n");
    }
  }
  
  // Usage
  const subsystemA = new SubsystemA();
  const subsystemB = new SubsystemB();
  const facade = new Facade(subsystemA, subsystemB);
  console.log(facade.operation());  // Output: Facade initializes subsystems:\nSubsystem A operation\nSubsystem B operation
  