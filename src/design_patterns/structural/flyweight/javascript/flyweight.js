class Flyweight {
    constructor(sharedState) {
      this._sharedState = sharedState;
    }
  
    operation(uniqueState) {
      return `Flyweight: Shared '${this._sharedState}' and unique '${uniqueState}'`;
    }
  }
  
  class FlyweightFactory {
    constructor(initialFlyweights) {
      this._flyweights = {};
      for (let state of initialFlyweights) {
        this._flyweights[this.getKey(state)] = new Flyweight(state);
      }
    }
  
    getKey(state) {
      return state.sort().join("_");
    }
  
    getFlyweight(sharedState) {
      const key = this.getKey(sharedState);
      if (!this._flyweights[key]) {
        console.log("FlyweightFactory: Can't find a flyweight, creating new one.");
        this._flyweights[key] = new Flyweight(sharedState);
      } else {
        console.log("FlyweightFactory: Reusing existing flyweight.");
      }
      return this._flyweights[key];
    }
  }
  
  // Usage
  const flyweightFactory = new FlyweightFactory([
    ["shared_state_1", "unique_state_1"],
    ["shared_state_2", "unique_state_2"],
    ["shared_state_3", "unique_state_3"],
  ]);
  
  const flyweight1 = flyweightFactory.getFlyweight("shared_state_1");
  console.log(flyweight1.operation("unique_state_1")); // Output: Flyweight: Shared 'shared_state_1' and unique 'unique_state_1'
  
  const flyweight2 = flyweightFactory.getFlyweight("shared_state_2");
  console.log(flyweight2.operation("unique_state_2")); // Output: Flyweight: Shared 'shared_state_2' and unique 'unique_state_2'
  
  const flyweight3 = flyweightFactory.getFlyweight("shared_state_1");
  console.log(flyweight3.operation("unique_state_3")); // Output: Flyweight: Shared 'shared_state_1' and unique 'unique_state_3'
  