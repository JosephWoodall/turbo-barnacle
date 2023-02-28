class Director {
    constructor(builder) {
        this.builder = builder;
    }

    constructCar() {
        this.builder.createNewCar();
        this.builder.addModel();
        this.builder.addTires();
        this.builder.addEngine();
    }

    getCar() {
        return this.builder.car;
    }
}

class Builder {
    constructor() {
        this.car = null;
    }

    createNewCar() {
        this.car = {};
    }
}

class SportsCarBuilder extends Builder {
    addModel() {
        this.car.model = "Sports car";
    }

    addTires() {
        this.car.tires = "High performance";
    }

    addEngine() {
        this.car.engine = "Turbocharged";
    }
}

class Car {
    constructor() {
        this.model = null;
        this.tires = null;
        this.engine = null;
    }

    toString() {
        return `${this.model} | ${this.tires} tires | ${this.engine} engine`;
    }
}

const builder = new SportsCarBuilder();
const director = new Director(builder);
director.constructCar();
const car = director.getCar();
console.log(car.toString());
