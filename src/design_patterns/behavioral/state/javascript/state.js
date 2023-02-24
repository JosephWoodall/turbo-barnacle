class State {
    handle(context) { }
}

class ConcreteStateA extends State {
    handle(context) {
        console.log("Handling request with ConcreteStateA");
        context.state = new ConcreteStateB();
    }
}

class ConcreteStateB extends State {
    handle(context) {
        console.log("Handling request with ConcreteStateB");
        context.state = new ConcreteStateA();
    }
}

class Context {
    constructor() {
        this.state = new ConcreteStateA();
    }

    request() {
        this.state.handle(this);
    }
}
