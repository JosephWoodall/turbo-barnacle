class Mediator {
    constructor() {
        this.colleague1 = new Colleague1(this);
        this.colleague2 = new Colleague2(this);
    }

    send(message, colleague) {
        if (colleague === this.colleague1) {
            this.colleague2.notify(message);
        } else {
            this.colleague1.notify(message);
        }
    }
}

class Colleague1 {
    constructor(mediator) {
        this.mediator = mediator;
    }

    send(message) {
        this.mediator.send(message, this);
    }

    notify(message) {
        console.log(`Colleague1 received message: ${message}`);
    }
}

class Colleague2 {
    constructor(mediator) {
        this.mediator = mediator;
    }

    send(message) {
        this.mediator.send(message, this);
    }

    notify(message) {
        console.log(`Colleague2 received message: ${message}`);
    }
}

const mediator = new Mediator();
const colleague1 = new Colleague1(mediator);
const colleague2 = new Colleague2(mediator);

colleague1.send("Hello from Colleague1!");
colleague2.send("Hi from Colleague2!");
