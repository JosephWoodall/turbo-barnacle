class Observer {
    update(subject) { }
}

class Subject {
    constructor() {
        this._observers = [];
    }

    addObserver(observer) {
        this._observers.push(observer);
    }

    removeObserver(observer) {
        const index = this._observers.indexOf(observer);
        if (index !== -1) {
            this._observers.splice(index, 1);
        }
    }

    notify() {
        this._observers.forEach((observer) => observer.update(this));
    }
}

class ConcreteObserver extends Observer {
    update(subject) {
        console.log(`Subject's state has changed to ${subject.state}`);
    }
}

class ConcreteSubject extends Subject {
    constructor() {
        super();
        this.state = null;
    }

    setState(state) {
        this.state = state;
        this.notify();
    }
}
