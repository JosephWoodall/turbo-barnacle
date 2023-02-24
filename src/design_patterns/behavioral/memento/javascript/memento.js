class Memento {
    constructor(state) {
        this._state = state;
    }

    get state() {
        return this._state;
    }
}

class Originator {
    constructor() {
        this._state = null;
    }

    setState(state) {
        this._state = state;
    }

    save() {
        return new Memento(this._state);
    }

    restore(memento) {
        this._state = memento.state;
    }
}

class Caretaker {
    constructor(originator) {
        this._mementos = [];
        this._originator = originator;
    }

    saveState() {
        this._mementos.push(this._originator.save());
    }

    restoreState() {
        if (this._mementos.length === 0) {
            return;
        }

        const memento = this._mementos.pop();
        this._originator.restore(memento);
    }
}
