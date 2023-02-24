class Handler {
    constructor(successor = null) {
        this.successor = successor;
    }

    handleRequest(request) {
        if (this.successor) {
            return this.successor.handleRequest(request);
        }
    }
}

class ConcreteHandler1 extends Handler {
    handleRequest(request) {
        if (request === 'request 1') {
            console.log('Handled by ConcreteHandler1');
        } else {
            super.handleRequest(request);
        }
    }
}

class ConcreteHandler2 extends Handler {
    handleRequest(request) {
        if (request === 'request 2') {
            console.log('Handled by ConcreteHandler2');
        } else {
            super.handleRequest(request);
        }
    }
}

class ConcreteHandler3 extends Handler {
    handleRequest(request) {
        if (request === 'request 3') {
            console.log('Handled by ConcreteHandler3');
        } else {
            super.handleRequest(request);
        }
    }
}

// Client code
const handler1 = new ConcreteHandler1(new ConcreteHandler2(new ConcreteHandler3()));
handler1.handleRequest('request 1');
handler1.handleRequest('request 2');
handler1.handleRequest('request 3');
handler1.handleRequest('unknown request');