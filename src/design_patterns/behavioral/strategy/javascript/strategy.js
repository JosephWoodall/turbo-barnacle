class Strategy {
    doOperation(num1, num2) { }
}

class OperationAdd extends Strategy {
    doOperation(num1, num2) {
        return num1 + num2;
    }
}

class OperationSubtract extends Strategy {
    doOperation(num1, num2) {
        return num1 - num2;
    }
}

class OperationMultiply extends Strategy {
    doOperation(num1, num2) {
        return num1 * num2;
    }
}

class Context {
    constructor(strategy) {
        this.strategy = strategy;
    }

    executeStrategy(num1, num2) {
        return this.strategy.doOperation(num1, num2);
    }
}
