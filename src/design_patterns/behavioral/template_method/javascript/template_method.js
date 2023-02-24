class AbstractClass {
    templateMethod() {
        this.operation1();
        this.operation2();
        this.operation3();
    }

    operation1() { }

    operation2() { }

    operation3() { }
}

class ConcreteClass extends AbstractClass {
    operation1() {
        console.log("ConcreteClass.operation1() called");
    }

    operation2() {
        console.log("ConcreteClass.operation2() called");
    }
}

class AnotherConcreteClass extends AbstractClass {
    operation1() {
        console.log("AnotherConcreteClass.operation1() called");
    }

    operation3() {
        console.log("AnotherConcreteClass.operation3() called");
    }
}
