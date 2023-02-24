class Visitor {
    visitElement(element) { }
}

class Element {
    accept(visitor) {
        visitor.visitElement(this);
    }
}

class ConcreteElementA extends Element {
    accept(visitor) {
        visitor.visitConcreteElementA(this);
    }
}

class ConcreteElementB extends Element {
    accept(visitor) {
        visitor.visitConcreteElementB(this);
    }
}

class ConcreteVisitorA extends Visitor {
    visitConcreteElementA(element) {
        console.log("ConcreteVisitorA visited ConcreteElementA");
    }

    visitElement(element) {
        console.log("ConcreteVisitorA visited Element");
    }
}

class ConcreteVisitorB extends Visitor {
    visitConcreteElementB(element) {
        console.log("ConcreteVisitorB visited ConcreteElementB");
    }

    visitElement(element) {
        console.log("ConcreteVisitorB visited Element");
    }
}
