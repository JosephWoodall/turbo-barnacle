// Abstract Factory
class GUIFactory {
    createButton() { }
    createCheckbox() { }
}

// Concrete Factory 1
class WinFactory extends GUIFactory {
    createButton() {
        return new WinButton();
    }

    createCheckbox() {
        return new WinCheckbox();
    }
}

// Concrete Factory 2
class MacFactory extends GUIFactory {
    createButton() {
        return new MacButton();
    }

    createCheckbox() {
        return new MacCheckbox();
    }
}

// Abstract Product A
class Button {
    paint() { }
}

// Concrete Product A1
class WinButton extends Button {
    paint() {
        return "Windows button painted";
    }
}

// Concrete Product A2
class MacButton extends Button {
    paint() {
        return "Mac button painted";
    }
}

// Abstract Product B
class Checkbox {
    paint() { }
}

// Concrete Product B1
class WinCheckbox extends Checkbox {
    paint() {
        return "Windows checkbox painted";
    }
}

// Concrete Product B2
class MacCheckbox extends Checkbox {
    paint() {
        return "Mac checkbox painted";
    }
}

// Client code
function clientCode(factory) {
    const button = factory.createButton();
    const checkbox = factory.createCheckbox();

    console.log(button.paint());
    console.log(checkbox.paint());
}

const factory1 = new WinFactory();
const factory2 = new MacFactory();

clientCode(factory1);
clientCode(factory2);
