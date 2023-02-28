class Animal {
    speak() {
        throw new Error('This method must be overwritten!');
    }
}

class Dog extends Animal {
    speak() {
        return 'Woof!';
    }
}

class Cat extends Animal {
    speak() {
        return 'Meow';
    }
}

class AnimalFactory {
    createAnimal(animalType) {
        switch (animalType) {
            case 'Dog':
                return new Dog();
            case 'Cat':
                return new Cat();
            default:
                return null;
        }
    }
}

const factory = new AnimalFactory();
const dog = factory.createAnimal('Dog');
console.log(dog.speak());  // Output: "Woof!"
const cat = factory.createAnimal('Cat');
console.log(cat.speak());  // Output: "Meow"
