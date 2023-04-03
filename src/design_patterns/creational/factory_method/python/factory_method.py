from abc import ABC, abstractmethod


class Animal(ABC):
    """ """
    @abstractmethod
    def speak(self):
        """ """
        pass


class Dog(Animal):
    """ """
    def speak(self):
        """ """
        return "Woof!"


class Cat(Animal):
    """ """
    def speak(self):
        """ """
        return "Meow"


class AnimalFactory:
    """ """
    def create_animal(self, animal_type):
        """

        :param animal_type: 

        """
        if animal_type == "Dog":
            return Dog()
        elif animal_type == "Cat":
            return Cat()
        else:
            return None


factory = AnimalFactory()
dog = factory.create_animal("Dog")
print(dog.speak())  # Output: "Woof!"
cat = factory.create_animal("Cat")
print(cat.speak())  # Output: "Meow"
