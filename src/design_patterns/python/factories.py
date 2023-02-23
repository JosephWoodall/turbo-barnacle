'''
Factories are a creational pattern.
'''


class Sandwich:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    def print(self):
        print(self.ingredients)


class SandwichFactory:
    def createHamSandwich(self):
        ingredients = ["bread", "cheese", "ham"]
        return Sandwich(ingredients)

    def createDeluxHamSandwich(self):
        ingredients = ["bread", "tomato", "lettuce", "cheese", "ham"]
        return Sandwich(ingredients)

    def createVeganSandwich(self):
        ingredients = ["bread", "cheese", "not-so-ham-ham"]
        return Sandwich(ingredients)


myOrder = SandwichFactory()
myOrder.createHamSandwich().print()
myOrder.createDeluxHamSandwich().print()
myOrder.createVeganSandwich().print()
