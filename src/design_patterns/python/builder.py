'''
Builders are a creational pattern, but we dont have to immediately pass in the parameters now
'''


class Sandwich:
    def __init__(self):
        self.bread = None
        self.meat = None
        self.cheese = None

    def setBread(self, breadStyle):
        self.bread = breadStyle

    def setMeat(self, meatStyle):
        self.meat = meatStyle

    def setCheese(self, cheeseStyle):
        self.cheese = cheeseStyle


class SandwichBuilder:
    def __init__(self):
        self.sandwich = Sandwich()

    def addBread(self, breadStyle):
        self.sandwich.setBread(breadStyle)
        return self

    def addMeat(self, meatStyle):
        self.sandwich.setMeat(meatStyle)
        return self

    def addCheese(self, cheeseStyle):
        self.sandwich.setCheese(cheeseStyle)
        return self

    def build(self):
        return self.sandwich


sandwich = SandwichBuilder().addBread(
    "wheat").addMeat("ham").addCheese("swiss").build()
