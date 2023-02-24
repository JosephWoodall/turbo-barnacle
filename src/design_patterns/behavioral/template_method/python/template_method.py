class AbstractClass:
    def template_method(self):
        self.operation1()
        self.operation2()
        self.operation3()

    def operation1(self):
        pass

    def operation2(self):
        pass

    def operation3(self):
        pass


class ConcreteClass(AbstractClass):
    def operation1(self):
        print("ConcreteClass.operation1() called")

    def operation2(self):
        print("ConcreteClass.operation2() called")


class AnotherConcreteClass(AbstractClass):
    def operation1(self):
        print("AnotherConcreteClass.operation1() called")

    def operation3(self):
        print("AnotherConcreteClass.operation3() called")
