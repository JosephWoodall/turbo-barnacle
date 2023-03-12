class SubsystemA:
    def operation_a(self) -> str:
        return "Subsystem A operation"

class SubsystemB:
    def operation_b(self) -> str:
        return "Subsystem B operation"

class Facade:
    def __init__(self, subsystem_a: SubsystemA, subsystem_b: SubsystemB):
        self._subsystem_a = subsystem_a
        self._subsystem_b = subsystem_b

    def operation(self) -> str:
        results = []
        results.append("Facade initializes subsystems:")
        results.append(self._subsystem_a.operation_a())
        results.append(self._subsystem_b.operation_b())
        return "\n".join(results)

# Usage
subsystem_a = SubsystemA()
subsystem_b = SubsystemB()
facade = Facade(subsystem_a, subsystem_b)
print(facade.operation())  # Output: Facade initializes subsystems:\nSubsystem A operation\nSubsystem B operation
