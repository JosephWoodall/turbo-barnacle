'''
Observer is a behavioral pattern (and is also called the PubSub pattern)
    - It's used beyond OOP, it's also used in distributed systems

There is a subject/publisher that is the source of events, and an observer/subscriber that is notified of events in real time 
'''

from abc import ABC, abstractmethod


class App:
    def __init__(self, name):
        self.name = name
        self.subscribers = []

    def subscribe(self, sub):
        self.subscribers.append(sub)

    def notify(self, event):
        for sub in self.subscribers:
            sub.sendNotification(self.name, event)


class AppSubscriber(ABC):
    def __init__(self, name):
        self.name = name

    def sendNotification(self, channel, event):
        print(
            f"User {self.name} received a notification from {channel}: {event}")


app = App("SaaSAppOne")

app.subscribe(AppSubscriber("sub1"))
app.subscribe(AppSubscriber("sub2"))
app.subscribe(AppSubscriber("sub3"))

app.notify("A new thing is here!")
