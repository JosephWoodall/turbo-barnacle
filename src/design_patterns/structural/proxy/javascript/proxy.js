class Subject {
    request() {}
  }
  
  class RealSubject extends Subject {
    request() {
      console.log("RealSubject: Handling request.");
    }
  }
  
  class Proxy extends Subject {
    constructor(realSubject) {
      super();
      this._realSubject = realSubject;
    }
  
    request() {
      if (this.checkAccess()) {
        this._realSubject.request();
        this.logAccess();
      }
    }
  
    checkAccess() {
      console.log("Proxy: Checking access before handling request.");
      return true;
    }
  
    logAccess() {
      console.log("Proxy: Logging the time of request.");
    }
  }
  
  // Usage
  const realSubject = new RealSubject();
  const proxy = new Proxy(realSubject);
  
  proxy.request();
  