class PastRunInspection:
    from metaflow import Flow
    run = Flow('Main').latest_run


if __name__ == "__main__":
    PastRunInspection()
