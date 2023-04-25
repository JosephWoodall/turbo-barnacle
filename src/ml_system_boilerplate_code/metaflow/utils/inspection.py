class PastRunInspection:
    from metaflow import Flow
    run = Flow('Main').latest_successful_run
    print(run)


if __name__ == "__main__":
    PastRunInspection()
