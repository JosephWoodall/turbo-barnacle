class PastRunInspection:
    """
    The PastRunInspection will return the last successful run for the user's namespace.
    """
    from metaflow import Flow
    run = Flow('Main').latest_successful_run
    print(run)


class PastRunStartCardInspection:
    """
    The PastRunStartCardInspection will return the card inspection for the last successful run for the user's namespace.
    """
    import os
    os.system('python3 main.py card view start')


if __name__ == "__main__":
    PastRunStartCardInspection()
