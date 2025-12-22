from time import time


class Reporter:
    def report_dict(self, data: dict, desc: str = "", inline: bool = True):
        if inline:
            if desc:
                print(f"[{desc}]", end="")
            print(" | ".join(f"{key}: {value}" for key, value in data.items()))
            return

        if desc:
            print("\n" + "=" * 20)
            print(desc)
            print("=" * 20)
        for key, value in data.items():
            print(f"{key}\t: {value}")
        print("-" * 20 + "\n")

    def start_timing(self, task_name: str, report_start: bool = False):
        self.task_name = task_name
        self.start_time = time()
        if report_start:
            print(f"[{self.task_name}] Started...")
    
    def end_timing(self):
        if self.start_time is None:
            print("[ERROR] Timer was not started.")
            return
        elapsed_time = time() - self.start_time
        print(f"[{self.task_name}] Completed in {elapsed_time:.2f} seconds")
        self.start_time = None
