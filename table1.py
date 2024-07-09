import pandas as pd
import pathlib


def main():
    results_dir = pathlib.Path("results")
    results = []
    for file in results_dir.glob("*/*/*"):
        result = pd.read_csv(file)
        result["random_seed"] = int(file.stem)
        result["method"] = file.parent.stem
        result["dataset"] = file.parents[1].stem
        results.append(result)

    results = pd.concat(results)

    print(results.groupby(["dataset", "method"]).mean().drop("random_seed", axis=1))



if __name__ == '__main__':
    main()