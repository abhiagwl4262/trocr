from Levenshtein import ratio as lev_ratio
import multiprocessing as mp

def main(gt, pred, result):
    ratio = lev_ratio(gt, pred)
    if ratio == 1:
        result["100%"] = result.get("100%", 0) + 1
    elif ratio >= .9 and ratio < 1:
        result["90% - <100%"] = result.get("90% - <100%", 0) + 1
    elif ratio >= .8 and ratio < .9:
        result["80% - <90%"] = result.get("80% - <90%", 0) + 1
    elif ratio >= .7 and ratio < .8:
        result["70% - <80%"] = result.get("70% - <80%", 0) + 1
    elif ratio >= .6 and ratio < .7:
        result["60% - <70%"] = result.get("60% - <70%", 0) + 1
    elif ratio >= .5 and ratio < .6:
        result["50% - <60%"] = result.get("50% - <60%", 0) + 1
    else:
        result["<50%"] = result.get("<50%", 0) + 1
    return result


if __name__ == '__main__':
    result_path = "data/name_insi/results.txt"
    gts   = []
    preds = []
    result = {}
    for line in open(result_path).readlines():
        line =  line.strip("\n")
        gt   = line.split("\t")[0].split("_")[0].lower()
        pred = line.split("\t")[-1].lower()
        print("gt:", gt, " | ", "pred:", pred)
        result = main(gt, pred, result)
        gts.append(gt)
        preds.append(pred)
    result = {k:round(v/len(gts), 2)*100 for k, v in result.items()}
    print("total eval samples:", len(gts))
    print(result)