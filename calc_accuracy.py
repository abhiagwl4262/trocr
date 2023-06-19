import Levenshtein

def main():
    return

if __name__ == '__main__':
    result_path = "custom_data/results.txt"
    f = open(result_path, "r")
    gts   = []
    preds = []
    for line in f.readlines():
        line =  line.strip("\n")
        
        gt   = line.split("\t")[0].split("_")[0]
        pred = line.split("\t")[-1]
        print(Levenshtein.distance(gt, pred))
        gts.append(gt)
        preds.append(pred)
