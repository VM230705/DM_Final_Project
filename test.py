import csv

fin = open('mining.csv', encoding='utf-8-sig', errors='ignore')
csv_reader = csv.reader(fin)
next(csv_reader)

total_cnt = 0
drop_cnt = 0
for row in csv_reader:
    total_cnt += 1
    try:
        missing = (float(row[3])*float(row[4]) < 1e-6)
    except:
        drop_cnt += 1
        continue
    drop_cnt += missing
    # print(f"{total_cnt}: {missing}")
    # if total_cnt >= 20:
    #   break
print(f"Drop rows: {drop_cnt}\nTotal rows: {total_cnt}\nDrop ratio: {drop_cnt/total_cnt}")

