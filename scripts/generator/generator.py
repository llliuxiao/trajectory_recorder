import gen_world_ca
import datetime
import random


def main():
    counter = 500
    set_size = 1000
    while counter < set_size:
        print(f'___________________________world{counter}___________________________')
        rows = 30 + random.randint(0, 20)
        cols = 30 + random.randint(0, 70)
        fill_pict = 0.17 + random.uniform(0.0, 0.03)
        result = gen_world_ca.main(counter, hash(datetime.datetime.now()),
                                   rows=rows, cols=cols, fill_pct=fill_pict, show_metrics=0)
        if result:
            counter += 1


if __name__ == "__main__":
    main()
