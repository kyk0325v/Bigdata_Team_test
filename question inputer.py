import random
import time


w = ["A bad workman blames his tools.", "A bird in the hand is worth two in the bush.", "A stitch in time saves nine.", "After death, to call the doctor.", "All is well that ends well.", "All that glitteres is not gold.", "Absence makes the heart grow fonder.", "Action speak louder than words.", "After a storm comes a calm.","All is fair in love and war."]

n = 1

print("[타자 게임] 준비되면 엔터!")
input()

start = time.time()
q = random.choice(w)

while n <= 5:
    print("*문제", n)
    print(q)
    x = input()
    if q == x:
        print("통과!")
        n = n + 1
        q = random.choice(w)
    else:
        print("오타! 다시도전!")

end = time.time()
et = end - start
et = format(et, ".2f")

print("타자시간:", et, "초")