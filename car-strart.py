from car import Car

c = Car('마이카', 15, 60)  # 이름, 연비, 연료탱크량
c.print()

c.chargeFuel(47)
c.print()

c.drive(100)
c.print()